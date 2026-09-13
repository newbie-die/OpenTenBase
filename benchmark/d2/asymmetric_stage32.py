#!/usr/bin/env python3
"""D2-v8 asymmetric Stage32 risk control (offline only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import loss_budgeted as v6
import progressive_shadow as shadow
import risk_ranked as v7

CAL_IDS = np.arange(500)
DEV_IDS = np.arange(500, 1000)
EPS = 1e-12
V7_AVERAGE_PROBES = 50.368
V7_RECALL_DELTA = -0.0054
STAGE16_SPEC = {"max_depth": 1, "n_estimators": 50, "learning_rate": 0.1}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def fit_stage16_oof(x, labels):
    probabilities = np.zeros((500, 3))
    for fold in range(5):
        valid = CAL_IDS[CAL_IDS % 5 == fold]; train = CAL_IDS[CAL_IDS % 5 != fold]
        model = v7.build_model("gradient_boosting", STAGE16_SPEC)
        model.fit(x[train], labels[train])
        probabilities[valid] = v7.aligned_probabilities(model, x[valid])
    return probabilities


def fit_stage16_development(x, labels):
    model = v7.build_model("gradient_boosting", STAGE16_SPEC)
    model.fit(x[:500], labels[:500])
    return model, v7.aligned_probabilities(model, x[500:])


def stage32_specs(family):
    if family == "logistic_regression":
        return [{"C": c, "unsafe_weight": weight} for c in (0.1, 1.0, 10.0)
                for weight in (2.0, 4.0, 8.0)]
    if family == "gradient_boosting":
        return [{"max_depth": depth, "n_estimators": estimators,
                 "learning_rate": rate, "unsafe_weight": weight}
                for depth in (1, 2) for estimators in (50, 100)
                for rate in (0.05, 0.1) for weight in (2.0, 4.0, 8.0)]
    if family == "random_forest":
        return [{"max_depth": depth, "min_samples_leaf": leaf,
                 "n_estimators": 150, "unsafe_weight": weight}
                for depth in (6, None) for leaf in (5, 15)
                for weight in (2.0, 4.0, 8.0)]
    raise ValueError(family)


def build_stage32(family, spec):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    weight = {0: 1.0, 1: spec["unsafe_weight"]}
    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=spec["C"], class_weight=weight, max_iter=3000, random_state=20260913))
    if family == "gradient_boosting":
        return GradientBoostingClassifier(max_depth=spec["max_depth"],
            n_estimators=spec["n_estimators"], learning_rate=spec["learning_rate"],
            random_state=20260913)
    return RandomForestClassifier(max_depth=spec["max_depth"],
        min_samples_leaf=spec["min_samples_leaf"], n_estimators=spec["n_estimators"],
        class_weight=weight, n_jobs=-1, random_state=20260913)


def fit_binary(model, family, x, labels, unsafe_weight):
    if family == "gradient_boosting":
        sample_weight = np.where(labels == 1, unsafe_weight, 1.0)
        model.fit(x, labels, sample_weight=sample_weight)
    else:
        model.fit(x, labels)
    return model


def unsafe_probability(model, x):
    probabilities = model.predict_proba(x)
    column = list(model.classes_).index(1)
    return probabilities[:, column]


def stage32_oof(family, spec, x, labels):
    output = np.zeros(500)
    for fold in range(5):
        valid = CAL_IDS[CAL_IDS % 5 == fold]; train = CAL_IDS[CAL_IDS % 5 != fold]
        model = build_stage32(family, spec)
        fit_binary(model, family, x[train], labels[train], spec["unsafe_weight"])
        output[valid] = unsafe_probability(model, x[valid])
    return output


def logit(probability):
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_calibrator(method, raw_oof, labels):
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1e6, max_iter=2000, random_state=20260913)
        model.fit(logit(raw_oof).reshape(-1, 1), labels)
        return model
    from sklearn.isotonic import IsotonicRegression
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_oof, labels)
    return model


def calibrate(calibrator, method, raw):
    if method == "platt":
        return calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]
    return calibrator.predict(raw)


def ece(labels, probabilities, bins=10):
    total = len(labels); value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (probabilities <= high if index == bins - 1 else probabilities < high)
        if mask.any():
            value += mask.sum() / total * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return float(value)


def unsafe_metrics(labels, probabilities, safe_threshold):
    from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix
    from sklearn.metrics import precision_score, recall_score, roc_auc_score
    # A Stage32 downgrade is allowed at p(unsafe) <= safe_threshold.
    predicted_unsafe = probabilities > safe_threshold
    return {
        "safe_probability_threshold": safe_threshold,
        "precision": float(precision_score(labels, predicted_unsafe, zero_division=0)),
        "recall": float(recall_score(labels, predicted_unsafe, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "expected_calibration_error_10bin": ece(labels, probabilities),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predicted_unsafe, labels=[0, 1]).tolist(),
    }


def thresholds(values, count=51):
    return sorted(set([-math.inf, math.inf] +
                      [float(value) for value in np.quantile(values, np.linspace(0, 1, count))]))


def actions_for(stage16_risk, unsafe32_probability, stage16_threshold, stage32_threshold):
    return np.where(stage16_risk <= stage16_threshold, 16,
                    np.where(unsafe32_probability <= stage32_threshold, 32, 64)).astype(int)


def policy_summary(ids, actions, loss16, loss32, recall64):
    actual = v6.selected_values(actions, loss16[ids], loss32[ids])
    total = float(actual.sum()); average = float(actions.mean())
    fold_deltas = {}
    for fold in range(5):
        local = np.asarray(ids) % 5 == fold
        fold_deltas[str(fold)] = -float(actual[local].mean())
    return {
        "queries": len(ids), "average_probes": average,
        "probe_reduction": 1.0 - average / 64.0,
        "probe_distribution": {str(probe): int((actions == probe).sum()) for probe in shadow.PROBES},
        "fixed64_mean_recall": float(recall64[ids].mean()),
        "mean_recall": float(recall64[ids].mean() - total / len(ids)),
        "recall_delta": -total / len(ids), "actual_total_recall_loss": total,
        "fold_recall_deltas": fold_deltas,
        "false_downgrade": {
            "lossy_action16": int(((actions == 16) & (loss16[ids] > EPS)).sum()),
            "unsafe_action32": int(((actions == 32) & (loss32[ids] > EPS)).sum()),
            "any_positive_selected_loss": int((actual > EPS).sum()),
            "selected_loss_at_least_0_2": int((actual >= 0.2 - EPS).sum()),
        },
    }


def tune_thresholds(stage16_risk, unsafe_probability, loss16, loss32, recall64):
    best = None
    for threshold16 in thresholds(stage16_risk):
        for threshold32 in thresholds(unsafe_probability):
            actions = actions_for(stage16_risk, unsafe_probability, threshold16, threshold32)
            actual = v6.selected_values(actions, loss16[:500], loss32[:500])
            delta = -float(actual.mean())
            fold_deltas = [-float(actual[CAL_IDS % 5 == fold].mean()) for fold in range(5)]
            if delta < -0.0025 - EPS or min(fold_deltas) < -0.005 - EPS:
                continue
            average = float(actions.mean())
            false = int((actual > EPS).sum()); severe = int((actual >= 0.2 - EPS).sum())
            # Joint movement of these thresholds performs risk-ranked swaps:
            # revoke marginal p16 actions while admitting lower-risk p32 actions.
            key = (average, severe, false, float(actual.sum()), -min(fold_deltas), threshold16, threshold32)
            if best is None or key < best[0]:
                best = (key, float(threshold16), float(threshold32), actions)
    if best is None:
        raise RuntimeError("fixed64 should always satisfy the calibration constraints")
    _, threshold16, threshold32, actions = best
    return threshold16, threshold32, policy_summary(CAL_IDS, actions, loss16, loss32, recall64)


def model_importance(model, names):
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.abs(estimator.coef_[0])
    return sorted(({"feature": name, "importance": float(value)} for name, value in zip(names, values)),
                  key=lambda row: row["importance"], reverse=True)


def tune_family(family, x32, feature_names, unsafe_labels, stage16_risk,
                loss16, loss32, recall64):
    best = None
    for spec in stage32_specs(family):
        raw_oof = stage32_oof(family, spec, x32, unsafe_labels)
        for method in ("platt", "isotonic"):
            calibrator = fit_calibrator(method, raw_oof, unsafe_labels[:500])
            calibrated = calibrate(calibrator, method, raw_oof)
            threshold16, threshold32, policy = tune_thresholds(
                stage16_risk, calibrated, loss16, loss32, recall64)
            metrics = unsafe_metrics(unsafe_labels[:500], calibrated, threshold32)
            key = (policy["average_probes"], policy["false_downgrade"]["selected_loss_at_least_0_2"],
                   policy["false_downgrade"]["any_positive_selected_loss"],
                   -metrics["recall"], metrics["expected_calibration_error_10bin"],
                   json.dumps(spec, sort_keys=True), method)
            if best is None or key < best[0]:
                best = (key, spec, method, calibrator, threshold16, threshold32, policy, metrics)
    _, spec, method, calibrator, threshold16, threshold32, policy, metrics = best
    model = build_stage32(family, spec)
    fit_binary(model, family, x32[:500], unsafe_labels[:500], spec["unsafe_weight"])
    raw_development = unsafe_probability(model, x32[500:])
    development_probability = calibrate(calibrator, method, raw_development)
    return {
        "family": family, "spec": spec, "calibration_method": method,
        "thresholds": {"stage16_risk": threshold16, "stage32_max_unsafe_probability": threshold32},
        "calibration_oof_policy": policy, "calibration_oof_unsafe32": metrics,
        "model": model, "calibrator": calibrator,
        "development_probability": development_probability,
        "feature_importance": model_importance(model, feature_names),
    }


def gate(metrics):
    recall_ok = metrics["recall_delta"] >= -0.005 - EPS
    if recall_ok and metrics["average_probes"] <= 45:
        return "Strong Go"
    if recall_ok and metrics["average_probes"] <= 50:
        return "Go"
    if ((recall_ok and metrics["average_probes"] <= 53) or
            (metrics["average_probes"] <= 50 and metrics["recall_delta"] >= -0.010 - EPS)):
        return "Marginal"
    return "No-Go"


def analyze(args):
    raw = shadow.read_raw(args.raw); old = shadow.read_centroids(args.old_features)
    geometry = v7.read_geometry(args.geometry); features = v7.build_features(raw, old, geometry)
    x16, x32 = features["extended16"], features["extended32"]
    loss16, loss32, recall64 = v6.losses_from_raw(raw)
    lost16 = v7.lost_neighbor_classes(loss16)
    unsafe32 = (loss32 > EPS).astype(int)

    stage16_oof_probability = fit_stage16_oof(x16, lost16)
    stage16_oof_risk = v7.risk_score(stage16_oof_probability, 2.0)
    tuned = []
    for family in ("logistic_regression", "gradient_boosting", "random_forest"):
        print(f"calibrating asymmetric Stage32 family={family}", flush=True)
        tuned.append(tune_family(family, x32, features["extended32_names"], unsafe32,
                                 stage16_oof_risk, loss16, loss32, recall64))

    # Development set evaluation starts only after all OOF choices are locked.
    stage16_model, stage16_development_probability = fit_stage16_development(x16, lost16)
    stage16_development_risk = v7.risk_score(stage16_development_probability, 2.0)
    results = []
    for item in tuned:
        probability = item.pop("development_probability")
        thresholds_value = item["thresholds"]
        actions = actions_for(stage16_development_risk, probability,
                              thresholds_value["stage16_risk"],
                              thresholds_value["stage32_max_unsafe_probability"])
        item["development_policy"] = policy_summary(DEV_IDS, actions, loss16, loss32, recall64)
        item["development_unsafe32"] = unsafe_metrics(unsafe32[500:], probability,
            thresholds_value["stage32_max_unsafe_probability"])
        item["gate"] = gate(item["development_policy"])
        item.pop("model"); item.pop("calibrator"); results.append(item)

    compliant = [item for item in results if item["development_policy"]["recall_delta"] >= -0.005 - EPS]
    best = (min(compliant, key=lambda item: (item["development_policy"]["average_probes"],
                                             item["development_policy"]["false_downgrade"]["any_positive_selected_loss"]))
            if compliant else max(results, key=lambda item: (item["development_policy"]["recall_delta"],
                                                              -item["development_policy"]["average_probes"])))
    best_metrics = best["development_policy"]
    improvement = {
        "average_probes_reduction": V7_AVERAGE_PROBES - best_metrics["average_probes"],
        "recall_delta_improvement": best_metrics["recall_delta"] - V7_RECALL_DELTA,
        "probe_reduction_change": best_metrics["probe_reduction"] - (1.0 - V7_AVERAGE_PROBES / 64.0),
    }
    report = {
        "status": "COMPLETE", "set_naming": {"calibration": [0, 499], "development": [500, 999]},
        "stage16_frozen_model": {"family": "gradient_boosting_multiclass", "spec": STAGE16_SPEC,
                                 "risk": "P(loss=1) + 2*P(loss>=2)"},
        "calibration_constraints": {"mean_recall_delta_min": -0.0025,
                                    "each_qid_mod_5_fold_recall_delta_min": -0.005},
        "models": results,
        "best_model": {"family": best["family"], "spec": best["spec"],
                       "calibration_method": best["calibration_method"],
                       "thresholds": best["thresholds"],
                       "calibration_oof_policy": best["calibration_oof_policy"],
                       "calibration_oof_unsafe32": best["calibration_oof_unsafe32"],
                       "development_policy": best_metrics,
                       "development_unsafe32": best["development_unsafe32"],
                       "feature_importance": best["feature_importance"], "gate": best["gate"]},
        "comparison_to_d2_v7": {"d2_v7": {"average_probes": V7_AVERAGE_PROBES,
                                             "recall_delta": V7_RECALL_DELTA},
                                 "d2_v8": {"average_probes": best_metrics["average_probes"],
                                             "recall_delta": best_metrics["recall_delta"]},
                                 "improvement": improvement},
        "go_achieved": best["gate"] in ("Go", "Strong Go"),
    }
    atomic_json(args.output, report)
    lines = ["# D2-v8 asymmetric Stage32 risk control", "",
             "Qids 500–999 are reported as the development set. All models, probability calibration, and thresholds were selected from qids 0–499 OOF predictions.", "",
             "| Stage32 model | Calibration | OOF avg/delta | Development avg/delta | 16/32/64 | Unsafe32 P/R | ROC-AUC | PR-AUC | ECE | False downgrade | Gate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in results:
        cal = item["calibration_oof_policy"]; dev = item["development_policy"]
        dist = dev["probe_distribution"]; metric = item["development_unsafe32"]
        lines.append(f"| {item['family']} | {item['calibration_method']} | {cal['average_probes']:.3f}/{cal['recall_delta']:+.4f} | {dev['average_probes']:.3f}/{dev['recall_delta']:+.4f} | {dist['16']}/{dist['32']}/{dist['64']} | {metric['precision']:.3f}/{metric['recall']:.3f} | {metric['roc_auc']:.3f} | {metric['pr_auc']:.3f} | {metric['expected_calibration_error_10bin']:.4f} | {dev['false_downgrade']['any_positive_selected_loss']} | {item['gate']} |")
    lines += ["", f"Best model: **{best['family']}**, gate **{best['gate']}**.", "",
              f"Versus D2-v7, average probes changes by {-improvement['average_probes_reduction']:+.3f} and Recall delta improves by {improvement['recall_delta_improvement']:+.4f}.", "",
              "Full fold constraints, confusion matrices, Brier scores, thresholds, and feature importance are in `summary.json`.", ""]
    args.output.with_name("report.md").write_text("\n".join(lines))
    print(json.dumps({"best_model": best["family"], "gate": best["gate"],
                      "average_probes": best_metrics["average_probes"],
                      "recall_delta": best_metrics["recall_delta"],
                      "distribution": best_metrics["probe_distribution"],
                      "vs_d2_v7": improvement}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, default=shadow.CENTROID_FEATURES)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
