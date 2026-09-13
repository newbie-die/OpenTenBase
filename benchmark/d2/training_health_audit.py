#!/usr/bin/env python3
"""Training health audit for the locked D2-v8 GradientBoosting policy."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np

import asymmetric_stage32 as v8
import loss_budgeted as v6
import progressive_shadow as shadow
import risk_ranked as v7

CAL_IDS = np.arange(500)
CURRENT_STAGE32_SPEC = {
    "learning_rate": 0.05, "max_depth": 1, "n_estimators": 50,
    "unsafe_weight": 4.0,
}
CURRENT_THRESHOLDS = {
    "stage16_risk": 0.28952659772434647,
    "stage32_max_unsafe_probability": 0.11764705882352941,
}
PATIENCE = 20
MIN_DELTA = 1e-4
MAX_ESTIMATORS = 500
EPS = 1e-12


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_inputs(args):
    raw = shadow.read_raw(args.raw)
    old = shadow.read_centroids(args.old_features)
    geometry = v7.read_geometry(args.geometry)
    features = v7.build_features(raw, old, geometry)
    loss16, loss32, recall64 = v6.losses_from_raw(raw)
    lost16 = v7.lost_neighbor_classes(loss16)
    unsafe32 = (loss32 > EPS).astype(int)
    return features, loss16, loss32, recall64, lost16, unsafe32


def fit_curve_model(x, labels):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(max_depth=CURRENT_STAGE32_SPEC["max_depth"],
        n_estimators=MAX_ESTIMATORS, learning_rate=CURRENT_STAGE32_SPEC["learning_rate"],
        random_state=20260913)
    weights = np.where(labels == 1, CURRENT_STAGE32_SPEC["unsafe_weight"], 1.0)
    model.fit(x, labels, sample_weight=weights)
    return model


def learning_curves(x, labels):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    rows = []
    summaries = []
    for fold in range(5):
        print(f"learning_curve fold={fold + 1}/5", flush=True)
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        model = fit_curve_model(x[train], labels[train])
        best_loss = math.inf; best_iteration = 0; wait = 0; stopped = MAX_ESTIMATORS
        best_train_loss = None; loss_at_50 = None
        for iteration, (train_probability, validation_probability) in enumerate(zip(
                model.staged_predict_proba(x[train]), model.staged_predict_proba(x[validation])), 1):
            train_loss = float(log_loss(labels[train], train_probability, labels=[0, 1]))
            validation_loss = float(log_loss(labels[validation], validation_probability, labels=[0, 1]))
            roc_auc = float(roc_auc_score(labels[validation], validation_probability[:, 1]))
            pr_auc = float(average_precision_score(labels[validation], validation_probability[:, 1]))
            rows.append({"fold": fold, "n_estimators": iteration,
                         "train_logloss": train_loss, "validation_logloss": validation_loss,
                         "roc_auc": roc_auc, "pr_auc": pr_auc})
            if iteration == 50:
                loss_at_50 = validation_loss
            if validation_loss < best_loss - MIN_DELTA:
                best_loss = validation_loss; best_iteration = iteration
                best_train_loss = train_loss; wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                stopped = iteration
                break
        final = rows[-1]
        summaries.append({
            "fold": fold, "train_queries": len(train), "validation_queries": len(validation),
            "best_n_estimators": best_iteration, "early_stop_iteration": stopped,
            "best_validation_logloss": best_loss,
            "validation_logloss_at_50": loss_at_50,
            "validation_gain_50_to_best": (loss_at_50 - best_loss) if loss_at_50 is not None else None,
            "train_logloss_at_best": best_train_loss,
            "train_logloss_at_stop": final["train_logloss"],
            "validation_logloss_at_stop": final["validation_logloss"],
            "post_best_train_improvement": best_train_loss - final["train_logloss"],
            "post_best_validation_worsening": final["validation_logloss"] - best_loss,
            "roc_auc_at_stop": final["roc_auc"], "pr_auc_at_stop": final["pr_auc"],
        })
    return rows, summaries


def current_oof(x16, x32, lost16, unsafe32):
    stage16_probability = v8.fit_stage16_oof(x16, lost16)
    stage16_risk = v7.risk_score(stage16_probability, 2.0)
    raw_unsafe = v8.stage32_oof("gradient_boosting", CURRENT_STAGE32_SPEC, x32, unsafe32)
    calibrator = v8.fit_calibrator("isotonic", raw_unsafe, unsafe32[:500])
    calibrated_unsafe = v8.calibrate(calibrator, "isotonic", raw_unsafe)
    return stage16_risk, raw_unsafe, calibrated_unsafe


def subset_policy(ids, stage16_risk, unsafe_probability, loss16, loss32, recall64,
                  thresholds=CURRENT_THRESHOLDS):
    actions = v8.actions_for(stage16_risk[ids], unsafe_probability[ids],
        thresholds["stage16_risk"], thresholds["stage32_max_unsafe_probability"])
    actual = v6.selected_values(actions, loss16[ids], loss32[ids])
    total = float(actual.sum())
    return {
        "queries": len(ids), "average_probes": float(actions.mean()),
        "recall_delta": -total / len(ids), "actual_total_recall_loss": total,
        "probe_distribution": {str(probe): int((actions == probe).sum()) for probe in shadow.PROBES},
        "probe_reduction": 1.0 - float(actions.mean()) / 64.0,
    }, actions


def fold_stability(stage16_risk, calibrated_unsafe, unsafe32,
                   loss16, loss32, recall64):
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
    rows = []
    threshold = CURRENT_THRESHOLDS["stage32_max_unsafe_probability"]
    for fold in range(5):
        ids = CAL_IDS[CAL_IDS % 5 == fold]
        policy, _ = subset_policy(ids, stage16_risk, calibrated_unsafe,
                                  loss16, loss32, recall64)
        labels = unsafe32[ids]; probability = calibrated_unsafe[ids]
        prediction = probability > threshold
        rows.append({
            "fold": fold, **policy,
            "unsafe32_precision": float(precision_score(labels, prediction, zero_division=0)),
            "unsafe32_recall": float(recall_score(labels, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probability)),
            "pr_auc": float(average_precision_score(labels, probability)),
        })
    fields = ("average_probes", "recall_delta", "unsafe32_precision",
              "unsafe32_recall", "roc_auc", "pr_auc")
    aggregate = {}
    for field in fields:
        values = [row[field] for row in rows]
        aggregate[field] = {"mean": statistics.fmean(values),
                            "std": statistics.pstdev(values),
                            "min": min(values), "max": max(values)}
    return rows, aggregate


def local_threshold_search(ids, stage16_risk, unsafe_probability,
                           loss16, loss32):
    best = None
    for threshold16 in v8.thresholds(stage16_risk[ids], count=31):
        for threshold32 in v8.thresholds(unsafe_probability[ids], count=31):
            actions = v8.actions_for(stage16_risk[ids], unsafe_probability[ids],
                                     threshold16, threshold32)
            actual = v6.selected_values(actions, loss16[ids], loss32[ids])
            delta = -float(actual.mean())
            if delta < -0.0025 - EPS:
                continue
            key = (float(actions.mean()), int((actual >= 0.2 - EPS).sum()),
                   int((actual > EPS).sum()), float(actual.sum()), threshold16, threshold32)
            if best is None or key < best[0]:
                best = (key, float(threshold16), float(threshold32), delta,
                        float(actions.mean()))
    if best is None:
        raise RuntimeError("fixed64 should make local threshold search feasible")
    return {"stage16_threshold": best[1], "unsafe32_threshold": best[2],
            "fold_local_recall_delta": best[3], "fold_local_average_probes": best[4]}


def threshold_stability(stage16_risk, calibrated_unsafe, loss16, loss32):
    rows = []
    for fold in range(5):
        ids = CAL_IDS[CAL_IDS % 5 == fold]
        rows.append({"fold": fold, **local_threshold_search(
            ids, stage16_risk, calibrated_unsafe, loss16, loss32)})
    aggregate = {}
    for field in ("stage16_threshold", "unsafe32_threshold"):
        values = [row[field] for row in rows]
        aggregate[field] = {"mean": statistics.fmean(values),
                            "std": statistics.pstdev(values),
                            "min": min(values), "max": max(values),
                            "range": max(values) - min(values)}
    return rows, aggregate


def deterministic_nested_train_ids(fold, fraction):
    candidates = CAL_IDS[CAL_IDS % 5 != fold]
    # Stable hash order provides nested subsets without random resampling.
    ordered = sorted((int(qid) for qid in candidates),
                     key=lambda qid: ((qid * 2654435761 + 20260913 + fold) & 0xffffffff, qid))
    return np.asarray(ordered[:round(len(ordered) * fraction)], dtype=int)


def learning_size_curve(x32, unsafe32, stage16_risk,
                        loss16, loss32, recall64):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        raw_oof = np.zeros(500)
        counts = []
        for fold in range(5):
            validation = CAL_IDS[CAL_IDS % 5 == fold]
            train = deterministic_nested_train_ids(fold, fraction)
            counts.append(len(train))
            model = v8.build_stage32("gradient_boosting", CURRENT_STAGE32_SPEC)
            v8.fit_binary(model, "gradient_boosting", x32[train], unsafe32[train],
                          CURRENT_STAGE32_SPEC["unsafe_weight"])
            raw_oof[validation] = v8.unsafe_probability(model, x32[validation])
        calibrator = v8.fit_calibrator("isotonic", raw_oof, unsafe32[:500])
        calibrated = v8.calibrate(calibrator, "isotonic", raw_oof)
        policy, _ = subset_policy(CAL_IDS, stage16_risk, calibrated,
                                  loss16, loss32, recall64)
        rows.append({
            "training_fraction": fraction,
            "training_queries_per_fold": counts,
            "roc_auc": float(roc_auc_score(unsafe32[:500], raw_oof)),
            "pr_auc": float(average_precision_score(unsafe32[:500], raw_oof)),
            "calibrated_ece": v8.ece(unsafe32[:500], calibrated),
            **policy,
        })
    return rows


def health_classification(curves, size_curve):
    best = [row["best_n_estimators"] for row in curves]
    gains = [row["validation_gain_50_to_best"] for row in curves
             if row["validation_gain_50_to_best"] is not None]
    worsening = [row["post_best_validation_worsening"] for row in curves]
    train_improvement = [row["post_best_train_improvement"] for row in curves]
    last_auc_gain = size_curve[-1]["roc_auc"] - size_curve[-2]["roc_auc"]
    last_pr_gain = size_curve[-1]["pr_auc"] - size_curve[-2]["pr_auc"]
    facts = {
        "median_best_n_estimators": statistics.median(best),
        "mean_validation_gain_50_to_best": statistics.fmean(gains) if gains else None,
        "mean_post_best_validation_worsening": statistics.fmean(worsening),
        "mean_post_best_train_improvement": statistics.fmean(train_improvement),
        "size_80_to_100_roc_auc_gain": last_auc_gain,
        "size_80_to_100_pr_auc_gain": last_pr_gain,
    }
    if (facts["median_best_n_estimators"] >= 80 and
            facts["mean_validation_gain_50_to_best"] is not None and
            facts["mean_validation_gain_50_to_best"] >= 0.005):
        category = "A. Undertrained"
    elif (facts["median_best_n_estimators"] <= 50 and
          facts["mean_post_best_validation_worsening"] >= 0.002 and
          facts["mean_post_best_train_improvement"] >= 0.002):
        category = "B. Overfitting"
    elif last_auc_gain >= 0.01 or last_pr_gain >= 0.015:
        category = "C. Data-limited"
    else:
        category = "D. Feature/model-limited"
    return category, facts


def analyze(args):
    features, loss16, loss32, recall64, lost16, unsafe32 = load_inputs(args)
    x16, x32 = features["extended16"], features["extended32"]
    curve_rows, curve_summary = learning_curves(x32, unsafe32[:500])

    stage16_risk, raw_unsafe, calibrated_unsafe = current_oof(
        x16, x32, lost16, unsafe32)
    current_policy, _ = subset_policy(CAL_IDS, stage16_risk, calibrated_unsafe,
                                      loss16, loss32, recall64)
    if (abs(current_policy["average_probes"] - 49.6) > EPS or
            abs(current_policy["recall_delta"] + 0.0022) > EPS):
        raise RuntimeError(f"failed to reproduce locked D2-v8 OOF policy: {current_policy}")
    folds, fold_aggregate = fold_stability(stage16_risk, calibrated_unsafe,
                                           unsafe32, loss16, loss32, recall64)
    threshold_rows, threshold_aggregate = threshold_stability(
        stage16_risk, calibrated_unsafe, loss16, loss32)
    size_curve = learning_size_curve(x32, unsafe32, stage16_risk,
                                     loss16, loss32, recall64)
    category, facts = health_classification(curve_summary, size_curve)

    with args.output.with_name("learning_curve.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=curve_rows[0].keys())
        writer.writeheader(); writer.writerows(curve_rows)
    report = {
        "status": "COMPLETE", "scope": "calibration qids 0..499 only",
        "policy_modified": False, "database_run": False,
        "locked_policy": {"stage16": {"spec": v8.STAGE16_SPEC,
                                        "risk": "P(loss=1)+2*P(loss>=2)"},
                          "stage32": {"spec": CURRENT_STAGE32_SPEC,
                                      "calibration": "isotonic"},
                          "thresholds": CURRENT_THRESHOLDS},
        "learning_curve": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                           "max_estimators": MAX_ESTIMATORS,
                           "folds": curve_summary},
        "fold_stability": {"folds": folds, "aggregate": fold_aggregate},
        "threshold_stability": {"definition": "fold-local diagnostic optimum under delta >= -0.0025; not applied to policy",
                                "folds": threshold_rows, "aggregate": threshold_aggregate},
        "learning_size_curve": {"definition": "nested deterministic training subsets; fixed 50-tree model and locked thresholds",
                                "rows": size_curve},
        "diagnosis": {"category": category, "facts": facts},
    }
    atomic_json(args.output, report)
    lines = ["# D2-v8 GradientBoosting training health audit", "",
             f"Diagnosis: **{category}**.", "",
             "## Early stopping", "",
             "| Fold | Best trees | Stop | Train loss@best | Validation loss@best | Validation gain 50→best | Post-best validation worsening |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in curve_summary:
        gain = row["validation_gain_50_to_best"]
        lines.append(f"| {row['fold']} | {row['best_n_estimators']} | {row['early_stop_iteration']} | {row['train_logloss_at_best']:.4f} | {row['best_validation_logloss']:.4f} | {gain if gain is not None else float('nan'):+.4f} | {row['post_best_validation_worsening']:+.4f} |")
    lines += ["", "## Fold stability", "",
              "| Fold | Avg probes | Recall delta | Unsafe P/R | ROC-AUC | PR-AUC |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in folds:
        lines.append(f"| {row['fold']} | {row['average_probes']:.3f} | {row['recall_delta']:+.4f} | {row['unsafe32_precision']:.3f}/{row['unsafe32_recall']:.3f} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} |")
    lines += ["", "## Learning-size curve", "",
              "| Training size | ROC-AUC | PR-AUC | Avg probes | Recall delta |",
              "|---:|---:|---:|---:|---:|"]
    for row in size_curve:
        lines.append(f"| {row['training_fraction']:.0%} | {row['roc_auc']:.3f} | {row['pr_auc']:.3f} | {row['average_probes']:.3f} | {row['recall_delta']:+.4f} |")
    lines += ["", "Full per-iteration curves, aggregate stability statistics, and fold-local thresholds are in the artifacts.", ""]
    args.output.with_name("report.md").write_text("\n".join(lines))
    print(json.dumps({"diagnosis": category,
                      "best_n_estimators": [row["best_n_estimators"] for row in curve_summary],
                      "fold_recall_delta_range": [fold_aggregate["recall_delta"]["min"],
                                                  fold_aggregate["recall_delta"]["max"]],
                      "size_80_to_100_auc_gain": facts["size_80_to_100_roc_auc_gain"],
                      "size_80_to_100_pr_gain": facts["size_80_to_100_pr_auc_gain"]}, indent=2))


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
