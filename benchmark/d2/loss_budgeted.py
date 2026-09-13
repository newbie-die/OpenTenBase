#!/usr/bin/env python3
"""D2-v6 loss-budgeted adaptive probing study (offline only)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

import progressive_shadow as shadow

EPS = 1e-12
CALIBRATION_IDS = np.arange(500)
TEST_IDS = np.arange(500, 1000)
SEVERE_LOSS = 0.2


def atomic_json(path: Path, value):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def losses_from_raw(raw):
    loss16, loss32, recall64 = [], [], []
    for qid in range(shadow.QUERY_COUNT):
        baseline = raw[(qid, 64)]["recall"]
        loss16.append(baseline - raw[(qid, 16)]["recall"])
        loss32.append(baseline - raw[(qid, 32)]["recall"])
        recall64.append(baseline)
    return np.asarray(loss16), np.asarray(loss32), np.asarray(recall64)


def budgeted_oracle(ids, loss16, loss32, recall64, budget_per_query=0.005):
    """Exact Pareto-frontier DP; Recall@10 losses are integer tenths."""
    # state value: total probes, count16, count32, count64, exact float loss
    frontier = {0: (0, 0, 0, 0, 0.0)}
    for qid in ids:
        choices = ((16, loss16[qid], 0), (32, loss32[qid], 1), (64, 0.0, 2))
        expanded = {}
        for cumulative_units, state in frontier.items():
            for probe, loss, slot in choices:
                units = int(round(float(loss) * 10.0))
                if abs(units / 10.0 - loss) > 1e-9:
                    raise ValueError(f"loss is not a Recall@10 tenth: qid={qid} loss={loss}")
                key = cumulative_units + units
                counts = [state[1], state[2], state[3]]
                counts[slot] += 1
                candidate = (state[0] + probe, counts[0], counts[1], counts[2], state[4] + float(loss))
                prior = expanded.get(key)
                if prior is None or (candidate[0], candidate[4]) < (prior[0], prior[4]):
                    expanded[key] = candidate
        # A state is dominated if another uses no more loss and no more probes.
        frontier = {}
        best_cost = math.inf
        for units in sorted(expanded):
            state = expanded[units]
            if state[0] < best_cost:
                frontier[units] = state
                best_cost = state[0]
    limit_units = math.floor(len(ids) * budget_per_query * 10.0 + EPS)
    feasible = [(state[0], state[4], units, state) for units, state in frontier.items()
                if units <= limit_units]
    if not feasible:
        raise RuntimeError("fixed64 must make the oracle budget feasible")
    _, _, units, chosen = min(feasible)
    total_probes, count16, count32, count64, actual_loss = chosen
    return {
        "queries": len(ids), "budget_per_query": budget_per_query,
        "total_loss_budget": len(ids) * budget_per_query,
        "actual_total_recall_loss": actual_loss,
        "loss_units_tenths": units,
        "average_probes": total_probes / len(ids),
        "probe_reduction": 1.0 - total_probes / (64.0 * len(ids)),
        "probe_distribution": {"16": count16, "32": count32, "64": count64},
        "mean_recall": statistics.fmean(recall64[qid] for qid in ids) - actual_loss / len(ids),
        "recall_delta": -actual_loss / len(ids),
    }


def model_specs(family):
    if family == "gradient_boosting_regressor":
        return [{"max_depth": depth, "n_estimators": estimators, "learning_rate": rate,
                 "loss": loss}
                for depth in (1, 2) for estimators in (50, 100)
                for rate in (0.05, 0.1) for loss in ("squared_error", "huber")]
    if family == "random_forest_regressor":
        return [{"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": 150}
                for depth in (4, 6, None) for leaf in (5, 15)]
    if family == "decision_tree_regressor":
        return [{"max_depth": depth, "min_samples_leaf": leaf}
                for depth in (3, 4, 5) for leaf in (5, 15, 30)]
    raise ValueError(family)


def build_model(family, spec):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    if family == "gradient_boosting_regressor":
        return GradientBoostingRegressor(max_depth=spec["max_depth"],
            n_estimators=spec["n_estimators"], learning_rate=spec["learning_rate"],
            loss=spec["loss"], random_state=20260913)
    if family == "random_forest_regressor":
        return RandomForestRegressor(max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"], n_estimators=spec["n_estimators"],
            n_jobs=-1, random_state=20260913)
    return DecisionTreeRegressor(max_depth=spec["max_depth"],
        min_samples_leaf=spec["min_samples_leaf"], random_state=20260913)


def oof_predictions(family, spec, x16, x32, loss16, loss32):
    predictions = [np.zeros(500), np.zeros(500)]
    for fold in range(5):
        valid = CALIBRATION_IDS[CALIBRATION_IDS % 5 == fold]
        train = CALIBRATION_IDS[CALIBRATION_IDS % 5 != fold]
        for x, target, output in ((x16, loss16, predictions[0]),
                                  (x32, loss32, predictions[1])):
            model = build_model(family, spec)
            model.fit(x[train], target[train])
            output[valid] = model.predict(x[valid])
    return predictions


def threshold_candidates(predictions):
    return sorted(set([-math.inf, math.inf] +
                      [float(value) for value in np.quantile(predictions, np.linspace(0, 1, 41))]))


def policy_actions(pred16, pred32, threshold16, threshold32):
    return np.where(pred16 <= threshold16, 16,
                    np.where(pred32 <= threshold32, 32, 64)).astype(int)


def selected_values(actions, value16, value32):
    return np.where(actions == 16, value16, np.where(actions == 32, value32, 0.0))


def policy_summary(ids, actions, pred16, pred32, loss16, loss32, recall64):
    actual = selected_values(actions, loss16[ids], loss32[ids])
    predicted = selected_values(actions, pred16, pred32)
    total_loss = float(actual.sum())
    severe = actual >= SEVERE_LOSS - EPS
    positive = actual > EPS
    counts = {str(probe): int((actions == probe).sum()) for probe in shadow.PROBES}
    average = float(actions.mean())
    return {
        "queries": len(ids), "average_probes": average,
        "probe_reduction": 1.0 - average / 64.0,
        "probe_distribution": counts,
        "fixed64_mean_recall": float(recall64[ids].mean()),
        "mean_recall": float(recall64[ids].mean() - total_loss / len(ids)),
        "recall_delta": -total_loss / len(ids),
        "actual_total_recall_loss": total_loss,
        "predicted_total_recall_loss": float(predicted.sum()),
        "positive_loss_downgrades": int(positive.sum()),
        "false_severe_downgrades": int(severe.sum()),
        "false_severe_query_ids": [int(qid) for qid in ids[severe]],
    }


def tune_policy(pred16, pred32, loss16, loss32, recall64, floor=-0.003):
    best = None
    for threshold16 in threshold_candidates(pred16):
        for threshold32 in threshold_candidates(pred32):
            actions = policy_actions(pred16, pred32, threshold16, threshold32)
            actual = selected_values(actions, loss16[:500], loss32[:500])
            recall_delta = -float(actual.mean())
            if recall_delta < floor - EPS:
                continue
            average = float(actions.mean())
            severe = int((actual >= SEVERE_LOSS - EPS).sum())
            positive = int((actual > EPS).sum())
            key = (average, severe, positive, float(actual.sum()), threshold16, threshold32)
            if best is None or key < best[0]:
                best = (key, threshold16, threshold32, actions)
    if best is None:
        raise RuntimeError("fixed64 must satisfy the calibration policy constraint")
    _, threshold16, threshold32, actions = best
    return float(threshold16), float(threshold32), policy_summary(
        CALIBRATION_IDS, actions, pred16, pred32, loss16, loss32, recall64)


def fit_test_predictions(family, spec, x16, x32, loss16, loss32):
    models, predictions = [], []
    for x, target in ((x16, loss16), (x32, loss32)):
        model = build_model(family, spec)
        model.fit(x[:500], target[:500])
        models.append(model)
        predictions.append(model.predict(x[500:]))
    return models, predictions


def feature_importance(model, names):
    return sorted(({"feature": name, "importance": float(value)}
                   for name, value in zip(names, model.feature_importances_)),
                  key=lambda item: item["importance"], reverse=True)


def tune_family(family, x16, x32, data, loss16, loss32, recall64):
    best = None
    for spec in model_specs(family):
        pred16, pred32 = oof_predictions(family, spec, x16, x32, loss16, loss32)
        threshold16, threshold32, calibration = tune_policy(
            pred16, pred32, loss16, loss32, recall64)
        mae16 = float(np.mean(np.abs(pred16 - loss16[:500])))
        mae32 = float(np.mean(np.abs(pred32 - loss32[:500])))
        key = (calibration["average_probes"], calibration["false_severe_downgrades"],
               calibration["positive_loss_downgrades"], calibration["actual_total_recall_loss"],
               mae16 + mae32, json.dumps(spec, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, spec, threshold16, threshold32, calibration,
                    {"loss16": mae16, "loss32": mae32, "mean": (mae16 + mae32) / 2.0})
    _, spec, threshold16, threshold32, calibration, calibration_mae = best
    models, test_predictions = fit_test_predictions(family, spec, x16, x32, loss16, loss32)
    return {
        "family": family, "spec": spec,
        "thresholds": {"loss16": threshold16, "loss32": threshold32},
        "calibration_oof": calibration, "calibration_oof_loss_mae": calibration_mae,
        "models": models, "test_predictions": test_predictions,
        "feature_importance": {
            "loss16": feature_importance(models[0], data["stage16_names"]),
            "loss32": feature_importance(models[1], data["stage32_names"]),
        },
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
    raw = shadow.read_raw(args.raw)
    centroids = shadow.read_centroids(args.centroid_features)
    frozen = shadow.load_phase_c(args.phase_c, args.system, args.round)
    for key, row in raw.items():
        if row["ids"] != frozen[key]["ids"] or abs(row["recall"] - frozen[key]["recall"]) > EPS:
            raise RuntimeError(f"raw/Phase C audit mismatch: {key}")
    data = shadow.make_features(raw, centroids)
    x16 = np.asarray(data["stage16"], dtype=float)
    x32 = np.asarray(data["stage32"], dtype=float)
    loss16, loss32, recall64 = losses_from_raw(raw)
    oracles = {
        "full_workload": budgeted_oracle(np.arange(1000), loss16, loss32, recall64),
        "calibration": budgeted_oracle(CALIBRATION_IDS, loss16, loss32, recall64),
    }
    print("budgeted oracle", json.dumps(oracles["full_workload"], sort_keys=True), flush=True)

    tuned = []
    for family in ("gradient_boosting_regressor", "random_forest_regressor", "decision_tree_regressor"):
        print(f"calibrating family={family}", flush=True)
        tuned.append(tune_family(family, x16, x32, data, loss16, loss32, recall64))

    # The only test evaluation starts here, after all specs and thresholds are locked.
    oracles["test"] = budgeted_oracle(TEST_IDS, loss16, loss32, recall64)
    results = []
    for item in tuned:
        pred16, pred32 = item.pop("test_predictions")
        thresholds = item["thresholds"]
        actions = policy_actions(pred16, pred32, thresholds["loss16"], thresholds["loss32"])
        item["test"] = policy_summary(TEST_IDS, actions, pred16, pred32,
                                      loss16, loss32, recall64)
        mae16 = float(np.mean(np.abs(pred16 - loss16[500:])))
        mae32 = float(np.mean(np.abs(pred32 - loss32[500:])))
        item["test_loss_mae"] = {"loss16": mae16, "loss32": mae32,
                                 "mean": (mae16 + mae32) / 2.0}
        item["gate"] = gate(item["test"])
        item.pop("models")
        results.append(item)
    compliant = [item for item in results if item["test"]["recall_delta"] >= -0.005 - EPS]
    best = (min(compliant, key=lambda item: (item["test"]["average_probes"],
                                             item["test"]["false_severe_downgrades"]))
            if compliant else max(results, key=lambda item: (item["test"]["recall_delta"],
                                                              -item["test"]["average_probes"])))
    report = {
        "status": "COMPLETE", "baseline_probes": 64,
        "split": {"calibration": [0, 499], "test": [500, 999], "fold": "query_id % 5"},
        "calibration_recall_delta_floor": -0.003, "test_recall_delta_floor": -0.005,
        "false_severe_downgrade_definition": "selected per-query Recall@10 loss >= 0.2",
        "budgeted_oracle": oracles, "models": results,
        "test_constraint_met_by_any_model": bool(compliant),
        "best_model": {"family": best["family"], "spec": best["spec"],
                       "thresholds": best["thresholds"], "test": best["test"],
                       "test_loss_mae": best["test_loss_mae"], "gate": best["gate"],
                       "feature_importance": best["feature_importance"]},
    }
    atomic_json(args.output, report)
    lines = ["# D2-v6 loss-budgeted adaptive probing", "",
             "All hyperparameters and policy thresholds were selected using five-fold OOF calibration on qids 0–499. Test qids 500–999 were evaluated once after locking.", "",
             "## Budgeted oracle", "",
             "| Workload | Avg probes | Recall delta | Reduction | 16/32/64 |",
             "|---|---:|---:|---:|---:|"]
    for name, oracle in oracles.items():
        dist = oracle["probe_distribution"]
        lines.append(f"| {name} | {oracle['average_probes']:.3f} | {oracle['recall_delta']:+.4f} | {oracle['probe_reduction']:.2%} | {dist['16']}/{dist['32']}/{dist['64']} |")
    lines += ["", "## Regressor policies", "",
              "| Model | OOF avg/delta | Test avg probes | Test Recall delta | Reduction | 16/32/64 | Total loss | MAE16/MAE32 | Severe | Gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in results:
        cal = item["calibration_oof"]; test = item["test"]; dist = test["probe_distribution"]
        mae = item["test_loss_mae"]
        lines.append(f"| {item['family']} | {cal['average_probes']:.3f}/{cal['recall_delta']:+.4f} | {test['average_probes']:.3f} | {test['recall_delta']:+.4f} | {test['probe_reduction']:.2%} | {dist['16']}/{dist['32']}/{dist['64']} | {test['actual_total_recall_loss']:.3f} | {mae['loss16']:.4f}/{mae['loss32']:.4f} | {test['false_severe_downgrades']} | {item['gate']} |")
    lines += ["", f"Best locked model: **{best['family']}**, gate **{best['gate']}**.", "",
              "Full thresholds, feature importance, predicted loss totals, and severe query IDs are in `summary.json`.", ""]
    args.output.with_name("report.md").write_text("\n".join(lines))
    print(json.dumps({"best_model": best["family"], "gate": best["gate"],
                      "average_probes": best["test"]["average_probes"],
                      "recall_delta": best["test"]["recall_delta"],
                      "distribution": best["test"]["probe_distribution"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-c", type=Path, default=shadow.PHASE_C)
    parser.add_argument("--centroid-features", type=Path, default=shadow.CENTROID_FEATURES)
    parser.add_argument("--system", default="vanilla_eq")
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
