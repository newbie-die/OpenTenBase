#!/usr/bin/env python3
"""D2-v8.1 GradientBoosting convergence completion (offline only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import asymmetric_stage32 as v8
import loss_budgeted as v6
import progressive_shadow as shadow
import risk_ranked as v7

CAL_IDS = np.arange(500)
DEV_IDS = np.arange(500, 1000)
MAX_ESTIMATORS = 1000
PATIENCE = 30
EPS = 1e-12
STAGE32_BASE_SPEC = {
    "learning_rate": 0.05,
    "max_depth": 1,
    "unsafe_weight": 4.0,
}
ORIGINAL_D2V8 = {
    "oof_average_probes": 49.6,
    "oof_recall_delta": -0.0022,
    "development_average_probes": 51.456,
    "development_recall_delta": -0.0050,
}


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


def curve_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        max_depth=STAGE32_BASE_SPEC["max_depth"],
        n_estimators=MAX_ESTIMATORS,
        learning_rate=STAGE32_BASE_SPEC["learning_rate"],
        random_state=20260913,
    )


def fit_fold_curve(fold, x, labels):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    validation = CAL_IDS[CAL_IDS % 5 == fold]
    train = CAL_IDS[CAL_IDS % 5 != fold]
    model = curve_model()
    weights = np.where(labels[train] == 1, STAGE32_BASE_SPEC["unsafe_weight"], 1.0)
    model.fit(x[train], labels[train], sample_weight=weights)

    rows = []
    best_loss = math.inf
    best_iteration = 0
    best_row = None
    wait = 0
    stop_iteration = MAX_ESTIMATORS
    stages = zip(model.staged_predict_proba(x[train]),
                 model.staged_predict_proba(x[validation]))
    for iteration, (train_probability, validation_probability) in enumerate(stages, 1):
        row = {
            "fold": fold,
            "n_estimators": iteration,
            "train_logloss": float(log_loss(
                labels[train], train_probability, labels=[0, 1])),
            "validation_logloss": float(log_loss(
                labels[validation], validation_probability, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(
                labels[validation], validation_probability[:, 1])),
            "pr_auc": float(average_precision_score(
                labels[validation], validation_probability[:, 1])),
        }
        rows.append(row)
        if row["validation_logloss"] < best_loss - EPS:
            best_loss = row["validation_logloss"]
            best_iteration = iteration
            best_row = row
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            stop_iteration = iteration
            break

    stop_row = rows[-1]
    after_30_index = best_iteration + PATIENCE
    after_30_row = (rows[after_30_index - 1]
                    if after_30_index <= len(rows) else None)
    after_30_loss = (float(after_30_row["validation_logloss"])
                     if after_30_row is not None else None)
    after_30_worsening = (after_30_loss - best_loss
                          if after_30_loss is not None else None)
    summary = {
        "fold": fold,
        "train_queries": int(len(train)),
        "validation_queries": int(len(validation)),
        "best_iteration": int(best_iteration),
        "stop_iteration": int(stop_iteration),
        "best_validation_loss": float(best_loss),
        "train_logloss_at_best": float(best_row["train_logloss"]),
        "roc_auc_at_best": float(best_row["roc_auc"]),
        "pr_auc_at_best": float(best_row["pr_auc"]),
        "validation_loss_at_stop": float(stop_row["validation_logloss"]),
        "validation_loss_30_trees_after_best": after_30_loss,
        "validation_worsening_30_trees_after_best": after_30_worsening,
        "reached_max_estimators": bool(stop_iteration == MAX_ESTIMATORS),
        "still_improving_at_cap": bool(
            stop_iteration == MAX_ESTIMATORS and best_iteration > MAX_ESTIMATORS - PATIENCE),
    }
    return rows, summary


def convergence_curves(x32, unsafe32):
    all_rows = []
    folds = []
    rows_by_fold = {}
    for fold in range(5):
        print(f"convergence fold={fold + 1}/5 max_estimators={MAX_ESTIMATORS}", flush=True)
        rows, summary = fit_fold_curve(fold, x32, unsafe32)
        all_rows.extend(rows)
        folds.append(summary)
        rows_by_fold[fold] = rows

    common_horizon = min(row["stop_iteration"] for row in folds)
    mean_rows = []
    for iteration in range(1, common_horizon + 1):
        values = [rows_by_fold[fold][iteration - 1] for fold in range(5)]
        mean_rows.append({
            "n_estimators": iteration,
            "mean_train_logloss": float(np.mean(
                [row["train_logloss"] for row in values])),
            "mean_validation_logloss": float(np.mean(
                [row["validation_logloss"] for row in values])),
            "mean_roc_auc": float(np.mean([row["roc_auc"] for row in values])),
            "mean_pr_auc": float(np.mean([row["pr_auc"] for row in values])),
        })
    global_row = min(mean_rows,
                     key=lambda row: (row["mean_validation_logloss"],
                                      row["n_estimators"]))
    converged = all(
        row["best_iteration"] < MAX_ESTIMATORS
        and row["stop_iteration"] < MAX_ESTIMATORS
        and not row["still_improving_at_cap"]
        for row in folds
    )
    return all_rows, folds, mean_rows, global_row, converged


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def stage32_spec(iteration):
    return {
        **STAGE32_BASE_SPEC,
        "n_estimators": int(iteration),
    }


def run_locked_policy(global_best, features, loss16, loss32, recall64,
                      lost16, unsafe32):
    x16 = features["extended16"]
    x32 = features["extended32"]
    spec = stage32_spec(global_best)

    print(f"policy OOF global_best_iteration={global_best}", flush=True)
    stage16_oof_probability = v8.fit_stage16_oof(x16, lost16)
    stage16_oof_risk = v7.risk_score(stage16_oof_probability, 2.0)
    raw_oof = v8.stage32_oof(
        "gradient_boosting", spec, x32, unsafe32)
    calibrator = v8.fit_calibrator("isotonic", raw_oof, unsafe32[:500])
    calibrated_oof = v8.calibrate(calibrator, "isotonic", raw_oof)
    threshold16, threshold32, oof_policy = v8.tune_thresholds(
        stage16_oof_risk, calibrated_oof, loss16, loss32, recall64)
    oof_unsafe = v8.unsafe_metrics(
        unsafe32[:500], calibrated_oof, threshold32)

    # No development labels or policy metrics are touched until the OOF
    # iteration, calibrator, and thresholds above are locked.
    print("development evaluation=1/1", flush=True)
    _, stage16_development_probability = v8.fit_stage16_development(
        x16, lost16)
    stage16_development_risk = v7.risk_score(
        stage16_development_probability, 2.0)

    final_model = v8.build_stage32("gradient_boosting", spec)
    v8.fit_binary(final_model, "gradient_boosting",
                  x32[:500], unsafe32[:500], spec["unsafe_weight"])
    raw_development = v8.unsafe_probability(final_model, x32[500:])
    development_probability = v8.calibrate(
        calibrator, "isotonic", raw_development)
    development_actions = v8.actions_for(
        stage16_development_risk,
        development_probability,
        threshold16,
        threshold32,
    )
    development_policy = v8.policy_summary(
        DEV_IDS, development_actions, loss16, loss32, recall64)
    development_unsafe = v8.unsafe_metrics(
        unsafe32[500:], development_probability, threshold32)

    recall_ok = development_policy["recall_delta"] >= -0.005 - EPS
    probes_ok = development_policy["average_probes"] <= 50.0 + EPS
    gate = "Go" if recall_ok and probes_ok else "Marginal"
    comparison = {
        "average_probes_change":
            development_policy["average_probes"]
            - ORIGINAL_D2V8["development_average_probes"],
        "average_probes_reduction":
            ORIGINAL_D2V8["development_average_probes"]
            - development_policy["average_probes"],
        "recall_delta_change":
            development_policy["recall_delta"]
            - ORIGINAL_D2V8["development_recall_delta"],
    }
    return {
        "stage32_spec": spec,
        "calibration_method": "isotonic",
        "thresholds": {
            "stage16_risk": threshold16,
            "stage32_max_unsafe_probability": threshold32,
        },
        "oof_policy": oof_policy,
        "oof_unsafe32": oof_unsafe,
        "development_policy": development_policy,
        "development_unsafe32": development_unsafe,
        "comparison_to_original_d2_v8": comparison,
        "gate": gate,
    }


def render_report(report):
    lines = [
        "# D2-v8.1 GradientBoosting convergence completion",
        "",
        f"Convergence: **{report['convergence']['status']}**.",
        "",
        "| Fold | Best iteration | Stop iteration | Best validation loss | Worsening after 30 trees |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in report["convergence"]["folds"]:
        worsening = row["validation_worsening_30_trees_after_best"]
        worsening_text = "N/A" if worsening is None else f"{worsening:+.6f}"
        lines.append(
            f"| {row['fold']} | {row['best_iteration']} | "
            f"{row['stop_iteration']} | {row['best_validation_loss']:.6f} | "
            f"{worsening_text} |"
        )
    lines.append("")
    if report["convergence"]["status"] == "CONVERGED":
        lines += [
            f"Global best iteration: **{report['convergence']['global_best_iteration']}** "
            f"(common horizon {report['convergence']['common_horizon']}).",
            "",
        ]
    else:
        lines += [
            "Global best iteration: **not selected** because convergence failed. "
            f"The common-prefix diagnostic minimum is "
            f"{report['convergence']['diagnostic_common_prefix_best_iteration']} "
            f"(common horizon {report['convergence']['common_horizon']}).",
            "",
        ]
    if report["policy_evaluated"]:
        policy = report["policy"]
        oof = policy["oof_policy"]
        dev = policy["development_policy"]
        change = policy["comparison_to_original_d2_v8"]
        lines += [
            "| Set | Average probes | Recall delta | 16/32/64 |",
            "|---|---:|---:|---:|",
            f"| Calibration OOF | {oof['average_probes']:.3f} | "
            f"{oof['recall_delta']:+.4f} | "
            f"{oof['probe_distribution']['16']}/"
            f"{oof['probe_distribution']['32']}/"
            f"{oof['probe_distribution']['64']} |",
            f"| Development | {dev['average_probes']:.3f} | "
            f"{dev['recall_delta']:+.4f} | "
            f"{dev['probe_distribution']['16']}/"
            f"{dev['probe_distribution']['32']}/"
            f"{dev['probe_distribution']['64']} |",
            "",
            f"Versus original D2-v8 development 51.456/-0.0050: "
            f"average probes change {change['average_probes_change']:+.3f}; "
            f"Recall delta change {change['recall_delta_change']:+.4f}.",
            "",
            f"Gate: **{policy['gate']}**.",
        ]
    else:
        lines += [
            "Policy and development evaluation were skipped because at least "
            "one fold did not converge before the 1000-tree cap.",
        ]
    lines.append("")
    return "\n".join(lines)


def analyze(args):
    features, loss16, loss32, recall64, lost16, unsafe32 = load_inputs(args)
    curve_rows, fold_summaries, mean_rows, global_row, converged = (
        convergence_curves(features["extended32"], unsafe32[:500])
    )
    write_csv(args.output.with_name("staged_curves.csv"), curve_rows)
    write_csv(args.output.with_name("mean_oof_curve.csv"), mean_rows)

    convergence = {
        "max_estimators": MAX_ESTIMATORS,
        "patience": PATIENCE,
        "min_delta": 0.0,
        "selection_scope":
            "iterations 1..minimum independent fold stop; all five folds present",
        "common_horizon": int(min(
            row["stop_iteration"] for row in fold_summaries)),
        "global_best_iteration":
            int(global_row["n_estimators"]) if converged else None,
        "diagnostic_common_prefix_best_iteration":
            int(global_row["n_estimators"]),
        "diagnostic_common_prefix_best_mean_validation_loss":
            float(global_row["mean_validation_logloss"]),
        "folds": fold_summaries,
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
    }
    report = {
        "status": "COMPLETE",
        "scope": {
            "calibration_qids": [0, 499],
            "development_qids": [500, 999],
            "development_evaluations": 0,
        },
        "database_run": False,
        "features_modified": False,
        "policy_design_modified": False,
        "original_d2_v8": ORIGINAL_D2V8,
        "convergence": convergence,
        "policy_evaluated": False,
    }

    if converged:
        policy = run_locked_policy(
            global_row["n_estimators"], features, loss16, loss32,
            recall64, lost16, unsafe32)
        report["policy"] = policy
        report["policy_evaluated"] = True
        report["scope"]["development_evaluations"] = 1
    else:
        report["gate"] = "NOT_EVALUATED"

    atomic_json(args.output, report)
    args.output.with_name("report.md").write_text(render_report(report))
    print(json.dumps({
        "convergence": convergence["status"],
        "global_best_iteration": convergence["global_best_iteration"],
        "fold_best_iterations": [
            row["best_iteration"] for row in fold_summaries],
        "fold_stop_iterations": [
            row["stop_iteration"] for row in fold_summaries],
        "policy_evaluated": report["policy_evaluated"],
        "oof": report.get("policy", {}).get("oof_policy"),
        "development": report.get("policy", {}).get("development_policy"),
        "gate": report.get("policy", {}).get("gate", report.get("gate")),
    }, indent=2))


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
