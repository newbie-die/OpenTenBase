#!/usr/bin/env python3
"""D2-v8.3 global five-fold OOF convergence and locked-policy evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import asymmetric_stage32 as v8
import loss_budgeted as v6
import progressive_shadow as shadow
import risk_ranked as v7

CAL_IDS = np.arange(500)
DEV_IDS = np.arange(500, 1000)
N_ESTIMATORS = 2000
ROLLING_WINDOW = 21
ROLLING_RADIUS = ROLLING_WINDOW // 2
TAIL_MATERIAL_GAIN = 0.001
EPS = 1e-12
STAGE32_BASE_SPEC = {
    "learning_rate": 0.05,
    "max_depth": 1,
    "unsafe_weight": 4.0,
}
REFERENCE_HASHES = {
    "feature_columns_sha256":
        "cf72f18f725d998a1dd122dea63b415ed50379a0c594cde15d513611ab9ad5b0",
    "feature_values_sha256":
        "59675da0b9418135d1074d4867c6f63af82e36934591513684b02bf2f831315c",
    "labels_sha256":
        "9068d31d20c370b056c4b8274b56cca719a4632aefe80d4c91d7c0acf72dde96",
}
ORIGINAL_D2V8 = {
    "development_average_probes": 51.456,
    "development_recall_delta": -0.0050,
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def hash_array(values, dtype):
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def hash_strings(values):
    payload = json.dumps(list(values), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_inputs(args):
    raw = shadow.read_raw(args.raw)
    old = shadow.read_centroids(args.old_features)
    geometry = v7.read_geometry(args.geometry)
    features = v7.build_features(raw, old, geometry)
    loss16, loss32, recall64 = v6.losses_from_raw(raw)
    lost16 = v7.lost_neighbor_classes(loss16)
    unsafe32 = (loss32 > EPS).astype(int)

    hashes = {
        "feature_columns_sha256":
            hash_strings(features["extended32_names"]),
        "feature_values_sha256":
            hash_array(features["extended32"][:500], "<f8"),
        "labels_sha256": hash_array(unsafe32[:500], "<i8"),
    }
    if hashes != REFERENCE_HASHES:
        raise RuntimeError(
            f"feature/label drift from D2-v8.2: {hashes}")
    return features, loss16, loss32, recall64, lost16, unsafe32, hashes


def build_curve_model():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(
        max_depth=STAGE32_BASE_SPEC["max_depth"],
        n_estimators=N_ESTIMATORS,
        learning_rate=STAGE32_BASE_SPEC["learning_rate"],
        random_state=20260913,
    )


def train_full_curves(x32, labels):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    all_rows = []
    rows_by_fold = {}
    for fold in range(5):
        print(f"training fold={fold + 1}/5 estimators={N_ESTIMATORS}",
              flush=True)
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        model = build_curve_model()
        weights = np.where(labels[train] == 1, 4.0, 1.0)
        model.fit(x32[train], labels[train], sample_weight=weights)

        fold_rows = []
        stages = zip(model.staged_predict_proba(x32[train]),
                     model.staged_predict_proba(x32[validation]))
        for iteration, (train_probability, validation_probability) in enumerate(stages, 1):
            row = {
                "fold": fold,
                "n_estimators": iteration,
                "train_logloss": float(log_loss(
                    labels[train], train_probability, labels=[0, 1])),
                "validation_logloss": float(log_loss(
                    labels[validation], validation_probability,
                    labels=[0, 1])),
                "roc_auc": float(roc_auc_score(
                    labels[validation], validation_probability[:, 1])),
                "pr_auc": float(average_precision_score(
                    labels[validation], validation_probability[:, 1])),
            }
            fold_rows.append(row)
            all_rows.append(row)
        if len(fold_rows) != N_ESTIMATORS:
            raise RuntimeError(
                f"fold {fold} produced {len(fold_rows)} stages")
        rows_by_fold[fold] = fold_rows
    return all_rows, rows_by_fold


def global_curve(rows_by_fold):
    rows = []
    for iteration in range(1, N_ESTIMATORS + 1):
        local = [rows_by_fold[fold][iteration - 1] for fold in range(5)]
        rows.append({
            "n_estimators": iteration,
            "mean_train_logloss": float(np.mean(
                [row["train_logloss"] for row in local])),
            "mean_validation_logloss": float(np.mean(
                [row["validation_logloss"] for row in local])),
            "mean_roc_auc": float(np.mean(
                [row["roc_auc"] for row in local])),
            "mean_pr_auc": float(np.mean(
                [row["pr_auc"] for row in local])),
            "rolling_mean_validation_logloss": None,
        })

    validation_losses = np.asarray(
        [row["mean_validation_logloss"] for row in rows])
    rolling = np.convolve(
        validation_losses,
        np.ones(ROLLING_WINDOW) / ROLLING_WINDOW,
        mode="valid",
    )
    for offset, value in enumerate(rolling):
        center_iteration = offset + ROLLING_RADIUS + 1
        rows[center_iteration - 1]["rolling_mean_validation_logloss"] = float(value)

    candidates = [
        row for row in rows
        if row["rolling_mean_validation_logloss"] is not None
    ]
    best = min(
        candidates,
        key=lambda row: (
            row["rolling_mean_validation_logloss"],
            row["n_estimators"],
        ),
    )
    return rows, best


def value_at(global_rows, iteration):
    if iteration < 1 or iteration > N_ESTIMATORS:
        return None
    return global_rows[iteration - 1]


def convergence_summary(global_rows, best, rows_by_fold):
    best_iteration = best["n_estimators"]
    last_full_center = N_ESTIMATORS - ROLLING_RADIUS
    tail_start = value_at(global_rows, 1800)
    tail_end = value_at(global_rows, last_full_center)
    tail_gain = (
        tail_start["rolling_mean_validation_logloss"]
        - tail_end["rolling_mean_validation_logloss"]
    )
    cap_limited = best_iteration >= 1800
    tail_still_improving = tail_gain > TAIL_MATERIAL_GAIN
    converged = not cap_limited and not tail_still_improving

    checkpoints = {}
    for offset in (0, 100, 200, 500):
        iteration = best_iteration + offset
        row = value_at(global_rows, iteration)
        checkpoints[f"best_plus_{offset}"] = (
            None if row is None else {
                "iteration": iteration,
                "mean_validation_logloss":
                    row["mean_validation_logloss"],
                "rolling_mean_validation_logloss":
                    row["rolling_mean_validation_logloss"],
            }
        )
    row_2000 = value_at(global_rows, 2000)
    per_fold = []
    for fold in range(5):
        row = rows_by_fold[fold][best_iteration - 1]
        per_fold.append({
            "fold": fold,
            "validation_logloss": row["validation_logloss"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
        })
    return {
        "status": "CONVERGED" if converged else "NOT_CONVERGED",
        "global_best_iteration": int(best_iteration),
        "rolling_window": ROLLING_WINDOW,
        "rolling_definition":
            "centered full window t-10..t+10; valid centers 11..1990",
        "loss_at_best": {
            "mean_validation_logloss":
                best["mean_validation_logloss"],
            "rolling_mean_validation_logloss":
                best["rolling_mean_validation_logloss"],
        },
        "loss_checkpoints": checkpoints,
        "loss_at_2000": {
            "mean_validation_logloss":
                row_2000["mean_validation_logloss"],
            "rolling_mean_validation_logloss": None,
        },
        "last_200_tree_check": {
            "start_iteration": 1800,
            "end_iteration": last_full_center,
            "rolling_loss_at_start":
                tail_start["rolling_mean_validation_logloss"],
            "rolling_loss_at_end":
                tail_end["rolling_mean_validation_logloss"],
            "validation_gain": float(tail_gain),
            "material_gain_threshold": TAIL_MATERIAL_GAIN,
            "still_materially_improving": bool(tail_still_improving),
        },
        "best_at_or_after_1800": bool(cap_limited),
        "per_fold_at_global_best": per_fold,
    }


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


def locked_policy(best_iteration, features, loss16, loss32, recall64,
                  lost16, unsafe32):
    x16 = features["extended16"]
    x32 = features["extended32"]
    spec = stage32_spec(best_iteration)

    print(f"regenerating OOF prediction estimators={best_iteration}",
          flush=True)
    stage16_oof_probability = v8.fit_stage16_oof(x16, lost16)
    stage16_oof_risk = v7.risk_score(stage16_oof_probability, 2.0)
    raw_oof = v8.stage32_oof(
        "gradient_boosting", spec, x32, unsafe32)
    calibrator = v8.fit_calibrator(
        "isotonic", raw_oof, unsafe32[:500])
    calibrated_oof = v8.calibrate(
        calibrator, "isotonic", raw_oof)
    threshold16, threshold32, oof_policy = v8.tune_thresholds(
        stage16_oof_risk, calibrated_oof,
        loss16, loss32, recall64)
    oof_unsafe = v8.unsafe_metrics(
        unsafe32[:500], calibrated_oof, threshold32)

    # The development set is touched only after all OOF choices are locked.
    print("development evaluation=1/1", flush=True)
    _, stage16_dev_probability = v8.fit_stage16_development(
        x16, lost16)
    stage16_dev_risk = v7.risk_score(
        stage16_dev_probability, 2.0)

    final_model = v8.build_stage32("gradient_boosting", spec)
    v8.fit_binary(
        final_model, "gradient_boosting",
        x32[:500], unsafe32[:500], spec["unsafe_weight"])
    raw_dev = v8.unsafe_probability(final_model, x32[500:])
    dev_probability = v8.calibrate(
        calibrator, "isotonic", raw_dev)
    dev_actions = v8.actions_for(
        stage16_dev_risk,
        dev_probability,
        threshold16,
        threshold32,
    )
    dev_policy = v8.policy_summary(
        DEV_IDS, dev_actions, loss16, loss32, recall64)
    dev_unsafe = v8.unsafe_metrics(
        unsafe32[500:], dev_probability, threshold32)

    recall_ok = dev_policy["recall_delta"] >= -0.005 - EPS
    probes_ok = dev_policy["average_probes"] <= 50.0 + EPS
    gate = "Go" if recall_ok and probes_ok else "Marginal"
    return {
        "stage32_spec": spec,
        "calibration_method": "isotonic",
        "thresholds": {
            "stage16_risk": threshold16,
            "stage32_max_unsafe_probability": threshold32,
        },
        "oof_policy": oof_policy,
        "oof_unsafe32": oof_unsafe,
        "development_policy": dev_policy,
        "development_unsafe32": dev_unsafe,
        "comparison_to_original_d2_v8": {
            "average_probes_change":
                dev_policy["average_probes"]
                - ORIGINAL_D2V8["development_average_probes"],
            "average_probes_reduction":
                ORIGINAL_D2V8["development_average_probes"]
                - dev_policy["average_probes"],
            "recall_delta_change":
                dev_policy["recall_delta"]
                - ORIGINAL_D2V8["development_recall_delta"],
        },
        "gate": gate,
    }


def render_report(report):
    convergence = report["convergence"]
    best = convergence["global_best_iteration"]
    lines = [
        "# D2-v8.3 Global OOF Convergence",
        "",
        f"Convergence: **{convergence['status']}**.",
        "",
        f"Global best iteration: **{best}**.",
        "",
        f"Mean validation loss at best: "
        f"{convergence['loss_at_best']['mean_validation_logloss']:.6f}; "
        f"21-tree rolling mean: "
        f"{convergence['loss_at_best']['rolling_mean_validation_logloss']:.6f}.",
        "",
        "| Fold | Validation loss | ROC-AUC | PR-AUC |",
        "|---:|---:|---:|---:|",
    ]
    for row in convergence["per_fold_at_global_best"]:
        lines.append(
            f"| {row['fold']} | {row['validation_logloss']:.6f} | "
            f"{row['roc_auc']:.6f} | {row['pr_auc']:.6f} |")
    lines += [
        "",
        f"Last-tail rolling validation gain "
        f"1800→1990: "
        f"{convergence['last_200_tree_check']['validation_gain']:.6f}.",
        "",
    ]
    if report["policy_evaluated"]:
        policy = report["policy"]
        oof = policy["oof_policy"]
        dev = policy["development_policy"]
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
            f"Gate: **{policy['gate']}**.",
        ]
    else:
        lines += [
            "Policy and development evaluation were skipped because the "
            "global curve did not converge.",
        ]
    lines.append("")
    return "\n".join(lines)


def analyze(args):
    (features, loss16, loss32, recall64,
     lost16, unsafe32, hashes) = load_inputs(args)
    curve_rows, rows_by_fold = train_full_curves(
        features["extended32"], unsafe32[:500])
    global_rows, best = global_curve(rows_by_fold)
    convergence = convergence_summary(
        global_rows, best, rows_by_fold)

    write_csv(args.output.with_name("fold_curves.csv"), curve_rows)
    write_csv(args.output.with_name("global_oof_curve.csv"), global_rows)

    report = {
        "status": "COMPLETE",
        "database_run": False,
        "features_modified": False,
        "labels_modified": False,
        "policy_design_modified": False,
        "data_hashes": hashes,
        "training": {
            "folds": 5,
            "calibration_qids": [0, 499],
            "n_estimators": N_ESTIMATORS,
            "per_fold_early_stopping": False,
            "learning_rate": 0.05,
            "max_depth": 1,
            "min_samples_leaf": 1,
            "subsample": 1.0,
            "random_state": 20260913,
            "loss": "log_loss",
            "sample_weight": {"safe": 1.0, "unsafe": 4.0},
        },
        "convergence": convergence,
        "policy_evaluated": False,
        "development_evaluations": 0,
        "original_d2_v8": ORIGINAL_D2V8,
    }
    if convergence["status"] == "CONVERGED":
        report["policy"] = locked_policy(
            convergence["global_best_iteration"],
            features, loss16, loss32, recall64,
            lost16, unsafe32)
        report["policy_evaluated"] = True
        report["development_evaluations"] = 1
    else:
        report["gate"] = "NOT_EVALUATED"

    atomic_json(args.output, report)
    args.output.with_name("report.md").write_text(
        render_report(report))
    print(json.dumps({
        "convergence": convergence["status"],
        "global_best_iteration":
            convergence["global_best_iteration"],
        "loss_at_best": convergence["loss_at_best"],
        "last_200_validation_gain":
            convergence["last_200_tree_check"]["validation_gain"],
        "policy_evaluated": report["policy_evaluated"],
        "oof": report.get("policy", {}).get("oof_policy"),
        "development":
            report.get("policy", {}).get("development_policy"),
        "gate": report.get("policy", {}).get(
            "gate", report.get("gate")),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--old-features", type=Path,
        default=shadow.CENTROID_FEATURES)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(
            f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
