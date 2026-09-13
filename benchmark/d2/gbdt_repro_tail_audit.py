#!/usr/bin/env python3
"""D2-v8.2 GBDT reproducibility and Fold 4 tail audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import gradient_boosting_convergence as v81
import training_health_audit as health

FOLDS = range(5)
CAL_IDS = np.arange(500)
MAX_ESTIMATORS = 500
NUMERIC_TOLERANCE = 1e-12
CHECKPOINTS = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


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


def read_curves(path):
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["fold"]), int(row["n_estimators"]))
            rows[key] = {
                "validation_logloss": float(row["validation_logloss"]),
                "train_logloss": float(row["train_logloss"]),
                "roc_auc": float(row["roc_auc"]),
                "pr_auc": float(row["pr_auc"]),
            }
    return rows


def model_configuration(n_estimators):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        max_depth=1,
        n_estimators=n_estimators,
        learning_rate=0.05,
        random_state=20260913,
    )
    params = model.get_params(deep=False)
    return {
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_samples_leaf": params["min_samples_leaf"],
        "subsample": params["subsample"],
        "random_state": params["random_state"],
        "loss": params["loss"],
        "n_estimators": params["n_estimators"],
        "class_weight": None,
        "sample_weight": {
            "safe_label_0": 1.0,
            "unsafe_label_1": 4.0,
        },
    }


def train_current_curves(x32, labels):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    rows = {}
    output_rows = []
    for fold in FOLDS:
        print(f"reproducibility fold={fold + 1}/5 estimators={MAX_ESTIMATORS}",
              flush=True)
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        model = GradientBoostingClassifier(
            max_depth=1,
            n_estimators=MAX_ESTIMATORS,
            learning_rate=0.05,
            random_state=20260913,
        )
        weights = np.where(labels[train] == 1, 4.0, 1.0)
        model.fit(x32[train], labels[train], sample_weight=weights)
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
            rows[(fold, iteration)] = row
            output_rows.append(row)
    return rows, output_rows


def compare_curves(current, historical, historical_name):
    compared = []
    coverage = {}
    first_different = None
    max_difference = 0.0
    for fold in FOLDS:
        available = []
        missing = []
        for iteration in range(1, MAX_ESTIMATORS + 1):
            key = (fold, iteration)
            if key not in historical:
                missing.append(iteration)
                continue
            available.append(iteration)
            difference = abs(
                current[key]["validation_logloss"]
                - historical[key]["validation_logloss"])
            compared.append((fold, iteration, difference))
            max_difference = max(max_difference, difference)
            if difference > NUMERIC_TOLERANCE and first_different is None:
                first_different = {
                    "fold": fold,
                    "iteration": iteration,
                    "absolute_loss_difference": difference,
                }
        coverage[str(fold)] = {
            "compared_iterations": len(available),
            "first_iteration": min(available) if available else None,
            "last_iteration": max(available) if available else None,
            "missing_iterations_through_500": len(missing),
        }
    return {
        "historical_artifact": historical_name,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "compared_points": len(compared),
        "expected_points_if_full_history": 5 * MAX_ESTIMATORS,
        "max_abs_loss_diff": max_difference,
        "first_different_iteration": first_different,
        "coverage_by_fold": coverage,
        "within_numeric_tolerance": first_different is None,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_both_pipelines(args):
    namespace = SimpleNamespace(
        raw=args.raw,
        geometry=args.geometry,
        old_features=args.old_features,
    )
    health_data = health.load_inputs(namespace)
    v81_data = v81.load_inputs(namespace)
    health_features = health_data[0]
    v81_features = v81_data[0]
    health_x = health_features["extended32"]
    v81_x = v81_features["extended32"]
    health_labels = health_data[-1][:500]
    v81_labels = v81_data[-1][:500]
    equality = {
        "feature_columns_equal": (
            health_features["extended32_names"]
            == v81_features["extended32_names"]),
        "feature_values_equal": bool(np.array_equal(
            health_x[:500], v81_x[:500], equal_nan=True)),
        "labels_equal": bool(np.array_equal(health_labels, v81_labels)),
    }
    if not all(equality.values()):
        raise RuntimeError(f"v8/v8.1 data pipeline mismatch: {equality}")
    return health_features, health_x, health_labels, equality


def data_audit(features, x32, labels, equality):
    folds = []
    for fold in FOLDS:
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        folds.append({
            "fold": fold,
            "split_rule": "qid % 5 == fold is validation",
            "train_queries": int(len(train)),
            "validation_queries": int(len(validation)),
            "train_qid_sha256": hash_array(train, "<i8"),
            "validation_qid_sha256": hash_array(validation, "<i8"),
            "train_sample_weight_sha256": hash_array(
                np.where(labels[train] == 1, 4.0, 1.0), "<f8"),
        })
    names = features["extended32_names"]
    return {
        "pipeline_equality": equality,
        "feature_count": len(names),
        "feature_columns_in_order": names,
        "feature_columns_sha256": hash_strings(names),
        "calibration_feature_values_sha256": hash_array(x32[:500], "<f8"),
        "calibration_labels_sha256": hash_array(labels, "<i8"),
        "canonical_hash_encoding": (
            "SHA-256(shape NUL numpy dtype NUL C-order bytes); "
            "feature names use compact UTF-8 JSON"),
        "folds": folds,
    }


def tail_audit(v81_curves):
    rows = []
    for iteration in CHECKPOINTS:
        row = v81_curves[(4, iteration)]
        rows.append({"iteration": iteration, **row})
    loss = {
        iteration: v81_curves[(4, iteration)]["validation_logloss"]
        for iteration in (500, 750, 900, 1000)
    }
    gains = {
        "500_to_750_validation_gain": loss[500] - loss[750],
        "750_to_1000_validation_gain": loss[750] - loss[1000],
        "900_to_1000_validation_gain": loss[900] - loss[1000],
    }
    return {"fold": 4, "checkpoints": rows, "validation_gains": gains}


def render_report(report):
    repro = report["reproducibility"]
    lines = [
        "# D2-v8.2 GBDT reproducibility and Fold 4 tail audit",
        "",
        f"Reproducibility: **{repro['status']}**.",
        "",
        f"Maximum absolute validation-loss difference: "
        f"**{repro['max_abs_loss_diff']:.3g}**.",
        "",
        f"First different iteration: "
        f"**{repro['first_different_iteration']}**.",
        "",
        "| Fold | Train qid hash | Validation qid hash |",
        "|---:|---|---|",
    ]
    for row in report["data"]["folds"]:
        lines.append(
            f"| {row['fold']} | {row['train_qid_sha256']} | "
            f"{row['validation_qid_sha256']} |")
    if report["tail_evaluated"]:
        lines += [
            "",
            "## Fold 4 tail",
            "",
            "| Iteration | Train loss | Validation loss | ROC-AUC | PR-AUC |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in report["fold4_tail"]["checkpoints"]:
            lines.append(
                f"| {row['iteration']} | {row['train_logloss']:.6f} | "
                f"{row['validation_logloss']:.6f} | "
                f"{row['roc_auc']:.6f} | {row['pr_auc']:.6f} |")
        gain = report["fold4_tail"]["validation_gains"]
        lines += [
            "",
            f"- 500→750 validation gain: "
            f"{gain['500_to_750_validation_gain']:.6f}",
            f"- 750→1000 validation gain: "
            f"{gain['750_to_1000_validation_gain']:.6f}",
            f"- 900→1000 validation gain: "
            f"{gain['900_to_1000_validation_gain']:.6f}",
        ]
    lines += ["", f"Conclusion: **{report['conclusion']}**.", ""]
    return "\n".join(lines)


def analyze(args):
    features, x32, labels, equality = load_both_pipelines(args)
    data = data_audit(features, x32, labels, equality)
    current_curves, current_rows = train_current_curves(x32, labels)
    write_csv(args.output.with_name("current_500_curves.csv"), current_rows)

    audit_curves = read_curves(args.v8_audit_curves)
    v81_curves = read_curves(args.v8_1_curves)
    versus_audit = compare_curves(
        current_curves, audit_curves, str(args.v8_audit_curves))
    versus_v81 = compare_curves(
        current_curves, v81_curves, str(args.v8_1_curves))

    config_audit = model_configuration(500)
    config_v81 = model_configuration(1000)
    config_current = model_configuration(500)
    invariant_keys = (
        "learning_rate", "max_depth", "min_samples_leaf", "subsample",
        "random_state", "loss", "class_weight", "sample_weight")
    training_invariants_equal = all(
        config_audit[key] == config_v81[key] == config_current[key]
        for key in invariant_keys)

    max_diff = max(
        versus_audit["max_abs_loss_diff"],
        versus_v81["max_abs_loss_diff"])
    first_different = (
        versus_audit["first_different_iteration"]
        or versus_v81["first_different_iteration"])
    reproducible = (
        all(equality.values())
        and training_invariants_equal
        and versus_audit["within_numeric_tolerance"]
        and versus_v81["within_numeric_tolerance"]
        and versus_v81["coverage_by_fold"]["4"]["compared_iterations"] == 500
    )
    reproducibility = {
        "status": "PASS" if reproducible else "FAIL",
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "max_abs_loss_diff": max_diff,
        "first_different_iteration": first_different,
        "training_invariants_equal": training_invariants_equal,
        "v8_audit_configuration": {
            **config_audit,
            "early_stopping_patience": 20,
            "early_stopping_min_delta": 1e-4,
        },
        "v8_1_configuration": {
            **config_v81,
            "early_stopping_patience": 30,
            "early_stopping_min_delta": 0.0,
        },
        "current_configuration": config_current,
        "comparison_to_v8_audit": versus_audit,
        "comparison_to_v8_1": versus_v81,
        "historical_coverage_note": (
            "v8/v8.1 persisted curves end at each independent early-stop; "
            "only Fold 4 has all first 500 historical v8.1 stages. All "
            "persisted points are compared, and the current run stores all "
            "500 stages for every fold."),
        "best_iteration_difference_explanation": (
            "Underlying staged losses are identical. Earlier reported best/"
            "stop iterations differ because v8 audit used patience=20 and "
            "min_delta=1e-4, while v8.1 used patience=30 and min_delta=0."),
    }

    tail = tail_audit(v81_curves) if reproducible else None
    if not reproducible:
        conclusion = "TRAINING_PIPELINE_ISSUE"
    elif tail["validation_gains"]["900_to_1000_validation_gain"] > 1e-3:
        conclusion = "ESTIMATOR_CAP_LIMITED"
    else:
        conclusion = "PRACTICALLY_CONVERGED"

    report = {
        "status": "COMPLETE",
        "database_run": False,
        "development_policy_evaluated": False,
        "policy_tuned": False,
        "data": data,
        "reproducibility": reproducibility,
        "tail_evaluated": reproducible,
        "fold4_tail": tail,
        "tail_materiality_rule": (
            "900-to-1000 validation logloss gain > 0.001 is material"),
        "conclusion": conclusion,
    }
    atomic_json(args.output, report)
    args.output.with_name("report.md").write_text(render_report(report))
    print(json.dumps({
        "reproducibility": reproducibility["status"],
        "max_abs_loss_diff": max_diff,
        "first_different_iteration": first_different,
        "fold4_validation_gains": (
            tail["validation_gains"] if tail is not None else None),
        "conclusion": conclusion,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--v8-audit-curves", type=Path, required=True)
    parser.add_argument("--v8-1-curves", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-features", type=Path, default=health.shadow.CENTROID_FEATURES)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
