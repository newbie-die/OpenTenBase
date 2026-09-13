#!/usr/bin/env python3
"""D2-v9R deterministic reconstruction of frozen D2-v8.3 scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

import asymmetric_stage32 as v8
import loss_budgeted as v6
import progressive_shadow as shadow
import risk_ranked as v7

CAL_IDS = np.arange(500)
DEV_IDS = np.arange(500, 1000)
N_ESTIMATORS = 1704
RANDOM_SEED = 20260913
TOLERANCE = 1e-9
STAGE32_SPEC = {
    "max_depth": 1,
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.05,
    "unsafe_weight": 4.0,
}
CSV_FIELDS = (
    "query_id", "fold", "recall16", "recall32", "recall64",
    "stage16_raw_score", "stage16_calibrated_score",
    "stage32_raw_probability", "stage32_calibrated_probability",
)
REFERENCE_HASHES = {
    "feature_columns_sha256":
        "cf72f18f725d998a1dd122dea63b415ed50379a0c594cde15d513611ab9ad5b0",
    "feature_values_sha256":
        "59675da0b9418135d1074d4867c6f63af82e36934591513684b02bf2f831315c",
    "labels_sha256":
        "9068d31d20c370b056c4b8274b56cca719a4632aefe80d4c91d7c0acf72dde96",
}
EPS = 1e-12


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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
    recall16 = recall64 - loss16
    recall32 = recall64 - loss32
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
        raise RuntimeError(f"D2-v8.3 input drift: {hashes}")
    return (features, recall16, recall32, recall64,
            lost16, unsafe32, hashes)


def reference_metrics(path):
    report = json.loads(path.read_text())
    if report["convergence"]["global_best_iteration"] != N_ESTIMATORS:
        raise RuntimeError("reference is not D2-v8.3 iteration 1704")
    return {
        int(row["fold"]): row
        for row in report["convergence"]["per_fold_at_global_best"]
    }


def train_stage32_folds(x32, labels, references):
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    models = []
    raw_oof = np.zeros(500, dtype=float)
    comparisons = []
    passed = True
    for fold in range(5):
        print(f"reconstruction_check fold={fold + 1}/5", flush=True)
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        model = v8.build_stage32("gradient_boosting", STAGE32_SPEC)
        v8.fit_binary(model, "gradient_boosting",
                      x32[train], labels[train],
                      STAGE32_SPEC["unsafe_weight"])
        probability = v8.unsafe_probability(model, x32[validation])
        raw_oof[validation] = probability
        probability2 = np.column_stack((1.0 - probability, probability))
        actual = {
            "validation_logloss": float(log_loss(
                labels[validation], probability2, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(
                labels[validation], probability)),
            "pr_auc": float(average_precision_score(
                labels[validation], probability)),
        }
        reference = references[fold]
        differences = {
            name + "_abs_diff": abs(actual[name] - reference[name])
            for name in ("validation_logloss", "roc_auc", "pr_auc")
        }
        fold_pass = all(value <= TOLERANCE
                        for value in differences.values())
        comparisons.append({
            "fold": fold,
            "actual": actual,
            "reference": {
                name: reference[name]
                for name in ("validation_logloss", "roc_auc", "pr_auc")
            },
            "differences": differences,
            "pass": fold_pass,
        })
        passed = passed and fold_pass
        models.append(model)
    return models, raw_oof, comparisons, passed


def train_stage16_folds(x16, labels):
    models = []
    probabilities = np.zeros((500, 3), dtype=float)
    for fold in range(5):
        print(f"stage16 reconstruction fold={fold + 1}/5", flush=True)
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        model = v7.build_model("gradient_boosting", v8.STAGE16_SPEC)
        model.fit(x16[train], labels[train])
        probabilities[validation] = v7.aligned_probabilities(
            model, x16[validation])
        models.append(model)
    return models, probabilities


def write_predictions(path, ids, recall16, recall32, recall64,
                      stage16_score, stage32_raw, stage32_calibrated):
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for local_index, qid in enumerate(ids):
            writer.writerow({
                "query_id": int(qid),
                "fold": int(qid % 5),
                "recall16": float(recall16[qid]),
                "recall32": float(recall32[qid]),
                "recall64": float(recall64[qid]),
                "stage16_raw_score": float(stage16_score[local_index]),
                "stage16_calibrated_score": float(stage16_score[local_index]),
                "stage32_raw_probability": float(stage32_raw[local_index]),
                "stage32_calibrated_probability":
                    float(stage32_calibrated[local_index]),
            })
    temporary.replace(path)


def dump_models(path, bundle):
    import joblib
    temporary = path.with_name("." + path.name + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, path)


def qid_splits():
    output = []
    for fold in range(5):
        validation = CAL_IDS[CAL_IDS % 5 == fold]
        train = CAL_IDS[CAL_IDS % 5 != fold]
        output.append({
            "fold": fold,
            "rule": "qid % 5 == fold is validation",
            "train_qids": [int(value) for value in train],
            "validation_qids": [int(value) for value in validation],
            "train_qid_sha256": hash_array(train, "<i8"),
            "validation_qid_sha256": hash_array(validation, "<i8"),
        })
    return output


def manifest_value(args, features, hashes, artifact_names,
                   stage16_model, stage32_model):
    import joblib
    import sklearn

    return {
        "status": "PASS",
        "purpose": "D2-v8.3 deterministic score reconstruction",
        "policy_search_performed": False,
        "database_run": False,
        "calibration_qids": [0, 499],
        "development_qids": [500, 999],
        "random_seed": RANDOM_SEED,
        "data_hashes": hashes,
        "inputs": {
            "raw": {"path": str(args.raw), "sha256": sha256_file(args.raw)},
            "geometry": {"path": str(args.geometry),
                         "sha256": sha256_file(args.geometry)},
            "old_features": {"path": str(args.old_features),
                             "sha256": sha256_file(args.old_features)},
            "v8_3_summary": {"path": str(args.v8_3_summary),
                             "sha256": sha256_file(args.v8_3_summary)},
        },
        "features": {
            "stage16_columns_in_order":
                list(features["extended16_names"]),
            "stage32_columns_in_order":
                list(features["extended32_names"]),
        },
        "labels": {
            "stage16": "clip(round((recall64-recall16)*10), 0, 2)",
            "stage32": "recall32 < recall64",
        },
        "stage16": {
            "model": "GradientBoostingClassifier multiclass",
            "parameters": stage16_model.get_params(deep=False),
            "raw_score": "P(loss=1) + 2*P(loss>=2)",
            "calibration": "identity",
            "calibrated_score":
                "same as raw_score; D2-v8.3 has no Stage16 calibrator",
        },
        "stage32": {
            "model": "GradientBoostingClassifier binary",
            "parameters": stage32_model.get_params(deep=False),
            "sample_weight": {
                "safe_label_0": 1.0,
                "unsafe_label_1": 4.0,
            },
            "calibration":
                "IsotonicRegression(out_of_bounds='clip') fit on OOF",
        },
        "qid_splits": qid_splits(),
        "model_bundle_contents": [
            "stage16_fold_models[5]",
            "stage32_fold_models[5]",
            "stage32_isotonic_calibrator",
            "stage16_full_model",
            "stage32_full_model",
        ],
        "artifacts": artifact_names,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def render_report(status, comparisons, artifacts=None):
    lines = [
        "# D2-v9R deterministic score reconstruction",
        "",
        f"Reconstruction: **{status}**.",
        "",
        "| Fold | Loss diff | ROC-AUC diff | PR-AUC diff | Result |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in comparisons:
        diff = row["differences"]
        lines.append(
            f"| {row['fold']} | "
            f"{diff['validation_logloss_abs_diff']:.3g} | "
            f"{diff['roc_auc_abs_diff']:.3g} | "
            f"{diff['pr_auc_abs_diff']:.3g} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |")
    if artifacts:
        lines += [
            "",
            f"OOF predictions: {artifacts['oof']['rows']} rows, "
            f"SHA256 {artifacts['oof']['sha256']}.",
            "",
            f"Development predictions: {artifacts['development']['rows']} rows, "
            f"SHA256 {artifacts['development']['sha256']}.",
            "",
            f"Model bundle SHA256: {artifacts['models']['sha256']}.",
        ]
    lines.append("")
    return "\n".join(lines)


def count_csv_rows(path):
    with path.open(newline="") as source:
        return sum(1 for _ in csv.DictReader(source))


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (features, recall16, recall32, recall64,
     lost16, unsafe32, hashes) = load_inputs(args)
    references = reference_metrics(args.v8_3_summary)

    stage32_models, raw_oof, comparisons, passed = train_stage32_folds(
        features["extended32"], unsafe32, references)
    if not passed:
        result = {
            "status": "FAIL",
            "reason": "D2-v8.3 fold metric mismatch",
            "tolerance": TOLERANCE,
            "comparisons": comparisons,
            "prediction_files_generated": False,
            "model_bundle_generated": False,
            "policy_search_performed": False,
            "development_inference_performed": False,
            "database_run": False,
        }
        atomic_json(args.output / "summary.json", result)
        (args.output / "report.md").write_text(
            render_report("FAIL", comparisons))
        print(json.dumps(result, indent=2))
        return

    stage16_models, stage16_oof_probability = train_stage16_folds(
        features["extended16"], lost16)
    stage16_oof_score = v7.risk_score(stage16_oof_probability, 2.0)
    calibrator = v8.fit_calibrator("isotonic", raw_oof, unsafe32[:500])
    calibrated_oof = v8.calibrate(calibrator, "isotonic", raw_oof)

    oof_path = args.output / "d2_v83_oof_predictions.csv"
    write_predictions(
        oof_path, CAL_IDS, recall16, recall32, recall64,
        stage16_oof_score, raw_oof, calibrated_oof)

    print("full-fit frozen models", flush=True)
    stage16_full_model = v7.build_model(
        "gradient_boosting", v8.STAGE16_SPEC)
    stage16_full_model.fit(features["extended16"][:500], lost16[:500])
    stage16_dev_probability = v7.aligned_probabilities(
        stage16_full_model, features["extended16"][500:])
    stage16_dev_score = v7.risk_score(stage16_dev_probability, 2.0)

    stage32_full_model = v8.build_stage32(
        "gradient_boosting", STAGE32_SPEC)
    v8.fit_binary(
        stage32_full_model, "gradient_boosting",
        features["extended32"][:500], unsafe32[:500],
        STAGE32_SPEC["unsafe_weight"])
    raw_dev = v8.unsafe_probability(
        stage32_full_model, features["extended32"][500:])
    calibrated_dev = v8.calibrate(calibrator, "isotonic", raw_dev)

    dev_path = args.output / "d2_v83_development_predictions.csv"
    write_predictions(
        dev_path, DEV_IDS, recall16, recall32, recall64,
        stage16_dev_score, raw_dev, calibrated_dev)

    models_path = args.output / "d2_v83_models.joblib"
    dump_models(models_path, {
        "stage16_fold_models": stage16_models,
        "stage32_fold_models": stage32_models,
        "stage32_isotonic_calibrator": calibrator,
        "stage16_full_model": stage16_full_model,
        "stage32_full_model": stage32_full_model,
    })

    artifact_names = {
        "oof_predictions": oof_path.name,
        "development_predictions": dev_path.name,
        "model_bundle": models_path.name,
        "manifest": "manifest.json",
    }
    manifest = manifest_value(
        args, features, hashes, artifact_names,
        stage16_full_model, stage32_full_model)
    manifest_path = args.output / "manifest.json"
    atomic_json(manifest_path, manifest)

    rows_oof = count_csv_rows(oof_path)
    rows_dev = count_csv_rows(dev_path)
    if rows_oof != 500 or rows_dev != 500:
        raise RuntimeError(
            f"prediction row mismatch: OOF={rows_oof} dev={rows_dev}")
    artifacts = {
        "oof": {"path": str(oof_path), "rows": rows_oof,
                "sha256": sha256_file(oof_path)},
        "development": {"path": str(dev_path), "rows": rows_dev,
                        "sha256": sha256_file(dev_path)},
        "models": {"path": str(models_path),
                   "sha256": sha256_file(models_path)},
        "manifest": {"path": str(manifest_path),
                     "sha256": sha256_file(manifest_path)},
    }
    result = {
        "status": "PASS",
        "tolerance": TOLERANCE,
        "comparisons": comparisons,
        "data_hashes": hashes,
        "artifacts": artifacts,
        "policy_search_performed": False,
        "development_inference_performed": True,
        "development_policy_evaluated": False,
        "database_run": False,
    }
    atomic_json(args.output / "summary.json", result)
    (args.output / "report.md").write_text(
        render_report("PASS", comparisons, artifacts))
    print(json.dumps({
        "reconstruction": "PASS",
        "max_metric_abs_diff": max(
            value for row in comparisons
            for value in row["differences"].values()),
        "oof_rows": rows_oof,
        "development_rows": rows_dev,
        "artifacts": artifacts,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--v8-3-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-features", type=Path,
                        default=shadow.CENTROID_FEATURES)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
