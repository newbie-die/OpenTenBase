#!/usr/bin/env python3
"""D2-v9 risk-budget calibrated policy search from frozen scores only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

BOOTSTRAPS = 10000
RANDOM_SEED = 20260913
UCB_QUANTILE = 0.95
CI_LOW = 0.025
CI_HIGH = 0.975
RISK_BUDGET = 0.005
EPS = 1e-12
PROBES = (16, 32, 64)
EXPECTED_COLUMNS = (
    "query_id", "fold", "recall16", "recall32", "recall64",
    "stage16_raw_score", "stage16_calibrated_score",
    "stage32_raw_probability", "stage32_calibrated_probability",
)
REFERENCE_SHA256 = {
    "oof":
        "778a2002c1a37132050f50df2a47ec0ff49c909b6dd00dedf6092a16d30245d0",
    "development":
        "179e06d47d46229cc0287aa8dffd34b4ad754afdfdd8acc2e89498e2078aac52",
}
V83_DEVELOPMENT = {
    "average_probes": 54.016,
    "recall_delta": -0.0038,
}


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


def read_predictions(path, expected_qids, expected_sha):
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"frozen prediction SHA mismatch: {path} {actual_sha}")
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected columns in {path}")
        rows = list(reader)
    qids = np.asarray([int(row["query_id"]) for row in rows])
    if not np.array_equal(qids, np.asarray(expected_qids)):
        raise ValueError(f"unexpected qid sequence in {path}")
    return {
        "query_id": qids,
        "fold": np.asarray([int(row["fold"]) for row in rows]),
        "recall16": np.asarray([float(row["recall16"]) for row in rows]),
        "recall32": np.asarray([float(row["recall32"]) for row in rows]),
        "recall64": np.asarray([float(row["recall64"]) for row in rows]),
        "risk16": np.asarray([
            float(row["stage16_calibrated_score"]) for row in rows]),
        "risk32": np.asarray([
            float(row["stage32_calibrated_probability"]) for row in rows]),
        "sha256": actual_sha,
    }


def threshold_values(scores):
    return [-math.inf] + sorted(float(value) for value in np.unique(scores))


def threshold_json(value):
    return "-inf" if math.isinf(value) and value < 0 else float(value)


def threshold_csv(value):
    return "-inf" if math.isinf(value) and value < 0 else repr(float(value))


def selected_actions(risk16, risk32, threshold16, threshold32):
    return np.where(
        risk16 <= threshold16,
        16,
        np.where(risk32 <= threshold32, 32, 64),
    ).astype(np.int16)


def selected_loss(actions, loss16, loss32):
    return np.where(actions == 16, loss16,
                    np.where(actions == 32, loss32, 0.0))


def downgrade_counts(actions, losses):
    return {
        "false_downgrade": int(
            ((actions < 64) & (losses > EPS)).sum()),
        "severe_downgrade": int(
            ((actions < 64) & (losses >= 0.2 - EPS)).sum()),
        "false_action16": int(
            ((actions == 16) & (losses > EPS)).sum()),
        "false_action32": int(
            ((actions == 32) & (losses > EPS)).sum()),
        "severe_action16": int(
            ((actions == 16) & (losses >= 0.2 - EPS)).sum()),
        "severe_action32": int(
            ((actions == 32) & (losses >= 0.2 - EPS)).sum()),
    }


def policy_row(index16, index32, threshold16, threshold32,
               actions, losses, bootstrap_means):
    distribution = {
        str(probe): int((actions == probe).sum())
        for probe in PROBES
    }
    counts = downgrade_counts(actions, losses)
    return {
        "index16": index16,
        "index32": index32,
        "stage16_threshold": threshold_json(threshold16),
        "stage32_threshold": threshold_json(threshold32),
        "average_probes": float(actions.mean()),
        "empirical_mean_loss": float(losses.mean()),
        "recall_delta": -float(losses.mean()),
        "bootstrap_ucb95": float(np.quantile(
            bootstrap_means, UCB_QUANTILE, method="linear")),
        "bootstrap_ci_low": float(np.quantile(
            bootstrap_means, CI_LOW, method="linear")),
        "bootstrap_ci_high": float(np.quantile(
            bootstrap_means, CI_HIGH, method="linear")),
        "count16": distribution["16"],
        "count32": distribution["32"],
        "count64": distribution["64"],
        **counts,
    }


def enumerate_policies(data):
    n = len(data["query_id"])
    loss16 = data["recall64"] - data["recall16"]
    loss32 = data["recall64"] - data["recall32"]
    thresholds16 = threshold_values(data["risk16"])
    thresholds32 = threshold_values(data["risk32"])

    rng = np.random.default_rng(RANDOM_SEED)
    probabilities = np.full(n, 1.0 / n)
    bootstrap_counts_integer = rng.multinomial(
        n, probabilities, size=BOOTSTRAPS).astype(np.uint16)
    bootstrap_hash = hash_array(bootstrap_counts_integer, "<u2")
    bootstrap_counts = bootstrap_counts_integer.astype(np.float64)

    stage32_loss_matrix = np.column_stack([
        np.where(data["risk32"] <= threshold, loss32, 0.0)
        for threshold in thresholds32
    ])
    bootstrap_sums = bootstrap_counts @ stage32_loss_matrix

    mask16 = np.zeros(n, dtype=bool)
    rows = []
    started = time.perf_counter()
    grouped16 = {
        value: np.flatnonzero(data["risk16"] == value)
        for value in thresholds16[1:]
    }
    for index16, threshold16 in enumerate(thresholds16):
        if index16 > 0:
            group = grouped16[threshold16]
            delta = np.empty((len(group), len(thresholds32)))
            for local, qid in enumerate(group):
                delta[local] = np.where(
                    data["risk32"][qid] <= np.asarray(thresholds32),
                    loss16[qid] - loss32[qid],
                    loss16[qid],
                )
            bootstrap_sums += bootstrap_counts[:, group] @ delta
            mask16[group] = True

        for index32, threshold32 in enumerate(thresholds32):
            actions = np.where(
                mask16, 16,
                np.where(data["risk32"] <= threshold32, 32, 64),
            ).astype(np.int16)
            losses = selected_loss(actions, loss16, loss32)
            means = bootstrap_sums[:, index32] / n
            if index16 in (0, len(thresholds16) - 1) and index32 in (
                    0, len(thresholds32) - 1):
                direct = bootstrap_counts @ losses / n
                if np.max(np.abs(direct - means)) > 1e-12:
                    raise RuntimeError("incremental bootstrap mismatch")
            rows.append(policy_row(
                index16, index32, threshold16, threshold32,
                actions, losses, means))

        if (index16 + 1) % 50 == 0 or index16 + 1 == len(thresholds16):
            elapsed = time.perf_counter() - started
            print(
                f"threshold16={index16 + 1}/{len(thresholds16)} "
                f"policies={len(rows)}/{len(thresholds16) * len(thresholds32)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    metadata = {
        "bootstrap_samples": BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "resample_count_matrix_sha256": bootstrap_hash,
        "stage16_thresholds": len(thresholds16),
        "stage32_thresholds": len(thresholds32),
        "policies": len(rows),
        "signed_loss": True,
    }
    return rows, thresholds16, thresholds32, metadata


def row_key_for_selection(row):
    return (
        row["average_probes"],
        row["bootstrap_ucb95"],
        row["empirical_mean_loss"],
        row["severe_downgrade"],
        row["false_downgrade"],
        row["index16"],
        row["index32"],
    )


def selected_policy(rows):
    acceptable = [
        row for row in rows
        if row["bootstrap_ucb95"] <= RISK_BUDGET + EPS
    ]
    if not acceptable:
        raise RuntimeError("fixed64 must be bootstrap acceptable")
    return min(acceptable, key=row_key_for_selection), len(acceptable)


def annotate_policy(row, data):
    threshold16 = (
        -math.inf if row["stage16_threshold"] == "-inf"
        else float(row["stage16_threshold"]))
    threshold32 = (
        -math.inf if row["stage32_threshold"] == "-inf"
        else float(row["stage32_threshold"]))
    actions = selected_actions(
        data["risk16"], data["risk32"], threshold16, threshold32)
    loss16 = data["recall64"] - data["recall16"]
    loss32 = data["recall64"] - data["recall32"]
    losses = selected_loss(actions, loss16, loss32)
    folds = []
    for fold in range(5):
        mask = data["fold"] == fold
        mean_loss = float(losses[mask].mean())
        folds.append({
            "fold": fold,
            "queries": int(mask.sum()),
            "average_probes": float(actions[mask].mean()),
            "mean_loss": mean_loss,
            "recall_delta": -mean_loss,
            "distribution": {
                str(probe): int(
                    ((actions == probe) & mask).sum())
                for probe in PROBES
            },
        })
    deltas = [item["recall_delta"] for item in folds]
    return {
        **row,
        "distribution": {
            str(probe): int((actions == probe).sum())
            for probe in PROBES
        },
        "folds": folds,
        "worst_fold_recall_delta": min(deltas),
        "fold_recall_delta_std": statistics.pstdev(deltas),
    }, actions, losses


def pareto_and_anchors(rows, selected):
    by_probes = {}
    for row in rows:
        average = row["average_probes"]
        current = by_probes.get(average)
        key = (
            row["bootstrap_ucb95"],
            row["empirical_mean_loss"],
            row["severe_downgrade"],
            row["false_downgrade"],
        )
        if current is None or key < (
                current["bootstrap_ucb95"],
                current["empirical_mean_loss"],
                current["severe_downgrade"],
                current["false_downgrade"]):
            by_probes[average] = row

    pareto_ids = set()
    best_risk = math.inf
    for average in sorted(by_probes):
        row = by_probes[average]
        if row["bootstrap_ucb95"] < best_risk - EPS:
            pareto_ids.add((row["index16"], row["index32"]))
            best_risk = row["bootstrap_ucb95"]

    safest = min(rows, key=lambda row: (
        row["bootstrap_ucb95"], row["empirical_mean_loss"],
        row["average_probes"]))
    closest50 = min(rows, key=lambda row: (
        abs(row["average_probes"] - 50.0),
        row["bootstrap_ucb95"], row["empirical_mean_loss"]))
    aggressive = min(rows, key=lambda row: (
        row["average_probes"], row["bootstrap_ucb95"]))
    anchors = {
        "safest": safest,
        "selected": selected,
        "closest_to_50": closest50,
        "most_aggressive": aggressive,
    }
    anchor_labels = {}
    for label, row in anchors.items():
        identifier = (row["index16"], row["index32"])
        anchor_labels.setdefault(identifier, []).append(label)

    keep = pareto_ids | set(anchor_labels)
    output = []
    for row in sorted(
            (row for row in rows
             if (row["index16"], row["index32"]) in keep),
            key=lambda row: (
                row["average_probes"], row["bootstrap_ucb95"])):
        identifier = (row["index16"], row["index32"])
        output.append({
            **row,
            "is_pareto": identifier in pareto_ids,
            "anchors": ";".join(anchor_labels.get(identifier, [])),
        })
    return output, {
        label: {
            "index16": row["index16"],
            "index32": row["index32"],
            "average_probes": row["average_probes"],
            "empirical_mean_loss": row["empirical_mean_loss"],
            "bootstrap_ucb95": row["bootstrap_ucb95"],
            "distribution": {
                "16": row["count16"],
                "32": row["count32"],
                "64": row["count64"],
            },
        }
        for label, row in anchors.items()
    }


def development_evaluation(data, locked):
    threshold16 = (
        -math.inf if locked["stage16_threshold"] == "-inf"
        else float(locked["stage16_threshold"]))
    threshold32 = (
        -math.inf if locked["stage32_threshold"] == "-inf"
        else float(locked["stage32_threshold"]))
    actions = selected_actions(
        data["risk16"], data["risk32"], threshold16, threshold32)
    loss16 = data["recall64"] - data["recall16"]
    loss32 = data["recall64"] - data["recall32"]
    losses = selected_loss(actions, loss16, loss32)
    mean_loss = float(losses.mean())
    counts = downgrade_counts(actions, losses)
    average = float(actions.mean())
    recall_delta = -mean_loss
    recall_ok = recall_delta >= -0.005 - EPS
    if recall_ok and average <= 45.0 + EPS:
        gate = "Strong Go"
    elif recall_ok and average <= 50.0 + EPS:
        gate = "Go"
    elif recall_ok or (
            average <= 50.0 + EPS and recall_delta >= -0.010 - EPS):
        gate = "Marginal"
    else:
        gate = "No-Go"
    return {
        "queries": len(actions),
        "average_probes": average,
        "mean_loss": mean_loss,
        "recall_delta": recall_delta,
        "distribution": {
            str(probe): int((actions == probe).sum())
            for probe in PROBES
        },
        **counts,
        "gate": gate,
        "comparison_to_d2_v8_3": {
            "average_probes_change":
                average - V83_DEVELOPMENT["average_probes"],
            "average_probes_reduction":
                V83_DEVELOPMENT["average_probes"] - average,
            "recall_delta_change":
                recall_delta - V83_DEVELOPMENT["recall_delta"],
        },
    }


def write_rows(path, rows):
    fields = list(rows[0].keys())
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flat_row(row):
    output = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            output[key] = json.dumps(
                value, sort_keys=True, separators=(",", ":"))
        else:
            output[key] = value
    return output


def render_report(report):
    oof = report["selected_oof_policy"]
    dev = report["development_policy"]
    lines = [
        "# D2-v9 Risk-Budget Calibrated Policy Search",
        "",
        f"Selected Stage16 threshold: {oof['stage16_threshold']}.",
        "",
        f"Selected Stage32 threshold: {oof['stage32_threshold']}.",
        "",
        "| Set | Avg probes | Recall delta | UCB95 | 16/32/64 |",
        "|---|---:|---:|---:|---:|",
        f"| OOF | {oof['average_probes']:.3f} | "
        f"{oof['recall_delta']:+.4f} | "
        f"{oof['bootstrap_ucb95']:.6f} | "
        f"{oof['distribution']['16']}/"
        f"{oof['distribution']['32']}/"
        f"{oof['distribution']['64']} |",
        f"| Development | {dev['average_probes']:.3f} | "
        f"{dev['recall_delta']:+.4f} | N/A | "
        f"{dev['distribution']['16']}/"
        f"{dev['distribution']['32']}/"
        f"{dev['distribution']['64']} |",
        "",
        f"Gate: **{dev['gate']}**.",
        "",
    ]
    return "\n".join(lines)


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)

    # Only OOF scores and labels are loaded before policy selection.
    oof = read_predictions(
        args.oof, range(500), REFERENCE_SHA256["oof"])
    policies, _, _, bootstrap = enumerate_policies(oof)
    selected, acceptable_count = selected_policy(policies)
    locked, _, _ = annotate_policy(selected, oof)
    frontier, anchors = pareto_and_anchors(policies, selected)

    all_path = args.output / "all_policies.csv"
    frontier_path = args.output / "pareto_frontier.csv"
    write_rows(all_path, [flat_row(row) for row in policies])
    write_rows(frontier_path, [flat_row(row) for row in frontier])

    # Thresholds are fully locked above. Development is read exactly once.
    print("development policy evaluation=1/1", flush=True)
    development = read_predictions(
        args.development, range(500, 1000),
        REFERENCE_SHA256["development"])
    dev_metrics = development_evaluation(development, locked)

    report = {
        "status": "COMPLETE",
        "model_trained": False,
        "database_run": False,
        "pgvector_modified": False,
        "development_evaluations": 1,
        "development_used_for_threshold_selection": False,
        "inputs": {
            "oof": {"path": str(args.oof), "sha256": oof["sha256"]},
            "development": {
                "path": str(args.development),
                "sha256": development["sha256"],
            },
        },
        "objective": {
            "minimize": "average_probes",
            "constraint": "bootstrap one-sided UCB95(mean signed loss) <= 0.005",
            "signed_loss": "recall64 - recall_selected; not clipped",
        },
        "bootstrap": bootstrap,
        "policies_enumerated": len(policies),
        "acceptable_policies": acceptable_count,
        "selected_oof_policy": locked,
        "development_policy": dev_metrics,
        "d2_v8_3_development": V83_DEVELOPMENT,
        "pareto": {
            "rows": len(frontier),
            "anchors": anchors,
            "path": str(frontier_path),
        },
    }
    summary_path = args.output / "summary.json"
    report_path = args.output / "report.md"
    atomic_json(summary_path, report)
    report_path.write_text(render_report(report))
    print(json.dumps({
        "selected_thresholds": {
            "stage16": locked["stage16_threshold"],
            "stage32": locked["stage32_threshold"],
        },
        "oof": {
            "average_probes": locked["average_probes"],
            "recall_delta": locked["recall_delta"],
            "bootstrap_ucb95": locked["bootstrap_ucb95"],
            "distribution": locked["distribution"],
        },
        "development": dev_metrics,
        "gate": dev_metrics["gate"],
        "artifacts": {
            "summary": str(summary_path),
            "all_policies": str(all_path),
            "pareto_frontier": str(frontier_path),
        },
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    analyze(args)


if __name__ == "__main__":
    main()
