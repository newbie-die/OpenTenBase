#!/usr/bin/env python3
"""D2-v10 oracle-gap mechanism attribution without model fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

import progressive_shadow as shadow
import risk_ranked as v7

EPS = 1e-12
BUDGET_PER_QUERY = 0.005
PROBES = (16, 32, 64)
PREDICTION_SHA = {
    "oof": "778a2002c1a37132050f50df2a47ec0ff49c909b6dd00dedf6092a16d30245d0",
    "development": "179e06d47d46229cc0287aa8dffd34b4ad754afdfdd8acc2e89498e2078aac52",
}
EXPECTED_ORACLE = {
    "oof": {"average_probes": 31.808, "distribution": {"16": 232, "32": 155, "64": 113}},
    "development": {"average_probes": 34.592, "distribution": {"16": 209, "32": 146, "64": 145}},
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_predictions(path, expected_qids, expected_sha):
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"prediction SHA mismatch: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    qids = np.asarray([int(row["query_id"]) for row in rows])
    if not np.array_equal(qids, np.asarray(expected_qids)):
        raise ValueError(f"qid mismatch: {path}")
    return {
        "qid": qids,
        "recall16": np.asarray([float(row["recall16"]) for row in rows]),
        "recall32": np.asarray([float(row["recall32"]) for row in rows]),
        "recall64": np.asarray([float(row["recall64"]) for row in rows]),
        "stage16_raw_score": np.asarray([float(row["stage16_raw_score"]) for row in rows]),
        "stage16_calibrated_score": np.asarray([float(row["stage16_calibrated_score"]) for row in rows]),
        "stage32_raw_probability": np.asarray([float(row["stage32_raw_probability"]) for row in rows]),
        "stage32_calibrated_probability": np.asarray([float(row["stage32_calibrated_probability"]) for row in rows]),
    }


def exact_budgeted_oracle(data):
    n = len(data["qid"])
    loss16 = data["recall64"] - data["recall16"]
    loss32 = data["recall64"] - data["recall32"]
    units16 = np.rint(loss16 * 10).astype(int)
    units32 = np.rint(loss32 * 10).astype(int)
    if (np.max(np.abs(units16 / 10 - loss16)) > 1e-9 or
            np.max(np.abs(units32 / 10 - loss32)) > 1e-9):
        raise ValueError("Recall losses are not integer tenths")
    if units16.min() < 0 or units32.min() < 0:
        raise ValueError("this exact bounded DP expects nonnegative losses")

    budget_units = int(math.floor(n * BUDGET_PER_QUERY * 10 + EPS))
    costs = {0: 0}
    parents = []
    for qid in range(n):
        next_costs = {}
        parent = {}
        for prior_units, prior_cost in costs.items():
            for probe, units in ((16, units16[qid]), (32, units32[qid]), (64, 0)):
                total_units = prior_units + int(units)
                if total_units > budget_units:
                    continue
                cost = prior_cost + probe
                if total_units not in next_costs or cost < next_costs[total_units]:
                    next_costs[total_units] = cost
                    parent[total_units] = (prior_units, probe)
        costs = next_costs
        parents.append(parent)

    final_units = min(costs, key=lambda units: (costs[units], units))
    actions = np.empty(n, dtype=np.int16)
    units = final_units
    for qid in range(n - 1, -1, -1):
        prior_units, probe = parents[qid][units]
        actions[qid] = probe
        units = prior_units
    losses = np.where(actions == 16, loss16,
                      np.where(actions == 32, loss32, 0.0))
    return {
        "actions": actions,
        "losses": losses,
        "average_probes": float(actions.mean()),
        "recall_delta": -float(losses.mean()),
        "distribution": {str(probe): int((actions == probe).sum()) for probe in PROBES},
        "loss_budget_units": budget_units,
        "actual_loss_units": final_units,
    }


def risk_policy_actions(data, thresholds):
    return np.where(
        data["stage16_calibrated_score"] <= thresholds["stage16"],
        16,
        np.where(
            data["stage32_calibrated_probability"] <= thresholds["stage32"],
            32, 64),
    ).astype(np.int16)


def feature_category(name):
    if name.startswith("stage"):
        return "risk_score"
    if name.startswith("r16_") or name == "candidates16":
        return "stage16_execution"
    if (name.startswith("r32_") or name == "candidates32" or
            name in ("overlap16_32", "replacements16_32",
                     "kth_rel_change", "mean_rel_change",
                     "max_rel_change", "candidate_increase",
                     "candidate_rel_increase")):
        return "stage32_execution"
    return "centroid_geometry"


def build_feature_table(args, prediction_data):
    raw = shadow.read_raw(args.raw)
    old = shadow.read_centroids(args.old_features)
    geometry = v7.read_geometry(args.geometry)
    built = v7.build_features(raw, old, geometry)
    qids = prediction_data["qid"]
    values = built["extended32"][qids]
    names = list(built["extended32_names"])
    extra_names = [
        "stage16_raw_score", "stage16_calibrated_score",
        "stage32_raw_probability", "stage32_calibrated_probability",
    ]
    extra = np.column_stack([prediction_data[name] for name in extra_names])
    result = np.column_stack((values, extra))
    if not np.isfinite(result).all():
        raise ValueError("non-finite feature value")
    return names + extra_names, result


def auc_for_groups(values_a, values_b):
    from scipy.stats import rankdata
    combined = np.concatenate((values_a, values_b))
    labels = np.concatenate((np.ones(len(values_a)), np.zeros(len(values_b))))
    ranks = rankdata(combined, method="average")
    n_pos = len(values_a)
    n_neg = len(values_b)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def feature_statistics(names, values, mask_a, mask_b):
    from scipy.stats import ks_2samp

    rows = []
    for column, name in enumerate(names):
        a = values[mask_a, column]
        b = values[mask_b, column]
        mean_a = float(a.mean())
        mean_b = float(b.mean())
        std_a = float(a.std(ddof=1)) if len(a) > 1 else 0.0
        std_b = float(b.std(ddof=1)) if len(b) > 1 else 0.0
        denominator = len(a) + len(b) - 2
        pooled = math.sqrt(
            ((len(a) - 1) * std_a ** 2 + (len(b) - 1) * std_b ** 2) / denominator
        ) if denominator > 0 else 0.0
        effect = (mean_a - mean_b) / pooled if pooled > EPS else 0.0
        raw_auc = auc_for_groups(a, b) if (len(a) and len(b)) else 0.5
        separation = max(raw_auc, 1.0 - raw_auc)
        direction = "higher_in_A" if raw_auc >= 0.5 else "lower_in_A"
        rows.append({
            "feature": name,
            "category": feature_category(name),
            "group_a_count": int(len(a)),
            "group_b_count": int(len(b)),
            "group_a_mean": mean_a,
            "group_a_std": std_a,
            "group_b_mean": mean_b,
            "group_b_std": std_b,
            "cohens_d_a_minus_b": float(effect),
            "ks_statistic": float(ks_2samp(a, b).statistic),
            "roc_auc_a_positive": raw_auc,
            "separation_auc": separation,
            "auc_direction": direction,
        })
    return rows


def transition_attribution(oracle, policy, n):
    cells = []
    total_gap = 0.0
    for oracle_probe in PROBES:
        for policy_probe in PROBES:
            count = int(((oracle == oracle_probe) & (policy == policy_probe)).sum())
            contribution = (policy_probe - oracle_probe) * count / n
            total_gap += contribution
            cells.append({
                "oracle_probe": oracle_probe,
                "policy_probe": policy_probe,
                "queries": count,
                "average_probe_gap_contribution": contribution,
            })
    return cells, total_gap


def analyze_set(name, data, features, feature_names, thresholds):
    oracle = exact_budgeted_oracle(data)
    expected = EXPECTED_ORACLE[name]
    if (abs(oracle["average_probes"] - expected["average_probes"]) > EPS or
            oracle["distribution"] != expected["distribution"]):
        raise RuntimeError(f"{name} oracle mismatch: {oracle}")

    policy = risk_policy_actions(data, thresholds)
    loss16 = data["recall64"] - data["recall16"]
    loss32 = data["recall64"] - data["recall32"]
    policy_losses = np.where(policy == 16, loss16,
                             np.where(policy == 32, loss32, 0.0))
    group_a = (oracle["actions"] < 64) & (policy == 64)
    group_b = (oracle["actions"] == 64) & (policy == 64)
    stats = feature_statistics(feature_names, features, group_a, group_b)
    transitions, transition_gap = transition_attribution(
        oracle["actions"], policy, len(policy))
    policy_average = float(policy.mean())
    exact_gap = policy_average - oracle["average_probes"]
    if abs(transition_gap - exact_gap) > EPS:
        raise RuntimeError("transition attribution does not sum to gap")

    oracle16_missed = group_a & (oracle["actions"] == 16)
    oracle32_missed = group_a & (oracle["actions"] == 32)
    saving16 = 48 * int(oracle16_missed.sum()) / len(policy)
    saving32 = 32 * int(oracle32_missed.sum()) / len(policy)
    missed_saving = saving16 + saving32
    theoretical_average = policy_average - missed_saving
    return {
        "set": name,
        "queries": len(policy),
        "oracle": {key: value for key, value in oracle.items()
                   if key not in ("actions", "losses")},
        "policy": {
            "average_probes": policy_average,
            "recall_delta": -float(policy_losses.mean()),
            "distribution": {str(probe): int((policy == probe).sum()) for probe in PROBES},
        },
        "group_a": {
            "definition": "oracle action 16/32 and policy action 64",
            "queries": int(group_a.sum()),
            "ratio": float(group_a.mean()),
            "oracle16_queries": int(oracle16_missed.sum()),
            "oracle32_queries": int(oracle32_missed.sum()),
            "strict_zero_loss_oracle16": int((oracle16_missed & (loss16 <= EPS)).sum()),
            "strict_zero_loss_oracle32": int((oracle32_missed & (loss32 <= EPS)).sum()),
            "average_probe_saving_if_replaced_by_oracle_action": missed_saving,
            "theoretical_policy_average_after_replacement": theoretical_average,
        },
        "group_b": {
            "definition": "oracle action 64 and policy action 64",
            "queries": int(group_b.sum()),
            "ratio": float(group_b.mean()),
            "absolute_average_probe_load": 64 * int(group_b.sum()) / len(policy),
            "oracle_gap_contribution": 0.0,
        },
        "gap": {
            "policy_minus_oracle_average_probes": exact_gap,
            "safe16_missed_opportunity": saving16,
            "safe32_missed_opportunity": saving32,
            "necessary64_same_action_gap_contribution": 0.0,
            "other_transition_net_contribution": exact_gap - missed_saving,
            "transition_cells": transitions,
        },
        "feature_statistics": stats,
        "_group_a_mask": group_a,
        "_group_b_mask": group_b,
        "_oracle_actions": oracle["actions"],
        "_policy_actions": policy,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_groups(path, data, result):
    with path.open("w", newline="") as target:
        fields = ("query_id", "group", "oracle_probe", "policy_probe")
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for index, qid in enumerate(data["qid"]):
            group = ("A" if result["_group_a_mask"][index]
                     else "B" if result["_group_b_mask"][index] else "other")
            writer.writerow({
                "query_id": int(qid),
                "group": group,
                "oracle_probe": int(result["_oracle_actions"][index]),
                "policy_probe": int(result["_policy_actions"][index]),
            })


def public_result(result):
    return {key: value for key, value in result.items() if not key.startswith("_")}


def render_report(report):
    primary = report["oof"]
    conclusion = report["conclusion"]
    top = conclusion["top_features"]
    lines = [
        "# D2-v10 Oracle Gap Attribution",
        "",
        "The cited 34.592 oracle belongs to development qids 500-999. "
        "The same-set OOF oracle is 31.808.",
        "",
        "| Set | Oracle avg | Policy avg | Gap | Group A | Group B |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in (report["oof"], report["development"]):
        lines.append(
            f"| {result['set']} | {result['oracle']['average_probes']:.3f} | "
            f"{result['policy']['average_probes']:.3f} | "
            f"{result['gap']['policy_minus_oracle_average_probes']:.3f} | "
            f"{result['group_a']['queries']} | {result['group_b']['queries']} |")
    lines += [
        "",
        "| Top feature | Category | Separation AUC | KS | Cohen d |",
        "|---|---|---:|---:|---:|",
    ]
    for row in top[:10]:
        lines.append(
            f"| {row['feature']} | {row['category']} | "
            f"{row['separation_auc']:.3f} | {row['ks_statistic']:.3f} | "
            f"{row['cohens_d_a_minus_b']:+.3f} |")
    lines += ["", f"Conclusion: **{conclusion['classification']}**.", ""]
    return "\n".join(lines)


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    policy_summary = json.loads(args.policy_summary.read_text())
    thresholds = {
        "stage16": float(policy_summary["selected_oof_policy"]["stage16_threshold"]),
        "stage32": float(policy_summary["selected_oof_policy"]["stage32_threshold"]),
    }

    oof_data = read_predictions(args.oof, range(500), PREDICTION_SHA["oof"])
    dev_data = read_predictions(args.development, range(500, 1000), PREDICTION_SHA["development"])
    feature_names_oof, features_oof = build_feature_table(args, oof_data)
    feature_names_dev, features_dev = build_feature_table(args, dev_data)
    if feature_names_oof != feature_names_dev:
        raise RuntimeError("feature-name mismatch")

    print("analyzing OOF oracle gap", flush=True)
    oof = analyze_set("oof", oof_data, features_oof, feature_names_oof, thresholds)
    print("analyzing development oracle gap", flush=True)
    development = analyze_set(
        "development", dev_data, features_dev, feature_names_dev, thresholds)

    oof_stats = oof["feature_statistics"]
    best_feature = max(oof_stats, key=lambda row: row["separation_auc"])
    risk_rows = [row for row in oof_stats if row["category"] == "risk_score"]
    best_risk = max(risk_rows, key=lambda row: row["separation_auc"])
    if best_feature["separation_auc"] < 0.7 and best_risk["separation_auc"] < 0.7:
        classification = "STATIC_FEATURE_INFORMATION_CEILING"
        exploitable = []
    else:
        classification = "EXISTING_FEATURE_OPPORTUNITY"
        exploitable = [
            row for row in sorted(
                oof_stats, key=lambda row: row["separation_auc"], reverse=True)
            if row["separation_auc"] >= 0.7
        ]
    top_features = sorted(
        oof_stats, key=lambda row: (
            row["separation_auc"], row["ks_statistic"]), reverse=True)

    oof_stats_path = args.output / "oof_group_a_b_feature_stats.csv"
    dev_stats_path = args.output / "development_group_a_b_feature_stats.csv"
    oof_groups_path = args.output / "oof_oracle_policy_groups.csv"
    dev_groups_path = args.output / "development_oracle_policy_groups.csv"
    write_csv(oof_stats_path, oof_stats)
    write_csv(dev_stats_path, development["feature_statistics"])
    write_groups(oof_groups_path, oof_data, oof)
    write_groups(dev_groups_path, dev_data, development)

    report = {
        "status": "COMPLETE",
        "model_trained": False,
        "policy_modified": False,
        "database_run": False,
        "set_scope_correction": {
            "requested_cross_set_comparison": {
                "oracle_development": 34.592,
                "policy_oof": 51.392,
                "arithmetic_gap": 16.8,
            },
            "valid_oof_comparison": {
                "oracle": 31.808,
                "policy": 51.392,
            },
            "valid_development_comparison": {
                "oracle": 34.592,
                "policy": 53.824,
            },
            "note": "gap attribution is performed only within the same query set",
        },
        "thresholds_frozen": thresholds,
        "oof": public_result(oof),
        "development": public_result(development),
        "conclusion": {
            "classification": classification,
            "best_single_feature": best_feature,
            "best_risk_score": best_risk,
            "top_features": top_features[:10],
            "features_with_separation_auc_at_least_0_7": exploitable,
            "criterion": "ceiling iff best single-feature separation AUC < 0.7 and best risk-score separation AUC < 0.7",
        },
        "artifacts": {
            "oof_feature_stats": str(oof_stats_path),
            "development_feature_stats": str(dev_stats_path),
            "oof_groups": str(oof_groups_path),
            "development_groups": str(dev_groups_path),
        },
    }
    summary_path = args.output / "summary.json"
    report_path = args.output / "report.md"
    atomic_json(summary_path, report)
    report_path.write_text(render_report(report))
    print(json.dumps({
        "oof": {
            "oracle_average_probes": oof["oracle"]["average_probes"],
            "policy_average_probes": oof["policy"]["average_probes"],
            "gap": oof["gap"]["policy_minus_oracle_average_probes"],
            "group_a": oof["group_a"],
            "group_b": oof["group_b"],
        },
        "development": {
            "oracle_average_probes": development["oracle"]["average_probes"],
            "policy_average_probes": development["policy"]["average_probes"],
            "gap": development["gap"]["policy_minus_oracle_average_probes"],
            "group_a": development["group_a"],
            "group_b": development["group_b"],
        },
        "classification": classification,
        "best_feature": best_feature,
        "best_risk": best_risk,
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--policy-summary", type=Path, required=True)
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
