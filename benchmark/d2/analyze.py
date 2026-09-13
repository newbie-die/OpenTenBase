#!/usr/bin/env python3
"""Offline D2 Oracle report and one-feature adaptive-rule search (stdlib only)."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

PROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RULE_PROBES = (16, 32, 64, 128)


def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("empty values")
    return values[math.ceil(q / 100 * len(values)) - 1]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def phase_c_recalls(path, system, round_no):
    recalls = defaultdict(dict)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["system"] == system and int(row["round"]) == round_no:
                recalls[int(row["query_id"])][int(row["probes"])] = float(row["recall_at_10"])
    if len(recalls) != 1000 or any(set(values) != set(PROBES) for values in recalls.values()):
        raise ValueError("Phase C must contain exactly 1000 queries and all nine probes for the selected system/round")
    return recalls


def oracle_report(recalls, target):
    oracle = {}
    unmet = []
    for qid, values in recalls.items():
        probe = next((probe for probe in PROBES if values[probe] >= target), None)
        if probe is None:
            # Preserve a bounded average while making the non-achievement explicit.
            probe = PROBES[-1]
            unmet.append(qid)
        oracle[qid] = probe
    values = list(oracle.values())
    average = statistics.fmean(values)
    return {
        "target": target,
        "queries": len(values),
        "average_oracle_probes": average,
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "probe_distribution": {str(probe): sum(value == probe for value in values) for probe in PROBES},
        "unmet_at_max_probes": len(unmet),
        "unmet_query_ids": unmet,
        "average_oracle_probes_is_lower_bound": bool(unmet),
        "potential_probe_reduction_is_upper_bound": bool(unmet),
        "fixed_probe_baseline": 64,
        "potential_probe_reduction": 1.0 - average / 64.0,
        "oracle_probes_by_query": oracle,
    }


def read_features(path):
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            qid = int(row["query_id"])
            if qid in rows:
                raise ValueError(f"duplicate feature query_id {qid}")
            rows[qid] = {key: float(value) if key != "query_id" else qid for key, value in row.items()}
    if set(rows) != set(range(1000)):
        raise ValueError("features CSV must contain query_id 0..999 exactly once")
    return rows


def candidate_thresholds(values, count=51):
    ordered = sorted(set(values))
    if not ordered:
        raise ValueError("no feature values")
    picks = {0, len(ordered) - 1}
    for i in range(count):
        picks.add(round(i * (len(ordered) - 1) / (count - 1)))
    # +infinity represents an empty easier bucket.
    return [ordered[i] for i in sorted(picks)] + [float.fromhex("0x1.fffffffffffffp+1023")]


def choose_probe(value, thresholds):
    t16, t32, t64 = thresholds
    if value >= t16:
        return 16
    if value >= t32:
        return 32
    if value >= t64:
        return 64
    return 128


def score(ids, features, recalls, thresholds):
    probes = [choose_probe(features[qid]["d2_d1"], thresholds) for qid in ids]
    observed = [recalls[qid][probe] for qid, probe in zip(ids, probes)]
    return {
        "queries": len(ids),
        "mean_recall_at_10": statistics.fmean(observed),
        "average_probes": statistics.fmean(probes),
        "probe_distribution": {str(p): probes.count(p) for p in RULE_PROBES},
        "recall_by_query": {str(qid): recall for qid, recall in zip(ids, observed)},
        "probes_by_query": {str(qid): probe for qid, probe in zip(ids, probes)},
    }


def search_rule(features, recalls, target, max_recall_delta):
    calibration = list(range(500))
    test = list(range(500, 1000))
    fixed = statistics.fmean(recalls[qid][64] for qid in calibration)
    floor = max(target, fixed - max_recall_delta)
    values = [features[qid]["d2_d1"] for qid in calibration]
    candidates = candidate_thresholds(values)
    best = None
    for i, t16 in enumerate(candidates):
        for j in range(i, len(candidates)):
            t32 = candidates[j]
            for k in range(j, len(candidates)):
                thresholds = (t16, t32, candidates[k])
                summary = score(calibration, features, recalls, thresholds)
                if summary["mean_recall_at_10"] + 1e-12 < floor:
                    continue
                key = (summary["average_probes"], -summary["mean_recall_at_10"], thresholds)
                if best is None or key < best[0]:
                    best = (key, thresholds, summary)
    if best is None:
        raise RuntimeError("no monotonic d2/d1 rule meets the calibration recall floor")
    _, thresholds, calibration_summary = best
    test_summary = score(test, features, recalls, thresholds)
    fixed_test = statistics.fmean(recalls[qid][64] for qid in test)
    result = {
        "feature": "d2_d1",
        "rule": "if d2/d1 >= ratio_16: 16; elif >= ratio_32: 32; elif >= ratio_64: 64; else: 128",
        "thresholds": {"ratio_16": thresholds[0], "ratio_32": thresholds[1], "ratio_64": thresholds[2]},
        "calibration_queries": [0, 499],
        "test_queries": [500, 999],
        "target_recall": target,
        "max_recall_delta_vs_fixed_64": max_recall_delta,
        "calibration_fixed_64_recall": fixed,
        "calibration_recall_floor": floor,
        "calibration": calibration_summary,
        "test_fixed_64_recall": fixed_test,
        "test": test_summary,
        "test_recall_delta_vs_fixed_64": test_summary["mean_recall_at_10"] - fixed_test,
        "test_probe_reduction_vs_fixed_64": 1.0 - test_summary["average_probes"] / 64.0,
        "sql_settings": [
            "SET ivfflat.probes = 128;",
            "SET ivfflat.adaptive_probes = on;",
            f"SET ivfflat.adaptive_probes_ratio_16 = {thresholds[0]:.17g};",
            f"SET ivfflat.adaptive_probes_ratio_32 = {thresholds[1]:.17g};",
            f"SET ivfflat.adaptive_probes_ratio_64 = {thresholds[2]:.17g};",
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default="vanilla_eq")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--max-recall-delta", type=float, default=0.005)
    args = parser.parse_args()
    recalls = phase_c_recalls(args.phase_c, args.system, args.round)
    report = {str(target): oracle_report(recalls, target) for target in (0.90, 0.95, 0.99)}
    gate = report["0.95"]["potential_probe_reduction"] >= 0.20
    result = {"phase_c": str(args.phase_c), "system": args.system, "round": args.round,
              "oracle": report, "continue_to_heuristic": gate,
              "gate": "target=0.95 potential_probe_reduction >= 0.20"}
    if args.features:
        if not gate:
            raise RuntimeError("Oracle gate failed; heuristic search is forbidden")
        result["heuristic"] = search_rule(read_features(args.features), recalls,
                                           args.target, args.max_recall_delta)
    save_json(args.output, result)
    for target, values in report.items():
        print(f"target={target} average_oracle_probes={values['average_oracle_probes']:.3f} "
              f"median={values['median']:.0f} p95={values['p95']:.0f} "
              f"potential_probe_reduction={values['potential_probe_reduction']:.2%}")
    if args.features:
        heuristic = result["heuristic"]
        print("heuristic test", f"recall={heuristic['test']['mean_recall_at_10']:.4f}",
              f"delta={heuristic['test_recall_delta_vs_fixed_64']:.4f}",
              f"average_probes={heuristic['test']['average_probes']:.3f}",
              f"reduction={heuristic['test_probe_reduction_vs_fixed_64']:.2%}")

if __name__ == "__main__":
    main()
