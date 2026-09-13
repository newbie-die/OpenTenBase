#!/usr/bin/env python3
"""Summarize D1-2 standalone kernel timing and apply the requested gate."""
import csv
import json
import statistics
import sys
from pathlib import Path


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csvwrite(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_results.py run_directory")
    root = Path(sys.argv[1])
    timing = list(csv.DictReader((root / "phase_d1_2_timing_raw.csv").open()))
    correctness = list(csv.DictReader((root / "phase_d1_2_correctness.csv").open()))
    if len(timing) != 15 * 9:
        raise RuntimeError(f"expected 135 timing rows, found {len(timing)}")
    if len(correctness) != 10:
        raise RuntimeError(f"expected 10 correctness rows, found {len(correctness)}")
    if any(int(row["false_abandon"]) or int(row["nonabandon_score_mismatch"]) for row in correctness):
        raise RuntimeError("correctness gate failed")

    correctness_index = {
        (row["workload"], int(row["block_size"])): row for row in correctness
    }
    groups = {}
    for row in timing:
        key = (row["workload"], row["config"])
        groups.setdefault(key, []).append(float(row["ns_per_candidate"]))

    no_abandon_overhead = {}
    baseline_no_abandon = statistics.median(groups[("no_abandon", "baseline")])
    for block in (64, 128):
        progressive = statistics.median(groups[("no_abandon", f"progressive_b{block}")])
        no_abandon_overhead[block] = progressive / baseline_no_abandon - 1.0

    summary = []
    for workload in ("no_abandon", "realistic_all", "realistic_p64", "realistic_p128", "realistic_p256"):
        probes = 0 if workload in ("no_abandon", "realistic_all") else int(workload.removeprefix("realistic_p"))
        baseline = statistics.median(groups[(workload, "baseline")])
        for config, block in (("baseline", 0), ("progressive_b64", 64), ("progressive_b128", 128)):
            values = groups[(workload, config)]
            median = statistics.median(values)
            if block:
                check = correctness_index[(workload, block)]
                records = int(check["records"])
                early = int(check["early_abandoned"])
                avoided = int(check["avoided_dimensions"])
                evaluated = int(check["evaluated_dimensions"])
                possible = records * 960
                dimension_avoid = avoided / possible
                mean_dims = evaluated / records
                early_ratio = early / records
                checks_per_candidate = float(check["checks_per_candidate"])
                speedup = baseline / median
                improvement = 1.0 - median / baseline
                overhead = no_abandon_overhead[block]
            else:
                dimension_avoid = 0.0
                mean_dims = 960.0
                early_ratio = 0.0
                checks_per_candidate = 0.0
                speedup = 1.0
                improvement = 0.0
                overhead = 0.0
            summary.append({
                "workload": workload,
                "probes": probes,
                "config": config,
                "block_size": block,
                "trials": len(values),
                "median_ns_per_candidate": median,
                "p25_ns_per_candidate": statistics.quantiles(values, n=4)[0],
                "p75_ns_per_candidate": statistics.quantiles(values, n=4)[2],
                "speedup_x": speedup,
                "kernel_improvement_ratio": improvement,
                "dimension_avoid_ratio": dimension_avoid,
                "mean_dims_evaluated": mean_dims,
                "early_abandon_ratio": early_ratio,
                "no_abandon_overhead_ratio": overhead,
                "threshold_checks_per_candidate": checks_per_candidate,
                "false_abandon": 0,
            })

    block_decisions = {}
    for block in (64, 128):
        row = next(
            item for item in summary
            if item["workload"] == "realistic_all" and item["block_size"] == block
        )
        improvement = row["kernel_improvement_ratio"]
        overhead = no_abandon_overhead[block]
        if improvement >= 0.20 and overhead <= 0.05:
            decision = "Strong Go"
        elif improvement >= 0.10 and overhead <= 0.05:
            decision = "Go"
        elif improvement < 0.03:
            decision = "No-Go"
        else:
            decision = "Marginal"
        block_decisions[block] = decision
    rank = {"No-Go": 0, "Marginal": 1, "Go": 2, "Strong Go": 3}
    overall = max(block_decisions.values(), key=lambda decision: rank[decision])
    eligible_blocks = [
        block for block in (64, 128) if no_abandon_overhead[block] <= 0.05
    ]
    recommended = max(
        eligible_blocks,
        key=lambda block: next(
            row["kernel_improvement_ratio"] for row in summary
            if row["workload"] == "realistic_all" and row["block_size"] == block
        ),
    ) if eligible_blocks else None
    best_observed = max(
        (64, 128),
        key=lambda block: next(
            row["kernel_improvement_ratio"] for row in summary
            if row["workload"] == "realistic_all" and row["block_size"] == block
        ),
    )

    assembly = json.loads((root / "phase_d1_2_assembly_audit.json").read_text())
    analysis = {
        "phase": "D1-2",
        "status": "COMPLETE",
        "decision": overall,
        "block_decisions": {str(key): value for key, value in block_decisions.items()},
        "recommended_block_size_for_d1_3": recommended,
        "best_observed_block_size": best_observed,
        "d1_3_gate_satisfied": recommended is not None,
        "gate_basis": "median realistic_all kernel ns/candidate and matching block no-abandon overhead",
        "gate_rule": "improvement >=20% and overhead <=5% => Strong Go; >=10% and overhead <=5% => Go; improvement <3% => No-Go; otherwise Marginal",
        "no_abandon_overhead": {str(key): value for key, value in no_abandon_overhead.items()},
        "realistic_results": [
            row for row in summary
            if row["workload"].startswith("realistic") and row["block_size"] in (64, 128)
        ],
        "correctness": {
            "sample_records": int(correctness[0]["records"]) if correctness else 0,
            "false_abandon": sum(int(row["false_abandon"]) for row in correctness),
            "nonabandon_score_mismatch": sum(int(row["nonabandon_score_mismatch"]) for row in correctness),
            "baseline_vs_production_bitwise_mismatch": int(correctness[0]["baseline_vs_production_bitwise_mismatch"]),
            "max_abs_score_difference": float(correctness[0]["max_abs_score_difference"]),
        },
        "assembly": assembly,
        "limitations": [
            "Standalone raw-kernel timing only; no PostgreSQL scan, tuplesort, visibility, fallback, or end-to-end latency claim.",
            "Realistic thresholds and candidates are a deterministic stratified reservoir sample from the completed D1-1 K40 stream.",
            "No production source or vector.so was modified.",
        ],
        "d1_3_started": False,
    }
    csvwrite(root / "phase_d1_2_summary.csv", summary)
    save(root / "phase_d1_2_analysis.json", analysis)
    print(json.dumps({
        "decision": overall,
        "block_decisions": block_decisions,
        "recommended_block_size_for_d1_3": recommended,
        "best_observed_block_size": best_observed,
        "no_abandon_overhead": no_abandon_overhead,
    }, indent=2))


if __name__ == "__main__":
    main()
