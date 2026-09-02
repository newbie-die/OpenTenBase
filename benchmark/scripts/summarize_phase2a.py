#!/usr/bin/env python3
import argparse, csv, statistics
from pathlib import Path

def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    rows = []
    for path in sorted(directory.glob("glove_l2_p*_b*_r*.csv")):
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                for key in ("probes", "sort_bound", "round", "min_returned_rows", "topk"):
                    row[key] = int(row[key])
                for key in ("recall_at_10", "p50_ms", "p95_ms", "p99_ms", "qps"):
                    row[key] = float(row[key])
                rows.append(row)
    if not rows:
        raise SystemExit(f"no Phase 2A configuration CSVs found in {directory}")
    summary = []
    for probes in sorted({row["probes"] for row in rows}):
        probe_rows = [row for row in rows if row["probes"] == probes]
        baseline = [row for row in probe_rows if row["sort_bound"] == 0]
        if not baseline:
            raise SystemExit(f"missing bound=0 baseline for probes={probes}")
        baseline_p95 = statistics.median(row["p95_ms"] for row in baseline)
        baseline_qps = statistics.median(row["qps"] for row in baseline)
        for bound in sorted({row["sort_bound"] for row in probe_rows}):
            group = [row for row in probe_rows if row["sort_bound"] == bound]
            p95 = statistics.median(row["p95_ms"] for row in group)
            qps = statistics.median(row["qps"] for row in group)
            minimum = min(row["min_returned_rows"] for row in group)
            summary.append({
                "probes": probes, "sort_bound": bound,
                "median_recall_at_10": statistics.median(row["recall_at_10"] for row in group),
                "median_p50_ms": statistics.median(row["p50_ms"] for row in group),
                "median_p95_ms": p95,
                "median_p99_ms": statistics.median(row["p99_ms"] for row in group),
                "median_qps": qps, "baseline_p95_ms": baseline_p95,
                "p95_improvement_pct": 100 * (baseline_p95 - p95) / baseline_p95,
                "baseline_qps": baseline_qps,
                "qps_improvement_pct": 100 * (qps - baseline_qps) / baseline_qps,
                "min_returned_rows": minimum,
                "status": "correct" if minimum == group[0]["topk"] else "unsafe/incorrect",
            })
    write_csv(directory / "phase2a_raw.csv", rows)
    write_csv(directory / "phase2a_summary.csv", summary)

if __name__ == "__main__":
    main()
