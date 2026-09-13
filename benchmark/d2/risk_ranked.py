#!/usr/bin/env python3
"""D2-v7 risk-ranked lost-neighbor budget allocation (offline/shadow only)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path

import numpy as np

import loss_budgeted as v6
import progressive_shadow as shadow

TRACE_RE = re.compile(r"IVFFLAT_ADAPTIVE\s+(.*)")
GEOMETRY_FIELDS = (
    "query_id", "d1", "d2", "d4", "d8", "d16", "d32", "d64",
    "d2_d1", "d4_d1", "d8_d1", "d16_d1", "d32_d1", "d64_d1",
    "d32_d16", "d64_d32", "gap", "gap32_16_d1", "gap64_32_d1",
    "centroid_slope", "centroid_curvature", "centroid_mean",
    "centroid_std", "centroid_range",
)
EXTENDED_NAMES = [
    "d32_d1", "d64_d1", "d32_d16", "d64_d32",
    "gap32_16_d1", "gap64_32_d1", "centroid_slope",
    "centroid_curvature", "centroid_mean", "centroid_std", "centroid_range",
]
CAL_IDS = np.arange(500)
TEST_IDS = np.arange(500, 1000)
EPS = 1e-12


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def geometry_derived(fields):
    distances = np.asarray([fields[name] for name in ("d1", "d2", "d4", "d8", "d16", "d32", "d64")])
    normalized = distances / distances[0]
    ranks = np.arange(7, dtype=float)
    slope = float(np.polyfit(ranks, normalized, 1)[0])
    curvature = float(np.diff(normalized, n=2).mean())
    return {
        "centroid_slope": slope,
        "centroid_curvature": curvature,
        "centroid_mean": float(distances.mean()),
        "centroid_std": float(distances.std()),
        "centroid_range": float(distances.max() - distances.min()),
    }


def parse_trace(notices):
    required = set(GEOMETRY_FIELDS[1:19])
    for notice in reversed(notices):
        match = TRACE_RE.search(notice)
        if not match:
            continue
        fields = {key: float(value) for key, value in shadow.FIELD_RE.findall(match.group(1))}
        if required <= set(fields):
            fields.update(geometry_derived(fields))
            return fields
    raise RuntimeError("missing complete d1..d64 IVFFLAT_ADAPTIVE trace")


def read_geometry(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != GEOMETRY_FIELDS:
            raise ValueError("unexpected geometry CSV header")
        for row in reader:
            qid = int(row["query_id"])
            if qid in rows:
                raise ValueError(f"duplicate geometry qid {qid}")
            rows[qid] = {key: (qid if key == "query_id" else float(value)) for key, value in row.items()}
    return rows


def collect(args):
    import psycopg2

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "centroid64.csv"
    rows = read_geometry(path)
    old = shadow.read_centroids(args.old_features)
    frozen = shadow.load_phase_c(args.phase_c, args.system, args.round)
    queries, _ = shadow.load_workload(shadow.QUERY_COUNT)
    literals = [shadow.vector_literal(query) for query in queries]
    started = time.perf_counter()
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname,
                            user=args.user, application_name="d2_v7_centroid64")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cur.execute("SET LOCAL lock_timeout=10000")
            cur.execute("LOCK TABLE gist_base IN SHARE MODE")
            shadow.configure(cur, 64)
            shadow.set_if_present(cur, "ivfflat.adaptive_probes_trace", "on", required=True)
            sql = "SELECT id FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10"
            cur.execute("EXPLAIN (FORMAT JSON) " + sql, (literals[0],))
            plan = cur.fetchone()[0]
            if "gist_ivf_l2" not in json.dumps(plan) or "Index Scan" not in json.dumps(plan):
                raise RuntimeError("expected gist_ivf_l2 Index Scan")
            atomic_json(args.output / "plan.json", plan)
            mode = "a" if path.exists() else "w"
            with path.open(mode, newline="", buffering=1) as target:
                writer = csv.DictWriter(target, fieldnames=GEOMETRY_FIELDS)
                if mode == "w":
                    writer.writeheader()
                for qid, literal in enumerate(literals):
                    if qid in rows:
                        continue
                    conn.notices.clear()
                    cur.execute(sql, (literal,))
                    ids = [int(row[0]) for row in cur.fetchall()]
                    if ids != frozen[(qid, 64)]["ids"]:
                        raise RuntimeError(f"Phase C p64 ID mismatch qid={qid}")
                    fields = parse_trace(conn.notices)
                    for name in ("d1", "d2_d1", "d4_d1", "d8_d1", "d16_d1", "gap"):
                        if not math.isclose(fields[name], old[qid][name], rel_tol=1e-12, abs_tol=1e-12):
                            raise RuntimeError(f"old centroid feature mismatch qid={qid} feature={name}")
                    row = {"query_id": qid, **{name: fields[name] for name in GEOMETRY_FIELDS[1:]}}
                    writer.writerow(row)
                    rows[qid] = row
                    done = len(rows); elapsed = time.perf_counter() - started
                    if done % args.progress_every == 0 or done == shadow.QUERY_COUNT:
                        eta = elapsed * (shadow.QUERY_COUNT - done) / done
                        value = {"status": "RUNNING", "query": f"{done}/{shadow.QUERY_COUNT}",
                                 "elapsed_seconds": elapsed, "eta_seconds": eta}
                        atomic_json(args.output / "progress.json", value)
                        print(" ".join(f"{key}={value[key]}" for key in value if key != "status"), flush=True)
            if set(rows) != set(range(shadow.QUERY_COUNT)):
                raise RuntimeError(f"geometry collection incomplete: {len(rows)}/1000")
            conn.commit()
            elapsed = time.perf_counter() - started
            atomic_json(args.output / "progress.json", {"status": "COMPLETE", "query": "1000/1000",
                        "elapsed_seconds": elapsed, "eta_seconds": 0.0})
            atomic_json(args.output / "manifest.json", {"status": "COMPLETE", "queries": 1000,
                        "probes": 64, "candidate_scan_added": False,
                        "source": "GetScanLists existing sorted centroid distances",
                        "features": list(GEOMETRY_FIELDS[1:]), "elapsed_seconds": elapsed})
    finally:
        conn.close()


def lost_neighbor_classes(loss):
    counts = np.rint(loss * 10.0).astype(int)
    if np.max(np.abs(counts / 10.0 - loss)) > 1e-9:
        raise ValueError("Recall@10 loss is not an integer neighbor count")
    # A lower-probe result can occasionally beat p64; this is zero lost neighbors.
    return np.clip(counts, 0, 2)


def build_features(raw, old_centroids, geometry):
    if set(geometry) != set(range(1000)):
        raise ValueError("centroid64 CSV must contain qids 0..999")
    base = shadow.make_features(raw, old_centroids)
    extended = np.asarray([[geometry[qid][name] for name in EXTENDED_NAMES]
                           for qid in range(1000)], dtype=float)
    return {
        "base16": np.asarray(base["stage16"], dtype=float),
        "base32": np.asarray(base["stage32"], dtype=float),
        "extended16": np.column_stack((np.asarray(base["stage16"], dtype=float), extended)),
        "extended32": np.column_stack((np.asarray(base["stage32"], dtype=float), extended)),
        "base16_names": base["stage16_names"], "base32_names": base["stage32_names"],
        "extended16_names": base["stage16_names"] + EXTENDED_NAMES,
        "extended32_names": base["stage32_names"] + EXTENDED_NAMES,
    }


def model_specs(family):
    if family == "logistic":
        return [{"C": c, "class_weight": weight} for c in (0.1, 1.0, 10.0)
                for weight in (None, "balanced")]
    if family == "gradient_boosting":
        return [{"max_depth": depth, "n_estimators": estimators, "learning_rate": rate}
                for depth in (1, 2) for estimators in (50, 100) for rate in (0.05, 0.1)]
    if family == "random_forest":
        return [{"max_depth": depth, "min_samples_leaf": leaf, "n_estimators": 150,
                 "class_weight": weight}
                for depth in (6, None) for leaf in (5, 15) for weight in (None, "balanced")]
    raise ValueError(family)


def build_model(family, spec):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if family == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"],
            class_weight=spec["class_weight"], max_iter=3000, random_state=20260913))
    if family == "gradient_boosting":
        return GradientBoostingClassifier(max_depth=spec["max_depth"],
            n_estimators=spec["n_estimators"], learning_rate=spec["learning_rate"],
            random_state=20260913)
    return RandomForestClassifier(max_depth=spec["max_depth"],
        min_samples_leaf=spec["min_samples_leaf"], n_estimators=spec["n_estimators"],
        class_weight=spec["class_weight"], n_jobs=-1, random_state=20260913)


def aligned_probabilities(model, x):
    probabilities = np.zeros((len(x), 3), dtype=float)
    values = model.predict_proba(x)
    for column, label in enumerate(model.classes_):
        probabilities[:, int(label)] = values[:, column]
    return probabilities


def oof_probabilities(family, spec, x16, x32, y16, y32):
    outputs = [np.zeros((500, 3)), np.zeros((500, 3))]
    for fold in range(5):
        valid = CAL_IDS[CAL_IDS % 5 == fold]; train = CAL_IDS[CAL_IDS % 5 != fold]
        for x, y, output in ((x16, y16, outputs[0]), (x32, y32, outputs[1])):
            model = build_model(family, spec)
            model.fit(x[train], y[train])
            output[valid] = aligned_probabilities(model, x[valid])
    return outputs


def risk_score(probabilities, severe_penalty):
    return probabilities[:, 1] + severe_penalty * probabilities[:, 2]


def allocation_frontier(risk16, risk32):
    """Exact minimum predicted risk for each saving (units of 16 probes)."""
    n = len(risk16); maximum = 3 * n
    prior = np.full(maximum + 1, np.inf); prior[0] = 0.0
    choices = np.full((n, maximum + 1), -1, dtype=np.int8)
    for index in range(n):
        current = prior.copy()
        choice = np.where(np.isfinite(prior), 0, -1).astype(np.int8)
        candidate32 = prior[:-2] + risk32[index]
        mask32 = candidate32 < current[2:] - EPS
        current[2:][mask32] = candidate32[mask32]
        choice[2:][mask32] = 1
        candidate16 = prior[:-3] + risk16[index]
        mask16 = candidate16 < current[3:] - EPS
        current[3:][mask16] = candidate16[mask16]
        choice[3:][mask16] = 2
        choices[index] = choice; prior = current
    return prior, choices


def allocate(frontier, choices, predicted_budget_count):
    feasible = np.flatnonzero(frontier <= predicted_budget_count + EPS)
    saving_units = int(feasible[-1])
    actions = np.full(choices.shape[0], 64, dtype=int)
    cursor = saving_units
    for index in range(choices.shape[0] - 1, -1, -1):
        choice = int(choices[index, cursor])
        if choice == 1:
            actions[index] = 32; cursor -= 2
        elif choice == 2:
            actions[index] = 16; cursor -= 3
        elif choice != 0:
            raise RuntimeError("invalid allocation backpointer")
    if cursor != 0:
        raise RuntimeError("allocation backtrack did not reach zero")
    return actions, float(frontier[saving_units])


def classification_metrics(labels, probabilities):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    prediction = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, prediction, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float((prediction == labels).mean()),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(("loss_0", "loss_1", "loss_2plus"))},
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    }


def policy_summary(ids, actions, probabilities16, probabilities32, severe_penalty,
                   risk_consumed, loss16, loss32, recall64):
    actual = v6.selected_values(actions, loss16[ids], loss32[ids])
    expected16 = risk_score(probabilities16, 2.0) / 10.0
    expected32 = risk_score(probabilities32, 2.0) / 10.0
    predicted = v6.selected_values(actions, expected16, expected32)
    total = float(actual.sum()); average = float(actions.mean())
    return {
        "queries": len(ids), "average_probes": average,
        "probe_reduction": 1.0 - average / 64.0,
        "probe_distribution": {str(p): int((actions == p).sum()) for p in shadow.PROBES},
        "fixed64_mean_recall": float(recall64[ids].mean()),
        "mean_recall": float(recall64[ids].mean() - total / len(ids)),
        "recall_delta": -total / len(ids), "actual_total_recall_loss": total,
        "predicted_total_recall_loss_minimum": float(predicted.sum()),
        "penalized_risk_budget_consumed_neighbor_count": risk_consumed,
        "severe_probability_penalty": severe_penalty,
        "positive_loss_downgrades": int((actual > EPS).sum()),
        "loss_at_least_0_2_downgrades": int((actual >= 0.2 - EPS).sum()),
        "loss_at_least_0_2_query_ids": [int(qid) for qid in ids[actual >= 0.2 - EPS]],
    }


def tune_allocation(prob16, prob32, loss16, loss32, recall64):
    best = None
    margins = np.linspace(0.00025, 0.006, 24)
    for penalty in (2.0, 3.0, 4.0, 6.0, 8.0, 12.0):
        risk16 = risk_score(prob16, penalty); risk32 = risk_score(prob32, penalty)
        frontier, choices = allocation_frontier(risk16, risk32)
        for margin in margins:
            actions, consumed = allocate(frontier, choices, margin * len(CAL_IDS) * 10.0)
            actual = v6.selected_values(actions, loss16[:500], loss32[:500])
            delta = -float(actual.mean())
            if delta < -0.003 - EPS:
                continue
            average = float(actions.mean()); severe = int((actual >= 0.2 - EPS).sum())
            key = (average, severe, int((actual > EPS).sum()), float(actual.sum()), penalty, margin)
            if best is None or key < best[0]:
                summary = policy_summary(CAL_IDS, actions, prob16, prob32, penalty,
                                         consumed, loss16, loss32, recall64)
                best = (key, penalty, float(margin), summary)
    if best is None:
        raise RuntimeError("no calibration allocation met the -0.003 safety margin")
    return best[1:]


def fit_test(family, spec, x16, x32, y16, y32):
    models, probabilities = [], []
    for x, y in ((x16, y16), (x32, y32)):
        model = build_model(family, spec); model.fit(x[:500], y[:500])
        models.append(model); probabilities.append(aligned_probabilities(model, x[500:]))
    return models, probabilities


def importance(model, names):
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.linalg.norm(estimator.coef_, axis=0)
    return sorted(({"feature": name, "importance": float(value)} for name, value in zip(names, values)),
                  key=lambda row: row["importance"], reverse=True)


def tune_family(family, feature_mode, features, y16, y32, loss16, loss32, recall64):
    x16 = features[f"{feature_mode}16"]; x32 = features[f"{feature_mode}32"]
    best = None
    for spec in model_specs(family):
        prob16, prob32 = oof_probabilities(family, spec, x16, x32, y16, y32)
        penalty, margin, calibration = tune_allocation(prob16, prob32, loss16, loss32, recall64)
        class_metrics = {"loss16": classification_metrics(y16[:500], prob16),
                         "loss32": classification_metrics(y32[:500], prob32)}
        key = (calibration["average_probes"], calibration["loss_at_least_0_2_downgrades"],
               calibration["positive_loss_downgrades"], calibration["actual_total_recall_loss"],
               -(class_metrics["loss16"]["macro_f1"] + class_metrics["loss32"]["macro_f1"]),
               json.dumps(spec, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, spec, penalty, margin, calibration, class_metrics)
    _, spec, penalty, margin, calibration, class_metrics = best
    models, test_probabilities = fit_test(family, spec, x16, x32, y16, y32)
    return {
        "family": family, "feature_mode": feature_mode, "spec": spec,
        "policy": {"severe_probability_penalty": penalty,
                   "predicted_mean_recall_loss_budget": margin,
                   "allocation": "exact minimum predicted risk for each total probe-saving level"},
        "calibration_oof": calibration, "calibration_oof_classification": class_metrics,
        "models": models, "test_probabilities": test_probabilities,
        "feature_importance": {
            "loss16": importance(models[0], features[f"{feature_mode}16_names"]),
            "loss32": importance(models[1], features[f"{feature_mode}32_names"]),
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
    raw = shadow.read_raw(args.raw); old = shadow.read_centroids(args.old_features)
    geometry = read_geometry(args.geometry); features = build_features(raw, old, geometry)
    loss16, loss32, recall64 = v6.losses_from_raw(raw)
    y16, y32 = lost_neighbor_classes(loss16), lost_neighbor_classes(loss32)
    full_oracle = v6.budgeted_oracle(np.arange(1000), loss16, loss32, recall64)
    calibration_oracle = v6.budgeted_oracle(CAL_IDS, loss16, loss32, recall64)
    tuned = []
    for family in ("logistic", "gradient_boosting", "random_forest"):
        for mode in ("base", "extended"):
            print(f"calibrating family={family} features={mode}", flush=True)
            tuned.append(tune_family(family, mode, features, y16, y32,
                                     loss16, loss32, recall64))

    # All specifications, penalties, and predicted budgets are locked above.
    test_oracle = v6.budgeted_oracle(TEST_IDS, loss16, loss32, recall64)
    results = []
    for item in tuned:
        prob16, prob32 = item.pop("test_probabilities")
        policy = item["policy"]; penalty = policy["severe_probability_penalty"]
        frontier, choices = allocation_frontier(risk_score(prob16, penalty),
                                                risk_score(prob32, penalty))
        budget = policy["predicted_mean_recall_loss_budget"] * len(TEST_IDS) * 10.0
        actions, consumed = allocate(frontier, choices, budget)
        item["test"] = policy_summary(TEST_IDS, actions, prob16, prob32, penalty,
                                      consumed, loss16, loss32, recall64)
        item["test_classification"] = {
            "loss16": classification_metrics(y16[500:], prob16),
            "loss32": classification_metrics(y32[500:], prob32),
        }
        item["oracle_gap_average_probes"] = item["test"]["average_probes"] - test_oracle["average_probes"]
        item["gate"] = gate(item["test"]); item.pop("models"); results.append(item)

    extended = [item for item in results if item["feature_mode"] == "extended"]
    compliant = [item for item in extended if item["test"]["recall_delta"] >= -0.005 - EPS]
    best = (min(compliant, key=lambda item: (item["test"]["average_probes"],
                                             item["test"]["loss_at_least_0_2_downgrades"]))
            if compliant else max(extended, key=lambda item: (item["test"]["recall_delta"],
                                                               -item["test"]["average_probes"])))
    ablation = {}
    for family in ("logistic", "gradient_boosting", "random_forest"):
        base = next(item for item in results if item["family"] == family and item["feature_mode"] == "base")
        ext = next(item for item in results if item["family"] == family and item["feature_mode"] == "extended")
        ablation[family] = {
            "average_probes_delta_extended_minus_base": ext["test"]["average_probes"] - base["test"]["average_probes"],
            "recall_delta_change_extended_minus_base": ext["test"]["recall_delta"] - base["test"]["recall_delta"],
            "severe_downgrade_delta_extended_minus_base": ext["test"]["loss_at_least_0_2_downgrades"] - base["test"]["loss_at_least_0_2_downgrades"],
        }
    report = {
        "status": "COMPLETE", "split": {"calibration": [0, 499], "test": [500, 999], "fold": "query_id % 5"},
        "lost_neighbor_classes": {"0": "max(0, round(loss*10)) == 0", "1": "round(loss*10) == 1",
                                  "2plus": "round(loss*10) >= 2"},
        "calibration_actual_recall_delta_floor": -0.003,
        "budgeted_oracle": {"full_workload": full_oracle, "calibration": calibration_oracle, "test": test_oracle},
        "models": results, "extended_geometry_ablation": ablation,
        "best_extended_model": {"family": best["family"], "spec": best["spec"],
            "policy": best["policy"], "test": best["test"], "test_classification": best["test_classification"],
            "oracle_gap_average_probes": best["oracle_gap_average_probes"], "gate": best["gate"],
            "feature_importance": best["feature_importance"]},
        "test_gate_met_by_any_extended_model": bool([item for item in extended if item["gate"] in ("Go", "Strong Go")]),
    }
    atomic_json(args.output, report)
    lines = ["# D2-v7 risk-ranked budget allocation", "",
             "All model and allocation parameters were selected with five-fold OOF calibration on qids 0–499. Test qids 500–999 were evaluated once after locking.", "",
             f"Test budgeted Oracle: **{test_oracle['average_probes']:.3f} probes**, delta {test_oracle['recall_delta']:+.4f}, distribution {test_oracle['probe_distribution']['16']}/{test_oracle['probe_distribution']['32']}/{test_oracle['probe_distribution']['64']}.", "",
             "| Model | Features | OOF avg/delta | Test avg | Test delta | Reduction | 16/32/64 | Actual/predicted loss | Severe | Oracle gap | Gate |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in results:
        cal = item["calibration_oof"]; test = item["test"]; dist = test["probe_distribution"]
        lines.append(f"| {item['family']} | {item['feature_mode']} | {cal['average_probes']:.3f}/{cal['recall_delta']:+.4f} | {test['average_probes']:.3f} | {test['recall_delta']:+.4f} | {test['probe_reduction']:.2%} | {dist['16']}/{dist['32']}/{dist['64']} | {test['actual_total_recall_loss']:.3f}/{test['predicted_total_recall_loss_minimum']:.3f} | {test['loss_at_least_0_2_downgrades']} | {item['oracle_gap_average_probes']:.3f} | {item['gate']} |")
    lines += ["", f"Best extended model: **{best['family']}**, gate **{best['gate']}**.", "",
              "Full multiclass metrics, confusion matrices, policies, feature importance, and d32/d64 ablation are in `summary.json`.", ""]
    args.output.with_name("report.md").write_text("\n".join(lines))
    print(json.dumps({"best_model": best["family"], "gate": best["gate"],
                      "average_probes": best["test"]["average_probes"],
                      "recall_delta": best["test"]["recall_delta"],
                      "oracle_gap": best["oracle_gap_average_probes"],
                      "distribution": best["test"]["probe_distribution"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    collector = sub.add_parser("collect")
    collector.add_argument("--output", type=Path, required=True)
    collector.add_argument("--old-features", type=Path, default=shadow.CENTROID_FEATURES)
    collector.add_argument("--phase-c", type=Path, default=shadow.PHASE_C)
    collector.add_argument("--system", default="vanilla_eq"); collector.add_argument("--round", type=int, default=1)
    collector.add_argument("--host", default="127.0.0.1"); collector.add_argument("--port", type=int, default=5432)
    collector.add_argument("--dbname", default="taskdb"); collector.add_argument("--user", default="dev")
    collector.add_argument("--progress-every", type=int, default=10)
    analyzer = sub.add_parser("analyze")
    analyzer.add_argument("--raw", type=Path, required=True); analyzer.add_argument("--geometry", type=Path, required=True)
    analyzer.add_argument("--output", type=Path, required=True)
    analyzer.add_argument("--old-features", type=Path, default=shadow.CENTROID_FEATURES)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        if args.output.exists(): raise RuntimeError(f"output already exists: {args.output}")
        analyze(args)


if __name__ == "__main__":
    main()
