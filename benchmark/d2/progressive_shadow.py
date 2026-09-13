#!/usr/bin/env python3
"""D2-v5 execution-aware progressive probing shadow study.

Collection is resumable and records the real p16/p32/p64 top-10 results. Analysis
uses qids 0..499 for five-fold calibration and evaluates qids 500..999 once.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from pathlib import Path

PROBES = (16, 32, 64)
QUERY_COUNT = 1000
DATASET = Path("/workspace/benchmark/data/gist1m/gist-960-euclidean.hdf5")
PHASE_C = Path("/workspace/benchmark/runs/20260909T083926Z/phase_c_artifacts/phase_c_raw.csv")
CENTROID_FEATURES = Path("/workspace/benchmark/runs/phase_d2_20260911T125412Z/features/features.csv")
PROFILE_RE = re.compile(r"IVFFLAT_PROFILE\s+(.*)")
FIELD_RE = re.compile(r"([a-z0-9_]+)=(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)")
RAW_FIELDS = ("probes", "query_id", "result_ids", "result_tids", "result_distances",
              "scanned_candidates", "recall_at_10")
EPS = 1e-12


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def vector_literal(value):
    return "[" + ",".join(str(float(x)) for x in value) + "]"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_phase_c(path: Path, system="vanilla_eq", round_no=1):
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["system"] != system or int(row["round"]) != round_no:
                continue
            probe = int(row["probes"])
            if probe not in PROBES:
                continue
            key = (int(row["query_id"]), probe)
            if key in rows:
                raise ValueError(f"duplicate Phase C row {key}")
            rows[key] = {
                "ids": [int(value) for value in row["result_ids"].split(";")],
                "recall": float(row["recall_at_10"]),
            }
    expected = {(qid, probe) for qid in range(QUERY_COUNT) for probe in PROBES}
    if set(rows) != expected:
        raise ValueError(f"Phase C coverage mismatch: missing={len(expected - set(rows))} extra={len(set(rows) - expected)}")
    return rows


def load_workload(count):
    import h5py
    import numpy as np
    with h5py.File(DATASET, "r") as source:
        queries = np.asarray(source["test"][:count], dtype=np.float32)
        truth = [set(int(value) for value in row) for row in source["neighbors"][:count, :10]]
    return queries, truth


def set_if_present(cur, name, value, required=False):
    cur.execute("SELECT setting FROM pg_settings WHERE name=%s", (name,))
    present = cur.fetchone() is not None
    if required and not present:
        raise RuntimeError(f"required GUC is missing: {name}")
    if present:
        cur.execute("SELECT set_config(%s,%s,false)", (name, str(value)))
        actual = cur.fetchone()[0]
        if actual != str(value):
            raise RuntimeError(f"failed to set {name}: expected={value} actual={actual}")


def configure(cur, probe):
    cur.execute("LOAD 'vector'")
    set_if_present(cur, "ivfflat.probes", probe, required=True)
    set_if_present(cur, "ivfflat.iterative_scan", "off", required=True)
    set_if_present(cur, "ivfflat.adaptive_probes", "off")
    set_if_present(cur, "ivfflat.adaptive_probes_trace", "off")
    set_if_present(cur, "ivfflat.distance_path", "generic")
    set_if_present(cur, "ivfflat.bounded_scan", "off")
    for statement in ("SET LOCAL enable_seqscan=off", "SET LOCAL enable_indexscan=on",
                      "SET LOCAL max_parallel_workers_per_gather=0", "SET LOCAL statement_timeout=0"):
        cur.execute(statement)


def parse_profile(notices):
    for notice in reversed(notices):
        match = PROFILE_RE.search(notice)
        if not match:
            continue
        fields = {key: float(value) for key, value in FIELD_RE.findall(match.group(1))}
        if {"probes", "candidates"} <= set(fields):
            return int(fields["probes"]), int(fields["candidates"])
    raise RuntimeError("missing IVFFLAT_PROFILE notice; the installed vector.so needs -DIVFFLAT_BENCH")


def read_completed(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"unexpected raw CSV header in {path}")
        for row in reader:
            key = (int(row["query_id"]), int(row["probes"]))
            if key in rows:
                raise ValueError(f"duplicate collected row {key}")
            rows[key] = row
    return rows


def write_progress(path, done, rows, started, status="RUNNING"):
    elapsed = time.perf_counter() - started
    eta = elapsed * (QUERY_COUNT * len(PROBES) - done) / done if done else None
    probes = [int(row["probes"]) for row in rows.values()]
    recalls = [float(row["recall_at_10"]) for row in rows.values()]
    config_no = min(len(PROBES), done // QUERY_COUNT + (1 if done % QUERY_COUNT else 0)) if done else 1
    value = {
        "status": status,
        "config": f"{config_no}/{len(PROBES)}",
        "query": f"{done}/{QUERY_COUNT * len(PROBES)}",
        "average_probes_so_far": statistics.fmean(probes) if probes else 0.0,
        "recall_so_far": statistics.fmean(recalls) if recalls else 0.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": 0.0 if status == "COMPLETE" else eta,
    }
    atomic_json(path, value)
    return value


def collect(args):
    import psycopg2

    if args.queries != QUERY_COUNT:
        raise ValueError("D2-v5 requires exactly 1000 queries")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "shadow_raw.csv"
    completed = read_completed(raw_path)
    frozen = load_phase_c(args.phase_c, args.system, args.round)
    queries, truth = load_workload(args.queries)
    literals = [vector_literal(query) for query in queries]
    started = time.perf_counter()
    conn = psycopg2.connect(host=args.host, port=args.port, dbname=args.dbname, user=args.user,
                            application_name="d2_v5_shadow_collection")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
            cur.execute("SET LOCAL lock_timeout=10000")
            cur.execute("LOCK TABLE gist_base IN SHARE MODE")
            cur.execute("SELECT oid::text,relfilenode::text,reloptions FROM pg_class WHERE oid='gist_ivf_l2'::regclass")
            index_identity = cur.fetchone()
            if index_identity is None or "lists=1000" not in (index_identity[2] or []):
                raise RuntimeError("gist_ivf_l2 with lists=1000 is required")
            configure(cur, 16)
            sql = ("SELECT id,ctid::text,embedding <-> %s::vector AS distance "
                   "FROM gist_base ORDER BY embedding <-> %s::vector LIMIT 10")
            cur.execute("EXPLAIN (FORMAT JSON) " + sql, (literals[0], literals[0]))
            plan = cur.fetchone()[0]
            if "gist_ivf_l2" not in json.dumps(plan) or "Index Scan" not in json.dumps(plan):
                raise RuntimeError("expected gist_ivf_l2 Index Scan")
            atomic_json(args.output / "plan.json", plan)
            atomic_json(args.output / "manifest.json", {
                "status": "RUNNING", "queries": args.queries, "probes": list(PROBES),
                "phase_c": str(args.phase_c), "phase_c_sha256": sha256(args.phase_c),
                "dataset": str(DATASET), "dataset_sha256": sha256(DATASET),
                "index_identity": index_identity, "split": {"calibration": [0, 499], "test": [500, 999]},
            })
            mode = "a" if raw_path.exists() else "w"
            with raw_path.open(mode, newline="", buffering=1) as target:
                writer = csv.DictWriter(target, fieldnames=RAW_FIELDS)
                if mode == "w":
                    writer.writeheader()
                for config_no, probe in enumerate(PROBES, 1):
                    configure(cur, probe)
                    for qid, literal in enumerate(literals):
                        key = (qid, probe)
                        if key in completed:
                            continue
                        conn.notices.clear()
                        cur.execute(sql, (literal, literal))
                        result = cur.fetchall()
                        actual_probe, candidates = parse_profile(conn.notices)
                        ids = [int(row[0]) for row in result]
                        tids = [row[1] for row in result]
                        distances = [float(row[2]) for row in result]
                        if actual_probe != probe:
                            raise RuntimeError(f"probe mismatch qid={qid}: expected={probe} actual={actual_probe}")
                        if len(ids) != 10 or len(set(ids)) != 10 or distances != sorted(distances):
                            raise RuntimeError(f"invalid ordered top10 qid={qid} probes={probe}")
                        if ids != frozen[key]["ids"]:
                            raise RuntimeError(f"Phase C ordered ID mismatch qid={qid} probes={probe}")
                        recall = len(set(ids) & truth[qid]) / 10.0
                        if abs(recall - frozen[key]["recall"]) > EPS:
                            raise RuntimeError(f"Phase C recall mismatch qid={qid} probes={probe}")
                        row = {
                            "probes": probe, "query_id": qid,
                            "result_ids": json.dumps(ids, separators=(",", ":")),
                            "result_tids": json.dumps(tids, separators=(",", ":")),
                            "result_distances": json.dumps(distances, separators=(",", ":")),
                            "scanned_candidates": candidates, "recall_at_10": format(recall, ".17g"),
                        }
                        writer.writerow(row)
                        completed[key] = row
                        done = len(completed)
                        if done % args.progress_every == 0 or done == QUERY_COUNT * len(PROBES):
                            value = write_progress(args.output / "progress.json", done, completed, started)
                            print(" ".join(f"{name}={value[name]}" for name in
                                  ("config", "query", "average_probes_so_far", "recall_so_far",
                                   "elapsed_seconds", "eta_seconds")), flush=True)
            expected = {(qid, probe) for qid in range(QUERY_COUNT) for probe in PROBES}
            if set(completed) != expected:
                raise RuntimeError(f"collection incomplete: {len(completed)}/{len(expected)}")
            conn.commit()
            manifest = json.loads((args.output / "manifest.json").read_text())
            manifest.update(status="COMPLETE", elapsed_seconds=time.perf_counter() - started)
            atomic_json(args.output / "manifest.json", manifest)
            write_progress(args.output / "progress.json", len(completed), completed, started, "COMPLETE")
    finally:
        conn.close()


def read_raw(path: Path):
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (int(row["query_id"]), int(row["probes"]))
            rows[key] = {
                "ids": json.loads(row["result_ids"]), "tids": json.loads(row["result_tids"]),
                "distances": json.loads(row["result_distances"]),
                "candidates": int(row["scanned_candidates"]), "recall": float(row["recall_at_10"]),
            }
    expected = {(qid, probe) for qid in range(QUERY_COUNT) for probe in PROBES}
    if set(rows) != expected:
        raise ValueError(f"shadow raw coverage mismatch: {len(rows)}/{len(expected)}")
    return rows


def read_centroids(path: Path):
    rows = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            qid = int(row["query_id"])
            rows[qid] = {key: float(value) for key, value in row.items() if key != "query_id"}
    if set(rows) != set(range(QUERY_COUNT)):
        raise ValueError("centroid features must contain query_id 0..999 exactly once")
    return rows


def stats(values):
    return min(values), statistics.fmean(values), statistics.pstdev(values), max(values)


def make_features(raw, centroids):
    centroid_names = ["d1", "d2_d1", "d4_d1", "d8_d1", "d16_d1", "gap"]
    stage16_names = centroid_names + ["r16_min", "r16_mean", "r16_std", "r16_max",
                                       "r16_d10_d1", "r16_spread", "candidates16"]
    stage32_names = stage16_names + ["r32_min", "r32_mean", "r32_std", "r32_max",
                                       "r32_d10_d1", "r32_spread", "candidates32",
                                       "overlap16_32", "replacements16_32", "kth_rel_change",
                                       "mean_rel_change", "max_rel_change", "candidate_increase",
                                       "candidate_rel_increase"]
    x_centroid, x16, x32, y16, y32 = [], [], [], [], []
    derived = {}
    for qid in range(QUERY_COUNT):
        centroid = [centroids[qid][name] for name in centroid_names]
        p16, p32, p64 = raw[(qid, 16)], raw[(qid, 32)], raw[(qid, 64)]
        s16 = stats(p16["distances"]); s32 = stats(p32["distances"])
        f16 = list(s16) + [s16[3] / max(s16[0], EPS), s16[3] - s16[0], p16["candidates"]]
        overlap = len(set(p16["ids"]) & set(p32["ids"]))
        f32 = list(s32) + [s32[3] / max(s32[0], EPS), s32[3] - s32[0], p32["candidates"],
                           overlap, 10 - overlap,
                           (s32[3] - s16[3]) / max(abs(s16[3]), EPS),
                           (s32[1] - s16[1]) / max(abs(s16[1]), EPS),
                           (s32[3] - s16[3]) / max(abs(s16[3]), EPS),
                           p32["candidates"] - p16["candidates"],
                           (p32["candidates"] - p16["candidates"]) / max(p16["candidates"], 1)]
        x_centroid.append(centroid); x16.append(centroid + f16); x32.append(centroid + f16 + f32)
        y16.append(int(p16["recall"] >= p64["recall"] - EPS))
        y32.append(int(p32["recall"] >= p64["recall"] - EPS))
        derived[qid] = {"overlap": overlap, "kth_rel_change": f32[9],
                        "recall16": p16["recall"], "recall32": p32["recall"], "recall64": p64["recall"]}
    return {"centroid_names": centroid_names, "stage16_names": stage16_names,
            "stage32_names": stage32_names, "centroid": x_centroid, "stage16": x16,
            "stage32": x32, "y16": y16, "y32": y32, "derived": derived}


def model_specs(family):
    if family == "logistic_regression":
        return [{"C": value, "class_weight": weight} for value in (0.1, 1.0, 10.0)
                for weight in (None, "balanced")]
    if family == "decision_tree":
        return [{"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
                for depth in (3, 4) for leaf in (10, 25) for weight in (None, "balanced")]
    if family == "gradient_boosting":
        return [{"max_depth": depth, "n_estimators": estimators, "learning_rate": rate}
                for depth in (1, 2) for estimators in (50, 100) for rate in (0.05, 0.1)]
    raise ValueError(family)


def build_model(family, spec):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    if family == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"],
            class_weight=spec["class_weight"], max_iter=2000, random_state=20260913))
    if family == "decision_tree":
        return DecisionTreeClassifier(max_depth=spec["max_depth"], min_samples_leaf=spec["min_samples_leaf"],
                                      class_weight=spec["class_weight"], random_state=20260913)
    return GradientBoostingClassifier(max_depth=spec["max_depth"], n_estimators=spec["n_estimators"],
                                      learning_rate=spec["learning_rate"], random_state=20260913)


def oof_probabilities(family, spec, x16, x32, y16, y32):
    import numpy as np
    result16 = np.zeros(500); result32 = np.zeros(500)
    ids = np.arange(500)
    for fold in range(5):
        valid = ids[ids % 5 == fold]; train = ids[ids % 5 != fold]
        for x, y, output in ((x16, y16, result16), (x32, y32, result32)):
            model = build_model(family, spec)
            model.fit(np.asarray(x)[train], np.asarray(y)[train])
            output[valid] = model.predict_proba(np.asarray(x)[valid])[:, 1]
    return result16, result32


def thresholds(values):
    import numpy as np
    return sorted(set([float("inf"), -float("inf")] +
                      [float(v) for v in np.quantile(values, np.linspace(0, 1, 31))]))


def cascade_actions(p16, p32, t16, t32):
    return [16 if a >= t16 else 32 if b >= t32 else 64 for a, b in zip(p16, p32)]


def action_summary(ids, actions, data, p16=None, p32=None, t16=None, t32=None):
    from sklearn.metrics import confusion_matrix
    recalls = [data["derived"][qid][f"recall{probe}"] for qid, probe in zip(ids, actions)]
    baseline = [data["derived"][qid]["recall64"] for qid in ids]
    y16 = [data["y16"][qid] for qid in ids]; y32 = [data["y32"][qid] for qid in ids]
    pred16 = [int(probe == 16) for probe in actions]
    pred32 = [int(probe <= 32) for probe in actions]
    def classification(y, pred):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}
    unsafe16 = sum(probe == 16 and not safe for probe, safe in zip(actions, y16))
    unsafe32 = sum(probe == 32 and not safe for probe, safe in zip(actions, y32))
    needs64 = sum(probe < 64 and not safe for probe, safe in zip(actions, y32))
    average = statistics.fmean(actions)
    value = {
        "queries": len(ids), "average_probes": average, "probe_reduction": 1.0 - average / 64.0,
        "mean_recall": statistics.fmean(recalls), "fixed64_mean_recall": statistics.fmean(baseline),
        "recall_delta": statistics.fmean(recalls) - statistics.fmean(baseline),
        "probe_distribution": {str(p): actions.count(p) for p in PROBES},
        "safe16": classification(y16, pred16), "safe32": classification(y32, pred32),
        "false_downgrade": {"unsafe_action16": unsafe16, "unsafe_action32": unsafe32,
                            "needs64_but_downgraded": needs64, "total": unsafe16 + unsafe32},
    }
    if p16 is not None:
        raw16 = [int(value >= t16) for value in p16]
        raw32 = [int(value >= t32) for value in p32]
        value["raw_classifier"] = {"safe16": classification(y16, raw16),
                                   "safe32": classification(y32, raw32)}
    return value


def tune_thresholds(p16, p32, data):
    ids = list(range(500)); best = None
    for t16 in thresholds(p16):
        for t32 in thresholds(p32):
            actions = cascade_actions(p16, p32, t16, t32)
            summary = action_summary(ids, actions, data)
            if summary["recall_delta"] < -0.005 - EPS:
                continue
            key = (summary["average_probes"], summary["false_downgrade"]["total"],
                   -summary["recall_delta"], -t16, -t32)
            if best is None or key < best[0]:
                best = (key, float(t16), float(t32), summary)
    if best is None:
        raise RuntimeError("fixed64 should always make the calibration constraint feasible")
    return best[1:]


def fit_predict(family, spec, x16, x32, y16, y32):
    import numpy as np
    models = []
    probabilities = []
    for x, y in ((x16, y16), (x32, y32)):
        model = build_model(family, spec)
        model.fit(np.asarray(x)[:500], np.asarray(y)[:500])
        probabilities.append(model.predict_proba(np.asarray(x)[500:])[:, 1])
        models.append(model)
    return models, probabilities


def importance(model, names):
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = abs(estimator.coef_[0])
    return sorted(({"feature": name, "importance": float(value)} for name, value in zip(names, values)),
                  key=lambda item: item["importance"], reverse=True)


def tune_family(family, mode, data):
    x16 = data["stage16"] if mode == "execution_aware" else data["centroid"]
    x32 = data["stage32"] if mode == "execution_aware" else data["centroid"]
    best = None
    for spec in model_specs(family):
        p16, p32 = oof_probabilities(family, spec, x16, x32, data["y16"], data["y32"])
        t16, t32, calibration = tune_thresholds(p16, p32, data)
        key = (calibration["average_probes"], calibration["false_downgrade"]["total"],
               -calibration["recall_delta"], json.dumps(spec, sort_keys=True))
        if best is None or key < best[0]:
            best = (key, spec, t16, t32, calibration, p16, p32)
    _, spec, t16, t32, calibration, oof16, oof32 = best
    models, test_probabilities = fit_predict(family, spec, x16, x32, data["y16"], data["y32"])
    return {"family": family, "feature_mode": mode, "spec": spec,
            "thresholds": {"safe16": t16, "safe32": t32}, "calibration": calibration,
            "models": models, "test_probabilities": test_probabilities,
            "oof_probabilities": [oof16, oof32],
            "feature_importance": {"safe16": importance(models[0], data["stage16_names"] if mode == "execution_aware" else data["centroid_names"]),
                                   "safe32": importance(models[1], data["stage32_names"] if mode == "execution_aware" else data["centroid_names"])}}


def simple_rule_calibration(data):
    import numpy as np
    # Production-friendly shadow rule: exact/high overlap plus a calibrated kth-change ceiling.
    ids = list(range(500)); values = [data["derived"][qid]["kth_rel_change"] for qid in ids]
    ceilings = sorted(set([float("inf"), -float("inf")] +
                          [float(v) for v in np.quantile(values, np.linspace(0, 1, 51))]))
    best = None
    for min_overlap in (9, 10):
        for ceiling in ceilings:
            actions = [32 if data["derived"][qid]["overlap"] >= min_overlap and
                       data["derived"][qid]["kth_rel_change"] <= ceiling else 64 for qid in ids]
            summary = action_summary(ids, actions, data)
            if summary["recall_delta"] < -0.005 - EPS:
                continue
            key = (summary["average_probes"], summary["false_downgrade"]["total"], -summary["recall_delta"])
            if best is None or key < best[0]:
                best = (key, min_overlap, ceiling, summary)
    return {"rule": "if overlap16_32 >= min_overlap and kth_rel_change <= ceiling: stop32; else continue64",
            "min_overlap": best[1], "kth_rel_change_ceiling": best[2], "calibration": best[3]}


def overlap_diagnostics(ids, data):
    result = {}
    for label, predicate in (("overlap_10_of_10", lambda value: value == 10),
                             ("overlap_at_least_9_of_10", lambda value: value >= 9)):
        selected = [qid for qid in ids if predicate(data["derived"][qid]["overlap"])]
        safe = sum(data["y32"][qid] for qid in selected)
        result[label] = {"queries": len(selected), "safe32": safe,
                         "safe32_precision": safe / len(selected) if selected else None}
    return result


def kth_diagnostics(train_ids, test_ids, data):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    train_values = np.asarray([data["derived"][qid]["kth_rel_change"] for qid in train_ids])
    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, 6)))
    def summarize(ids):
        values = np.asarray([data["derived"][qid]["kth_rel_change"] for qid in ids])
        labels = np.asarray([data["y32"][qid] for qid in ids])
        bins = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (values >= low) & (values <= high if high == edges[-1] else values < high)
            bins.append({"low": float(low), "high": float(high), "queries": int(mask.sum()),
                         "safe32_rate": float(labels[mask].mean()) if mask.any() else None})
        return {"pearson_with_safe32": float(np.corrcoef(values, labels)[0, 1]),
                "auc_kth_change": float(roc_auc_score(labels, values)), "bins": bins}
    return {"bin_edges_locked_on_calibration": [float(value) for value in edges],
            "calibration": summarize(train_ids), "test": summarize(test_ids)}


def gate(metrics):
    average = metrics["average_probes"]
    recall_ok = metrics["recall_delta"] >= -0.005 - EPS
    if recall_ok and average <= 45:
        return "Strong Go"
    if recall_ok and average <= 50:
        return "Go"
    if 50 < average <= 53:
        return "Marginal"
    return "No-Go"


def analyze(args):
    raw = read_raw(args.raw); centroids = read_centroids(args.centroid_features)
    frozen = load_phase_c(args.phase_c, args.system, args.round)
    for key, row in raw.items():
        if row["ids"] != frozen[key]["ids"] or abs(row["recall"] - frozen[key]["recall"]) > EPS:
            raise RuntimeError(f"raw/Phase C audit mismatch: {key}")
    data = make_features(raw, centroids)
    tuned = []
    for family in ("logistic_regression", "decision_tree", "gradient_boosting"):
        for mode in ("centroid_only", "execution_aware"):
            print(f"calibrating family={family} features={mode}", flush=True)
            tuned.append(tune_family(family, mode, data))
    simple = simple_rule_calibration(data)

    # This block is the only test-set evaluation. All specs and thresholds are now locked.
    test_ids = list(range(500, 1000))
    results = []
    for item in tuned:
        p16, p32 = item.pop("test_probabilities")
        t16, t32 = item["thresholds"]["safe16"], item["thresholds"]["safe32"]
        actions = cascade_actions(p16, p32, t16, t32)
        item["test"] = action_summary(test_ids, actions, data, p16, p32, t16, t32)
        item["gate"] = gate(item["test"])
        item.pop("models"); item.pop("oof_probabilities")
        results.append(item)
    simple_actions = [32 if data["derived"][qid]["overlap"] >= simple["min_overlap"] and
                      data["derived"][qid]["kth_rel_change"] <= simple["kth_rel_change_ceiling"] else 64
                      for qid in test_ids]
    simple["test"] = action_summary(test_ids, simple_actions, data); simple["gate"] = gate(simple["test"])
    execution = [item for item in results if item["feature_mode"] == "execution_aware"]
    compliant = [item for item in execution if item["test"]["recall_delta"] >= -0.005 - EPS]
    best = (min(compliant, key=lambda item: (item["test"]["average_probes"],
                                             item["test"]["false_downgrade"]["total"]))
            if compliant else max(execution, key=lambda item: (item["test"]["recall_delta"],
                                                                -item["test"]["average_probes"])))
    improvements = {}
    for family in ("logistic_regression", "decision_tree", "gradient_boosting"):
        centroid = next(item for item in results if item["family"] == family and item["feature_mode"] == "centroid_only")
        aware = next(item for item in results if item["family"] == family and item["feature_mode"] == "execution_aware")
        improvements[family] = {"average_probes_delta_execution_minus_centroid": aware["test"]["average_probes"] - centroid["test"]["average_probes"],
                                "recall_delta_improvement": aware["test"]["recall_delta"] - centroid["test"]["recall_delta"]}
    report = {
        "status": "COMPLETE", "split": {"calibration": [0, 499], "test": [500, 999], "fold": "query_id % 5"},
        "recall_constraint": -0.005, "test_constraint_met_by_any_execution_aware_model": bool(compliant),
        "models": results, "best_execution_aware": {"family": best["family"],
            "spec": best["spec"], "thresholds": best["thresholds"], "test": best["test"], "gate": best["gate"],
            "feature_importance": best["feature_importance"]},
        "simple_overlap_kth_rule": simple,
        "overlap_analysis": {"calibration": overlap_diagnostics(range(500), data),
                             "test": overlap_diagnostics(range(500, 1000), data)},
        "kth_distance_change_analysis": kth_diagnostics(list(range(500)), test_ids, data),
        "execution_aware_vs_centroid_only": improvements,
    }
    atomic_json(args.output, report)
    markdown = ["# D2-v5 execution-aware progressive probing", "",
                f"Best model: **{best['family']}**; gate: **{best['gate']}**.", "",
                "| Model | Features | Avg probes | Recall delta | Reduction | 16/32/64 | False downgrade | Gate |",
                "|---|---|---:|---:|---:|---:|---:|---|"]
    for item in results:
        m = item["test"]; d = m["probe_distribution"]
        markdown.append(f"| {item['family']} | {item['feature_mode']} | {m['average_probes']:.3f} | {m['recall_delta']:+.4f} | {m['probe_reduction']:.2%} | {d['16']}/{d['32']}/{d['64']} | {m['false_downgrade']['total']} | {item['gate']} |")
    markdown += ["", "## Locked best thresholds", "",
                 f"- safe16 probability: `{best['thresholds']['safe16']:.17g}`",
                 f"- safe32 probability: `{best['thresholds']['safe32']:.17g}`",
                 "", "Full metrics, confusion matrices, diagnostics, and feature importance are in `summary.json`.", ""]
    args.output.with_name("report.md").write_text("\n".join(markdown))
    print(json.dumps({"best_model": best["family"], "gate": best["gate"],
                      "average_probes": best["test"]["average_probes"],
                      "recall_delta": best["test"]["recall_delta"],
                      "probe_distribution": best["test"]["probe_distribution"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--phase-c", type=Path, default=PHASE_C)
    collect_parser.add_argument("--system", default="vanilla_eq")
    collect_parser.add_argument("--round", type=int, default=1)
    collect_parser.add_argument("--queries", type=int, default=QUERY_COUNT)
    collect_parser.add_argument("--host", default="127.0.0.1")
    collect_parser.add_argument("--port", type=int, default=5432)
    collect_parser.add_argument("--dbname", default="taskdb")
    collect_parser.add_argument("--user", default="dev")
    collect_parser.add_argument("--progress-every", type=int, default=10)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--raw", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--phase-c", type=Path, default=PHASE_C)
    analyze_parser.add_argument("--centroid-features", type=Path, default=CENTROID_FEATURES)
    analyze_parser.add_argument("--system", default="vanilla_eq")
    analyze_parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    collect(args) if args.command == "collect" else analyze(args)


if __name__ == "__main__":
    main()
