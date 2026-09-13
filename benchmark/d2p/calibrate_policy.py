#!/usr/bin/env python3
"""Tune the frozen D2-P static baseline and four shallow-tree policies."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

SEED = 20260913
TREE_CONFIGS = ((2, 20), (2, 40), (3, 20), (3, 40))
CENTROID = ("d1", "d2", "d4", "d8", "d16", "d32", "d64", "d2_d1",
            "d4_d1", "d8_d1", "d16_d1", "d32_d1", "d64_d1", "d32_d16",
            "d64_d32", "gap", "gap32_16_d1", "gap64_32_d1")
STAGE16 = CENTROID + (
    "s16_top10_min", "s16_top10_mean", "s16_top10_std", "s16_top10_max",
    "s16_top10_spread", "s16_candidates", "s16_pages", "s16_replacements",
    "s16_replacement_rate", "s16_candidates_per_page")
STAGE32 = STAGE16 + (
    "s32_top10_min", "s32_top10_mean", "s32_top10_std", "s32_top10_max",
    "s32_top10_spread", "s32_candidates", "s32_pages", "s32_replacements",
    "s32_replacement_rate", "s32_candidates_per_page", "top10_overlap_16_32",
    "top10_replaced_16_32", "kth10_relative_change_16_32",
    "mean10_relative_change_16_32", "candidate_growth_16_32")


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def parse_floats(value):
    return [float(item) for item in value.split(";") if item]


def fmean(values):
    return statistics.fmean(values) if values else 0.0


def safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 1e-30 else 0.0


def stage_stats(row, stage):
    distances = parse_floats(row[f"stage{stage}_topk_distances"])[:10]
    if len(distances) != 10:
        raise ValueError(f"qid={row['query_id']} stage={stage}: missing Top10")
    mean = fmean(distances)
    std = math.sqrt(fmean([(value - mean) ** 2 for value in distances]))
    candidates = float(row[f"stage{stage}_candidates"])
    pages = float(row[f"stage{stage}_pages"])
    replacements = float(row[f"stage{stage}_replacements"])
    return distances, {
        f"s{stage}_top10_min": min(distances), f"s{stage}_top10_mean": mean,
        f"s{stage}_top10_std": std, f"s{stage}_top10_max": max(distances),
        f"s{stage}_top10_spread": max(distances) - min(distances),
        f"s{stage}_candidates": candidates, f"s{stage}_pages": pages,
        f"s{stage}_replacements": replacements,
        f"s{stage}_replacement_rate": safe_ratio(replacements, candidates),
        f"s{stage}_candidates_per_page": safe_ratio(candidates, pages),
    }


def feature_rows(rows):
    output = {}
    for row in rows:
        qid = int(row["query_id"])
        features = {name: float(row[name]) for name in CENTROID}
        d16, s16 = stage_stats(row, 16)
        d32, s32 = stage_stats(row, 32)
        features.update(s16)
        features.update(s32)
        tids16 = row["stage16_topk_tids"].split(";")[:10]
        tids32 = row["stage32_topk_tids"].split(";")[:10]
        overlap = len(set(tids16) & set(tids32))
        features.update({
            "top10_overlap_16_32": float(overlap),
            "top10_replaced_16_32": float(10 - overlap),
            "kth10_relative_change_16_32": safe_ratio(d16[9] - d32[9], d16[9]),
            "mean10_relative_change_16_32": safe_ratio(fmean(d16) - fmean(d32), fmean(d16)),
            "candidate_growth_16_32": safe_ratio(float(row["stage32_candidates"]) -
                                                  float(row["stage16_candidates"]),
                                                  float(row["stage16_candidates"])),
        })
        output[qid] = features
    return output


def phase_c_recalls(path):
    candidates = {}
    priority = {"vanilla_eq": 0, "final": 1, "phase2b": 2, "phase2a": 3}
    for row in read_csv(path):
        probes = int(row["probes"])
        if probes not in (16, 32, 64) or int(row["round"]) != 1:
            continue
        if row["system"] not in priority:
            continue
        key = (int(row["query_id"]), probes)
        rank = priority[row["system"]]
        if key not in candidates or rank < candidates[key][0]:
            candidates[key] = (rank, float(row["recall_at_10"]))
    recalls = {qid: {p: candidates[(qid, p)][1] for p in (16, 32, 64)}
               for qid in range(1000)}
    return recalls


def load_fixed(path, probes):
    rows = read_csv(path / f"fixed{probes}.csv")
    if len(rows) != 1000 or {int(row["query_id"]) for row in rows} != set(range(1000)):
        raise ValueError(f"fixed{probes}.csv must contain qids 0..999")
    return {int(row["query_id"]): row for row in rows}


def tree_json(tree, names):
    from sklearn.tree import _tree
    def node(index):
        if tree.tree_.feature[index] == _tree.TREE_UNDEFINED:
            counts = tree.tree_.value[index][0]
            total = float(sum(counts))
            return {"safe_probability": float(counts[1] / total if total else 0.0),
                    "samples": int(tree.tree_.n_node_samples[index])}
        feature = names[tree.tree_.feature[index]]
        return {"feature": feature, "threshold": float(tree.tree_.threshold[index]),
                "if_le": node(tree.tree_.children_left[index]),
                "if_gt": node(tree.tree_.children_right[index])}
    return node(0)


def emit_if_else(node, indent="    "):
    if "safe_probability" in node:
        return [f"{indent}return {node['safe_probability']:.17g};"]
    lines = [f"{indent}if (f.{node['feature']} <= {node['threshold']:.17g})"]
    lines += emit_if_else(node["if_le"], indent + "    ")
    lines += [f"{indent}else"]
    lines += emit_if_else(node["if_gt"], indent + "    ")
    return lines


def policy_metrics(qids, p16, p32, threshold16, threshold32, recalls, shadow):
    actions = {}
    for qid in qids:
        actions[qid] = 16 if p16[qid] >= threshold16 else (32 if p32[qid] >= threshold32 else 64)
    selected_recall = [recalls[qid][actions[qid]] for qid in qids]
    baseline = [recalls[qid][64] for qid in qids]
    candidates = [float(shadow[qid][f"stage{actions[qid]}_candidates"]) for qid in qids]
    pages = [float(shadow[qid][f"stage{actions[qid]}_pages"]) for qid in qids]
    distance_calls = [float(shadow[qid][f"stage{actions[qid]}_distance_calls"]) for qid in qids]
    distribution = {str(p): sum(actions[qid] == p for qid in qids) for p in (16, 32, 64)}
    return {
        "average_probes": fmean([actions[qid] for qid in qids]),
        "average_candidates": fmean(candidates), "average_pages": fmean(pages),
        "average_distance_calls": fmean(distance_calls), "mean_recall": fmean(selected_recall),
        "baseline_recall": fmean(baseline),
        "recall_delta": fmean(selected_recall) - fmean(baseline),
        "distribution": distribution,
    }, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--phase-c", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    shadow_rows = read_csv(args.shadow)
    if len(shadow_rows) != 1000:
        raise ValueError("shadow calibration must contain exactly 1000 rows")
    shadow = {int(row["query_id"]): row for row in shadow_rows}
    if set(shadow) != set(range(1000)):
        raise ValueError("shadow calibration must contain qids 0..999")
    features = feature_rows(shadow_rows)
    recalls = phase_c_recalls(args.phase_c)
    recall64 = fmean([recalls[qid][64] for qid in range(1000)])
    target = recall64 - 0.005

    static_probe, static_rows = None, None
    tested_static = [{"probes": 16, "mean_recall": fmean([recalls[q][16] for q in range(1000)])},
                     {"probes": 32, "mean_recall": fmean([recalls[q][32] for q in range(1000)])}]
    if fmean([recalls[qid][32] for qid in range(1000)]) >= target:
        static_probe = 32
    else:
        for probes in (40, 48, 56):
            path = args.fixed_dir / f"fixed{probes}.csv"
            if not path.exists():
                raise FileNotFoundError(f"need sequential static calibration: {path}")
            rows = load_fixed(args.fixed_dir, probes)
            probe_recall = fmean([float(rows[qid]["recall_at_10"]) for qid in range(1000)])
            tested_static.append({"probes": probes, "mean_recall": probe_recall})
            if probe_recall >= target:
                static_probe, static_rows = probes, rows
                break
    if static_probe is None:
        static_probe = 64
        tested_static.append({"probes": 64, "mean_recall": recall64})
    if static_probe in (16, 32, 64):
        static_recall = fmean([recalls[qid][static_probe] for qid in range(1000)])
        static_candidates = fmean([float(shadow[qid][f"stage{static_probe}_candidates"])
                                   for qid in range(1000)])
    else:
        static_recall = fmean([float(static_rows[qid]["recall_at_10"]) for qid in range(1000)])
        static_candidates = fmean([float(static_rows[qid]["candidates"]) for qid in range(1000)])
    static = {"p_static": static_probe, "target_recall": target,
              "fixed64_recall": recall64, "mean_recall": static_recall,
              "recall_delta": static_recall - recall64,
              "average_probes": float(static_probe), "average_candidates": static_candidates,
              "tested_probes": tested_static}
    (args.output / "static_tuned.json").write_text(json.dumps(static, indent=2, sort_keys=True) + "\n")

    qids = list(range(1000))
    x16 = np.asarray([[features[qid][name] for name in STAGE16] for qid in qids])
    x32 = np.asarray([[features[qid][name] for name in STAGE32] for qid in qids])
    y16 = np.asarray([recalls[qid][16] >= recalls[qid][64] for qid in qids], dtype=int)
    y32 = np.asarray([recalls[qid][32] >= recalls[qid][64] for qid in qids], dtype=int)
    candidates = []
    for depth, leaf in TREE_CONFIGS:
        probabilities16 = np.zeros(1000)
        probabilities32 = np.zeros(1000)
        fold_metrics = []
        for fold in range(5):
            validation = np.asarray([qid % 5 == fold for qid in qids])
            train = ~validation
            models = []
            for matrix, labels, probabilities in ((x16, y16, probabilities16),
                                                   (x32, y32, probabilities32)):
                model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf,
                                               random_state=SEED)
                model.fit(matrix[train], labels[train])
                probabilities[validation] = model.predict_proba(matrix[validation])[:, 1]
                models.append(model)
            fold_metrics.append({"fold": fold, "validation_qids": int(validation.sum())})
        p16 = {qid: float(probabilities16[qid]) for qid in qids}
        p32 = {qid: float(probabilities32[qid]) for qid in qids}
        thresholds16 = sorted(set(p16.values()), reverse=True) + [math.inf]
        thresholds32 = sorted(set(p32.values()), reverse=True) + [math.inf]
        best = None
        for t16 in thresholds16:
            for t32 in thresholds32:
                metrics, actions = policy_metrics(qids, p16, p32, t16, t32, recalls, shadow)
                if metrics["recall_delta"] < -0.005 - 1e-12:
                    continue
                key = (metrics["average_candidates"], metrics["average_probes"], -metrics["recall_delta"])
                if best is None or key < best[0]:
                    best = (key, t16, t32, metrics, actions)
        if best is None:
            raise RuntimeError("fixed64 should always be a feasible policy")
        _, t16, t32, metrics, actions = best
        candidates.append({"max_depth": depth, "min_samples_leaf": leaf,
                           "threshold16": t16, "threshold32": t32,
                           "oof_metrics": metrics, "oof_actions": actions,
                           "oof_stage16_scores": p16, "oof_stage32_scores": p32})

    winner = min(candidates, key=lambda item: (item["oof_metrics"]["average_candidates"],
                                                item["oof_metrics"]["average_probes"]))
    final16 = DecisionTreeClassifier(max_depth=winner["max_depth"],
                                     min_samples_leaf=winner["min_samples_leaf"],
                                     random_state=SEED).fit(x16, y16)
    final32 = DecisionTreeClassifier(max_depth=winner["max_depth"],
                                     min_samples_leaf=winner["min_samples_leaf"],
                                     random_state=SEED).fit(x32, y32)
    tree16 = tree_json(final16, STAGE16)
    tree32 = tree_json(final32, STAGE32)
    policy = {
        "version": "D2-P", "calibration_qid_range": [0, 999],
        "seed": SEED, "fold": "query_id % 5",
        "model": "DecisionTreeClassifier", "max_depth": winner["max_depth"],
        "min_samples_leaf": winner["min_samples_leaf"], "class_weight": None,
        "stage16_features": list(STAGE16), "stage32_features": list(STAGE32),
        "stage16_tree": tree16, "stage32_tree": tree32,
        "stage16_threshold": winner["threshold16"],
        "stage32_threshold": winner["threshold32"],
        "oof_metrics": winner["oof_metrics"], "p_static": static_probe,
        "target_recall": target,
        "go_vs_static": (winner["oof_metrics"]["recall_delta"] >= -0.005 and
                          winner["oof_metrics"]["average_candidates"] < static_candidates),
    }
    policy_path = args.output / "d2p_policy.json"
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    (args.output / "d2p_policy.sha256").write_text(f"{digest}  {policy_path.name}\n")
    c_lines = ["/* Generated D2-P policy; no sklearn runtime dependency. */",
               "static double D2PStage16SafeProbability(D2PFeatures f)", "{"]
    c_lines += emit_if_else(tree16)
    c_lines += ["}", "", "static double D2PStage32SafeProbability(D2PFeatures f)", "{"]
    c_lines += emit_if_else(tree32)
    c_lines += ["}", "", f"#define D2P_STAGE16_THRESHOLD {winner['threshold16']:.17g}",
                f"#define D2P_STAGE32_THRESHOLD {winner['threshold32']:.17g}"]
    (args.output / "d2p_policy.c.inc").write_text("\n".join(c_lines) + "\n")
    oof_path = args.output / "d2p_oof_predictions.csv"
    with oof_path.open("w", newline="") as target_file:
        fields = ("query_id", "fold", "recall16", "recall32", "recall64", "loss16",
                  "loss32", "safe16", "safe32", "stage16_score", "stage32_score",
                  "selected_probes")
        writer = csv.DictWriter(target_file, fieldnames=fields)
        writer.writeheader()
        for qid in qids:
            writer.writerow({"query_id": qid, "fold": qid % 5,
                             "recall16": recalls[qid][16], "recall32": recalls[qid][32],
                             "recall64": recalls[qid][64],
                             "loss16": recalls[qid][64] - recalls[qid][16],
                             "loss32": recalls[qid][64] - recalls[qid][32],
                             "safe16": int(bool(y16[qid])), "safe32": int(bool(y32[qid])),
                             "stage16_score": winner["oof_stage16_scores"][qid],
                             "stage32_score": winner["oof_stage32_scores"][qid],
                             "selected_probes": winner["oof_actions"][qid]})
    fixed64_metrics = {
        "mean_recall": recall64, "recall_delta": 0.0, "average_probes": 64.0,
        "average_candidates": fmean([float(shadow[q]["stage64_candidates"]) for q in qids]),
        "average_pages": fmean([float(shadow[q]["stage64_pages"]) for q in qids]),
        "average_distance_calls": fmean([float(shadow[q]["stage64_distance_calls"]) for q in qids])}
    static_metrics = {"mean_recall": static_recall, "recall_delta": static_recall - recall64,
                      "average_probes": float(static_probe), "average_candidates": static_candidates}
    if static_probe in (16, 32, 64):
        static_metrics.update({
            "average_pages": fmean([float(shadow[q][f"stage{static_probe}_pages"]) for q in qids]),
            "average_distance_calls": fmean([float(shadow[q][f"stage{static_probe}_distance_calls"]) for q in qids])})
    else:
        static_metrics.update({
            "average_pages": fmean([float(static_rows[q]["pages"]) for q in qids]),
            "average_distance_calls": fmean([float(static_rows[q]["distance_calls"]) for q in qids])})
    sanity = {"fixed64": fixed64_metrics, "static_tuned": static_metrics,
              "d2p": winner["oof_metrics"]}
    (args.output / "calibration_sanity.json").write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
    report = {"static": static, "tree_candidates": [
        {key: value for key, value in item.items() if not key.startswith("oof_")} |
        {"oof_metrics": item["oof_metrics"]} for item in candidates],
        "selected_policy": policy, "policy_sha256": digest, "sanity": sanity}
    (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
