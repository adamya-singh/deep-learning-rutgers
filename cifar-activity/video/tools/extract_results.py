"""Freeze experiment results for the film (read-only; standard library only).

    python3 tools/extract_results.py

Reads ../runs/** (logs, metrics.jsonl, comparison.json, queue-state.json) and
writes video/public/data/results.json plus a verbatim copy of the source JSON
files under video/data/frozen/. It never runs report scripts and never writes
outside video/. Re-run it to refresh after the follow-up queue progresses.
"""
import json
import re
import shutil
import statistics
import time
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
RUNS = VIDEO.parent / "runs"
OUT = VIDEO / "public" / "data"
FROZEN = VIDEO / "data" / "frozen"
LINE = re.compile(r"epoch\s+(\d+)/(\d+) loss=([\d.]+) train=([\d.]+)% test=([\d.]+)%")


def log_curve(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        m = LINE.search(line)
        if m:
            rows.append({"epoch": int(m[1]), "loss": float(m[3]), "train": float(m[4]), "test": float(m[5])})
    return rows


def metrics(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def mean_curve(curves, key):
    n = min(len(c) for c in curves)
    return [round(statistics.fmean(c[i][key] for c in curves), 3) for i in range(n)]


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def main():
    FROZEN.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in ("augmentation/comparison.json", "depth/comparison.json", "followup/comparison.json",
                "followup/queue-state.json", "depth/queue-state.json", "results.csv"):
        src = RUNS / rel
        if src.exists():
            shutil.copyfile(src, FROZEN / rel.replace("/", "__"))

    r = {"snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # Classroom model: per-epoch test accuracy from the run logs.
    r["classroom"] = {
        "baseline_b64": log_curve(RUNS / "baseline.log"),
        "tuned_b512_10ep": log_curve(RUNS / "tuned-b512.log"),
        "b512_80ep": log_curve(RUNS / "step-matched-b512.log"),
        "updates": {"baseline_b64": 7820, "tuned_b512_10ep": 980, "b512_80ep": 7840},
    }

    # Early augmentation suite (test accuracy, exploratory).
    aug = json.loads((RUNS / "augmentation" / "comparison.json").read_text())
    r["augmentation"] = {"completed": aug["completed"], "recipes": []}
    for rec in aug["recipes"]:
        runs = rec["runs"]
        gaps = [x.get("train_accuracy", x.get("final", {}).get("train/eval_accuracy", 0)) for x in runs] if runs else []
        r["augmentation"]["recipes"].append({
            "recipe": rec["recipe"], "mean": round(rec["test_accuracy_mean"], 2),
            "sd": round(rec["test_accuracy_sample_std"], 2),
            "delta_vs_none": [round(v, 2) for v in rec["paired_differences"]["none"]]})
    curves = {}
    for name in ("none", "flip", "crop-flip"):
        seeds = [log_curve(RUNS / "augmentation" / name / f"seed-{s}" / "run.log") for s in (0, 1, 2)]
        curves[name] = {"test": mean_curve(seeds, "test"), "train": mean_curve(seeds, "train")}
    r["augmentation"]["curves_seed_mean"] = curves
    md = (RUNS / "augmentation" / "comparison.md").read_text()
    gaps = {}
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and cells[0] in ("none", "flip", "crop-flip", "crop"):
            gaps[cells[0]] = float(cells[5])
    r["augmentation"]["train_test_gap_pp"] = gaps

    # Depth sweep (validation, final-10-epoch mean).
    dep = json.loads((RUNS / "depth" / "comparison.json").read_text())
    r["depth"] = {"selection": dep["selection"], "plain_decision": dep["plain_decision"], "candidates": [],
                  "test_results": [{"seed": t["config"]["seed"], "accuracy": t["accuracy"]} for t in dep["test_results"]]}
    accs = [t["accuracy"] for t in dep["test_results"]]
    r["depth"]["test_mean"] = round(statistics.fmean(accs), 2)
    r["depth"]["test_sd"] = round(sd(accs), 2)
    for c in dep["candidates"]:
        seeds = [metrics(RUNS / "depth" / c["variant"] / f"conv-{c['depth']}" / f"seed-{s}" / "metrics.jsonl") for s in (0, 1, 2)]
        r["depth"]["candidates"].append({
            "variant": c["variant"], "depth": c["depth"], "mean": round(c["mean_score"], 2),
            "sd": round(c["score_std"], 2), "params": c["runs"][0]["params"],
            "val_curve": mean_curve(seeds, "eval/validation_accuracy"),
            "train_curve": mean_curve(seeds, "train/eval_accuracy"),
            "seed_val_curves": [[round(x["eval/validation_accuracy"], 2) for x in s] for s in seeds],
            "final_train": round(statistics.fmean(s[-1]["train/eval_accuracy"] for s in seeds), 2)})
    lr = metrics(RUNS / "depth" / "plain" / "conv-2" / "seed-0" / "metrics.jsonl")
    r["depth"]["lr_schedule"] = [x["optimizer/lr"] for x in lr]
    r["depth"]["steps_per_epoch"] = lr[0]["performance/optimizer_steps"]

    # Follow-up queue (validation; test only if artifacts exist).
    q = json.loads((RUNS / "followup" / "queue-state.json").read_text())
    fu = json.loads((RUNS / "followup" / "comparison.json").read_text())
    r["followup"] = {"phase": q.get("phase"), "jobs": {k: v["status"] for k, v in q["jobs"].items()},
                     "selection": fu.get("selection"), "test_results_available": bool(fu.get("test_results")),
                     "candidates": []}
    if fu.get("test_results"):
        accs = [t["accuracy"] for t in fu["test_results"]]
        r["followup"]["test_results"] = [{"seed": t.get("seed", t.get("config", {}).get("seed")), "accuracy": t["accuracy"],
                                          "loss": t.get("loss")} for t in fu["test_results"]]
        # Cross-check against the per-seed test summaries written by the runner (read-only).
        sel = (fu.get("selection") or {}).get("candidate")
        checks = []
        for t in r["followup"]["test_results"]:
            path = RUNS / "followup" / f"{sel}/seed-{t['seed']}" / "test-summary.json"
            if path.exists():
                checks.append(abs(json.loads(path.read_text())["accuracy"] - t["accuracy"]) < 1e-9)
                shutil.copyfile(path, FROZEN / f"followup__{sel}__seed-{t['seed']}__test-summary.json")
        r["followup"]["test_summary_verified"] = bool(checks) and all(checks) and len(checks) == len(accs)
        r["followup"]["test_mean"] = round(statistics.fmean(accs), 2)
        r["followup"]["test_sd"] = round(sd(accs), 2) if len(accs) > 1 else None
    for c in fu["candidates"]:
        name = c["candidate"]
        root = RUNS / ("depth/residual/conv-32" if name == "winner-none" else f"followup/{name}")
        seeds_done = [s for s in c["seeds"]]
        seed_metrics = {s: metrics(root / f"seed-{s}" / "metrics.jsonl") for s in seeds_done}
        per_seed = []
        for s, rows in seed_metrics.items():
            last10 = [x["eval/validation_accuracy"] for x in rows[-10:]]
            per_seed.append({"seed": s, "final10": round(statistics.fmean(last10), 2),
                             "peak": round(max(x["eval/validation_accuracy"] for x in rows), 2),
                             "final_epoch": round(rows[-1]["eval/validation_accuracy"], 2),
                             "epochs": len(rows)})
        r["followup"]["candidates"].append({
            "candidate": name, "seeds": seeds_done, "mean": round(c["mean_score"], 2),
            "sd": round(c["score_std"], 2) if c.get("score_std") is not None else None,
            "paired_gain": [round(v, 2) for v in c["paired_gain_pp"]], "params": c["runs"][0]["params"],
            "per_seed": per_seed,
            "val_curve": mean_curve(list(seed_metrics.values()), "eval/validation_accuracy"),
            "train_curve": mean_curve(list(seed_metrics.values()), "train/eval_accuracy")})
    # Live progress of any running seed (not a result; shown only as status).
    running = {}
    for k, status in r["followup"]["jobs"].items():
        if status == "running":
            rows = metrics(RUNS / "followup" / k / "metrics.jsonl")
            running[k] = rows[-1]["epoch"] if rows else 0
    r["followup"]["running_epochs"] = running

    (OUT / "results.json").write_text(json.dumps(r, indent=1))
    fc = {c["candidate"]: (c["mean"], c["sd"], c["seeds"]) for c in r["followup"]["candidates"]}
    print("snapshot", r["snapshot_utc"], "phase", r["followup"]["phase"], "running", running)
    print("followup", fc)
    print("depth test", r["depth"]["test_mean"], r["depth"]["test_sd"])


if __name__ == "__main__":
    main()
