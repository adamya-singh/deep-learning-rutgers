"""Discover completed training runs and render architecture/change figures for skill-written notes.

This helper deliberately does not write the prose. Entries are drafted with
summarize-experiment-as-adamya, then registered with `record`.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

import depth_cnn as base
import run_depth_queue as depth_queue
import run_augmentation_queue as aug_queue

HERE = base.HERE
NOTES = HERE / "notes/research-log"
FIGURES = HERE / "figures/experiments"
REGISTRY = NOTES / "entries.json"


def entry_date(path):
    return datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("America/New_York")).strftime("%b %-d, %Y")


def collect():
    """Use completed artifacts only. Smoke/benchmark runs and live checkpoints are excluded."""
    records = []
    manifest = json.loads((HERE / "augmentation_experiments.json").read_text())
    augmentation = {}
    for recipe, seed in aug_queue.jobs(manifest):
        directory = HERE / "runs/augmentation" / recipe / f"seed-{seed}"
        path = directory / "summary.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        expected = aug_queue.expected_config(manifest, recipe, seed, row["config"]["device"])
        if not aug_queue.completed(directory, expected, manifest["epochs"]):
            continue
        record = {"key": f"augmentation/{recipe}/seed-{seed}", "suite": "augmentation", "recipe": recipe,
                  "seed": seed, "directory": str(directory), "summary": row, "date": entry_date(path),
                  "architecture": {"kind": "legacy", "depth": 2, "variant": "plain", "batch_norm": False},
                  "metric": "test accuracy", "score": row["test_accuracy"], "train_accuracy": row["train_accuracy"],
                  "loss": row["test_loss"], "seconds": row["elapsed_seconds"], "epochs": row["epochs"],
                  "batch_size": row["config"]["batch_size"], "params": 545098, "url": row["wandb_url"]}
        augmentation[recipe, seed] = record
        records.append(record)
    for record in records:
        if record["recipe"] == "none":
            record["reference"] = None
        else:
            recipe = "none" if record["recipe"] in ("crop", "flip", "crop-flip") else "crop-flip"
            record["reference"] = reference(augmentation.get((recipe, record["seed"])))

    # Historical full training runs predate summary.json. Recover their final metrics from checkpoints.
    historical = {}
    for name in ("baseline", "tuned-b512", "step-matched-b512"):
        directory = HERE / "runs" / name
        path = directory / "last.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = state["config"]
        metrics = state["pending_metrics"]
        record = {"key": f"historical/{name}", "suite": "historical", "recipe": "none", "seed": cfg["seed"],
                  "directory": str(directory), "date": entry_date(path), "architecture": {"kind": "legacy", "depth": 2,
                  "variant": "plain", "batch_norm": False}, "metric": "test accuracy", "score": metrics["eval/test_accuracy"],
                  "train_accuracy": metrics["train/eval_accuracy"], "loss": metrics["eval/test_loss"],
                  "seconds": state["elapsed"], "epochs": state["epoch"], "batch_size": cfg["batch_size"],
                  "params": 545098, "steps": state["steps"], "config": cfg,
                  "url": f"https://wandb.ai/{cfg['entity']}/{cfg['project']}/runs/{state['wandb_id']}",
                  "reference": reference(historical.get("baseline" if name == "tuned-b512" else "tuned-b512"))}
        historical[name] = record
        records.append(record)

    depth_root = HERE / "runs/depth"
    frozen_path = depth_root / "manifest.json"
    depth_records = {}
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        m = frozen["manifest"]
        completed = depth_queue.records_at(m, depth_root, frozen["device"], frozen["no_wandb"])
        for variant, depth in depth_queue.candidates(m):
            for seed in m["seeds"]:
                row = completed.get((variant, depth, seed))
                if row is None:
                    continue
                previous = m["depths"][m["depths"].index(depth) - 1] if depth != 2 else None
                comparator = depth_records.get(("plain", depth if variant == "residual" else previous, seed))
                record = modern_record(f"depth/{variant}/conv-{depth}/seed-{seed}", "depth", "none", seed,
                                       depth_root / depth_queue.key(variant, depth, seed), row,
                                       {"kind": "depth", "depth": depth, "variant": variant, "batch_norm": True}, reference(comparator))
                depth_records[variant, depth, seed] = record
                records.append(record)
    follow_root = HERE / "runs/followup"
    specs_path = follow_root / "specs.json"
    if specs_path.exists():
        import run_followup_queue as follow_queue
        document = json.loads(specs_path.read_text())
        frozen = json.loads((follow_root / "manifest.json").read_text())
        completed = follow_queue.read_records(document, follow_root, frozen["device"], frozen["no_wandb"])
        selected = document["selection"]
        for spec in document["candidates"]:
            for seed in document["seeds"]:
                row = completed.get((spec["id"], seed))
                if row is None:
                    continue
                comparator = depth_records.get((selected["variant"], selected["depth"], seed))
                architecture = ({"kind": "resnet18", "depth": 18, "variant": "residual", "batch_norm": True}
                                if spec["architecture"] == "resnet18-cifar" else
                                {"kind": "depth", "depth": selected["depth"], "variant": selected["variant"], "batch_norm": True})
                records.append(modern_record(f"followup/{spec['id']}/seed-{seed}", "followup", spec["augmentation"], seed,
                                             follow_root / follow_queue.job_key(spec["id"], seed), row, architecture, reference(comparator)))
    for record in records:
        record["hash"] = base.digest(record)
    return records


def reference(record):
    if record is None:
        return None
    return {k: record[k] for k in ("key", "score", "metric", "train_accuracy", "seconds", "epochs", "batch_size", "architecture")}


def modern_record(key, suite, recipe, seed, directory, row, architecture, comparator):
    final = row["final"]
    return {"key": key, "suite": suite, "recipe": recipe, "seed": seed, "directory": str(directory),
            "summary": row, "date": entry_date(directory / "summary.json"), "architecture": architecture,
            "metric": "validation score (last ten epochs)", "score": row["score"],
            "train_accuracy": final["train/eval_accuracy"], "final_validation_accuracy": final["eval/validation_accuracy"],
            "loss": final["eval/validation_loss"], "seconds": final["performance/elapsed_seconds"], "epochs": row["epochs"],
            "batch_size": (row["config"]["manifest"] if suite == "depth" else row["config"]["spec"]["training"])["batch_size"],
            "params": row["params"], "url": row["wandb_url"], "reference": comparator}


def registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def known_entries():
    found = registry()
    for path in NOTES.glob("[0-9]*.md"):
        tagged = re.search(r"<!-- experiment-key: ([^ ]+) -->", path.read_text())
        if tagged:
            found.setdefault(tagged[1], {"entry": str(path.relative_to(HERE))})
        match = re.fullmatch(r"\d+-(.+)-seed-(\d+)\.md", path.name)
        if match and match[1] in aug_queue.RECIPES:
            found.setdefault(f"augmentation/{match[1]}/seed-{match[2]}", {"entry": str(path.relative_to(HERE))})
        elif match and match[1].startswith("augmentation-") and match[1][13:] in aug_queue.RECIPES:
            found.setdefault(f"augmentation/{match[1][13:]}/seed-{match[2]}", {"entry": str(path.relative_to(HERE))})
        depth = re.fullmatch(r"\d+-depth-(plain|residual)-conv-(\d+)-seed-(\d+)\.md", path.name)
        if depth:
            found.setdefault(f"depth/{depth[1]}/conv-{depth[2]}/seed-{depth[3]}", {"entry": str(path.relative_to(HERE))})
        follow = re.fullmatch(r"\d+-followup-(.+)-seed-(\d+)\.md", path.name)
        if follow:
            found.setdefault(f"followup/{follow[1]}/seed-{follow[2]}", {"entry": str(path.relative_to(HERE))})
    return found


def next_number():
    return 1 + max((int(p.name.split("-")[0]) for p in NOTES.glob("[0-9]*.md")), default=0)


def slug(key):
    return key.replace("/", "-")


def render(record):
    """Bottom-to-top rounded pastel blocks, repeated stacks, and orange change callouts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set(xlim=(0, 10), ylim=(0, 14)); ax.axis("off")
    colors = {"conv": "#d5e8f4", "norm": "#fff2cc", "pool": "#f5d8de", "head": "#d9cce3", "input": "#f4dfc8", "aug": "#e1efd9"}
    orange = "#c65a10"
    arch = record["architecture"]
    comparator = record.get("reference")
    previous = comparator["architecture"] if comparator else None
    is_resnet = arch["kind"] == "resnet18"
    is_residual = arch["variant"] == "residual"
    is_aug = record["suite"] in ("augmentation", "followup") and record["recipe"] != "none"
    residual_changed = is_residual and (not previous or previous["variant"] != "residual")
    depth_changed = previous is not None and previous["depth"] != arch["depth"]
    title = "CIFAR ResNet-18" if is_resnet else f"{arch['depth']}-convolution {'residual' if is_residual else 'plain'} CNN"
    ax.text(5, 13.65, title, ha="center", fontsize=19, weight="bold")
    ax.text(5, 13.3, f"{record['params']:,} parameters · {record['suite']} · seed {record['seed']}", ha="center", fontsize=12)
    def box(y, label, kind="conv", changed=False, height=.5, width=3.6, x=3.0):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=.12",
                              facecolor=colors[kind], edgecolor=orange if changed else "#222", linewidth=2 if changed else 1.1)
        ax.add_patch(patch); ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=11)
    def arrow(y0, y1, x=4.8, color="#222"):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=color))
    def annotation(y, lines):
        ax.text(7.3, y, lines, va="center", fontsize=10, color=orange)
        ax.annotate("", xy=(6.65, y), xytext=(7.15, y), arrowprops={"arrowstyle": "->", "color": orange})
    def stack(y, label, channels, repeats, shortcut=False, changed=False, transition="", height=2.25):
        ax.add_patch(FancyBboxPatch((2.65, y), 4.3, height, boxstyle="round,pad=0.06,rounding_size=.15",
                                  facecolor="#f2f2f2", edgecolor="#888", linewidth=1))
        ax.text(2.35, y + height / 2, label, ha="right", va="center", fontsize=11)
        if arch["depth"] == 2 and not is_resnet:
            box(y + .2, f"3×3 Conv {transition}", changed=False)
            box(y + .9, "BatchNorm + ReLU" if arch["batch_norm"] else "ReLU", "norm",
                changed=arch["batch_norm"] and record["suite"] == "depth")
            arrow(y + .7, y + .9)
        else:
            box(y + .15, "3×3 Conv → BatchNorm → ReLU", changed=changed)
            box(y + .85, "3×3 Conv → BatchNorm", changed=changed)
            arrow(y + .65, y + .85)
            ax.text(6.75, y + .95, f"×{repeats}", fontsize=13, va="center", ha="left", color=orange if changed else "#222")
            if shortcut:
                ax.add_patch(Circle((4.8, y + 1.65), .11, facecolor="white", edgecolor=orange if residual_changed or is_resnet else "#222"))
                ax.text(4.8, y + 1.65, "+", ha="center", va="center", fontsize=12)
                arrow(y + 1.35, y + 1.53)
                color = orange if residual_changed or is_resnet else "#222"
                ax.plot([4.8, 2.85, 2.85, 4.6], [y - .15, y - .15, y + 1.65, y + 1.65], color=color, linewidth=1.4)
                ax.annotate("", xy=(4.69, y + 1.65), xytext=(4.4, y + 1.65), arrowprops={"arrowstyle": "->", "color": color})
                ax.text(2.72, y + .65, "shortcut", rotation=90, ha="right", va="center", fontsize=9, color=color)
                ax.text(4.8, y + 1.97, "ReLU", ha="center", va="center", fontsize=11)
                arrow(y + 1.76, y + 1.85)
            else:
                box(y + 1.55, "ReLU", "norm")
                arrow(y + 1.35, y + 1.55)
        ax.text(7.25, y + height - .05, channels, ha="left", fontsize=10)
    box(.25, "RGB image · 3 × 32 × 32", "input")
    recipe = record["recipe"]
    if recipe in ("crop", "flip", "crop-flip"):
        augmentation_label = {"crop": "Padded crop", "flip": "Horizontal flip", "crop-flip": "Padded crop + horizontal flip"}[recipe]
    elif recipe == "resized-crop":
        augmentation_label = "Resized crop + horizontal flip"
    else:
        augmentation_label = "Crop + flip\n+ " + recipe.replace("-", " ")
    box(.95, augmentation_label if is_aug else "Clean training view", "aug", changed=is_aug)
    arrow(.75, .95)
    box(1.65, "Normalize RGB channels", "norm")
    arrow(1.45, 1.65)
    if is_aug:
        annotation(1.2, "CHANGE: training views\n" + record["recipe"] +
                   ("\nNew architecture above" if is_resnet else "\nNetwork architecture unchanged"))
    if is_resnet:
        box(2.35, "3×3 Conv, 64 → BN → ReLU", changed=True)
        arrow(2.15, 2.35)
        y = 3.2
        for i, (channels, size) in enumerate(((64, 32), (128, 16), (256, 8), (512, 4))):
            box(y, f"Residual stage {i+1} · 2 blocks", changed=True, height=.65)
            ax.text(2.8, y + .32, f"{channels} × {size} × {size}", ha="right", fontsize=10)
            arrow(2.85 if i == 0 else y - .45, y)
            y += 1.15
        box(8.3, "Global average pool → 512", "pool", changed=True)
        arrow(7.3, 8.3)
        box(9.1, "Linear 512 → 10", "head", changed=True)
        arrow(8.8, 9.1)
        annotation(4.7, "CHANGE: four wider stages\n64 / 128 / 256 / 512 channels\nProjection shortcuts at transitions")
        annotation(8.55, "CHANGE: global averaging\nreplaces flatten + dense head")
        ax.text(4.8, 10.1, "Each block: Conv–BN–ReLU–Conv–BN + shortcut → ReLU", ha="center", fontsize=10)
        ax.text(4.8, 10.5, "No initial max pool; downsample only at stages 2, 3, and 4", ha="center", fontsize=10)
    else:
        repeats = max(1, arch["depth"] // 4)
        arrow(2.15, 2.8)
        stack(2.65, "Stage 1", "32 × 32 × 32", repeats, is_residual, depth_changed, "3 → 32")
        box(5.2, "MaxPool 2×2 → 32 × 16 × 16", "pool")
        arrow(4.05 if arch["depth"] == 2 else 4.7, 5.2)
        arrow(5.7, 6.35)
        stack(6.2, "Stage 2", "64 × 16 × 16", repeats, is_residual, depth_changed, "32 → 64")
        box(8.75, "MaxPool 2×2 → 64 × 8 × 8", "pool")
        arrow(7.6 if arch["depth"] == 2 else 8.25, 8.75)
        box(9.5, "Flatten → 4096", "pool")
        arrow(9.25, 9.5)
        box(10.25, "Linear 4096 → 128 → ReLU", "head")
        arrow(10, 10.25)
        box(11.0, "Linear 128 → 10", "head")
        arrow(10.75, 11)
        if depth_changed:
            annotation(4.0, f"CHANGE: depth {previous['depth']} → {arch['depth']}\n{repeats} two-conv blocks per stage\nChannels and pooling fixed")
        if residual_changed:
            annotation(7.75, "CHANGE: add shortcut sums\nZero-pad channels at transitions\nNo added trainable parameters")
        if record["suite"] == "depth" and arch["depth"] == 2:
            annotation(3.8, "CHANGE: add BatchNorm\nRemove convolution biases\nFresh training recipe and split")
    if record["suite"] == "historical":
        label = "Reference classroom recipe" if not comparator else ("CHANGE: batch 64 → 512" if record["epochs"] == 10 else "CHANGE: train 10 → 80 epochs")
        annotation(9.75, label + "\nArchitecture stays identical")
    if record["recipe"] == "none" and record["suite"] == "augmentation":
        annotation(1.2, "CONTROL: no augmentation\nArchitecture unchanged\nFresh paired-seed reference")
    ax.text(4.8, 11.9 if not is_resnet else 11.5, "10 class logits → cross-entropy loss", ha="center", fontsize=12)
    if not is_resnet:
        arrow(11.5, 11.75)
    ax.text(5, .03, "Orange marks the experimental change. Gray envelopes and pastel blocks follow the existing Transformer-paper figure style.", ha="center", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    prefix = FIGURES / slug(record["key"])
    for extension in ("svg", "png", "pdf"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return {ext: str(prefix.with_suffix("." + ext).relative_to(HERE)) for ext in ("svg", "png", "pdf")}


def attach_and_record(record, entry):
    text = entry.read_text()
    for required in ("**TL;DR:**", "**Problem.**", "*How we know:*", "**Experiment.**", "*What we hope to learn:*", "**Outcome.**", "**Takeaway:**"):
        if required not in text:
            raise ValueError(f"entry does not follow the experiment skill: missing {required}")
    images = render(record)
    marker = "<!-- experiment-architecture -->"
    if marker in text:
        text = text.split(marker)[0].rstrip()
    key_marker = f"<!-- experiment-key: {record['key']} -->"
    if key_marker not in text:
        text += "\n\n" + key_marker
    from os.path import relpath
    links = {ext: relpath(HERE / path, entry.parent) for ext, path in images.items()}
    text += f"\n\n{marker}\n![Architecture and experimental change]({links['png']})\n\n"
    text += f"Figure exports: [SVG]({links['svg']}) · [PDF]({links['pdf']}). Orange marks what changed; the diagram shows the trained architecture.\n"
    entry.write_text(text)
    import fcntl
    with (HERE / "runs/experiment-log.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = registry()
        saved[record["key"]] = {"entry": str(entry.relative_to(HERE)), "record_hash": record["hash"], "figures": images,
                                 "skill": "summarize-experiment-as-adamya"}
        base.atomic_json(REGISTRY, saved)


def update_index(records):
    saved = known_entries()
    lines = ["# CIFAR-10 experiment log", "", "One skill-written entry per completed training run. Each entry includes a bottom-to-top architecture diagram "
             "with orange callouts identifying its experimental change. Entries follow summarize-experiment-as-adamya and its voice corpus.", "",
             "Augmentation and historical runs report final test accuracy. Depth and follow-up runs report validation-only "
             "scores over the last ten epochs; these scores are not interchangeable. Smoke/benchmark runs are excluded.", "",
             "| Entry | Experiment | Metric | Score |", "| --- | --- | --- | ---: |"]
    indexed = []
    for record in records:
        item = saved.get(record["key"])
        if item:
            path = HERE / item["entry"]
            indexed.append((int(path.name.split("-")[0]), path, record))
    for number, path, record in sorted(indexed):
        lines.append(f"| [{number:02d}]({path.name}) | {record['key']} | {record['metric']} | {record['score']:.2f}% |")
    pending = [r["key"] for r in records if r["key"] not in saved or not saved[r["key"]].get("figures")]
    lines.extend(["", f"Completed training runs discovered: {len(records)}. Entries with registered diagrams: "
                  f"{sum(bool(saved.get(r['key'], {}).get('figures')) for r in records)}.", "",
                  "New completed runs are checked by the chat's experiment-log heartbeat. Each is drafted with the skill, "
                  "then its figure is generated and the index refreshed. Pending work is listed by `experiment_log.py pending`.", ""])
    (NOTES / "README.md").write_text("\n".join(lines))
    return pending


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("pending", "facts", "render", "record", "index"))
    p.add_argument("--key")
    p.add_argument("--entry", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    records = collect()
    if a.action == "pending":
        saved = known_entries()
        missing = [r for r in records if r["key"] not in saved or saved[r["key"]].get("record_hash") != r["hash"]
                   or not (HERE / saved[r["key"]]["entry"]).exists()
                   or not all((HERE / f).exists() for f in saved[r["key"]].get("figures", {}).values())
                   or not saved[r["key"]].get("figures")]
        print(json.dumps({"completed_runs": len(records), "next_number": next_number(),
                          "pending": [{"key": r["key"], "entry": saved.get(r["key"], {}).get("entry"), "score": r["score"]} for r in missing]}, indent=2))
    elif a.action == "index":
        print(json.dumps({"pending": update_index(records)}))
    else:
        record = next(r for r in records if r["key"] == a.key)
        if a.action == "facts": print(json.dumps(record, indent=2))
        elif a.action == "render": print(json.dumps(render(record)))
        else:
            if a.entry is None: p.error("record requires --entry")
            attach_and_record(record, a.entry.resolve())
            update_index(records)
            print(f"Registered {record['key']}: {a.entry}")


if __name__ == "__main__":
    main()
