"""Publish local execution and batch benchmarks as a W&B comparison table."""
import json
from pathlib import Path

import wandb


def main():
    records = []
    for stage, path in (("execution", Path("runs/benchmarks/benchmarks.jsonl")),
                        ("batch size", Path("runs/batch-benchmarks/benchmarks.jsonl"))):
        latest = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["batch_size"], row["precision"], row["layout"], row["compile"])
            latest[key] = row
        records.extend((stage, row) for row in latest.values())
    columns = ["stage", "batch", "precision", "layout", "compiled", "train images/s",
               "eval images/s", "estimated 10 epochs (s)", "setup (s)",
               "peak memory (MB)", "FP32 logit difference", "FP32 prediction agreement"]
    table = wandb.Table(columns=columns)
    for stage, row in records:
        table.add_data(stage, row["batch_size"], row["precision"], row["layout"], row["compile"],
                       round(row["train_images_per_second"]), round(row["eval_images_per_second"]),
                       round(row["estimated_10_epoch_seconds"], 2), round(row["setup_seconds"], 2),
                       round(row["peak_gpu_memory_mb"]), row["fp32_mean_abs_logit_difference"],
                       row["fp32_prediction_agreement"])
    with wandb.init(entity="7adamyasingh-rutgers-university", project="cifar-activity",
                    group="benchmark", name="RTX 3090 throughput benchmarks",
                    id="rjjf0tny", resume="allow", tags=["benchmark", "cifar10"]) as run:
        run.log({"benchmarks/results": table})
        run.summary["benchmarks/winning_batch_size"] = 512
        run.summary["benchmarks/winning_precision"] = "fp32"
        run.summary["benchmarks/winning_layout"] = "contiguous"
        print(run.url)


if __name__ == "__main__":
    main()
