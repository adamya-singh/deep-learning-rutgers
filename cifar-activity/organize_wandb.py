"""Create the curated CIFAR-10 project workspace. Run after the first W&B run."""

import wandb_workspaces.reports.v2 as charts
import wandb_workspaces.workspaces as workspaces


ENTITY = "7adamyasingh-rutgers-university"
PROJECT = "cifar-activity"
WORKSPACE_URL = "https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=sno1vf8ff7t"


def line(name, x, *metrics):
    return charts.LinePlot(title=name, x=x, y=list(metrics))


def main():
    sections = [
        workspaces.Section(name="Learning", panels=[
            line("Loss by epoch", "epoch", "train/loss", "train/eval_loss", "eval/test_loss"),
            line("Accuracy by epoch", "epoch", "train/eval_accuracy", "eval/test_accuracy"),
            line("Accuracy by training time", "performance/elapsed_seconds", "eval/test_accuracy"),
        ], is_open=True),
        workspaces.Section(name="Performance", panels=[
            line("Training throughput", "epoch", "performance/train_images_per_second"),
            line("Time per epoch", "epoch", "performance/train_seconds", "performance/eval_seconds",
                 "performance/checkpoint_seconds"),
            line("GPU memory", "epoch", "system/peak_gpu_memory_mb"),
        ], is_open=True),
        workspaces.Section(name="System", panels=[
            line("GPU utilization", "epoch", "system/gpu_utilization_pct"),
            line("GPU memory use", "epoch", "system/gpu_memory_used_mb"),
            line("Power and temperature", "epoch", "system/gpu_power_w", "system/gpu_temperature_c"),
        ], is_open=False),
        workspaces.Section(name="Diagnostics", panels=[
            charts.MediaBrowser(media_keys=["diagnostics/confusion_matrix", "diagnostics/misclassified"]),
        ], is_open=False),
        workspaces.Section(name="Comparison", panels=[
            line("Test accuracy", "epoch", "eval/test_accuracy"),
            line("Elapsed time", "epoch", "performance/elapsed_seconds"),
            line("Optimizer updates", "epoch", "performance/optimizer_steps"),
        ], is_open=True),
    ]
    workspace = workspaces.Workspace.from_url(WORKSPACE_URL)
    workspace.sections = sections
    workspace.runset_settings = workspaces.RunsetSettings(
        filters='Group != "smoke" and Group != "benchmark"')
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
