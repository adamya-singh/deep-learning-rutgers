"""Organize the default and saved CIFAR-10 workspaces, including run views."""

import copy
import json
import wandb_workspaces.reports.v2 as charts
import wandb_workspaces.workspaces as workspaces
import wandb
from wandb_workspaces.workspaces.internal import View, WorkspaceViewspec, execute_graphql


ENTITY = "7adamyasingh-rutgers-university"
PROJECT = "cifar-activity"
WORKSPACE_URL = "https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=sno1vf8ff7t"
DEFAULT_WORKSPACE_URL = "https://wandb.ai/7adamyasingh-rutgers-university/cifar-activity?nw=nwuser7adamyasingh"


def line(name, x, *metrics, unit=None):
    axis_titles = {"epoch": "Epoch", "performance/elapsed_seconds": "Elapsed train + evaluation time (s)",
                   "performance/optimizer_steps": "Optimizer updates"}
    metric_titles = {"train/eval_accuracy": "Train", "eval/test_accuracy": "Test",
                     "eval/validation_accuracy": "Validation", "eval/validation_loss": "Validation",
                     "train/eval_loss": "Train", "eval/test_loss": "Test",
                     "train/loss": "Optimization loss"}
    return charts.LinePlot(
        title=name, x=x, y=list(metrics), title_x=axis_titles.get(x, x), title_y=unit,
        smoothing_type="none", smoothing_show_original=True, aggregate=False,
        max_runs_to_show=100, legend_position="south", font_size="medium",
        line_titles={key: metric_titles[key] for key in metrics if key in metric_titles},
        line_marks={key: "dashed" for key in metrics if key.startswith("train/")},
    )


def sections():
    return [
        workspaces.Section(name="Learning — convergence and generalization", panels=[
            line("Clean training accuracy", "epoch", "train/eval_accuracy", unit="Accuracy (%)"),
            line("Validation accuracy", "epoch", "eval/validation_accuracy", unit="Accuracy (%)"),
            line("Test accuracy", "epoch", "eval/test_accuracy", unit="Accuracy (%)"),
            line("Clean training loss", "epoch", "train/eval_loss", unit="Cross-entropy"),
            line("Validation loss", "epoch", "eval/validation_loss", unit="Cross-entropy"),
            line("Test loss", "epoch", "eval/test_loss", unit="Cross-entropy"),
            line("Train, validation and test accuracy", "epoch", "train/eval_accuracy", "eval/validation_accuracy", "eval/test_accuracy", unit="Accuracy (%)"),
            line("Train, validation and test loss", "epoch", "train/eval_loss", "eval/validation_loss", "eval/test_loss", unit="Cross-entropy"),
            line("Accuracy vs optimizer updates", "performance/optimizer_steps", "train/eval_accuracy", "eval/validation_accuracy", "eval/test_accuracy", unit="Accuracy (%)"),
            line("Test loss vs optimizer updates", "performance/optimizer_steps", "eval/test_loss", unit="Cross-entropy"),
            line("Accuracy vs elapsed time", "performance/elapsed_seconds", "train/eval_accuracy", "eval/validation_accuracy", "eval/test_accuracy", unit="Accuracy (%)"),
            line("Optimization loss", "epoch", "train/loss", unit="Cross-entropy"),
            line("Training batch accuracy", "epoch", "train/batch_accuracy", "train/mixed_target_accuracy", unit="Accuracy (%)"),
        ], is_open=True, pinned=True,
            layout_settings=workspaces.SectionLayoutSettings(columns=3, rows=5)),
        workspaces.Section(name="Final results — test appears only when evaluated", panels=[
            charts.ScalarChart(title="Final training accuracy", metric="final/train_accuracy"),
            charts.ScalarChart(title="Final validation accuracy", metric="final/validation_accuracy"),
            charts.ScalarChart(title="Final test accuracy", metric="final/test_accuracy"),
            charts.ScalarChart(title="Final training loss", metric="final/train_loss"),
            charts.ScalarChart(title="Final validation loss", metric="final/validation_loss"),
            charts.ScalarChart(title="Final test loss", metric="final/test_loss"),
            charts.ScalarChart(title="Selection score (last 10 validation epochs)", metric="final/validation_score"),
        ], is_open=True),
        workspaces.Section(name="Optimization", panels=[
            line("Learning rate", "epoch", "optimizer/lr"),
            line("Optimizer updates", "epoch", "performance/optimizer_steps", unit="Updates"),
        ], is_open=True),
        workspaces.Section(name="Performance", panels=[
            line("Training throughput", "epoch", "performance/train_images_per_second", unit="Images / s"),
            line("Time per epoch", "epoch", "performance/train_seconds", "performance/eval_seconds",
                 "performance/checkpoint_seconds", unit="Seconds"),
            line("Elapsed training and evaluation time", "epoch", "performance/elapsed_seconds", unit="Seconds"),
            line("Peak allocated GPU memory", "epoch", "system/peak_gpu_memory_mb", "performance/peak_gpu_memory_mb", unit="MB"),
        ], is_open=True),
        workspaces.Section(name="System", panels=[
            line("GPU utilization", "epoch", "system/gpu_utilization_pct", unit="Utilization (%)"),
            line("GPU memory use", "epoch", "system/gpu_memory_used_mb", unit="MB"),
            line("GPU power", "epoch", "system/gpu_power_w", unit="Watts"),
            line("GPU temperature", "epoch", "system/gpu_temperature_c", unit="°C"),
        ], is_open=False),
        workspaces.Section(name="Diagnostics", panels=[
            charts.MediaBrowser(media_keys=["diagnostics/confusion_matrix_table", "diagnostics/misclassified"]),
        ], is_open=False),
        workspaces.Section(name="Throughput benchmark results", panels=[
            charts.MediaBrowser(media_keys=["benchmarks/results"]),
        ], is_open=True),
    ]


def load_workspace(url):
    # The public URL validator excludes personal views; load the existing view
    # by its exact name so saving retains its ID and updates the default layout.
    if url == DEFAULT_WORKSPACE_URL:
        response = execute_graphql(wandb.Api(), """
            query($entity: String!, $project: String!) {
                project(name: $project, entityName: $entity) {
                    allViews(viewType: "project-view") {
                        edges { node { id name displayName spec } }
                    }
                }
            }
        """, {"entity": ENTITY, "project": PROJECT})
        node = next(edge["node"] for edge in response["project"]["allViews"]["edges"]
                    if edge["node"]["name"] == "nw-nwuser7adamyasingh-w")
        return workspaces.Workspace._from_model(View(
            entity=ENTITY, project=PROJECT, name=node["name"], id=node["id"],
            display_name=node["displayName"], spec=WorkspaceViewspec.model_validate_json(node["spec"])))
    return workspaces.Workspace.from_url(url)


def save_run_layout(workspace):
    """Run workspaces use a flat spec, unlike project workspaces' section wrapper.

    Save the project-wide personal and shared defaults. W&B uses these for all
    runs in this project, including runs created after this script executes.
    """
    project_spec = workspace._to_model().spec.model_dump(by_alias=True, exclude_none=True)
    section = project_spec["section"]
    spec = {key: copy.deepcopy(section[key]) for key in
            ("panelBankConfig", "settings")}
    spec["workspaceSettings"] = {"shouldAutoGeneratePanels": False,
                                 "showEmptySections": True, "sortAlphabetically": False}
    spec["panels"] = {"views": {}, "tabs": []}
    for saved_section in spec["panelBankConfig"]["sections"]:
        for panel in saved_section.get("panels", []):
            panel.get("config", {})["singleRun"] = True
    api = wandb.Api()
    existing = execute_graphql(api, """
        query($entity: String!, $project: String!) {
            project(name: $project, entityName: $entity) {
                allViews(viewType: "run-view") { edges { node { id name spec } } }
            }
        }
    """, {"entity": ENTITY, "project": PROJECT})
    nodes = {e["node"]["name"]: e["node"]
             for e in existing["project"]["allViews"]["edges"]}
    for name in ("default", "nw-nwuser7adamyasingh-w"):
        result = execute_graphql(api, """
            mutation($id: ID, $entity: String!, $project: String!, $name: String!, $spec: String!) {
                upsertView(input: {id: $id, entityName: $entity, projectName: $project,
                    name: $name, type: "run-view", displayName: "CIFAR-10: consistent run metrics",
                    spec: $spec, createdUsing: WANDB_SDK}) { view { id name spec } }
            }
        """, {"id": nodes.get(name, {}).get("id"), "entity": ENTITY, "project": PROJECT,
              "name": name, "spec": json.dumps(spec)})
        actual = json.loads(result["upsertView"]["view"]["spec"])
        assert actual["panelBankConfig"]["sections"] == spec["panelBankConfig"]["sections"]
        print(f"Updated and verified run layout: {name}")


def persist_manual_mode(workspace):
    # Modern W&B clients read workspaceSettings before the legacy settings block.
    # Set both, so logging a newly introduced metric cannot regenerate panels.
    model = workspace._to_model()
    spec = model.spec.model_dump(by_alias=True, exclude_none=True)
    spec['section']['settings']['shouldAutoGeneratePanels'] = False
    spec['section']['workspaceSettings'] = {
        'shouldAutoGeneratePanels': False, 'showEmptySections': True,
        'sortAlphabetically': False,
        'linePlot': {'xAxis': 'epoch', 'smoothingType': 'none', 'showLegend': True}}
    spec['section']['panelBankConfig']['settings']['showEmptySections'] = True
    for section in spec['section']['panelBankConfig']['sections']:
        section['isPanelsAuto'] = False
    execute_graphql(wandb.Api(), """
        mutation($id: ID!, $entity: String!, $project: String!, $name: String!, $spec: String!) {
            upsertView(input: {id: $id, entityName: $entity, projectName: $project,
                name: $name, type: "project-view", spec: $spec, createdUsing: WANDB_SDK}) { view { id } }
        }
    """, {'id': model.id, 'entity': ENTITY, 'project': PROJECT,
          'name': model.name, 'spec': json.dumps(spec)})


def layout_signature(items):
    return [(s.name, [p.title if hasattr(p, 'title') else type(p).__name__ for p in s.panels])
            for s in items]


def ensure_layout():
    expected = layout_signature(sections())
    for url in (DEFAULT_WORKSPACE_URL, WORKSPACE_URL):
        workspace = load_workspace(url)
        # Replace stale automatic panel overrides as well as the panel list.
        object.__setattr__(workspace, '_auto_generate_panels', False)
        object.__setattr__(workspace, '_stashed_panel_config_overrides', None)
        object.__setattr__(workspace, '_stashed_panel_placement_overrides', None)
        if layout_signature(workspace.sections) != expected:
            print(f'Layout drift detected; restoring shared chart order: {url}', flush=True)
            main()
            return
    project_bank = workspace._to_model().spec.model_dump(by_alias=True, exclude_none=True)[
        'section']['panelBankConfig']['sections']
    def bank_signature(bank):
        return [(s['name'], [(p['viewType'], p.get('config', {}).get('chartTitle'),
                             p.get('config', {}).get('metrics'), p.get('config', {}).get('xAxis'))
                            for p in s.get('panels', [])]) for s in bank]
    result = execute_graphql(wandb.Api(), """
        query($entity: String!, $project: String!) {
            project(name: $project, entityName: $entity) {
                allViews(viewType: "run-view") { edges { node { name spec } } }
            }
        }
    """, {'entity': ENTITY, 'project': PROJECT})
    nodes = {e['node']['name']: json.loads(e['node']['spec'])
             for e in result['project']['allViews']['edges']}
    for name in ('default', 'nw-nwuser7adamyasingh-w'):
        if (name not in nodes or bank_signature(nodes[name]['panelBankConfig']['sections']) !=
                bank_signature(project_bank) or
                nodes[name].get('workspaceSettings', {}).get('shouldAutoGeneratePanels') is not False):
            save_run_layout(workspace)
            return


def main():
    for url in (DEFAULT_WORKSPACE_URL, WORKSPACE_URL):
        workspace = load_workspace(url)
        workspace.sections = sections()
        workspace.settings = workspaces.WorkspaceSettings(
            x_axis="epoch", smoothing_type="none", sort_panels_alphabetically=False,
            remove_legends_from_panels=False, max_runs=100)
        workspace.runset_settings = workspaces.RunsetSettings(
            pinned_columns=["run:displayName", "config:batch_size.value", "config:epochs.value",
                            "summary:final/train_accuracy", "summary:final/validation_score",
                            "summary:final/validation_accuracy", "summary:final/test_accuracy", "summary:final/test_loss",
                            "summary:final/elapsed_seconds", "summary:final/optimizer_steps"])
        # One atomic save: an intermediate automatic view could let the frontend
        # regenerate its panels before the modern manual-mode flag is written.
        persist_manual_mode(workspace)
        saved = load_workspace(url)
        assert saved.sections[0].pinned and len(saved.sections[0].panels) == 13
        assert layout_signature(saved.sections) == layout_signature(sections())
        assert not saved.runset_settings.filters
        print(f"Updated and verified: {workspace.url}")
    save_run_layout(workspace)


if __name__ == "__main__":
    main()
