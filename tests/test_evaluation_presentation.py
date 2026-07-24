"""
===============================================================================
test_evaluation_presentation.py
===============================================================================
Verify structured result figures and the recovered complete evaluation widget.

Responsibilities:
  - Validate staged chronology, exact boundaries, and selected-state markers.
  - Reconstruct all result views from compact public artifact fixtures.
  - Exercise model/view controls and replacement-safe widget updates.
  - Prove score aggregation and missing-artifact failures are deterministic.

Design principles:
  - Matplotlib uses a headless backend and tests inspect artist structure
    rather than pixels.
  - Tiny fixtures preserve the real artifact schema and maintained stage
    order.

Boundaries:
  - Tests never train models, run inference, access datasets, or load
    checkpoints.
===============================================================================
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.config.config_paths import ProjectPaths
from src.evaluation import evaluation_plots
from src.presentation import presentation_widgets

RUN_ID = "verified_fixture"


def _history_row(epoch: int, val_wauc: float, *, stage: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "epoch": epoch,
        "train_loss": 0.5 - epoch * 0.01,
        "train_acc": 0.45 + epoch * 0.01,
        "val_loss": 0.51 - epoch * 0.01,
        "val_acc": 0.44 + epoch * 0.01,
        "val_wauc": val_wauc,
    }
    if stage is not None:
        row["stage"] = stage
    return row


def _staged_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in evaluation_plots.EFFICIENTNET_STAGE_ORDER:
        epochs = 3 if stage == "feature_8" else 2 if stage == "head_stem" else 1
        for epoch in range(1, epochs + 1):
            score = 0.55 + epoch * 0.001
            if stage == "feature_8" and epoch == 3:
                score = 0.8
            rows.append(_history_row(epoch, score, stage=stage))
    return pd.DataFrame(rows)


def _write_public_run(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths(tmp_path)
    run_root = paths.artifacts / "alaska2" / RUN_ID
    tiny_history = pd.DataFrame([_history_row(epoch, 0.58 + epoch * 0.01) for epoch in range(1, 5)])
    efficient_history = _staged_history()
    comparison = pd.DataFrame(
        [
            {
                "model": "TinyCNN",
                "training_groups": 6,
                "training_images": 24,
                "best_validation_weighted_auc": 0.62,
                "final_test_accuracy": 0.48,
                "final_test_weighted_auc": 0.59,
                "selected_checkpoint": "/ignored/tiny.pt",
                "runtime_seconds": 125.0,
            },
            {
                "model": "EfficientNet-B0",
                "training_groups": 6,
                "training_images": 24,
                "best_validation_weighted_auc": 0.8,
                "final_test_accuracy": 0.26,
                "final_test_weighted_auc": 0.58,
                "selected_checkpoint": "/ignored/efficient.pt",
                "runtime_seconds": 185.0,
            },
        ]
    )
    run_root.mkdir(parents=True)
    comparison.to_csv(run_root / "comparison.csv", index=False)

    for key, history, accuracy, wauc in (
        ("tinycnn", tiny_history, 0.48, 0.59),
        ("efficientnet_b0", efficient_history, 0.26, 0.58),
    ):
        history_path = run_root / key / "histories" / "training_history.csv"
        evaluation_path = run_root / key / "evaluation"
        history_path.parent.mkdir(parents=True)
        evaluation_path.mkdir(parents=True)
        history.to_csv(history_path, index=False)
        (evaluation_path / "test_metrics.json").write_text(
            json.dumps(
                {
                    "confusion_matrix": [[4, 1], [2, 5]],
                    "evaluation_pass_count": 1,
                    "sample_count": 12,
                    "test_accuracy": accuracy,
                    "test_loss": 0.35,
                    "test_weighted_auc": wauc,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame({"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.7, 1.0]}).to_csv(
            evaluation_path / "roc_data.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "bin_left": [0.0, 0.5],
                "bin_right": [0.5, 1.0],
                "cover_count": [4, 1],
                "stego_count": [2, 5],
            }
        ).to_csv(evaluation_path / "score_distribution.csv", index=False)
    return paths


def test_stage_order_global_epochs_boundaries_and_disconnected_lines() -> None:
    history = _staged_history()
    timeline = evaluation_plots.prepare_history(
        history.sample(frac=1.0, random_state=7),
        stage_order=evaluation_plots.EFFICIENTNET_STAGE_ORDER,
    )

    assert tuple(interval.name for interval in timeline.intervals) == evaluation_plots.EFFICIENTNET_STAGE_ORDER
    assert [(interval.start, interval.end) for interval in timeline.intervals[:3]] == [(1, 2), (3, 5), (6, 6)]
    feature_eight = timeline.dataframe.loc[timeline.dataframe["stage"] == "feature_8"]
    assert feature_eight[["epoch", "global_epoch"]].to_records(index=False).tolist() == [(1, 3), (2, 4), (3, 5)]

    figure = evaluation_plots.plot_history(
        history,
        selected_epoch=3,
        selected_stage="feature_8",
        stage_order=evaluation_plots.EFFICIENTNET_STAGE_ORDER,
    )
    intervals = [(interval.start, interval.end) for interval in timeline.intervals]
    for axis in figure.axes:
        data_lines = [line for line in axis.lines if line.get_gid() == "history-series"]
        assert data_lines
        for line in data_lines:
            x_values = np.asarray(line.get_xdata(), dtype=float)
            assert any(x_values.min() >= start and x_values.max() <= end for start, end in intervals)
        boundaries = [
            float(np.asarray(line.get_xdata(), dtype=float)[0])
            for line in axis.lines
            if line.get_gid() == "stage-boundary"
        ]
        assert boundaries == [interval.start - 0.5 for interval in timeline.intervals[1:]]
    plt.close(figure)


def test_selected_markers_identify_efficientnet_and_tinycnn_states() -> None:
    staged = _staged_history()
    efficient_figure = evaluation_plots.plot_history(
        staged,
        selected_epoch=3,
        selected_stage="feature_8",
        stage_order=evaluation_plots.EFFICIENTNET_STAGE_ORDER,
    )
    efficient_selected = [line for line in efficient_figure.axes[2].lines if line.get_gid() == "selected-state"]
    assert len(efficient_selected) == 1
    assert np.asarray(efficient_selected[0].get_xdata()).tolist() == [5, 5]
    assert any(collection.get_gid() == "selected-state-marker" for collection in efficient_figure.axes[2].collections)
    assert "feature_8, stage epoch 3" in efficient_figure.axes[2].texts[-1].get_text()
    plt.close(efficient_figure)

    tiny = pd.DataFrame([_history_row(epoch, 0.58 + epoch * 0.01) for epoch in range(1, 5)])
    tiny_figure = evaluation_plots.plot_history(tiny, selected_epoch=4)
    tiny_selected = [line for line in tiny_figure.axes[2].lines if line.get_gid() == "selected-state"]
    assert np.asarray(tiny_selected[0].get_xdata()).tolist() == [4, 4]
    assert "epoch 4" in tiny_figure.axes[2].texts[-1].get_text()
    plt.close(tiny_figure)


def test_compact_result_loading_and_missing_artifact_error(tmp_path: Path) -> None:
    paths = _write_public_run(tmp_path)
    results = evaluation_plots.load_evaluation_results(RUN_ID, paths=paths)

    assert [result.key for result in results.models] == ["tinycnn", "efficientnet_b0"]
    assert results.model("tinycnn").selected_epoch == 4
    assert results.model("efficientnet_b0").selected_stage == "feature_8"
    assert results.model("efficientnet_b0").selected_epoch == 3
    assert results.model("tinycnn").confusion_matrix.tolist() == [[4, 1], [2, 5]]
    assert results.model("tinycnn").roc_data.columns.tolist() == ["fpr", "tpr"]
    assert int(results.model("tinycnn").score_distribution[["cover_count", "stego_count"]].to_numpy().sum()) == 12
    assert "selected_checkpoint" not in evaluation_plots.comparison_table(results).columns

    missing = paths.artifacts / "alaska2" / RUN_ID / "tinycnn" / "evaluation" / "roc_data.csv"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match=r"Required public evaluation artifact.*roc_data\.csv"):
        evaluation_plots.load_evaluation_results(RUN_ID, paths=paths)


@pytest.mark.parametrize("model_key", ["tinycnn", "efficientnet_b0"])
@pytest.mark.parametrize(
    "view_key",
    [key for key, _ in presentation_widgets.EVALUATION_VIEW_TITLES],
)
def test_every_evaluation_view_renders_from_compact_structured_evidence(
    tmp_path: Path,
    model_key: str,
    view_key: str,
) -> None:
    paths = _write_public_run(tmp_path)
    results = evaluation_plots.load_evaluation_results(RUN_ID, paths=paths)
    run_root = paths.artifacts / "alaska2" / RUN_ID

    assert not (paths.root / "assets").exists()
    for model in ("tinycnn", "efficientnet_b0"):
        assert not (run_root / model / "checkpoints").exists()
        assert not (run_root / model / "predictions").exists()

    rendered = presentation_widgets._evaluation_view(results, model_key, view_key)
    if view_key in {"history", "confusion", "roc", "scores"}:
        assert isinstance(rendered, Figure)
        assert rendered.axes
        plt.close(rendered)
    else:
        assert isinstance(rendered, pd.DataFrame)
        assert not rendered.empty
    assert not plt.get_fignums()


def test_score_distribution_aggregation_preserves_class_counts() -> None:
    distribution = evaluation_plots.aggregate_score_distribution(
        np.array([0.0, 0.2, 0.8, 1.0]),
        np.array([0, 1, 0, 1]),
        bin_edges=np.array([0.0, 0.5, 1.0]),
    )
    assert distribution.to_dict(orient="list") == {
        "bin_left": [0.0, 0.5],
        "bin_right": [0.5, 1.0],
        "cover_count": [1, 1],
        "stego_count": [1, 1],
    }


def test_evaluation_widget_owns_one_display_and_one_render_per_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_public_run(tmp_path)
    rendered: list[tuple[str, str]] = []
    displayed: list[tuple[str | None, object]] = []
    clears: list[tuple[str, bool]] = []
    active_figures_at_display: list[int] = []
    context_stack: list[str] = []
    registrations: Counter[tuple[int, object]] = Counter()
    removals: Counter[tuple[int, object]] = Counter()

    def callback_key(handler: Any) -> tuple[int, object]:
        return id(getattr(handler, "__self__", None)), getattr(handler, "__func__", handler)

    def maintained(handler: Any) -> bool:
        function = getattr(handler, "__func__", handler)
        return getattr(function, "__module__", "") == presentation_widgets.__name__

    def fake_view(
        results: evaluation_plots.EvaluationResults,
        model_key: str,
        view_key: str,
    ) -> Figure | pd.DataFrame:
        del results
        rendered.append((model_key, view_key))
        if view_key in {"metrics", "comparison"}:
            return pd.DataFrame({"value": [view_key]})
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [0.0, 1.0])
        return figure

    def fake_display(value: object) -> None:
        displayed.append((context_stack[-1] if context_stack else None, value))
        if isinstance(value, Figure):
            active_figures_at_display.append(len(plt.get_fignums()))

    original_enter = ipywidgets.Output.__enter__
    original_exit = ipywidgets.Output.__exit__
    original_observe = ipywidgets.Widget.observe
    original_unobserve = ipywidgets.Widget.unobserve
    original_on_click = ipywidgets.Button.on_click

    def record_enter(output: ipywidgets.Output):
        entered = original_enter(output)
        context_stack.append(output.model_id)
        return entered

    def record_exit(output: ipywidgets.Output, *args: Any):
        try:
            return original_exit(output, *args)
        finally:
            assert context_stack.pop() == output.model_id

    def record_clear(*args: Any, **kwargs: Any) -> None:
        assert len(context_stack) == 1
        clears.append((context_stack[-1], bool(kwargs.get("wait", False))))

    def record_observe(widget: ipywidgets.Widget, handler: Any, *args: Any, **kwargs: Any) -> None:
        if maintained(handler):
            registrations[callback_key(handler)] += 1
        original_observe(widget, handler, *args, **kwargs)

    def record_unobserve(widget: ipywidgets.Widget, handler: Any, *args: Any, **kwargs: Any) -> None:
        if maintained(handler):
            removals[callback_key(handler)] += 1
        original_unobserve(widget, handler, *args, **kwargs)

    def record_on_click(
        button: ipywidgets.Button,
        callback: Any,
        remove: bool = False,
    ) -> None:
        if maintained(callback):
            target = removals if remove else registrations
            target[callback_key(callback)] += 1
        original_on_click(button, callback, remove=remove)

    monkeypatch.setattr(presentation_widgets, "_evaluation_view", fake_view)
    monkeypatch.setattr(presentation_widgets, "display", fake_display)
    monkeypatch.setattr(presentation_widgets, "clear_output", record_clear)
    monkeypatch.setattr(ipywidgets.Output, "__enter__", record_enter)
    monkeypatch.setattr(ipywidgets.Output, "__exit__", record_exit)
    monkeypatch.setattr(ipywidgets.Widget, "observe", record_observe)
    monkeypatch.setattr(ipywidgets.Widget, "unobserve", record_unobserve)
    monkeypatch.setattr(ipywidgets.Button, "on_click", record_on_click)

    open_text = "open-evaluation-fixture"
    close_text = "close-evaluation-fixture"
    widget = presentation_widgets.make_evaluation_widget(
        RUN_ID,
        paths=paths,
        open_button_text=open_text,
        close_button_text=close_text,
    )
    descendants = tuple(_walk_widgets(widget))
    open_button = next(
        item for item in descendants if isinstance(item, ipywidgets.Button) and item.description == open_text
    )
    close_button = next(
        item for item in descendants if isinstance(item, ipywidgets.Button) and item.description == close_text
    )
    model_selector = next(item for item in descendants if isinstance(item, ipywidgets.ToggleButtons))
    tabs = next(item for item in descendants if isinstance(item, ipywidgets.Tab))

    _assert_unique_widget_models(widget)
    output_by_tab: list[ipywidgets.Output] = []
    for tab_child in tabs.children:
        tab_outputs = [item for item in _walk_widgets(tab_child) if isinstance(item, ipywidgets.Output)]
        assert len(tab_outputs) == 1
        output_by_tab.append(tab_outputs[0])
    assert len({output.model_id for output in output_by_tab}) == len(output_by_tab)
    original_parents = _output_parents(widget)
    child_mutations = _watch_child_mutations(widget)

    def assert_ownership() -> None:
        _assert_unique_widget_models(widget)
        assert _output_parents(widget) == original_parents
        assert not child_mutations

    def assert_result(action, expected: tuple[str, str], output: ipywidgets.Output) -> None:
        render_count = len(rendered)
        clear_count = len(clears)
        display_count = len(displayed)
        figure_count = len(active_figures_at_display)
        action()
        assert rendered[render_count:] == [expected]
        assert clears[clear_count:] == [(output.model_id, True)]
        publications = displayed[display_count:]
        assert len(publications) == 1
        assert publications[0][0] == output.model_id
        if expected[1] in {"metrics", "comparison"}:
            assert isinstance(publications[0][1], pd.DataFrame)
            assert len(active_figures_at_display) == figure_count
        else:
            assert isinstance(publications[0][1], Figure)
            assert active_figures_at_display[figure_count:] == [1]
        assert not plt.get_fignums()
        assert_ownership()

    assert not displayed
    assert not rendered
    presentation_widgets.display(widget)
    assert displayed == [(None, widget)]

    model_values = [option[1] if isinstance(option, tuple) else option for option in model_selector.options]
    assert_result(open_button.click, (str(model_values[0]), "history"), output_by_tab[0])
    counts = (len(rendered), len(clears), len(displayed))
    open_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts
    assert_ownership()

    assert_result(
        lambda: setattr(tabs, "selected_index", 1),
        (str(model_values[0]), "confusion"),
        output_by_tab[1],
    )
    metrics_index = next(
        index
        for index, (view_key, _) in enumerate(presentation_widgets.EVALUATION_VIEW_TITLES)
        if view_key == "metrics"
    )
    assert_result(
        lambda: setattr(tabs, "selected_index", metrics_index),
        (str(model_values[0]), "metrics"),
        output_by_tab[metrics_index],
    )
    assert_result(
        lambda: setattr(model_selector, "value", model_values[-1]),
        (str(model_values[-1]), "metrics"),
        output_by_tab[metrics_index],
    )

    counts = (len(rendered), len(clears), len(displayed))
    model_selector.value = model_values[-1]
    tabs.selected_index = metrics_index
    assert (len(rendered), len(clears), len(displayed)) == counts

    render_count = len(rendered)
    display_count = len(displayed)
    clear_count = len(clears)
    close_button.click()
    assert clears[clear_count:] == [(output_by_tab[metrics_index].model_id, True)]
    assert len(rendered) == render_count
    assert len(displayed) == display_count
    assert not plt.get_fignums()
    assert_ownership()

    comparison_index = next(
        index
        for index, (view_key, _) in enumerate(presentation_widgets.EVALUATION_VIEW_TITLES)
        if view_key == "comparison"
    )
    tabs.selected_index = comparison_index
    model_selector.value = model_values[0]
    assert len(rendered) == render_count

    assert_result(
        open_button.click,
        (str(model_values[0]), "comparison"),
        output_by_tab[comparison_index],
    )
    counts = (len(rendered), len(clears), len(displayed))
    open_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts

    widget.close()
    assert registrations == removals
    assert all(item.comm is None for item in descendants)
    tabs.selected_index = 0
    model_selector.value = model_values[-1]
    open_button.click()
    close_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts
    assert not plt.get_fignums()
    assert not child_mutations


def _walk_widgets(root: ipywidgets.Widget):
    yield root
    for child in getattr(root, "children", ()):
        yield from _walk_widgets(child)


def _assert_unique_widget_models(root: ipywidgets.Widget) -> None:
    locations: dict[str, list[str]] = defaultdict(list)

    def visit(widget: ipywidgets.Widget, path: str, ancestors: set[str]) -> None:
        locations[widget.model_id].append(path)
        if widget.model_id in ancestors:
            return
        for index, child in enumerate(getattr(widget, "children", ())):
            visit(child, f"{path}.children[{index}]", {*ancestors, widget.model_id})

    visit(root, "root", set())
    assert not {model_id: paths for model_id, paths in locations.items() if len(paths) > 1}


def _output_parents(root: ipywidgets.Widget) -> dict[str, str]:
    parents: dict[str, str] = {}
    for parent in _walk_widgets(root):
        for child in getattr(parent, "children", ()):
            if isinstance(child, ipywidgets.Output):
                assert child.model_id not in parents
                parents[child.model_id] = parent.model_id
    return parents


def _watch_child_mutations(root: ipywidgets.Widget) -> list[str]:
    mutations: list[str] = []
    for widget in tuple(_walk_widgets(root)):
        if "children" not in widget.traits():
            continue

        def record(_: dict[str, Any], model_id: str = widget.model_id) -> None:
            mutations.append(model_id)

        widget.observe(record, names="children")
    return mutations


def test_changed_modules_expose_the_maintained_public_surface() -> None:
    assert "prepare_history" in evaluation_plots.__all__
    assert "load_evaluation_results" in evaluation_plots.__all__
    assert "make_evaluation_widget" in presentation_widgets.__all__
    assert "EVALUATION_VIEW_TITLES" in presentation_widgets.__all__
