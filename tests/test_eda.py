"""
===============================================================================
test_eda.py
===============================================================================
Verify spatial color-channel exploratory-analysis contracts.

Responsibilities:
  - Exercise indexed z-score assignment for nonconsecutive dataframe indexes.
  - Reject multidimensional z-score outputs instead of flattening them.

Design principles:
  - Small deterministic image doubles isolate dataframe behavior from image I/O.

Boundaries:
  - Widget rendering and scientific plots are not snapshot-tested here.
===============================================================================
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import ipywidgets
import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy.stats import zscore

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config.config_paths import ProjectPaths
from src.eda import eda_channels, eda_controls, eda_examples
from src.presentation import presentation_cache, presentation_widgets


class _FakeImage:
    def __init__(self, value: float) -> None:
        self._value = value

    def __enter__(self) -> _FakeImage:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def convert(self, color_space: str) -> np.ndarray:
        return np.full((2, 2, 3), self._value, dtype=np.float64)


def test_outlier_z_scores_preserve_index_and_numeric_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    image_means = {
        "/Cover/a.jpg": 10.0,
        "/JMiPOD/b.jpg": 20.0,
        "/Cover/c.jpg": 30.0,
    }

    def fake_open(path: str) -> _FakeImage:
        try:
            return _FakeImage(image_means[path])
        except KeyError as error:
            raise OSError("unreadable fixture image") from error

    original_setitem = pd.DataFrame.__setitem__
    assignments: list[tuple[object, pd.Index, pd.Series]] = []

    def recording_setitem(frame, key, value) -> None:
        if key == "z_score":
            assert isinstance(value, pd.Series)
            assignments.append((key, frame.index.copy(), value.copy()))
        original_setitem(frame, key, value)

    monkeypatch.setattr(eda_channels.Image, "open", fake_open)
    monkeypatch.setattr(pd.DataFrame, "__setitem__", recording_setitem)

    frame = pd.DataFrame(
        {
            "path": ["/Cover/a.jpg", "/JMiPOD/b.jpg", "/Cover/c.jpg", "/broken/d.jpg"],
            "label_name": ["Cover", "JMiPOD", "Cover", "JMiPOD"],
        },
        index=pd.Index([11, 23, 37, 41]),
    )
    before = frame.copy(deep=True)

    outlier_widget = eda_channels.show_outliers_by_channel(frame, z_thresh=3.0)

    pd.testing.assert_frame_equal(frame, before)
    assert len(assignments) == 1
    column_name, target_index, assigned = assignments[0]
    assert column_name == "z_score"
    expected_index = pd.Index([11, 23, 37])
    pd.testing.assert_index_equal(target_index, expected_index)
    assert len(assigned) == len(expected_index)
    pd.testing.assert_index_equal(assigned.index, expected_index)
    assert assigned.dtype == np.dtype("float64")
    expected_z_scores = np.asarray(zscore([10.0, 20.0, 30.0]), dtype=np.float64)
    np.testing.assert_allclose(assigned.to_numpy(), expected_z_scores)
    assert not assigned.isna().any()
    outlier_widget.close()
    assert not plt.get_fignums()

    monkeypatch.setattr(eda_channels, "zscore", lambda values: np.zeros((len(values), 1)))
    with pytest.raises(ValueError, match="one-dimensional"):
        eda_channels.show_outliers_by_channel(frame, z_thresh=3.0)


def test_complete_eda_plot_catalog_preserves_section_order_and_cache_ids() -> None:
    frame = pd.DataFrame({"source_id": ["a", "a", "b"]})
    catalog = presentation_widgets.make_eda_plot_specs(frame, "PD12M", seed=17)

    assert tuple(catalog) == tuple(key for key, _ in presentation_widgets.EDA_SECTION_TITLES)
    assert tuple(len(catalog[key]) for key, _ in presentation_widgets.EDA_SECTION_TITLES) == (3, 2, 6, 5)
    specs = tuple(spec for section in catalog.values() for spec in section)
    assert [spec.cache_name for spec in specs] == [f"pd12m_plot_{index:03d}" for index in range(16)]
    assert [spec.cache_name for spec in specs if spec.prebuild_cache] == [
        "pd12m_plot_001",
        "pd12m_plot_002",
        "pd12m_plot_005",
        "pd12m_plot_006",
        "pd12m_plot_007",
        "pd12m_plot_008",
        "pd12m_plot_009",
        "pd12m_plot_011",
        "pd12m_plot_012",
        "pd12m_plot_013",
        "pd12m_plot_014",
    ]
    assert all(spec.cache_seed == 17 for spec in specs)
    assert all(spec.source_groups == 2 for spec in specs)
    assert all(spec.image_count == 3 for spec in specs)
    assert catalog["stats"][2].cache_parameters == {"color_space": "YCbCr", "dataset_name": "PD12M"}
    assert catalog["stats"][-1].cache_parameters == {"dataset_name": "PD12M", "z_thresh": 3.0}


def test_runtime_cache_resolves_only_generated_figures(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    assert presentation_cache.resolve_cached_figure_path("ALASKA2", "view", paths=paths) is None

    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    destination = presentation_cache.save_figure(figure, "ALASKA2", "view", paths=paths)
    plt.close(figure)

    assert destination == presentation_cache.figure_path("ALASKA2", "view", paths=paths)
    assert destination.is_file()
    assert presentation_cache.resolve_cached_figure_path("ALASKA2", "view", paths=paths) == destination


def test_dataset_cache_namespaces_never_fallback_and_ignore_placeholder(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    alaska2_cache = paths.dataset_cache("alaska2")
    alaska2_cache.mkdir(parents=True)
    (alaska2_cache / ".gitkeep").touch()

    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])
    pd12m = presentation_cache.save_figure(figure, "pd12m", "same-name", paths=paths)
    plt.close(figure)

    assert presentation_cache.figure_path("alaska2", "same-name", paths=paths) == alaska2_cache / "same-name.png"
    assert pd12m == paths.dataset_cache("pd12m") / "same-name.png"
    assert presentation_cache.resolve_cached_figure_path("pd12m", "same-name", paths=paths) == pd12m
    assert presentation_cache.resolve_cached_figure_path("alaska2", "same-name", paths=paths) is None
    assert presentation_cache.resolve_cached_figure_path("alaska2", ".gitkeep", paths=paths) is None


def test_all_versioned_pd12m_plots_match_manifest_and_resolve_without_rendering() -> None:
    paths = ProjectPaths(Path(__file__).parents[1])
    frame = pd.DataFrame({"source_id": [f"{index:03d}" for index in range(500) for _ in range(4)]})
    catalog = presentation_widgets.make_eda_plot_specs(frame, "PD12M", seed=42)
    specs = tuple(spec for section in catalog.values() for spec in section if spec.prebuild_cache)
    manifest_path = paths.dataset_cache("pd12m") / presentation_cache.CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["plots"]) == {f"{spec.cache_name}.png" for spec in specs}
    for spec in specs:
        expected = paths.dataset_cache("pd12m") / f"{spec.cache_name}.png"
        assert (
            presentation_cache.resolve_cached_figure_path(
                "pd12m",
                spec.cache_name,
                paths=paths,
                renderer=spec.cache_renderer,
                parameters=spec.cache_parameters,
                seed=spec.cache_seed,
                source_groups=spec.source_groups,
                image_count=spec.image_count,
            )
            == expected
        )


def test_cache_rejects_changed_parameters_content_and_version(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    cache_arguments: dict[str, Any] = {
        "paths": paths,
        "renderer": "fixture.renderer:abc123",
        "parameters": {"channel": "Y"},
        "seed": 42,
        "source_groups": 500,
        "image_count": 2_000,
    }
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [0.0, 1.0])
    destination = presentation_cache.save_figure(
        figure,
        "pd12m",
        "versioned-view",
        **cache_arguments,
    )
    plt.close(figure)

    assert (
        presentation_cache.resolve_cached_figure_path(
            "pd12m",
            "versioned-view",
            **cache_arguments,
        )
        == destination
    )
    changed = {**cache_arguments, "parameters": {"channel": "Cb"}}
    assert presentation_cache.resolve_cached_figure_path("pd12m", "versioned-view", **changed) is None

    destination.write_bytes(destination.read_bytes() + b"corrupt")
    assert presentation_cache.resolve_cached_figure_path("pd12m", "versioned-view", **cache_arguments) is None

    figure, axis = plt.subplots()
    axis.plot([1.0, 0.0], [0.0, 1.0])
    presentation_cache.save_figure(figure, "pd12m", "versioned-view", **cache_arguments)
    plt.close(figure)
    manifest_path = destination.parent / presentation_cache.CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_version"] = -1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert presentation_cache.resolve_cached_figure_path("pd12m", "versioned-view", **cache_arguments) is None


@pytest.mark.parametrize("dataset_name", ["pd12m", "alaska2"])
def test_cached_eda_section_generates_then_reuses_runtime_figure(
    dataset_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    rendered: list[Figure] = []
    displayed: list[object] = []

    def render() -> Figure:
        figure, axis = plt.subplots()
        axis.plot([0.0, 1.0], [1.0, 0.0])
        rendered.append(figure)
        return figure

    monkeypatch.setattr(presentation_cache, "default_paths", lambda: paths)
    monkeypatch.setattr(presentation_widgets, "display", displayed.append)
    monkeypatch.setattr(presentation_widgets, "clear_output", lambda *, wait: None)

    section: Any = presentation_widgets.make_dropdown_section(
        [presentation_widgets.PlotSpec("view", render, "view")],
        dataset_name,
        use_cache=True,
    )
    section.activate()
    cached = presentation_cache.figure_path(dataset_name, "view", paths=paths)
    assert cached.is_file()
    assert len(rendered) == 1
    assert len(displayed) == 1
    assert not plt.get_fignums()

    section.deactivate()
    section.activate()
    assert len(rendered) == 1
    assert len(displayed) == 2
    section.close()
    assert not plt.get_fignums()


def test_eda_application_owns_one_display_and_one_render_per_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
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

    def plot(name: str):
        def render() -> Figure:
            rendered.append(name)
            figure, axis = plt.subplots()
            axis.plot([0.0, 1.0], [0.0, 1.0])
            return figure

        return render

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
        callback,
        remove: bool = False,
    ) -> None:
        if maintained(callback):
            target = removals if remove else registrations
            target[callback_key(callback)] += 1
        original_on_click(button, callback, remove=remove)

    monkeypatch.setattr(presentation_widgets, "display", fake_display)
    monkeypatch.setattr(presentation_widgets, "clear_output", record_clear)
    monkeypatch.setattr(ipywidgets.Output, "__enter__", record_enter)
    monkeypatch.setattr(ipywidgets.Output, "__exit__", record_exit)
    monkeypatch.setattr(ipywidgets.Widget, "observe", record_observe)
    monkeypatch.setattr(ipywidgets.Widget, "unobserve", record_unobserve)
    monkeypatch.setattr(ipywidgets.Button, "on_click", record_on_click)

    first_section = presentation_widgets.make_dropdown_section(
        [
            presentation_widgets.PlotSpec("first-a", plot("first-a"), "first-a"),
            presentation_widgets.PlotSpec("first-b", plot("first-b"), "first-b"),
        ],
        "fixture",
        description="first-view",
    )
    second_section = presentation_widgets.make_dropdown_section(
        [
            presentation_widgets.PlotSpec("second-a", plot("second-a"), "second-a"),
            presentation_widgets.PlotSpec("second-b", plot("second-b"), "second-b"),
        ],
        "fixture",
        description="second-view",
    )
    open_text = "open-eda-fixture"
    close_text = "close-eda-fixture"
    root = presentation_widgets.make_lazy_tabs(
        [first_section, second_section],
        open_button_text=open_text,
        close_button_text=close_text,
    )
    descendants = tuple(_walk_widgets(root))
    open_button = next(
        item for item in descendants if isinstance(item, ipywidgets.Button) and item.description == open_text
    )
    close_button = next(
        item for item in descendants if isinstance(item, ipywidgets.Button) and item.description == close_text
    )
    first_dropdown = next(
        item for item in descendants if isinstance(item, ipywidgets.Dropdown) and item.description == "first-view"
    )
    tabs = next(item for item in descendants if isinstance(item, ipywidgets.Tab))
    second_dropdown = next(
        item for item in descendants if isinstance(item, ipywidgets.Dropdown) and item.description == "second-view"
    )

    _assert_unique_widget_models(root)
    output_by_tab: list[ipywidgets.Output] = []
    for tab_child in tabs.children:
        tab_outputs = [item for item in _walk_widgets(tab_child) if isinstance(item, ipywidgets.Output)]
        assert len(tab_outputs) == 1
        output_by_tab.append(tab_outputs[0])
    assert len({output.model_id for output in output_by_tab}) == len(output_by_tab)
    original_parents = _output_parents(root)
    child_mutations = _watch_child_mutations(root)

    def assert_ownership() -> None:
        _assert_unique_widget_models(root)
        assert _output_parents(root) == original_parents
        assert not child_mutations

    def assert_result(action, expected_render: str, output: ipywidgets.Output) -> None:
        render_count = len(rendered)
        clear_count = len(clears)
        display_count = len(displayed)
        action()
        assert rendered[render_count:] == [expected_render]
        assert clears[clear_count:] == [(output.model_id, True)]
        publications = displayed[display_count:]
        assert len(publications) == 1
        assert publications[0][0] == output.model_id
        assert isinstance(publications[0][1], Figure)
        assert not plt.get_fignums()
        assert_ownership()

    assert not displayed
    assert not rendered
    presentation_widgets.display(root)
    assert displayed == [(None, root)]

    assert_result(open_button.click, "first-a", output_by_tab[0])
    counts = (len(rendered), len(clears), len(displayed))
    open_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts
    assert_ownership()

    assert_result(lambda: setattr(first_dropdown, "value", 1), "first-b", output_by_tab[0])
    assert_result(lambda: setattr(tabs, "selected_index", 1), "second-a", output_by_tab[1])

    counts = (len(rendered), len(clears), len(displayed))
    first_dropdown.value = 0
    assert (len(rendered), len(clears), len(displayed)) == counts
    assert_ownership()

    assert_result(lambda: setattr(second_dropdown, "value", 1), "second-b", output_by_tab[1])
    assert_result(lambda: setattr(tabs, "selected_index", 0), "first-a", output_by_tab[0])

    render_count = len(rendered)
    display_count = len(displayed)
    clear_count = len(clears)
    close_button.click()
    assert clears[clear_count:] == [(output_by_tab[0].model_id, True)]
    assert len(rendered) == render_count
    assert len(displayed) == display_count
    assert_ownership()

    first_dropdown.value = 1
    assert len(rendered) == render_count
    assert_result(open_button.click, "first-b", output_by_tab[0])
    counts = (len(rendered), len(clears), len(displayed))
    open_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts

    assert active_figures_at_display == [1] * len(rendered)
    assert not plt.get_fignums()

    counts = (len(rendered), len(clears), len(displayed))
    root.close()
    assert registrations == removals
    assert all(item.comm is None for item in descendants)
    first_dropdown.value = 0
    tabs.selected_index = 1
    open_button.click()
    close_button.click()
    assert (len(rendered), len(clears), len(displayed)) == counts
    assert not plt.get_fignums()
    assert not child_mutations


def test_image_grid_controller_uses_pure_figures_and_closes_them() -> None:
    frame = pd.DataFrame(
        {
            "path": [f"/missing/{label}/{index}.jpg" for label in ("Cover", "JMiPOD") for index in range(4)],
            "label_name": [label for label in ("Cover", "JMiPOD") for _ in range(4)],
        }
    )

    figure = eda_examples.plot_image_grid(frame, rows=2, cols=1)
    assert isinstance(figure, Figure)
    plt.close(figure)

    widget = eda_examples.make_image_grid_widget(frame, rows=2, cols=1)
    descendants = tuple(_walk_widgets(widget))
    _assert_unique_widget_models(widget)
    assert not any(isinstance(item, ipywidgets.Output) for item in descendants)
    child_mutations = _watch_child_mutations(widget)
    class_selector = next(item for item in descendants if isinstance(item, ipywidgets.Dropdown))
    page_selector = next(item for item in descendants if isinstance(item, ipywidgets.BoundedIntText))

    image = next(item for item in descendants if isinstance(item, ipywidgets.Image))
    updates: list[bytes] = []
    image.observe(lambda change: updates.append(change["new"]), names="value")
    initial_value = image.value
    assert initial_value
    assert not plt.get_fignums()
    page_selector.value = 1
    assert len(updates) == 1
    assert image.value != initial_value
    page_value = image.value
    other_class = next(value for value in class_selector.options if value != class_selector.value)
    class_selector.value = other_class
    assert len(updates) == 2
    assert image.value != page_value
    assert not plt.get_fignums()
    assert not child_mutations
    final_value = image.value
    widget.close()
    assert all(item.comm is None for item in descendants)
    page_selector.value = 0
    assert image.value == final_value
    assert len(updates) == 2
    assert not plt.get_fignums()


def test_controller_initial_render_failure_disposes_owned_widgets(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"path": ["/missing/Cover/0.jpg"], "label_name": ["Cover"]})
    disposed_trees: list[bool] = []
    original_close = eda_controls.FigureController.close

    def record_close(controller: eda_controls.FigureController) -> None:
        if getattr(controller, "_disposed", False):
            original_close(controller)
            return
        descendants = tuple(_walk_widgets(controller))
        original_close(controller)
        disposed_trees.append(all(item.comm is None for item in descendants))

    def fail_render(*args: Any, **kwargs: Any) -> Figure:
        del args, kwargs
        raise RuntimeError("forced initial render failure")

    monkeypatch.setattr(eda_controls.FigureController, "close", record_close)
    monkeypatch.setattr(eda_examples, "_plot_image_grid_page", fail_render)

    with pytest.raises(RuntimeError, match="forced initial render failure"):
        eda_examples.make_image_grid_widget(frame)

    assert disposed_trees == [True]
    assert not plt.get_fignums()


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
