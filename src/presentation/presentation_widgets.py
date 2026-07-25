"""
===============================================================================
presentation_widgets.py
===============================================================================
Notebook plot controls and complete evaluation presentation.

Responsibilities:
  - Describe exploratory plots with typed PlotSpec values.
  - Reuse validated dataset-specific EDA PNGs or compute figures lazily.
  - Compose dropdown sections into open/close tab panels.
  - Present every verified model-evaluation view in one artifact-driven
    widget.

Design principles:
  - The notebook supplies EDA data while verified result widgets load only
    maintained repository-visible evidence.
  - One shared model selector and tab layout replace per-model result panels.

Boundaries:
  - No scientific calculation, training, inference, or test evaluation is
    implemented here.

Notes:
  - German button text can be supplied by the protected academic notebook.
===============================================================================
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import ipywidgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, clear_output, display
from matplotlib.figure import Figure

from ..config.config_paths import ProjectPaths
from ..eda import eda_channels, eda_dct, eda_examples, eda_overview
from ..evaluation import evaluation_plots
from .presentation_cache import resolve_cached_figure_path, sanitize_name, save_figure

__all__ = [
    "EDA_SECTION_TITLES",
    "EVALUATION_VIEW_TITLES",
    "PlotSpec",
    "make_dropdown_section",
    "make_eda_plot_specs",
    "make_evaluation_widget",
    "make_lazy_tabs",
    "make_plot_factory",
]

EDA_SECTION_TITLES: Final[tuple[tuple[str, str], ...]] = (
    ("overview", "1. Übersicht"),
    ("examples", "2. Bildbeispiele"),
    ("stats", "3. Farbkanalstatistik"),
    ("dct", "4. DCT-Analyse"),
)
EVALUATION_VIEW_TITLES: Final[tuple[tuple[str, str], ...]] = (
    ("history", "Lernverlauf"),
    ("confusion", "Konfusionsmatrix"),
    ("roc", "ROC-Kurve"),
    ("scores", "Score-Verteilung"),
    ("metrics", "Kennzahlen"),
    ("comparison", "Modellvergleich"),
)


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """Describe one lazily rendered and optionally cached notebook plot.

    Parameters
    ----------
    title
        Human-readable dropdown title.
    render
        Zero-argument callable producing the display object.
    cache_name
        Stable sanitized cache identifier.
    """

    title: str
    render: Callable[[], Any]
    cache_name: str
    cache_renderer: str = ""
    cache_parameters: dict[str, object] | None = None
    cache_seed: int = 42
    source_groups: int | None = None
    image_count: int | None = None
    prebuild_cache: bool = True


def _renderer_cache_identity(function: Callable[..., Any]) -> str:
    module_name = getattr(function, "__module__", "<unknown>")
    qualified_name = getattr(function, "__qualname__", getattr(function, "__name__", "<callable>"))
    source_path = inspect.getsourcefile(function)
    if source_path is not None and Path(source_path).is_file():
        source = Path(source_path).read_bytes()
    else:
        try:
            source = inspect.getsource(function).encode()
        except (OSError, TypeError):
            source = f"{module_name}.{qualified_name}".encode()
    digest = hashlib.sha256(source).hexdigest()[:16]
    return f"{module_name}.{qualified_name}:{digest}"


def make_plot_factory(
    dataframe: pd.DataFrame,
    dataset_name: str,
    *,
    seed: int = 42,
) -> Callable[..., PlotSpec]:
    """Bind a concise PlotSpec factory to one dataframe and dataset name.

    Parameters
    ----------
    dataframe
        Dataframe passed as the first plot-function argument.
    dataset_name
        Dataset name passed when the plot callable accepts it.

    Returns
    -------
    Callable[..., PlotSpec]
        Factory assigning stable cache names and lazy render callables.
    """
    source_groups = int(dataframe["source_id"].nunique()) if "source_id" in dataframe else None
    image_count = len(dataframe)
    counter = 0

    def create(
        title: str,
        function: Callable[..., Any],
        cache_name: str | None = None,
        *,
        prebuild_cache: bool = True,
        **kwargs: Any,
    ) -> PlotSpec:
        nonlocal counter
        if "dataset_name" in inspect.signature(function).parameters:
            kwargs.setdefault("dataset_name", dataset_name)
        name = cache_name or f"{sanitize_name(dataset_name)}_plot_{counter:03d}"
        counter += 1
        return PlotSpec(
            title=title,
            render=lambda: function(dataframe, **kwargs),
            cache_renderer=_renderer_cache_identity(function),
            cache_parameters=dict(kwargs),
            cache_seed=seed,
            source_groups=source_groups,
            image_count=image_count,
            cache_name=sanitize_name(name),
            prebuild_cache=prebuild_cache,
        )

    return create


def make_eda_plot_specs(
    dataframe: pd.DataFrame,
    dataset_name: str,
    *,
    seed: int,
) -> dict[str, tuple[PlotSpec, ...]]:
    """Build the complete ordered EDA catalog for notebook and cache generation."""
    plot = make_plot_factory(dataframe, dataset_name, seed=seed)
    return {
        "overview": (
            plot("1-1. Struktur & Statistik", eda_overview.show_dataset_overview, prebuild_cache=False),
            plot("1-2. Klassenverteilung", eda_overview.plot_class_distribution),
            plot("1-3. JPEG-Qualitätsverteilung", eda_overview.plot_jpeg_quality_distribution),
        ),
        "examples": (
            plot("2-1. Bildraster pro Klasse", eda_examples.make_image_grid_widget, prebuild_cache=False),
            plot(
                "2-2. Vergleich Cover vs. Stego",
                eda_examples.make_cover_stego_comparison_widget,
                prebuild_cache=False,
            ),
        ),
        "stats": (
            plot("3-1. Pixelwert-Histogramme (YCbCr)", eda_channels.plot_pixel_histograms),
            plot("3-2. Bild-Mittelwertverteilung", eda_channels.plot_image_mean_distribution),
            plot(
                "3-3. KDE & Boxplot - YCbCr",
                eda_channels.plot_kde_and_boxplot,
                color_space="YCbCr",
            ),
            plot("3-4. Korrelation YCbCr-Kanäle", eda_channels.plot_channel_correlation),
            plot(
                "3-5. KDE & Boxplot - RGB",
                eda_channels.plot_kde_and_boxplot,
                color_space="RGB",
            ),
            plot(
                "3-6. Ausreisser (Z-Score)",
                eda_channels.show_outliers_by_channel,
                prebuild_cache=False,
                z_thresh=3.0,
            ),
        ),
        "dct": (
            plot("4-1. DCT-Quantisierung (Cover + Δ)", eda_dct.plot_dct_avg_and_delta),
            plot("4-2. Anzahl DCT-Flips pro Bild", eda_dct.plot_flip_counts),
            plot(
                "4-3. Verteilung und Saldo der DCT-Flips im Y-Kanal (AC, ±1)",
                eda_dct.plot_flip_direction_overview,
            ),
            plot("4-4. Flip-Verteilung nach DCT-Index", eda_dct.plot_flip_position_heatmap),
            plot(
                "4-5. Flip-Masken (Heat-map & Overlay)",
                eda_dct.make_cover_stego_flipmask_widget,
                prebuild_cache=False,
            ),
        ),
    }


def _display_result(result: Any) -> None:
    if isinstance(result, Figure):
        try:
            display(result)
        finally:
            plt.close(result)
    elif isinstance(result, str):
        print(result)
    elif result is not None:
        display(result)


def _clear_result(output: ipywidgets.Output) -> None:
    with output:
        clear_output(wait=True)


def _replace_result(output: ipywidgets.Output, result: Any) -> None:
    with output:
        clear_output(wait=True)
        _display_result(result)


def _dispose_result(result: Any) -> None:
    if isinstance(result, ipywidgets.Widget):
        result.close()


def _ensure_unique_widget_models(widgets: Sequence[ipywidgets.Widget]) -> None:
    locations: dict[str, list[str]] = {}

    def visit(widget: ipywidgets.Widget, path: str, ancestors: set[str]) -> None:
        locations.setdefault(widget.model_id, []).append(path)
        if widget.model_id in ancestors:
            return
        for index, child in enumerate(getattr(widget, "children", ())):
            visit(child, f"{path}.children[{index}]", {*ancestors, widget.model_id})

    for index, widget in enumerate(widgets):
        visit(widget, f"sections[{index}]", set())
    duplicates = {model_id: paths for model_id, paths in locations.items() if len(paths) > 1}
    if duplicates:
        model_id, paths = next(iter(duplicates.items()))
        raise ValueError(f"Widget model {model_id} occurs at multiple child-tree locations: {paths}.")


class _PlotSection(ipywidgets.VBox):
    """Own one deferred EDA selection and its replaceable output."""

    def __init__(
        self,
        plots: Sequence[PlotSpec],
        dataset_name: str,
        *,
        description: str,
        use_cache: bool,
    ) -> None:
        self._plots = tuple(plots)
        self._dataset_name = dataset_name
        self._use_cache = use_cache
        self._last_index: int | None = None
        self._current_result: Any = None
        self._active = False
        self._disposed = False
        self.dropdown = ipywidgets.Dropdown(
            options=[(plot.title, index) for index, plot in enumerate(plots)],
            description=description,
            style={"description_width": "initial"},
        )
        self.output = ipywidgets.Output()
        super().__init__([self.dropdown, self.output])
        self.dropdown.observe(self._selection_changed, names="value")

    def _selection_changed(self, _: dict[str, Any]) -> None:
        self.render_selected()

    def render_selected(self) -> None:
        """Replace the active output with the currently selected EDA view."""
        if not self._active or self._disposed:
            return
        selected = self.dropdown.value
        if selected is None:
            return
        index = int(selected)
        if self._last_index == index:
            return
        spec = self._plots[index]
        cached = (
            resolve_cached_figure_path(
                self._dataset_name,
                spec.cache_name,
                renderer=spec.cache_renderer,
                parameters=spec.cache_parameters,
                seed=spec.cache_seed,
                source_groups=spec.source_groups,
                image_count=spec.image_count,
            )
            if self._use_cache
            else None
        )
        if cached is not None:
            result: Any = Image(filename=str(cached))
        else:
            result = spec.render()
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, Figure) and self._use_cache:
                figure = result
                try:
                    destination = save_figure(
                        figure,
                        self._dataset_name,
                        spec.cache_name,
                        renderer=spec.cache_renderer,
                        parameters=spec.cache_parameters,
                        seed=spec.cache_seed,
                        source_groups=spec.source_groups,
                        image_count=spec.image_count,
                    )
                    result = Image(filename=str(destination))
                finally:
                    plt.close(figure)
        _dispose_result(self._current_result)
        _replace_result(self.output, result)
        self._current_result = result
        self._last_index = index

    def activate(self) -> None:
        """Enable interaction and render exactly one current view."""
        if self._active or self._disposed:
            return
        self._active = True
        self.render_selected()

    def clear(self) -> None:
        """Clear this section's permanently owned output once."""
        if self._disposed:
            return
        _clear_result(self.output)

    def deactivate(self) -> None:
        """Detach visible content and reset the next activation."""
        if not self._active and self._last_index is None:
            return
        self._active = False
        _dispose_result(self._current_result)
        self._current_result = None
        self._last_index = None

    def close(self) -> None:
        """Dispose the section and unregister its stable observer."""
        if self._disposed:
            return
        self._disposed = True
        self._active = False
        _dispose_result(self._current_result)
        self._current_result = None
        self.dropdown.unobserve(self._selection_changed, names="value")
        self.dropdown.close()
        self.output.close()
        super().close()


def make_dropdown_section(
    plots: Sequence[PlotSpec],
    dataset_name: str,
    *,
    description: str = "Plot:",
    use_cache: bool = False,
) -> ipywidgets.VBox:
    """Build a lazy dropdown for a non-empty sequence of plot specifications.

    Parameters
    ----------
    plots
        Ordered plot specifications.
    dataset_name
        Cache namespace for the section.
    description
        Dropdown label.
    use_cache
        Whether to reuse and write validated dataset-specific PNG cache entries.

    Returns
    -------
    ipywidgets.VBox
        Dropdown and lazy output widget.

    Raises
    ------
    ValueError
        If no plot specifications are provided.
    """
    if not plots:
        raise ValueError("At least one plot is required.")
    return _PlotSection(
        plots,
        dataset_name,
        description=description,
        use_cache=use_cache,
    )


class _LazyTabs(ipywidgets.VBox):
    """Activate EDA tabs whose outputs each have one permanent owner."""

    def __init__(
        self,
        sections: Sequence[ipywidgets.Widget],
        *,
        tab_titles: Sequence[str] | None,
        open_button_text: str,
        close_button_text: str,
    ) -> None:
        self._sections = tuple(sections)
        _ensure_unique_widget_models(self._sections)
        self._is_open = False
        self._disposed = False

        self.open_button = ipywidgets.Button(description=open_button_text, button_style="primary")
        self.close_button = ipywidgets.Button(description=close_button_text, button_style="danger")
        self.tabs = ipywidgets.Tab(children=self._sections)
        titles = tab_titles or [f"Tab {index + 1}" for index in range(len(self._sections))]
        for index, title in enumerate(titles):
            self.tabs.set_title(index, title)
        self.panel = ipywidgets.VBox([self.close_button, self.tabs])
        self.panel.layout.display = "none"
        super().__init__([self.open_button, self.panel])

        self.tabs.observe(self._change_tab, names="selected_index")
        self.open_button.on_click(self._show_panel)
        self.close_button.on_click(self._show_button)

    def _plot_section(self, index: object) -> _PlotSection | None:
        if not isinstance(index, int) or not 0 <= index < len(self._sections):
            return None
        section = self._sections[index]
        return section if isinstance(section, _PlotSection) else None

    def _show_panel(self, _: object | None = None) -> None:
        if self._is_open or self._disposed:
            return
        self._is_open = True
        self.open_button.layout.display = "none"
        self.panel.layout.display = ""
        section = self._plot_section(self.tabs.selected_index)
        if section is not None:
            section.activate()

    def _show_button(self, _: object | None = None) -> None:
        if not self._is_open or self._disposed:
            return
        section = self._plot_section(self.tabs.selected_index)
        if section is not None:
            section.clear()
            section.deactivate()
        self.panel.layout.display = "none"
        self.open_button.layout.display = ""
        self._is_open = False

    def _change_tab(self, change: dict[str, Any]) -> None:
        if not self._is_open or self._disposed:
            return
        old_section = self._plot_section(change.get("old"))
        if old_section is not None:
            old_section.deactivate()
        new_section = self._plot_section(change.get("new"))
        if new_section is not None:
            new_section.activate()

    def close(self) -> None:
        """Dispose the application and make every registered handler inert."""
        if self._disposed:
            return
        self._disposed = True
        self._is_open = False
        self.tabs.unobserve(self._change_tab, names="selected_index")
        self.open_button.on_click(self._show_panel, remove=True)
        self.close_button.on_click(self._show_button, remove=True)
        for section in self._sections:
            section.close()
        self.tabs.close()
        self.close_button.close()
        self.panel.close()
        self.open_button.close()
        super().close()


def make_lazy_tabs(
    sections: Sequence[ipywidgets.Widget],
    *,
    tab_titles: Sequence[str] | None = None,
    open_button_text: str = "Open",
    close_button_text: str = "Close",
) -> ipywidgets.VBox:
    """Build a lazily opened tab panel from notebook sections.

    Parameters
    ----------
    sections
        Non-empty sequence of widget sections.
    tab_titles
        Optional title matching each section.
    open_button_text
        Text shown on the panel-open button.
    close_button_text
        Text shown on the panel-close button.

    Returns
    -------
    ipywidgets.VBox
        Stable root containing one open button and one reusable tab panel.

    Raises
    ------
    ValueError
        If sections are empty or title counts do not match.
    """
    if not sections:
        raise ValueError("At least one section is required.")
    if tab_titles is not None and len(tab_titles) != len(sections):
        raise ValueError("tab_titles length must match sections length.")
    return _LazyTabs(
        sections,
        tab_titles=tab_titles,
        open_button_text=open_button_text,
        close_button_text=close_button_text,
    )


def _evaluation_view(
    results: evaluation_plots.EvaluationResults,
    model_key: str,
    view_key: str,
) -> Figure | pd.DataFrame:
    result = results.model(model_key)
    if view_key == "history":
        return evaluation_plots.plot_history(
            result.history.dataframe,
            title=f"{result.display_name} - chronological training history",
            selected_epoch=result.selected_epoch,
            selected_stage=result.selected_stage,
            stage_order=(evaluation_plots.EFFICIENTNET_STAGE_ORDER if result.selected_stage is not None else None),
        )
    if view_key == "confusion":
        return evaluation_plots.plot_confusion_matrix(result.confusion_matrix)
    if view_key == "roc":
        return evaluation_plots.plot_roc_curves(
            [
                {
                    "fpr": result.roc_data["fpr"].to_numpy(),
                    "tpr": result.roc_data["tpr"].to_numpy(),
                    "wauc": float(result.metrics["test_weighted_auc"]),
                    "label": result.display_name,
                }
            ]
        )
    if view_key == "scores":
        return evaluation_plots.plot_score_distribution(result.score_distribution)
    if view_key == "metrics":
        return evaluation_plots.metrics_table(result)
    if view_key == "comparison":
        return evaluation_plots.comparison_table(results)
    raise KeyError(f"Unknown evaluation view: {view_key}")


class _EvaluationWidget(ipywidgets.VBox):
    """Own one output and stable callbacks for every evaluation view."""

    def __init__(
        self,
        results: evaluation_plots.EvaluationResults,
        *,
        open_button_text: str,
        close_button_text: str,
    ) -> None:
        self._results = results
        self._last_selection: tuple[str, str] | None = None
        self._is_open = False
        self._disposed = False

        self.model_selector = ipywidgets.ToggleButtons(
            options=[(result.display_name, result.key) for result in results.models],
            description="Modell:",
            style={"description_width": "initial"},
        )
        self.outputs = tuple(ipywidgets.Output() for _ in EVALUATION_VIEW_TITLES)
        self.tabs = ipywidgets.Tab(children=self.outputs)
        for index, (_, title) in enumerate(EVALUATION_VIEW_TITLES):
            self.tabs.set_title(index, title)

        self.open_button = ipywidgets.Button(description=open_button_text, button_style="primary")
        self.close_button = ipywidgets.Button(description=close_button_text, button_style="danger")
        self.panel = ipywidgets.VBox([self.close_button, self.model_selector, self.tabs])
        self.panel.layout.display = "none"
        super().__init__([self.open_button, self.panel])

        self.model_selector.observe(self._render, names="value")
        self.tabs.observe(self._render, names="selected_index")
        self.open_button.on_click(self._show_panel)
        self.close_button.on_click(self._show_button)

    def _render(self, _: dict[str, Any] | None = None) -> None:
        if not self._is_open or self._disposed:
            return
        selected_index = self.tabs.selected_index
        if selected_index is None:
            return
        view_key = EVALUATION_VIEW_TITLES[selected_index][0]
        model_key = str(self.model_selector.value)
        selection = (model_key, view_key)
        if selection == self._last_selection:
            return
        result = _evaluation_view(self._results, model_key, view_key)
        _replace_result(self.outputs[selected_index], result)
        self._last_selection = selection

    def _show_panel(self, _: object | None = None) -> None:
        if self._is_open or self._disposed:
            return
        self._is_open = True
        self.open_button.layout.display = "none"
        self.panel.layout.display = ""
        self._render()

    def _show_button(self, _: object | None = None) -> None:
        if not self._is_open or self._disposed:
            return
        selected_index = self.tabs.selected_index
        if selected_index is not None:
            _clear_result(self.outputs[selected_index])
        self.panel.layout.display = "none"
        self.open_button.layout.display = ""
        self._last_selection = None
        self._is_open = False

    def close(self) -> None:
        """Dispose the application and make every registered handler inert."""
        if self._disposed:
            return
        self._disposed = True
        self._is_open = False
        self.model_selector.unobserve(self._render, names="value")
        self.tabs.unobserve(self._render, names="selected_index")
        self.open_button.on_click(self._show_panel, remove=True)
        self.close_button.on_click(self._show_button, remove=True)
        self.model_selector.close()
        for output in self.outputs:
            output.close()
        self.tabs.close()
        self.close_button.close()
        self.panel.close()
        self.open_button.close()
        super().close()


def make_evaluation_widget(
    run_id: str,
    *,
    paths: ProjectPaths | None = None,
    open_button_text: str = "Gesamtevaluation anzeigen",
    close_button_text: str = "Schliessen",
) -> ipywidgets.VBox:
    """Build the single complete artifact-driven evaluation widget.

    Parameters
    ----------
    run_id
        Verified run-directory name under artifacts/alaska2.
    paths
        Optional project path contract for tests or alternate clones.
    open_button_text
        Label of the lazy panel-open button.
    close_button_text
        Label of the panel-close button.

    Returns
    -------
    ipywidgets.VBox
        Open/close panel with one model selector, evaluation-view tabs, and
        replacement-safe outputs.

    Raises
    ------
    FileNotFoundError
        If a required Git-visible structured result artifact is absent.
    ValueError
        If a structured result violates its maintained schema.

    Notes
    -----
    Construction and interaction only read compact public artifacts. They
    never access datasets, checkpoints, or full prediction files.
    """
    results = evaluation_plots.load_evaluation_results(run_id, paths=paths)
    return _EvaluationWidget(
        results,
        open_button_text=open_button_text,
        close_button_text=close_button_text,
    )
