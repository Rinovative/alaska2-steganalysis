"""Disposable controls for pure EDA figure renderers."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from typing import Any

import ipywidgets
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

__all__: list[str] = []


class FigureController(ipywidgets.VBox):
    """Render figures into one stable image model without display calls."""

    def __init__(self, controls: Sequence[ipywidgets.Widget]) -> None:
        _ensure_unique_controls(controls)
        self.image = ipywidgets.Image(format="png")
        self._observers: list[tuple[ipywidgets.Widget, Callable[[dict[str, Any]], None], str]] = []
        self._clicks: list[tuple[ipywidgets.Button, Callable[[ipywidgets.Button], None]]] = []
        self._disposed = False
        super().__init__([*controls, self.image])

    def set_figure(self, figure: Figure) -> None:
        """Replace the image bytes and close the source figure."""
        try:
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight")
            self.image.value = buffer.getvalue()
        finally:
            plt.close(figure)

    def register_observer(
        self,
        widget: ipywidgets.Widget,
        handler: Callable[[dict[str, Any]], None],
        *,
        names: str,
    ) -> None:
        """Register one observer that is removed during disposal."""
        widget.observe(handler, names=names)
        self._observers.append((widget, handler, names))

    def on_click(
        self,
        button: ipywidgets.Button,
        handler: Callable[[ipywidgets.Button], None],
    ) -> None:
        """Register one click handler that is removed during disposal."""
        button.on_click(handler)
        self._clicks.append((button, handler))

    def close(self) -> None:
        """Unregister handlers and dispose every child widget exactly once."""
        if self._disposed:
            return
        self._disposed = True
        for widget, handler, names in self._observers:
            widget.unobserve(handler, names=names)
        for button, handler in self._clicks:
            button.on_click(handler, remove=True)
        self._observers.clear()
        self._clicks.clear()
        for child in _owned_widgets(self):
            child.close()
        super().close()


def _ensure_unique_controls(controls: Sequence[ipywidgets.Widget]) -> None:
    seen: set[int] = set()

    def visit(widget: ipywidgets.Widget) -> None:
        identity = id(widget)
        if identity in seen:
            raise ValueError("A widget instance cannot occur at multiple control-tree locations.")
        seen.add(identity)
        for child in getattr(widget, "children", ()):
            visit(child)

    for control in controls:
        visit(control)


def _owned_widgets(root: ipywidgets.Widget) -> tuple[ipywidgets.Widget, ...]:
    found: list[ipywidgets.Widget] = []
    seen: set[int] = {id(root)}

    def visit(widget: ipywidgets.Widget) -> None:
        related: list[ipywidgets.Widget] = list(getattr(widget, "children", ()))
        for trait_name in ("layout", "style"):
            trait_widget = getattr(widget, trait_name, None)
            if isinstance(trait_widget, ipywidgets.Widget):
                related.append(trait_widget)
        for child in related:
            identity = id(child)
            if identity in seen:
                continue
            seen.add(identity)
            found.append(child)
            visit(child)

    visit(root)
    return tuple(reversed(found))
