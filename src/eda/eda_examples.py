from pathlib import Path

import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from PIL import Image


def plot_image_grid(df: pd.DataFrame, dataset_name: str = "", rows: int = 4, cols: int = 4) -> widgets.VBox:
    """
    Interaktives Widget zur Anzeige von Bildrastern (z.B. 4x4) pro Klasse mit Navigation.

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'
        dataset_name (str): Optionaler Titel
        rows (int): Anzahl Zeilen im Raster
        cols (int): Anzahl Spalten im Raster

    Returns:
        VBox: Widget mit Auswahl + Blättern + Ausgabe
    """
    df = df.copy()
    df["label_name"] = df["label_name"].astype("category")
    classes = df["label_name"].cat.categories.tolist()
    initial_class = classes[0]
    grouped = {cls: df[df["label_name"] == cls].reset_index(drop=True) for cls in classes}
    imgs_per_page = rows * cols
    max_steps = {cls: len(grouped[cls]) // imgs_per_page for cls in classes}

    dropdown = widgets.Dropdown(options=classes, description="Klasse:")
    idx_box = widgets.BoundedIntText(value=0, min=0, max=max_steps[initial_class], description="Seite:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btns = widgets.HBox([idx_box, btn_prev, btn_next])

    out = widgets.Output()

    def render(cls, page_idx):
        with out:
            clear_output(wait=True)
            subset = grouped[cls]
            start = page_idx * imgs_per_page
            end = start + imgs_per_page
            rows_subset = subset.iloc[start:end]

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
            fig.suptitle(f"{cls} - Seite {page_idx} - {dataset_name}", fontsize=14)

            axes = axes.flatten()
            for ax in axes:
                ax.axis("off")

            for ax, (_, row) in zip(axes, rows_subset.iterrows()):
                try:
                    img = Image.open(row["path"])
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Fehler", ha="center", va="center")

            plt.tight_layout()
            plt.show()

    def on_change_class(change):
        idx_box.max = max_steps[change["new"]]
        idx_box.value = 0
        render(change["new"], 0)

    def on_change_idx(change):
        render(dropdown.value, change["new"])

    def step(delta):
        new_val = max(0, min(idx_box.max, idx_box.value + delta))
        idx_box.value = new_val

    dropdown.observe(on_change_class, names="value")
    idx_box.observe(on_change_idx, names="value")
    btn_prev.on_click(lambda _: step(-1))
    btn_next.on_click(lambda _: step(1))

    # Initialanzeige
    render(dropdown.value, idx_box.value)

    return widgets.VBox([dropdown, btns, out])


def plot_cover_stego_comparison(df: pd.DataFrame, dataset_name: str = "") -> widgets.VBox:
    """
    Zeigt Cover + alle drei Stego-Varianten nebeneinander (1x4 Grid) für eine Bildgruppe.

    Args:
        df (pd.DataFrame): DataFrame mit 'path' und 'label_name'
        dataset_name (str): Optionaler Titel

    Returns:
        VBox: Interaktives Bedienfeld mit Navigation
    """
    df = df.copy()
    df["filename"] = df["path"].apply(lambda p: Path(p).name)
    df["base_name"] = df["filename"].str.extract(r"(\d+)\.jpg")

    LABELS = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
    complete_groups = df.groupby("base_name")["label_name"].nunique().loc[lambda g: g == 4].index.sort_values().tolist()

    idx_input = widgets.BoundedIntText(value=0, min=0, max=len(complete_groups) - 1, description="Index:")
    btn_prev = widgets.Button(description="←", layout=widgets.Layout(width="40px"))
    btn_next = widgets.Button(description="→", layout=widgets.Layout(width="40px"))
    btn_row = widgets.HBox([idx_input, btn_prev, btn_next])
    out = widgets.Output()

    def show(idx: int):
        base_id = complete_groups[idx]
        group = df[df["base_name"] == base_id]

        with out:
            clear_output(wait=True)
            if group["label_name"].nunique() < 4:
                print(f"Inkomplette Bildgruppe: {base_id}")
                return

            paths = {lbl: group[group["label_name"] == lbl]["path"].iloc[0] for lbl in LABELS}

            fig = plt.figure(figsize=(22, 6), constrained_layout=True)
            spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
            axes = [fig.add_subplot(spec[0, i]) for i in range(4)]

            fig.suptitle(f"Vergleich – ID {base_id} – {dataset_name}", fontsize=16)

            for ax, lbl in zip(axes, LABELS):
                try:
                    img = Image.open(paths[lbl]).convert("RGB")
                    ax.imshow(img)
                    ax.set_title(lbl)
                except Exception:
                    ax.text(0.5, 0.5, "Fehler", ha="center", va="center")
                ax.axis("off")

            plt.show()

    def go_relative(delta: int):
        new_idx = max(0, min(len(complete_groups) - 1, idx_input.value + delta))
        idx_input.value = new_idx

    btn_prev.on_click(lambda _: go_relative(-1))
    btn_next.on_click(lambda _: go_relative(1))
    idx_input.observe(lambda change: show(change["new"]), names="value")

    show(0)
    return widgets.VBox([btn_row, out])
