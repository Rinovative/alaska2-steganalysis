import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_dct_class_heatmaps(df: pd.DataFrame, dataset_name: str = "", force_recompute: bool = False) -> plt.Figure:
    """
    Zeigt Heatmaps der mittleren DCT-Quantisierungstabellen pro Klasse (Y-Komponente).

    Args:
        df (pd.DataFrame): DataFrame mit Spalten 'label_name' und 'q_y_00' bis 'q_y_63'.
        dataset_name (str): Titel für den Plot.
        force_recompute (bool): Nicht verwendet, für API-Kompatibilität.

    Returns:
        plt.Figure: Matplotlib-Figur mit Heatmaps.
    """
    q_cols = [f"q_y_{i:02d}" for i in range(64)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, cls in zip(axes, df["label_name"].cat.categories):
        avg_q = df[df["label_name"] == cls][q_cols].mean().values.reshape(8, 8)
        sns.heatmap(avg_q, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax, cbar=False)
        ax.set_title(cls)

    fig.suptitle(f"Mittlere DCT-Y-Quantisierung pro Klasse – {dataset_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_dct_delta_matrix(df: pd.DataFrame, dataset_name: str = "", force_recompute: bool = False) -> plt.Figure:
    """
    Zeigt die Differenz der mittleren Y-DCT-Quantisierung zwischen Cover und Stego-Bildern.

    Args:
        df (pd.DataFrame): DataFrame mit Spalten 'label_name' und 'q_y_00' bis 'q_y_63'.
        dataset_name (str): Titel für den Plot.
        force_recompute (bool): Nicht verwendet, für API-Kompatibilität.

    Returns:
        plt.Figure: Matplotlib-Figur mit Differenzheatmaps.
    """
    q_cols = [f"q_y_{i:02d}" for i in range(64)]
    cover_mean = df[df["label_name"] == "Cover"][q_cols].mean().values.reshape(8, 8)

    stego_classes = [cls for cls in df["label_name"].cat.categories if cls != "Cover"]
    fig, axes = plt.subplots(1, len(stego_classes), figsize=(5 * len(stego_classes), 4))

    if len(stego_classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, stego_classes):
        cls_mean = df[df["label_name"] == cls][q_cols].mean().values.reshape(8, 8)
        delta = cls_mean - cover_mean
        sns.heatmap(delta, cmap="coolwarm", center=0, annot=True, fmt=".1f", ax=ax, cbar=True)
        ax.set_title(f"{cls} – Δ zu Cover")
        ax.axis("off")

    fig.suptitle(f"Differenz der mittleren DCT-Y-Quantisierung – {dataset_name}", fontsize=14)
    plt.tight_layout()
    return fig


def plot_qtable_heatmaps(df: pd.DataFrame, group_col: str = "jpeg_quality", dataset_name: str = "") -> plt.Figure:
    """
    Zeigt für jede Gruppe (z. B. Klasse oder JPEG-Qualität) die mittlere Y-Quantisierungstabelle als 8x8-Heatmap.
    """
    fig, axs = plt.subplots(1, df[group_col].nunique(), figsize=(15, 4))

    for ax, (group, group_df) in zip(axs, df.groupby(group_col)):
        q_values = group_df[[f"q_y_{i:02d}" for i in range(64)]].mean().values.reshape(8, 8)
        sns.heatmap(q_values, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{group}")

    fig.suptitle(f"Mittelwerte der Y-Quantisierungstabellen – gruppiert nach {group_col} – {dataset_name}")
    plt.tight_layout()
    return fig


def plot_dct_class_variance_heatmaps(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Varianz der 64 Y-Quantisierungseinträge (q_y_00 bis q_y_63) für jede Klasse
    als 8×8-Heatmap pro Klasse.

    Args:
        df (pd.DataFrame): DataFrame mit q_y_00 bis q_y_63 und 'label_name'.
        dataset_name (str): Optionaler Name für den Plot-Titel.

    Returns:
        plt.Figure: Matplotlib-Figur mit den Varianz-Heatmaps.
    """
    qtable_cols = [f"q_y_{i:02d}" for i in range(64)]
    classes = df["label_name"].cat.categories if isinstance(df["label_name"].dtype, pd.CategoricalDtype) else sorted(df["label_name"].unique())

    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 5))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        subset = df[df["label_name"] == cls]
        var_vector = subset[qtable_cols].var().values.reshape(8, 8)
        sns.heatmap(var_vector, ax=ax, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
        ax.set_title(f"{cls}")

    fig.suptitle(f"Varianz der DCT-Y-Quantisierung pro Klasse – {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    return fig
