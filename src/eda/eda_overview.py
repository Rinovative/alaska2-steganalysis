# src/eda/eda_overview.py

import io

import matplotlib.pyplot as plt
import pandas as pd


def show_dataset_overview(df: pd.DataFrame, dataset_name: str = "") -> str:
    """
    Gibt kombinierte Textausgabe von df.info() und df.describe(),
    sowie gruppierte Statistiken nach label_name zurück.

    Returns:
        str: Formatierter Markdown-String
    """
    output = io.StringIO()

    print(f"## Datensatzübersicht: {dataset_name}", file=output)

    # Struktur
    print("\n### Struktur ausgewählter Spalten", file=output)
    selected_cols = ["path", "label_name", "jpeg_quality", "width", "height", "q_y_00", "q_y_63"]
    df[selected_cols].info(buf=output)

    # Gesamte Statistik
    print("\n### Gesamte Statistik ausgewählter Spalten", file=output)
    summary_cols = [
        "jpeg_quality",
        "width",
        "height",
        "mode",
        "q_y_00",
        "q_y_01",
        "q_y_02",
        "q_y_03",
        "q_y_10",
        "q_y_11",
        "q_y_20",
        "q_y_21",
        "q_y_30",
        "q_y_31",
        "q_y_40",
        "q_y_41",
        "q_y_50",
        "q_y_51",
        "q_y_60",
        "q_y_61",
        "q_y_62",
        "q_y_63",
    ]
    print(df[summary_cols].describe().to_string(), file=output)

    # Gruppierte Statistik nach Klasse
    if "label_name" in df.columns:
        print("\n### Gruppierte Statistik nach Klasse ausgewählter Spalten", file=output)

        # Optional: Nur bestimmte Spalten anzeigen
        summary_cols = ["width", "height", "mode", "q_y_00", "q_y_63"]
        df_summary = df[summary_cols + ["label_name"]]

        grouped = df_summary.groupby("label_name", observed=False).describe()
        print(grouped.to_string(), file=output)

    return output.getvalue()


def plot_class_distribution(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Anzahl Bilder pro Klasse (Cover, JMiPOD, JUNIWARD, UERD).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df["label_name"].value_counts().sort_index().plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
    ax.set_title(f"Klassenverteilung – {dataset_name}")
    ax.set_xlabel("Klasse")
    ax.set_ylabel("Anzahl Bilder")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    return fig


def plot_jpeg_quality_distribution(df: pd.DataFrame, dataset_name: str = "") -> plt.Figure:
    """
    Zeigt die Verteilung der JPEG-Qualitätsstufen (75, 90, 95) pro Klasse.
    Erwartet, dass `jpeg_quality` und `label_name` im DataFrame vorhanden sind.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby(["jpeg_quality", "label_name"], observed=False).size().unstack().plot(kind="bar", ax=ax)
    ax.set_title(f"JPEG-Qualitätsverteilung pro Klasse – {dataset_name}")
    ax.set_xlabel("JPEG-Qualitätsstufe")
    ax.set_ylabel("Anzahl Bilder")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Klasse")
    return fig
