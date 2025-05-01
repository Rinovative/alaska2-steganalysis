# src/eda/eda_overview.py

import matplotlib.pyplot as plt
import pandas as pd


def show_dataset_overview(df: pd.DataFrame, dataset_name: str = "", force_recompute: bool = False) -> str:
    """
    Gibt die kombinierte Textausgabe von df.info() und df.describe() als String zurück,
    z.B. für Ausgabe im toggle()-basierenden EDA-Widget.

    Returns:
        str: Formatiertes Text-Reporting für Struktur und Statistik
    """
    return df.describe()


def plot_class_distribution(df: pd.DataFrame, dataset_name: str = "ALASKA2") -> plt.Figure:
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


def plot_jpeg_quality_distribution(df: pd.DataFrame, dataset_name: str = "ALASKA2") -> plt.Figure:
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
