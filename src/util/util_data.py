from __future__ import annotations

# Standard library
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Dict, Optional

# Third-party libraries
import clip
import conseal as cl
import faiss
import jpeglib as jpeglib
import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def download_synthetic_PD12M(
    alaska2_path: str = "data/raw/alaska2-image-steganalysis/Cover",
    cover_path: str = "data/raw/PD12M/Cover",
    base_path: str = "data/raw/PD12M/",
    force_download: bool = False,
) -> str:
    """
    Lädt den vorbereiteten PD12M-Ersatzdatensatz (DCT-basiert, synthetisch)
    automatisch von Hugging Face herunter, falls ALASKA2 nicht verfügbar ist.

    Args:
        alaska2_path (str): Pfad zum ALASKA2-Cover-Ordner.
        cover_path (str): Pfad zum PD12M-Cover-Ordner.
        base_path (str): Zielpfad für den PD12M-Datensatz.
        force_download (bool): Wenn True, wird PD12M heruntergeladen, auch wenn ALASKA2 vorhanden ist.

    Returns:
        str: Statusmeldung.
    """
    alaska_dir = Path(alaska2_path)
    cover_dir = Path(cover_path)
    base_dir = Path(base_path)
    zip_filename = "pd12m_synthetic_stegano.zip"
    dataset_url = "https://huggingface.co/datasets/Rinovative/pd12m_dct_based_synthetic_stegano/resolve/main/pd12m_dct_based_synthetic_stegano.zip"

    # 1. Abbruchbedingung, wenn ALASKA2 vorhanden und kein force
    if alaska_dir.exists() and any(alaska_dir.glob("*.jpg")) and not force_download:
        return "✅ ALASKA2 vorhanden – kein Download nötig."

    # 2. Abbruchbedingung, wenn PD12M-Cover bereits Bilder enthält und kein force
    if cover_dir.exists() and any(cover_dir.glob("*.jpg")) and not force_download:
        return "✅ PD12M-Cover bereits befüllt – kein Download nötig."

    # 3. Bei force_download den Zielordner löschen, falls er existiert
    if force_download and base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 4. ZIP herunterladen (nur wenn nicht bereits vorhanden)
    zip_path = Path(zip_filename)
    downloaded = False
    if not zip_path.exists():
        urllib.request.urlretrieve(dataset_url, zip_filename)
        downloaded = True

    # 5. Entpacken
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_dir)

    # 6. Ordner umbenennen und verschieben
    extracted_folder = base_dir / "pd12m_dct_based_synthetic_stegano"

    # Falls der Hauptordner existiert, verschieben wir die Unterordner und den Cover-Ordner
    if extracted_folder.exists():
        # Verschieben der Unterordner in die Zielordner
        for old_folder in extracted_folder.glob("synthetic_*"):
            if old_folder.is_dir():
                # Entferne das 'synthetic_' Präfix aus den Ordnernamen
                new_folder_name = old_folder.name.replace("synthetic_", "")
                new_folder_path = base_dir / new_folder_name
                shutil.move(str(old_folder), str(new_folder_path))

        # Verschieben des Cover-Ordners direkt in den Zielordner
        cover_folder = extracted_folder / "Cover"
        if cover_folder.exists():
            shutil.move(str(cover_folder), str(base_dir / cover_folder.name))

        # Hauptordner löschen, falls leer
        if not any(extracted_folder.iterdir()):
            extracted_folder.rmdir()

    # 7. ZIP löschen
    if downloaded and zip_path.exists():
        zip_path.unlink()

    return f"✅ PD12M-Datensatz {'heruntergeladen und ' if downloaded else ''}erfolgreich entpackt und vorbereitet in {base_dir}."


def build_pd12m_like_reference(
    cover_count: int = 500,
    scan_limit: int = 5_000,
    out_root: str = "data/raw/PD12M",
    ref_dir: str = "data/raw/alaska2-image-steganalysis/Cover",
    batch_size: int = 32,
    ref_count: int = 300,
    initial_fetch: int = 10_000,
    force_new_generation: bool = False,
) -> str:
    """
    Erzeugt einen ALASKA2-ähnlichen CC0-Datensatz aus PD12M:
    - bricht ab, falls in out_root/Cover schon Bilder existieren
    - berechnet CLIP-Embeddings von Referenzbildern direkt
    - führt kNN auf PD12M-URLs im Embedding-Raum durch
    - speichert die Top-Bilder verkleinert auf 512x512 als JPEG.
    """
    # 0) Check ob schon Bilder vorhanden
    dataset_dir = Path(out_root) / "Cover"
    if dataset_dir.exists() and any(dataset_dir.glob("*.jpg")) and not force_new_generation:
        return f"✅ In '{dataset_dir}' existieren bereits Bilder. Keine neue Generierung nötig."

    # 1) CLIP-Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # 2) Referenz-Embeddings direkt berechnen
    ref_list = list(Path(ref_dir).glob("*.jpg"))
    k = min(len(ref_list), ref_count)
    ref_paths = random.sample(ref_list, k)

    with torch.no_grad():
        embs = []
        with tqdm(total=k, desc="Referenz-Embeddings") as pbar:
            for i in range(0, k, batch_size):
                batch = ref_paths[i : i + batch_size]
                imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
                embs.append(model.encode_image(imgs).cpu())
                pbar.update(len(batch))
    ref_emb = torch.cat(embs, dim=0).numpy().astype("float32")
    faiss.normalize_L2(ref_emb)

    # 3) PD12M-URLs (initial_fetch, dann shuffle, dann scan_limit)
    ds_stream = load_dataset("Spawning/PD12M", split="train", streaming=True)
    raw_urls = [row["url"] for row in tqdm(islice(ds_stream, initial_fetch), total=initial_fetch, desc="PD12M-URLs laden (raw)")]

    random.shuffle(raw_urls)
    urls = raw_urls[:scan_limit]

    # 4) Batch-Embedding + Clustering
    def download_and_preprocess(url: str) -> Optional[torch.Tensor]:
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return preprocess(img)
        except Exception:
            return None

    clusters = defaultdict(list)

    for i in tqdm(range(0, len(urls), batch_size), desc="Batch Embedding"):
        batch_urls = urls[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(download_and_preprocess, url) for url in batch_urls]
            imgs = [f.result() for f in futures if f.result() is not None]

        if not imgs:
            continue

        batch_tensor = torch.stack(imgs).to(device)

        with torch.no_grad():
            emb_batch = model.encode_image(batch_tensor).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb_batch)

        sims_matrix = emb_batch @ ref_emb.T  # shape: (batch_size, ref_count)
        max_sims = sims_matrix.max(axis=1)
        max_indices = sims_matrix.argmax(axis=1)

        for sim, url, ref_idx in zip(max_sims, batch_urls, max_indices):
            clusters[int(ref_idx)].append((float(sim), url))

    # Nun pro Cluster die Top-N auswählen
    cluster_count = len(clusters)
    per_cluster = max(1, cover_count // cluster_count)

    selected = []
    for ref_idx, items in clusters.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        selected += [url for _, url in items_sorted[:per_cluster]]

    # Falls noch nicht genug → Rest auffüllen
    if len(selected) < cover_count:
        all_candidates = []
        for items in clusters.values():
            all_candidates.extend(items)
        all_candidates_sorted = sorted(all_candidates, key=lambda x: -x[0])
        needed = cover_count - len(selected)
        additional = [url for _, url in all_candidates_sorted if url not in selected][:needed]
        selected += additional

    best_urls = selected[:cover_count]

    # 5) Top-URLs final speichern mit garantierter JPEG-Kompression
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(tqdm(best_urls, desc="Speichern")):
        try:
            r = requests.get(url, timeout=5)
            original = Image.open(BytesIO(r.content)).convert("RGB")

            img = Image.fromarray(np.array(original))
            img = img.resize((512, 512), Image.LANCZOS)

            quality = random.choice([75, 90, 95])
            img.save(dataset_dir / f"{idx + 1:05d}.jpg", "JPEG", quality=quality)

        except Exception:
            continue

    return f"✅ {len(best_urls)} Cover in '{dataset_dir}' gespeichert."


def generate_conseal_stego(
    cover_path: str = "data/raw/PD12M/Cover",
    difficulty: float = 0.4,
    seed: int = 42,
    force_new_generation: bool = False,
) -> str:
    """
    Erzeugt aus Cover‐JPEGs drei Stego‐Varianten mit conseal und speichert sie in Unterordnern.

    - UERD: Uniform Embedding Revisited Distortion
    - JUNIWARD: JPEG‐UNIWARD
    - JMiPOD: nsF5 als Ersatz für JMiPOD

    Args:
        cover_path (str): Pfad zum Cover‐Ordner mit *.jpg.
        difficulty (float): Embedding‐Rate bzw. alpha (0.0–1.0).
        seed (int): Basis‐Seed für RNG (pro Bild wird i hinzuaddiert).
        force_new_generation (bool): Wenn True, werden die Stego‐Ordner immer neu befüllt,
                                     auch wenn schon Bilder vorhanden sind.

    Returns:
        str: Statusmeldung, z. B. wie viele Bilder generiert wurden oder warum abgebrochen wurde.
    """
    cover_path = Path(cover_path)
    stego_base = cover_path.parent

    variant_fns = {
        "UERD": lambda im0, jpeg, d, s: cl.uerd.simulate_single_channel(y0=jpeg.Y, qt=jpeg.qt[0], alpha=d, seed=s),
        "JUNIWARD": lambda im0, jpeg, d, s: cl.juniward.simulate_single_channel(x0=im0.spatial[..., 0], y0=jpeg.Y, qt=jpeg.qt[0], alpha=d, seed=s),
        "JMiPOD": lambda im0, jpeg, d, s: cl.nsF5.simulate_single_channel(y0=jpeg.Y, alpha=d, seed=s),
    }

    # Ordner für jede Variante anlegen
    variant_paths = {name: stego_base / name for name in variant_fns}
    for p in variant_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # ggf. überspringen
    if not force_new_generation and all(p.exists() and any(p.glob("*.jpg")) for p in variant_paths.values()):
        return "✅ Stego‐Ordner existieren bereits und enthalten Bilder. Keine neue Generierung nötig."

    files = sorted(cover_path.glob("*.jpg"))
    if not files:
        return f"❌ Keine Bilder in {cover_path}"

    for i, path in enumerate(tqdm(files, desc="Stego mit conseal")):
        im0 = jpeglib.read_spatial(str(path), jpeglib.JCS_GRAYSCALE)
        jpeg = jpeglib.read_dct(str(path))

        for name, fn in variant_fns.items():
            jpeg_variant = jpeg.copy()
            jpeg_variant.Y = fn(im0, jpeg, difficulty, seed + i)
            jpeg_variant.write_dct(str(variant_paths[name] / path.name))

    return f"✅ {len(files)} Bilder generiert in {', '.join(variant_fns.keys())} (difficulty={difficulty})"


def build_file_index(
    dataset_root: str,
    class_labels: Dict[str, int],
    subsample_percent: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Erstellt einen DataFrame mit Bild‐Pfaden und Klassen­namen (ohne Metadaten).

    - Liest alle Dateinamen aus dem Cover-Ordner.
    - Wählt optional einen prozentualen Anteil davon (Subsampling).
    - Kombiniert diese Namen mit den anderen Klassen­ordnern,
      so dass in jeder Klasse die identischen Dateinamen vorkommen.

    Returns
    -------
    pd.DataFrame  Spalten: 'path', 'label_name'
    """
    random.seed(seed)
    root = Path(dataset_root)

    # 1) Alle Dateinamen im Cover-Ordner sammeln
    cover_folder = root / "Cover"
    all_filenames = sorted(p.name for p in cover_folder.glob("*.jpg"))

    # 2) Zufälliges Subsampling (identische Auswahl für alle Klassen!)
    if 0 < subsample_percent < 1.0:
        k = int(len(all_filenames) * subsample_percent)
        filenames = random.sample(all_filenames, k)
    else:
        filenames = all_filenames

    # 3) Dateipfade & Klassen auflisten
    records = []
    for class_name in class_labels:
        class_dir = root / class_name
        for name in filenames:
            records.append(
                {
                    "path": str(class_dir / name),
                    "label_name": class_name,
                }
            )

    return pd.DataFrame(records)


def add_jpeg_metadata(df: pd.DataFrame, quiet: bool = False) -> pd.DataFrame:
    """
    Ergänzt zu einem DataFrame mit einer 'path'-Spalte sämtliche JPEG-Metadaten.

    Für jedes Bild werden erzeugt:
    - 'jpeg_quality': heuristische Schätzung (75, 90, 95)
    - 'width', 'height', 'mode'
    - 'q_y_00' … 'q_y_63'  (64 Einträge der Y-Quantisierungstabelle)

    Hinweis zur Qualitäts­schätzung:
        mean(q_y)  <   8  → 95
        8 ≤ mean < 20  → 90
        sonst          → 75
    """
    meta_rows = []
    iterator = df["path"]
    if not quiet:
        iterator = tqdm(iterator, desc="Metadaten extrahieren")

    for path in iterator:
        meta = {
            "jpeg_quality": -1,
            "width": -1,
            "height": -1,
            "mode": "unknown",
            **{f"q_y_{i:02d}": -1 for i in range(64)},
        }

        try:
            with Image.open(path) as img:
                meta["width"], meta["height"] = img.size
                meta["mode"] = img.mode

                if getattr(img, "quantization", None):
                    q_y = img.quantization.get(0, [-1] * 64)
                    q_mean = np.mean(q_y)

                    if q_mean < 8:
                        meta["jpeg_quality"] = 95
                    elif q_mean < 20:
                        meta["jpeg_quality"] = 90
                    else:
                        meta["jpeg_quality"] = 75

                    for i, q in enumerate(q_y[:64]):
                        meta[f"q_y_{i:02d}"] = q
        except Exception:
            # Bild konnte nicht gelesen werden → Platzhalter bleiben -1
            pass

        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    return pd.concat([df.reset_index(drop=True), meta_df], axis=1)


def split_dataset_by_filename(
    df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splittet das Dataset so, dass alle Versionen eines Bildes
    (z.B. Cover/JMiPOD/JUNIWARD/UERD) zusammen im gleichen Split landen.

    Args:
        df (pd.DataFrame): DataFrame mit Spalten 'path' und 'label'.
        train_size (float): Anteil des Trainingssets.
        val_size (float): Anteil des Validierungssets.
        test_size (float): Anteil des Testsets.
        seed (int): Random-Seed für Reproduzierbarkeit.

    Returns:
        (train_df, val_df, test_df): Getrennte DataFrames für Training, Validation und Test.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits müssen zusammen 1.0 ergeben."

    # Extrahiere Dateinamen (z.B. '00001.jpg')
    df = df.copy()
    df["filename"] = df["path"].apply(lambda x: Path(x).name)

    # Eindeutige Bildnamen
    unique_filenames = df["filename"].unique()

    # Split der Bildnamen (nicht der Zeilen!)
    train_val_filenames, test_filenames = train_test_split(unique_filenames, test_size=test_size, random_state=seed)

    train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=val_size / (train_size + val_size), random_state=seed)

    # Zuordnung der Splits
    train_df = df[df["filename"].isin(train_filenames)].drop(columns=["filename"])
    val_df = df[df["filename"].isin(val_filenames)].drop(columns=["filename"])
    test_df = df[df["filename"].isin(test_filenames)].drop(columns=["filename"])

    return train_df, val_df, test_df
