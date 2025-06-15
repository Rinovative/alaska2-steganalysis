from __future__ import annotations

# Standard library
import heapq
import random
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path

# Third-party libraries
import clip
import faiss
import jpegio as jio
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
    if dataset_dir.exists() and any(dataset_dir.glob("*.jpg")):
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

    # 3) PD12M-URLs (erste scan_limit)
    ds_stream = load_dataset("Spawning/PD12M", split="train", streaming=True)
    urls = [row["url"] for row in tqdm(islice(ds_stream, scan_limit), total=scan_limit, desc="PD12M-URLs laden")]

    # 4) Batch-Embedding + kNN-Heap
    def download_and_preprocess(url: str) -> torch.Tensor | None:
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return preprocess(img)
        except Exception:
            return None

    heap: list[tuple[float, str]] = []
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

        sims = (emb_batch @ ref_emb.T).max(axis=1)
        for sim, url in zip(sims, batch_urls):
            entry = (float(sim), url)
            if len(heap) < cover_count:
                heapq.heappush(heap, entry)
            else:
                heapq.heappushpop(heap, entry)

    best_urls = [u for _, u in sorted(heap, key=lambda t: -t[0])]

    # 5) Top-URLs final speichern mit garantierter JPEG-Kompression
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(tqdm(best_urls, desc="Speichern")):
        try:
            r = requests.get(url, timeout=5)
            original = Image.open(BytesIO(r.content)).convert("RGB")

            # Garantierte Neukodierung: konvertieren zu Array → Image
            img = Image.fromarray(np.array(original))
            img = img.resize((512, 512), Image.LANCZOS)

            # Zufällige JPEG-Qualität aus {75, 90, 95}
            quality = random.choice([75, 90, 95])

            # Speichern
            img.save(dataset_dir / f"{idx + 1:05d}.jpg", "JPEG", quality=quality)

        except Exception:
            continue

    return f"✅ {len(best_urls)} Cover in '{dataset_dir}' gespeichert."


def _pick_payload_indices(mask: np.ndarray, payload_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Hilfsfunktion: wähle zufällig Positionen aus `mask` (True-Elements),
    so dass ungefähr `payload_rate × mask.sum()` Koef­fi­zienten modifiziert werden.
    """
    candidates = np.column_stack(np.where(mask))
    n_bits = int(np.round(payload_rate * len(candidates)))
    if n_bits == 0:
        return np.empty((0, 2), dtype=int)
    sel = rng.choice(len(candidates), n_bits, replace=False)
    return candidates[sel]


def generate_stego_variants(
    cover_path: str | Path = "data/raw/PD12M/Cover",
    stego_base_path: str | Path = "data/raw/PD12M",
    payload_rates: dict[str, float] | None = None,
    seed: int = 42,
) -> str:
    """
    Erzeugt JMiPOD-, JUNIWARD- und UERD-ähnliche Varianten durch ±1-Flips
    quantisierter AC-DCT-Koeffizienten OHNE JPEG-Rekompression.

    ▸ **Quantisierungstabellen bleiben unverändert**
    ▸ **Farbinformation (Cb/Cr) bleibt unberührt**
    ▸ **Payload-Rate** = Bits pro *nicht-null* AC-Koeffizient

    Args
    ----
    cover_path
        Ordner mit Cover-JPEGs (z. B. „…/PD12M/Cover“).
    stego_base_path
        Basisordner; es werden Unterordner „JMiPOD“, „JUNIWARD“, „UERD“
        angelegt.
    payload_rates
        Dict mit Payload-Raten pro Algorithmus (Default ≈ ALASKA2-Level).
        Beispiel: ``{"JMiPOD": 0.4, "JUNIWARD": 0.4, "UERD": 0.4}``
    seed
        RNG-Seed.

    Returns
    -------
    str
        Statusmeldung.
    """
    rng = np.random.default_rng(seed)

    cover_folder = Path(cover_path)
    jmipod_folder = Path(stego_base_path) / "JMiPOD"
    juniward_folder = Path(stego_base_path) / "JUNIWARD"
    uerd_folder = Path(stego_base_path) / "UERD"
    if all(folder.exists() and any(folder.glob("*.jpg")) for folder in [jmipod_folder, juniward_folder, uerd_folder]):
        return "✅ Stego-Ordner existieren bereits und enthalten Bilder. Keine neue Generierung nötig."

    for f in (jmipod_folder, juniward_folder, uerd_folder):
        f.mkdir(parents=True, exist_ok=True)

    # Nutzlast-Raten
    if payload_rates is None:
        payload_rates = {"JMiPOD": 0.4, "JUNIWARD": 0.4, "UERD": 0.4}

    cover_images = sorted(cover_folder.glob("*.jpg"))
    if not cover_images:
        return f"❌ Keine Cover-Bilder in {cover_folder} gefunden."

    # -----------------------------------------------------------
    for cover_img in tqdm(cover_images, desc="Erzeuge Stego-Varianten"):
        jpeg = jio.read(str(cover_img))
        y_coef = jpeg.coef_arrays[0]  # Y-Kanal (quanti­siert, int16)
        h, w = y_coef.shape

        # Masks: True = potentiell änder­bar
        nz_mask = y_coef != 0  # keine DC=0-Flips
        dc_mask = np.zeros_like(y_coef, dtype=bool)
        dc_mask[0::8, 0::8] = True  # DC-Koeff. ausschliessen
        ac_mask = nz_mask & ~dc_mask

        # Frequenzpositionen (u+v)-Matrix für einfache JUNIWARD-Gewichtung
        u = np.tile(np.arange(h)[:, None], (1, w))
        v = np.tile(np.arange(w)[None, :], (h, 1))
        freq_sum = (u % 8) + (v % 8)  # 0 (DC) … 14 (HF)

        # ---------- JMiPOD ----------
        # mittlere Frequenzen (2–6) bevorzugen
        jm_mask_mid = (freq_sum >= 2) & (freq_sum <= 6) & ac_mask
        idx_jm = _pick_payload_indices(jm_mask_mid, payload_rates["JMiPOD"], rng)
        y_jm = y_coef.copy()
        y_jm[idx_jm[:, 0], idx_jm[:, 1]] += rng.choice([-1, 1], size=len(idx_jm))

        # ---------- JUNIWARD ----------
        # hochfrequente (>6) und texturreiche Blöcke (AC-Energie)
        hf_mask = (freq_sum > 6) & ac_mask
        # einfache Textur-Schätzung: Varianz im 8×8-Block > Schwelle
        energy = np.abs(y_coef).reshape(h // 8, 8, w // 8, 8).sum(axis=(1, 3))  # Block-Energie
        high_energy_blocks = energy > np.percentile(energy, 70)
        block_mask = np.repeat(np.repeat(high_energy_blocks, 8, axis=0), 8, axis=1)
        ju_mask = hf_mask & block_mask
        idx_ju = _pick_payload_indices(ju_mask, payload_rates["JUNIWARD"], rng)
        y_ju = y_coef.copy()
        y_ju[idx_ju[:, 0], idx_ju[:, 1]] += rng.choice([-1, 1], size=len(idx_ju))

        # ---------- UERD ----------
        # breit gestreut, zufällig
        ue_mask = ac_mask
        idx_ue = _pick_payload_indices(ue_mask, payload_rates["UERD"], rng)
        y_ue = y_coef.copy()
        y_ue[idx_ue[:, 0], idx_ue[:, 1]] += rng.choice([-1, 1], size=len(idx_ue))

        # ---------- Speichern ----------
        for algo, y_mod, out_folder in [
            ("JMiPOD", y_jm, jmipod_folder),
            ("JUNIWARD", y_ju, juniward_folder),
            ("UERD", y_ue, uerd_folder),
        ]:
            jpeg_mod = jio.read(str(cover_img))
            jpeg_mod.coef_arrays[0] = y_mod.astype(np.int16)
            out_path = out_folder / cover_img.name
            jio.write(jpeg_mod, str(out_path))

    return f"✅ {len(cover_images)} Bilder verarbeitet – Stego-Varianten liegen in {stego_base_path}/(JMiPOD|JUNIWARD|UERD)"


def prepare_dataset(dataset_root: str, class_labels: dict, subsample_percent: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Erstellt ein Datenset aus einem JPEG-Steganalyse-Datensatz,
    inklusive optionaler Subsampling-Logik und vollständiger Metadatenextraktion.

    Für jedes Bild werden folgende Informationen extrahiert:
    - 'path': Pfad zur JPEG-Datei
    - 'label_name': Klassenname (Cover, JMiPOD, ...)
    - 'jpeg_quality': geschätzte JPEG-Qualitätsstufe (75, 90, 95), basierend auf dem Mittelwert der Y-Quantisierungstabelle
    - 'width', 'height': Bildauflösung
    - 'mode': Farbraum-Modus (z. B. 'YCbCr')
    - 'q_y_00' bis 'q_y_63': alle 64 Einträge der Y-Quantisierungstabelle

    Hinweis: Die JPEG-Qualitätsstufe wird heuristisch geschätzt, da sie nicht explizit gespeichert ist.
    Die Zuweisung erfolgt basierend auf typischen Mittelwerten aus libjpeg-Standardtabellen.

    Args:
        dataset_root (str): Pfad zum Verzeichnis mit den Klassenordnern.
        class_labels (dict): Mapping von Klassenordnern zu numerischen Labels.
        subsample_percent (float): Anteil der Bildnamen, die verwendet werden sollen (z.B. 0.10 = 10% oder 1.0 = 100%).
        seed (int): Seed für die Zufallsauswahl.

    Returns:
        pd.DataFrame: Aufbereiteter Datensatz mit Metadaten.
    """
    random.seed(seed)
    dataset_root = Path(dataset_root)

    cover_folder = dataset_root / "Cover"
    all_images = sorted([img.name for img in cover_folder.glob("*.jpg")])

    if subsample_percent < 1.0:
        sample_size = int(len(all_images) * subsample_percent)
        selected_image_names = random.sample(all_images, sample_size)
    else:
        selected_image_names = all_images

    image_paths = []
    label_names = []

    for class_name in class_labels:
        class_folder = dataset_root / class_name
        for img_name in selected_image_names:
            img_path = class_folder / img_name
            image_paths.append(str(img_path))
            label_names.append(class_name)

    df = pd.DataFrame(
        {
            "path": image_paths,
            "label_name": label_names,
        }
    )

    # Metadaten effizient extrahieren
    meta_data = []
    for path in tqdm(df["path"], desc="Extrahiere Metadaten"):
        result = {"jpeg_quality": -1, "width": -1, "height": -1, "mode": "unknown", **{f"q_y_{i:02d}": -1 for i in range(64)}}

        try:
            with Image.open(path) as img:
                result["width"], result["height"] = img.size
                result["mode"] = img.mode

                if hasattr(img, "quantization") and img.quantization:
                    q_y = img.quantization.get(0, [-1] * 64)
                    q_mean = np.mean(q_y)
                    if q_mean < 8:  # empirisch: mean ≈ 5 → Qualität 95
                        result["jpeg_quality"] = 95
                    elif q_mean < 20:  # empirisch: mean ≈ 11 → Qualität 90
                        result["jpeg_quality"] = 90
                    else:  # empirisch: mean ≈ 29 → Qualität 75
                        result["jpeg_quality"] = 75
                    # result["jpeg_quality"] = q_mean
                    for i in range(min(64, len(q_y))):
                        result[f"q_y_{i:02d}"] = q_y[i]
        except Exception:
            pass

        meta_data.append(result)

    meta_df = pd.DataFrame(meta_data)
    df = pd.concat([df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


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
