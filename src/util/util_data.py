import heapq
import random
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
from pathlib import Path

import clip
import cv2
import faiss
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


def build_pd12m_like_alaska2(
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
    - berechnet CLIP-Embeddings von Referenzbildern direkt (ohne Speichern)
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

    # 5) Top-URLs final speichern
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(tqdm(best_urls, desc="Speichern")):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(dataset_dir / f"{idx + 1:05d}.jpg", "JPEG", quality=90)
        except Exception:
            continue

    return f"✅ {len(best_urls)} Cover in '{dataset_dir}' gespeichert."


# _SESSION = requests.Session()
# def sample_random_pd12m(
#     out_dir: str = "data/raw/PD12M/Dataset",
#     num_samples: int = 1000,
#     scan_limit: int | None = None,
#     oversample_factor: int = 2,
#     timeout: int = 10,
#     max_workers: int = 32,
#     chunk_size: int = 65536,
# ) -> str:
#     """
#     Zieht num_samples zufällige Bilder aus den ersten scan_limit PD12M-URLs
#     und speichert sie parallel als rohe JPEG-Bytes (kein PIL) in out_dir.
#     scan_limit = num_samples * oversample_factor, falls None.
#     """
#     # 1) scan_limit berechnen
#     if scan_limit is None:
#         scan_limit = num_samples * oversample_factor

#     # 2) die ersten scan_limit URLs
#     from datasets import load_dataset
#     ds = load_dataset("Spawning/PD12M", split="train", streaming=True)
#     urls = [row["url"] for row in islice(ds, scan_limit)]

#     # 3) Stichprobe
#     if num_samples > len(urls):
#         num_samples = len(urls)
#     selected = random.sample(urls, num_samples)

#     # 4) Zielordner
#     out_path = Path(out_dir)
#     out_path.mkdir(parents=True, exist_ok=True)

#     # 5) Download-&-Save-Funktion (rohe Bytes)
#     def _download_and_save(url_idx):
#         url, idx = url_idx
#         fn = out_path / f"{idx:05d}.jpg"
#         try:
#             resp = _SESSION.get(url, stream=True, timeout=timeout)
#             resp.raise_for_status()
#             with open(fn, "wb") as f:
#                 for chunk in resp.iter_content(chunk_size=chunk_size):
#                     if chunk:
#                         f.write(chunk)
#             return True
#         except Exception:
#             return False

#     # 6) Parallel mit ThreadPoolExecutor
#     saved = 0
#     tasks = ((url, i) for i, url in enumerate(selected, start=1))
#     with ThreadPoolExecutor(max_workers=max_workers) as exe:
#         for ok in exe.map(_download_and_save, tasks):
#             if ok:
#                 saved += 1

#     return f"✅ {saved}/{num_samples} Bilder in '{out_path}' gespeichert (aus {scan_limit} URLs)."

# def preprocess_covers(
#     src_dir: str = "data/raw/PD12M/Dataset",
#     dst_dir: str = "data/raw/PD12M/Cover",
#     target_size: tuple[int, int] = (512, 512),
#     quality: int = 90
# ) -> str:
#     """
#     Resize + JPEG-Neukomprimierung aller Bilder mit OpenCV.
#     """
#     src = Path(src_dir)
#     dst = Path(dst_dir)
#     dst.mkdir(parents=True, exist_ok=True)

#     files = list(src.glob("*.jpg"))
#     for img_path in files:
#         img = cv2.imread(str(img_path))
#         if img is None:
#             continue
#         img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
#         out_path = str(dst / img_path.name)
#         cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

#     return f"✅ {len(files)} Bilder in {dst} neu skaliert und als JPEG gespeichert."


def generate_stego_variants(cover_path: str = "data/raw/PD12M/Cover", stego_base_path: str = "data/raw/PD12M/") -> str:
    """
    Erzeugt drei Stego-Varianten (JMiPOD, JUNIWARD, UERD) aus den Cover-Bildern,
    falls die Varianten noch nicht existieren.

    Args:
        cover_folder_path (str): Pfad zum Cover-Ordner.
        stego_base_path (str): Basis-Ordner für die Stego-Ordner.

    Returns:
        str: Statusmeldung.
    """
    cover_folder = Path(cover_path)
    jmipod_folder = Path(stego_base_path) / "JMiPOD"
    juniward_folder = Path(stego_base_path) / "JUNIWARD"
    uerd_folder = Path(stego_base_path) / "UERD"

    # Falls Stego-Ordner bereits existieren und nicht leer sind → Abbruch
    if all(folder.exists() and any(folder.glob("*.jpg")) for folder in [jmipod_folder, juniward_folder, uerd_folder]):
        return "✅ Stego-Ordner existieren bereits und enthalten Bilder. Keine neue Generierung nötig."

    # Zielordner erstellen, falls sie fehlen
    jmipod_folder.mkdir(parents=True, exist_ok=True)
    juniward_folder.mkdir(parents=True, exist_ok=True)
    uerd_folder.mkdir(parents=True, exist_ok=True)

    cover_images = list(cover_folder.glob("*.jpg"))
    if not cover_images:
        return f"❌ Keine Cover-Bilder im Ordner {cover_folder} gefunden."

    for cover_path in cover_images:
        img = cv2.imread(str(cover_path))
        img_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycc)

        y = y.astype(np.float32)
        block_size = 8
        h, w = y.shape

        # Erzeuge leere Matrizen für Varianten
        jmipod_y = np.zeros_like(y)
        juniward_y = np.zeros_like(y)
        uerd_y = np.zeros_like(y)

        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = y[i : i + block_size, j : j + block_size]
                if block.shape == (8, 8):
                    dct_block = cv2.dct(block)

                    # Kopiere für jede Variante
                    jmipod_block = dct_block.copy()
                    juniward_block = dct_block.copy()
                    uerd_block = dct_block.copy()

                    # Modifikationen
                    jmipod_block[2:6, 2:6] += np.random.normal(0, 0.5, (4, 4))
                    juniward_block[4:8, 0:4] *= 1.05
                    mask = np.random.rand(8, 8) < 0.05
                    uerd_block += mask * np.random.normal(0, 1.0, (8, 8))

                    # IDCT zurück
                    jmipod_block = np.clip(cv2.idct(jmipod_block), 0, 255)
                    juniward_block = np.clip(cv2.idct(juniward_block), 0, 255)
                    uerd_block = np.clip(cv2.idct(uerd_block), 0, 255)

                    # Blöcke einfügen
                    jmipod_y[i : i + block_size, j : j + block_size] = jmipod_block
                    juniward_y[i : i + block_size, j : j + block_size] = juniward_block
                    uerd_y[i : i + block_size, j : j + block_size] = uerd_block

        # Zurück in YCrCb und BGR
        img_jmipod = cv2.cvtColor(cv2.merge([jmipod_y.astype(np.uint8), cr, cb]), cv2.COLOR_YCrCb2BGR)
        img_juniward = cv2.cvtColor(cv2.merge([juniward_y.astype(np.uint8), cr, cb]), cv2.COLOR_YCrCb2BGR)
        img_uerd = cv2.cvtColor(cv2.merge([uerd_y.astype(np.uint8), cr, cb]), cv2.COLOR_YCrCb2BGR)

        # Speichern
        cv2.imwrite(str(jmipod_folder / cover_path.name), img_jmipod)
        cv2.imwrite(str(juniward_folder / cover_path.name), img_juniward)
        cv2.imwrite(str(uerd_folder / cover_path.name), img_uerd)

    return f"✅ {len(cover_images)} Cover-Bilder erfolgreich verarbeitet und Stego-Varianten erzeugt."


def prepare_dataset(dataset_root: str, class_labels: dict, subsample_percent: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    Erstellt ein Datenset aus einem JPEG-Steganalyse-Datensatz,
    indem optional nur ein Teil der Bildnamen zufällig ausgewählt wird.

    Args:
        dataset_root (Path): Pfad zum Verzeichnis mit den Klassenordnern.
        class_labels (dict): Mapping von Klassenordnern zu numerischen Labels.
        subsample_percent (float): Anteil der Bildnamen, die verwendet werden sollen (z.B. 0.10 = 10% oder 1.0 = 100%).
        seed (int): Seed für die Zufallsauswahl.

    Returns:
        pd.DataFrame: DataFrame mit Spalten 'path' und 'label', zufällig durchmischt.
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
    labels = []

    for class_name, label in class_labels.items():
        class_folder = dataset_root / class_name
        for img_name in selected_image_names:
            img_path = class_folder / img_name
            image_paths.append(str(img_path))
            labels.append(label)

    df = pd.DataFrame({"path": image_paths, "label": labels})

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


# -------------------------------------------------------------------------------
# def generate_and_save_ref_embeddings(
#     ref_dir: str = "data/raw/alaska2-image-steganalysis/Cover",
#     output_path: str = "data/raw/ref_emb.npy",
#     ref_count: int = 300,
#     batch_size: int = 32,
# ):
#     """
#     Berechnet CLIP-Embeddings für bis zu ref_count zufällige Cover-Bilder
#     und speichert sie als NumPy-Array in output_path.
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     model.eval()

#     ref_list = list(Path(ref_dir).glob("*.jpg"))
#     k = min(len(ref_list), ref_count)
#     ref_paths = random.sample(ref_list, k)

#     with torch.no_grad():
#         embs = []
#         for i in range(0, k, batch_size):
#             batch = ref_paths[i : i + batch_size]
#             imgs = torch.stack([
#                 preprocess(Image.open(p).convert("RGB"))
#                 for p in batch
#             ]).to(device)
#             embs.append(model.encode_image(imgs).cpu())
#     ref_emb = torch.cat(embs, dim=0).numpy().astype("float32")

#     out_path = Path(output_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     np.save(out_path, ref_emb)
#     print(f"✅ Gespeichert: {k} Referenz-Embeddings in '{out_path}'")


# def build_pd12m_like_alaska2(
#     cover_count: int = 500,
#     scan_limit: int = 10_000,
#     out_root: str = "data/raw/PD12M_like_alaska2",
#     ref_emb_path: str = "data/raw/ref_emb.npy",
#     batch_size: int = 16,
# ) -> str:
#     """
#     Erzeugt einen ALASKA2-ähnlichen CC0-Datensatz aus PD12M:
#     - lädt (oder erzeugt) ref_emb.npy
#     - führt kNN auf PD12M-URLs im Embedding-Raum durch
#     - speichert die Tops in out_root/Dataset/
#     """
#     # 1) CLIP-Setup
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     disable_caching()
#     model.eval()

#     # 2) Referenz-Embeddings laden (oder erzeugen)
#     ref_path = Path(ref_emb_path)
#     if not ref_path.exists():
#         generate_and_save_ref_embeddings(
#             ref_dir=str(Path(ref_path.parent) / "alaska2-image-steganalysis/Cover"),
#             output_path=str(ref_path),
#             ref_count=300,
#             batch_size=32
#         )
#     ref_emb = np.load(ref_path).astype("float32")
#     faiss.normalize_L2(ref_emb)

#     # 3) PD12M-URLs (erste scan_limit)
#     ds_stream = load_dataset("Spawning/PD12M", split="train", streaming=True)
#     urls = [row["url"] for row in islice(ds_stream, scan_limit)]

#     # 4) Batch-Embedding + kNN-Heap
#     heap: list[tuple[float, str]] = []
#     t0 = time.time()
#     for i in range(0, len(urls), batch_size):
#         batch_urls = urls[i : i + batch_size]
#         imgs = []
#         for url in batch_urls:
#             r = requests.get(url, timeout=5)
#             imgs.append(preprocess(Image.open(BytesIO(r.content)).convert("RGB")))
#         batch_tensor = torch.stack(imgs).to(device)

#         with torch.no_grad():
#             emb_batch = model.encode_image(batch_tensor).cpu().numpy().astype("float32")
#         faiss.normalize_L2(emb_batch)

#         sims = (emb_batch @ ref_emb.T).max(axis=1)
#         for sim, url in zip(sims, batch_urls):
#             entry = (float(sim), url)
#             if len(heap) < cover_count:
#                 heapq.heappush(heap, entry)
#             else:
#                 heapq.heappushpop(heap, entry)
#     print(f"⏱ Embedding+kNN für {len(urls)} Bilder: {time.time()-t0:.2f} s")

#     best_urls = [u for _, u in sorted(heap, key=lambda t: -t[0])]

#     # 5) Top-URLs final speichern
#     dataset_dir = Path(out_root) / "Dataset"
#     dataset_dir.mkdir(parents=True, exist_ok=True)
#     t1 = time.time()
#     for idx, url in enumerate(best_urls, 1):
#         r = requests.get(url, timeout=5)
#         img = Image.open(BytesIO(r.content)).convert("RGB")
#         img.save(dataset_dir / f"{idx:05d}.jpg", "JPEG", quality=95)
#     print(f"⏱ Finaler Download+Save: {time.time()-t1:.2f} s")

#     return f"✅ {len(best_urls)} Cover in {dataset_dir} gespeichert."
