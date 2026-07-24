"""
===============================================================================
data_synthetic.py
===============================================================================
Explicit and safe PD12M proxy preparation.

Responsibilities:
  - Download and validate the curated PD12M proxy with bounded network
    operations.
  - Select PD12M references without losing URL-to-embedding alignment.
  - Generate JMiPOD-compatible, JUNIWARD, and UERD variants transactionally.
  - Report success and failure causes instead of silently accepting partial
    data.

Design principles:
  - Potentially destructive replacement is allowed only below the configured
    generated data root and only after a complete staging dataset has been
    validated.

Boundaries:
  - Generation dependencies are optional and imported inside the functions
    that need them. Importing src never downloads data or imports the
    generation stack.

Notes:
  - The public JMiPOD compatibility variant is technically generated with
    nsF5. It is not scientifically equivalent to the ALASKA2 JMiPOD class.
===============================================================================
"""

from __future__ import annotations

import random
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from itertools import islice
from pathlib import Path
from typing import Final, cast

from PIL import Image
from tqdm.auto import tqdm

from ..config.config_paths import ProjectPaths, default_paths

__all__ = [
    "PreparationReport",
    "build_pd12m_reference",
    "download_pd12m_proxy",
    "generate_synthetic_stego",
    "validate_proxy_dataset",
]

_PROXY_URL: Final[str] = (
    "https://huggingface.co/datasets/Rinovative/pd12m_dct_based_synthetic_stegano/"
    "resolve/main/pd12m_dct_based_synthetic_stegano.zip"
)
_PROXY_VARIANTS: Final[tuple[str, ...]] = ("Cover", "JMiPOD", "JUNIWARD", "UERD")


@dataclass(frozen=True, slots=True)
class PreparationReport:
    """Describe an explicit synthetic-data preparation operation.

    Parameters
    ----------
    status
        Stable outcome category such as existing, downloaded, or generated.
    message
        Human-readable operation summary.
    saved_images
        Number of image files validated or written.
    failures
        Failure counts grouped by cause.
    """

    status: str
    message: str
    saved_images: int = 0
    failures: dict[str, int] = field(default_factory=dict)


def _assert_safe_target(target: Path, data_root: Path) -> None:
    resolved_target = target.resolve()
    resolved_root = data_root.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(f"Destructive target must be a child of {resolved_root}: {resolved_target}")
    if resolved_target == Path(resolved_target.anchor):
        raise ValueError("Filesystem roots are never valid dataset targets.")


def _jpg_names(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.jpg") if path.is_file()}


def _is_placeholder_only_directory(path: Path) -> bool:
    if not path.is_dir() or path.is_symlink():
        return False
    entries = tuple(path.iterdir())
    return len(entries) == 1 and entries[0].name == ".gitkeep" and entries[0].is_file() and not entries[0].is_symlink()


def _install_prepared_proxy(
    prepared: Path,
    target: Path,
    *,
    replace_placeholder: bool,
) -> None:
    if not replace_placeholder:
        if target.exists():
            shutil.rmtree(target)
        prepared.replace(target)
        return

    if not _is_placeholder_only_directory(target):
        raise ValueError(f"PD12M placeholder changed before installation and will not be modified: {target}")
    backup = target.parent / f"{prepared.parent.name}-placeholder-backup"
    if backup.exists():
        raise FileExistsError(f"Refusing to overwrite an existing PD12M rollback path: {backup}")
    target.replace(backup)
    try:
        prepared.replace(target)
    except OSError as install_error:
        try:
            backup.replace(target)
        except OSError as rollback_error:
            raise RuntimeError(
                f"PD12M installation and placeholder rollback failed; the original placeholder remains at {backup}."
            ) from rollback_error
        raise OSError(
            f"PD12M installation failed; the original placeholder was restored at {target}."
        ) from install_error
    shutil.rmtree(backup)


def validate_proxy_dataset(root: str | Path) -> int:
    """Validate exact source-name parity across all synthetic proxy variants.

    Parameters
    ----------
    root
        PD12M proxy root containing the four public class directories.

    Returns
    -------
    int
        Number of complete source groups.

    Raises
    ------
    ValueError
        If a class is missing, empty, or has mismatched filenames.
    """
    dataset_root = Path(root)
    names: dict[str, set[str]] = {}
    for variant in _PROXY_VARIANTS:
        directory = dataset_root / variant
        if not directory.is_dir():
            raise ValueError(f"Missing synthetic variant directory: {directory}")
        names[variant] = _jpg_names(directory)
        if not names[variant]:
            raise ValueError(f"Synthetic variant is empty: {directory}")
    reference = names["Cover"]
    mismatches = {variant: len(reference.symmetric_difference(values)) for variant, values in names.items()}
    mismatches = {variant: count for variant, count in mismatches.items() if count}
    if mismatches:
        raise ValueError(f"Incomplete synthetic source groups: {mismatches}")
    return len(reference)


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_resolved = destination.resolve()
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError(f"ZIP symlinks are not allowed: {member.filename}")
        output = (destination / member_path).resolve()
        if output != destination_resolved and destination_resolved not in output.parents:
            raise ValueError(f"ZIP member escapes destination: {member.filename}")
    archive.extractall(destination)


def _find_variant(source: Path, candidates: tuple[str, ...]) -> Path:
    matches = [
        path for candidate in candidates for path in source.rglob(candidate) if path.is_dir() and _jpg_names(path)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one populated directory from {candidates}, found {len(matches)}.")
    return matches[0]


def download_pd12m_proxy(
    *,
    paths: ProjectPaths | None = None,
    force: bool = False,
    url: str = _PROXY_URL,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
) -> PreparationReport:
    """Download, normalize, and transactionally install the curated PD12M proxy.

    Parameters
    ----------
    paths
        Optional project paths used to constrain the generated-data target.
    force
        Whether a complete existing proxy may be replaced.
    url
        HTTPS archive location.
    connect_timeout
        Maximum connection setup time in seconds.
    read_timeout
        Maximum response read wait in seconds.

    Returns
    -------
    PreparationReport
        Validated existing or newly downloaded proxy outcome.

    Raises
    ------
    ValueError
        If the target or archive layout is unsafe or incomplete.
    requests.RequestException
        If the bounded network request fails.
    zipfile.BadZipFile
        If the response is not a valid ZIP archive.

    Notes
    -----
    The archive is staged and validated before an existing generated target is replaced.
    """
    paths = paths or default_paths()
    target = paths.pd12m
    _assert_safe_target(target, paths.data)
    placeholder_only = _is_placeholder_only_directory(target)
    if target.exists() and not force and not placeholder_only:
        try:
            groups = validate_proxy_dataset(target)
        except ValueError as exc:
            if target.is_dir():
                entries = ", ".join(sorted(path.name for path in target.iterdir())) or "<empty>"
            else:
                entries = "<non-directory target>"
            raise ValueError(
                f"Existing PD12M target is neither a complete proxy nor the tracked placeholder-only "
                f"directory and will not be modified: {target}. Current entries: {entries}."
            ) from exc
        return PreparationReport("existing", f"Validated existing PD12M proxy at {target}.", groups * 4)

    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pd12m-download-", dir=target.parent) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "proxy.zip"
        extracted = temporary / "extracted"
        prepared = temporary / "prepared"
        extracted.mkdir()
        prepared.mkdir()

        with requests.get(
            url,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)

        source_directories = {
            "Cover": _find_variant(extracted, ("Cover",)),
            "JMiPOD": _find_variant(extracted, ("synthetic_JMiPOD", "JMiPOD", "synthetic_nsF5", "nsF5")),
            "JUNIWARD": _find_variant(extracted, ("synthetic_JUNIWARD", "JUNIWARD")),
            "UERD": _find_variant(extracted, ("synthetic_UERD", "UERD")),
        }
        for variant, source in source_directories.items():
            shutil.copytree(source, prepared / variant)
        if placeholder_only:
            shutil.copy2(target / ".gitkeep", prepared / ".gitkeep")
        else:
            (prepared / ".gitkeep").touch()
        groups = validate_proxy_dataset(prepared)

        _install_prepared_proxy(
            prepared,
            target,
            replace_placeholder=placeholder_only and not force,
        )
    return PreparationReport("downloaded", f"Downloaded and validated PD12M proxy at {target}.", groups * 4)


def build_pd12m_reference(
    *,
    paths: ProjectPaths | None = None,
    cover_count: int = 500,
    scan_limit: int = 5_000,
    reference_count: int = 300,
    initial_fetch: int = 10_000,
    batch_size: int = 32,
    seed: int = 42,
    force: bool = False,
    request_timeout: float = 10.0,
) -> PreparationReport:
    """Select and transactionally save an ALASKA2-like PD12M Cover subset.

    Parameters
    ----------
    paths
        Optional project paths that define source references and output target.
    cover_count
        Number of final PD12M Cover images.
    scan_limit
        Maximum shuffled candidate URLs to embed.
    reference_count
        Maximum ALASKA2 Cover references used for similarity.
    initial_fetch
        Maximum streaming rows examined before candidate shuffling.
    batch_size
        Embedding and concurrent fetch batch size.
    seed
        Local selection, filename, and JPEG-quality seed.
    force
        Whether an existing Cover subset may be replaced.
    request_timeout
        Timeout in seconds for each candidate request.

    Returns
    -------
    PreparationReport
        Validated existing or newly generated Cover-subset outcome.

    Raises
    ------
    ValueError
        If counts, target safety, or existing partial data are invalid.
    FileNotFoundError
        If ALASKA2 Cover references are unavailable.
    RuntimeError
        If embedding or transactional image saving cannot reach the requested count.

    Notes
    -----
    Heavy generation dependencies remain lazy and this operation performs network access.
    """
    if min(cover_count, scan_limit, reference_count, initial_fetch, batch_size) <= 0:
        raise ValueError("All count parameters must be positive.")
    paths = paths or default_paths()
    dataset_target = paths.pd12m
    cover_target = dataset_target / "Cover"
    _assert_safe_target(dataset_target, paths.data)
    if cover_target.exists() and not force:
        count = len(_jpg_names(cover_target))
        if count == cover_count:
            return PreparationReport("existing", f"Validated {count} existing PD12M covers.", count)
        raise ValueError(
            f"Existing PD12M Cover set has {count}/{cover_count} JPEG files; "
            "pass force=True to replace it transactionally."
        )

    reference_paths = sorted((paths.alaska2 / "Cover").glob("*.jpg"))
    if not reference_paths:
        raise FileNotFoundError("ALASKA2 Cover images are required for reference-guided selection.")

    # Optional native/model packages do not publish complete Pyright-readable metadata.
    import faiss  # pyright: ignore[reportMissingImports]
    import requests
    import torch
    from transformers import CLIPModel, CLIPProcessor  # pyright: ignore[reportMissingImports]

    # datasets exports this runtime factory dynamically from its package root.
    from datasets import load_dataset  # pyright: ignore[reportAttributeAccessIssue]

    rng = random.Random(seed)
    selected_references = rng.sample(reference_paths, min(reference_count, len(reference_paths)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    cast(torch.nn.Module, model).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    def encode_images(images: list[Image.Image]) -> torch.Tensor:
        processed = processor(
            images=images,
            # The processor stub omits its runtime-supported image-batch overload.
            return_tensors="pt",  # pyright: ignore[reportCallIssue]
        )
        pixel_values = processed.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("CLIP processor did not return a pixel_values tensor.")
        with torch.inference_mode():
            features = model.get_image_features(
                # The model stub is narrower than the processor's validated tensor output.
                pixel_values=pixel_values.to(device),  # pyright: ignore[reportArgumentType]
            )
        if not isinstance(features, torch.Tensor):
            raise TypeError("CLIP model did not return an image-feature tensor.")
        return features.cpu()

    reference_batches: list[torch.Tensor] = []
    for start in range(0, len(selected_references), batch_size):
        images: list[Image.Image] = []
        for path in selected_references[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        reference_batches.append(encode_images(images))
    reference_embeddings = torch.cat(reference_batches).numpy().astype("float32")
    faiss.normalize_L2(reference_embeddings)

    stream = load_dataset("Spawning/PD12M", split="train", streaming=True)
    raw_urls = [row["url"] for row in islice(stream, initial_fetch) if row.get("url")]
    rng.shuffle(raw_urls)
    urls = raw_urls[:scan_limit]
    failures: Counter[str] = Counter()
    clusters: defaultdict[int, list[tuple[float, str]]] = defaultdict(list)

    def fetch(url: str) -> tuple[str, Image.Image] | None:
        try:
            response = requests.get(url, timeout=request_timeout)
            response.raise_for_status()
            with Image.open(BytesIO(response.content)) as image:
                return url, image.convert("RGB")
        except requests.RequestException:
            failures["network"] += 1
        except (OSError, ValueError):
            failures["image_decode"] += 1
        return None

    for start in tqdm(range(0, len(urls), batch_size), desc="Embedding PD12M candidates"):
        batch_urls = urls[start : start + batch_size]
        with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
            fetched = [item for item in executor.map(fetch, batch_urls) if item is not None]
        if not fetched:
            continue
        successful_urls, fetched_images = zip(*fetched, strict=True)
        embeddings = encode_images(list(fetched_images)).numpy().astype("float32")
        faiss.normalize_L2(embeddings)
        similarity = embeddings @ reference_embeddings.T
        for row_index, url in enumerate(successful_urls):
            reference_index = int(similarity[row_index].argmax())
            clusters[reference_index].append((float(similarity[row_index, reference_index]), url))

    if not clusters:
        raise RuntimeError(f"No PD12M candidate embeddings succeeded. Failures: {dict(failures)}")
    per_cluster = max(1, cover_count // len(clusters))
    selected_urls: list[str] = []
    for candidates in clusters.values():
        selected_urls.extend(url for _, url in sorted(candidates, reverse=True)[:per_cluster])
    selected_urls = list(dict.fromkeys(selected_urls))
    if len(selected_urls) < cover_count:
        ranked = sorted(
            (candidate for candidates in clusters.values() for candidate in candidates),
            reverse=True,
        )
        selected_set = set(selected_urls)
        for _, url in ranked:
            if url in selected_set:
                continue
            selected_urls.append(url)
            selected_set.add(url)
            if len(selected_urls) == cover_count:
                break
    selected_urls = selected_urls[:cover_count]
    if len(selected_urls) < cover_count:
        failures["insufficient_candidates"] += cover_count - len(selected_urls)

    dataset_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pd12m-covers-", dir=dataset_target.parent) as temporary_name:
        staged_cover = Path(temporary_name) / "Cover"
        staged_cover.mkdir()
        saved = 0
        for url in tqdm(selected_urls, desc="Saving PD12M covers"):
            try:
                response = requests.get(url, timeout=request_timeout)
                response.raise_for_status()
                with Image.open(BytesIO(response.content)) as source:
                    image = source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
                    image.save(
                        staged_cover / f"{saved + 1:05d}.jpg",
                        format="JPEG",
                        quality=rng.choice((75, 90, 95)),
                    )
                saved += 1
            except requests.RequestException:
                failures["save_network"] += 1
            except (OSError, ValueError):
                failures["save_decode"] += 1
        if saved != cover_count:
            raise RuntimeError(f"Saved {saved}/{cover_count} covers; no partial dataset installed. {dict(failures)}")
        if cover_target.exists():
            shutil.rmtree(cover_target)
        dataset_target.mkdir(parents=True, exist_ok=True)
        staged_cover.replace(cover_target)

    return PreparationReport(
        "generated",
        f"Saved {cover_count} reference-guided PD12M covers at {cover_target}.",
        cover_count,
        dict(failures),
    )


def generate_synthetic_stego(
    *,
    paths: ProjectPaths | None = None,
    difficulty: float = 0.4,
    seed: int = 42,
    force: bool = False,
) -> PreparationReport:
    """Generate JMiPOD-compatible, JUNIWARD, and UERD variants transactionally.

    Parameters
    ----------
    paths
        Optional project paths constraining the generated-data target.
    difficulty
        Embedding payload fraction in the interval ``(0, 1]``.
    seed
        Base seed offset deterministically for each Cover image.
    force
        Whether complete or partial generated variants may be replaced.

    Returns
    -------
    PreparationReport
        Validated existing or newly generated stego outcome.

    Raises
    ------
    ValueError
        If difficulty, target safety, or existing partial variants are invalid.
    FileNotFoundError
        If the PD12M Cover directory has no JPEG files.
    RuntimeError
        If any staged variant fails or source-name parity is incomplete.

    Notes
    -----
    The public JMiPOD compatibility class is technically generated with nsF5.
    """
    if not 0 < difficulty <= 1:
        raise ValueError("difficulty must be in (0, 1].")
    paths = paths or default_paths()
    root = paths.pd12m
    _assert_safe_target(root, paths.data)
    cover_directory = root / "Cover"
    covers = sorted(cover_directory.glob("*.jpg"))
    if not covers:
        raise FileNotFoundError(f"No cover JPEG files found in {cover_directory}")

    variant_directories = {name: root / name for name in _PROXY_VARIANTS[1:]}
    existing_complete = all(
        _jpg_names(directory) == {path.name for path in covers} for directory in variant_directories.values()
    )
    if existing_complete and not force:
        return PreparationReport("existing", f"Validated existing synthetic variants at {root}.", len(covers) * 3)
    if not force and any(directory.exists() for directory in variant_directories.values()):
        raise ValueError("Partial synthetic variants exist; pass force=True to replace them transactionally.")

    # These optional generation packages do not publish Pyright-readable metadata.
    import conseal  # pyright: ignore[reportMissingImports]
    import jpeglib  # pyright: ignore[reportMissingImports]
    import numpy as np

    root.parent.mkdir(parents=True, exist_ok=True)
    failures: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix=".pd12m-stego-", dir=root.parent) as temporary_name:
        staging = Path(temporary_name)
        for variant in variant_directories:
            (staging / variant).mkdir()
        for index, cover_path in enumerate(tqdm(covers, desc="Generating synthetic stego")):
            try:
                spatial = jpeglib.read_spatial(str(cover_path), jpeglib.JCS_GRAYSCALE)
                spatial_values = np.asarray(spatial.spatial)
                jpeg = jpeglib.read_dct(str(cover_path))
                outputs = {
                    "JMiPOD": conseal.nsF5.simulate_single_channel(
                        y0=jpeg.Y,
                        alpha=difficulty,
                        seed=seed + index,
                    ),
                    "JUNIWARD": conseal.juniward.simulate_single_channel(
                        x0=spatial_values[..., 0],
                        y0=jpeg.Y,
                        qt=jpeg.qt[0],
                        alpha=difficulty,
                        seed=seed + index,
                    ),
                    "UERD": conseal.uerd.simulate_single_channel(
                        y0=jpeg.Y,
                        qt=jpeg.qt[0],
                        alpha=difficulty,
                        seed=seed + index,
                    ),
                }
                for variant, coefficients in outputs.items():
                    output_jpeg = jpeg.copy()
                    output_jpeg.Y = np.array(coefficients, copy=True)
                    output_jpeg.write_dct(str(staging / variant / cover_path.name))
            except (OSError, ValueError, RuntimeError, AttributeError) as error:
                failures[type(error).__name__] += 1

        expected_names = {path.name for path in covers}
        staged_names = {variant: _jpg_names(staging / variant) for variant in variant_directories}
        if failures or any(names != expected_names for names in staged_names.values()):
            raise RuntimeError(
                f"Synthetic generation incomplete; existing variants were not modified. "
                f"Counts: { {name: len(values) for name, values in staged_names.items()} }, "
                f"failures: {dict(failures)}"
            )
        root.mkdir(parents=True, exist_ok=True)
        for variant, destination in variant_directories.items():
            if destination.exists():
                shutil.rmtree(destination)
            (staging / variant).replace(destination)

    groups = validate_proxy_dataset(root)
    return PreparationReport(
        "generated",
        f"Generated JMiPOD-compatible, JUNIWARD, and UERD variants for {groups} covers.",
        groups * 3,
    )
