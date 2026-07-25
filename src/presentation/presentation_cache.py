"""Validated, dataset-specific EDA figure caching."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

from matplotlib.figure import Figure
from PIL import Image

from ..config.config_paths import DatasetCacheNamespace, ProjectPaths, default_paths

__all__ = [
    "CACHE_MANIFEST_NAME",
    "CACHE_RENDERER_VERSION",
    "CACHE_SCHEMA_VERSION",
    "figure_path",
    "resolve_cached_figure_path",
    "sanitize_name",
    "save_figure",
]

_UNSAFE_NAME = re.compile(r"[^a-z0-9_.-]+")
CACHE_MANIFEST_NAME: Final = "manifest.json"
CACHE_SCHEMA_VERSION: Final = 1
CACHE_RENDERER_VERSION: Final = "eda-renderer-v1"


def sanitize_name(value: str) -> str:
    """Normalize text into a stable lowercase filename component."""
    normalized = value.lower().replace("\u2013", "-").replace("\u2014", "-")
    result = _UNSAFE_NAME.sub("_", normalized).strip("._")
    if not result:
        raise ValueError("Name does not contain a usable filename component.")
    return result


def _dataset_namespace(dataset_name: str) -> DatasetCacheNamespace:
    namespace = sanitize_name(dataset_name)
    if namespace not in {"alaska2", "pd12m"}:
        raise ValueError("Figure caching requires the resolved dataset namespace 'alaska2' or 'pd12m'.")
    return cast(DatasetCacheNamespace, namespace)


def figure_path(
    dataset_name: str,
    figure_name: str,
    *,
    paths: ProjectPaths | None = None,
    create_parent: bool = False,
) -> Path:
    """Build a PNG path inside the resolved dataset's cache namespace."""
    project_paths = paths or default_paths()
    destination = project_paths.dataset_cache(_dataset_namespace(dataset_name)) / f"{sanitize_name(figure_name)}.png"
    if create_parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _manifest_path(dataset_name: str, *, paths: ProjectPaths | None = None) -> Path:
    project_paths = paths or default_paths()
    return project_paths.dataset_cache(_dataset_namespace(dataset_name)) / CACHE_MANIFEST_NAME


def _normalized_parameters(parameters: Mapping[str, object] | None) -> dict[str, Any]:
    try:
        serialized = json.dumps(dict(parameters or {}), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Cache parameters must be JSON-serializable.") from error
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise ValueError("Cache parameters must serialize to an object.")
    return cast(dict[str, Any], value)


def _cache_fingerprint(
    dataset_name: str,
    figure_name: str,
    *,
    renderer: str,
    parameters: Mapping[str, object] | None,
    seed: int,
    source_groups: int | None,
    image_count: int | None,
) -> str:
    payload = {
        "cache_version": CACHE_SCHEMA_VERSION,
        "dataset_source": _dataset_namespace(dataset_name),
        "image_count": image_count,
        "parameters": _normalized_parameters(parameters),
        "plot_id": sanitize_name(figure_name),
        "renderer": renderer,
        "renderer_version": CACHE_RENDERER_VERSION,
        "seed": seed,
        "source_groups": source_groups,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _compatible_manifest(
    dataset_name: str,
    *,
    paths: ProjectPaths | None,
    seed: int,
    source_groups: int | None,
    image_count: int | None,
) -> dict[str, Any] | None:
    manifest = _read_manifest(_manifest_path(dataset_name, paths=paths))
    if manifest is None:
        return None
    expected = {
        "cache_version": CACHE_SCHEMA_VERSION,
        "dataset_source": _dataset_namespace(dataset_name),
        "image_count": image_count,
        "renderer_version": CACHE_RENDERER_VERSION,
        "seed": seed,
        "source_groups": source_groups,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    return manifest if isinstance(manifest.get("plots"), dict) else None


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            f"{json.dumps(manifest, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_cached_figure_path(
    dataset_name: str,
    figure_name: str,
    *,
    paths: ProjectPaths | None = None,
    renderer: str = "",
    parameters: Mapping[str, object] | None = None,
    seed: int = 42,
    source_groups: int | None = None,
    image_count: int | None = None,
) -> Path | None:
    """Resolve an existing figure only when its manifest entry is valid."""
    runtime = figure_path(dataset_name, figure_name, paths=paths)
    if not runtime.is_file():
        return None
    manifest = _compatible_manifest(
        dataset_name,
        paths=paths,
        seed=seed,
        source_groups=source_groups,
        image_count=image_count,
    )
    if manifest is None:
        return None
    plots = cast(dict[str, Any], manifest["plots"])
    entry = plots.get(runtime.name)
    if not isinstance(entry, dict):
        return None
    fingerprint = _cache_fingerprint(
        dataset_name,
        figure_name,
        renderer=renderer or sanitize_name(figure_name),
        parameters=parameters,
        seed=seed,
        source_groups=source_groups,
        image_count=image_count,
    )
    expected_size = entry.get("bytes")
    try:
        valid_file = (
            isinstance(expected_size, int)
            and runtime.stat().st_size == expected_size
            and entry.get("sha256") == _sha256(runtime)
        )
    except OSError:
        return None
    return runtime if entry.get("fingerprint") == fingerprint and valid_file else None


def save_figure(
    figure: Figure,
    dataset_name: str,
    figure_name: str,
    *,
    paths: ProjectPaths | None = None,
    renderer: str = "",
    parameters: Mapping[str, object] | None = None,
    seed: int = 42,
    source_groups: int | None = None,
    image_count: int | None = None,
) -> Path:
    """Save a PNG and its validity metadata in the resolved dataset cache."""
    destination = figure_path(
        dataset_name,
        figure_name,
        paths=paths,
        create_parent=True,
    )
    figure.savefig(destination, bbox_inches="tight")
    with Image.open(destination) as image:
        image.load()
        width, height = image.size

    manifest_path = _manifest_path(dataset_name, paths=paths)
    manifest = _compatible_manifest(
        dataset_name,
        paths=paths,
        seed=seed,
        source_groups=source_groups,
        image_count=image_count,
    )
    if manifest is None:
        manifest = {
            "cache_version": CACHE_SCHEMA_VERSION,
            "dataset_source": _dataset_namespace(dataset_name),
            "image_count": image_count,
            "plots": {},
            "renderer_version": CACHE_RENDERER_VERSION,
            "seed": seed,
            "source_groups": source_groups,
        }
    plots = cast(dict[str, Any], manifest["plots"])
    normalized_parameters = _normalized_parameters(parameters)
    renderer_name = renderer or sanitize_name(figure_name)
    plots[destination.name] = {
        "bytes": destination.stat().st_size,
        "fingerprint": _cache_fingerprint(
            dataset_name,
            figure_name,
            renderer=renderer_name,
            parameters=normalized_parameters,
            seed=seed,
            source_groups=source_groups,
            image_count=image_count,
        ),
        "height": height,
        "parameters": normalized_parameters,
        "plot_id": sanitize_name(figure_name),
        "renderer": renderer_name,
        "sha256": _sha256(destination),
        "width": width,
    }
    _write_manifest(manifest_path, manifest)
    return destination
