"""
===============================================================================
test_evaluation_and_synthetic.py
===============================================================================
Verify one-pass evaluation and safe synthetic-data preparation contracts.

Responsibilities:
  - Prove evaluation consumes each sample once while producing binary
    metrics.
  - Exercise proxy state classification, transactional installation, and safe ZIP extraction.

Design principles:
  - Fixtures remain minimal and deterministic.
  - Optional generation dependencies are avoided unless a short-circuit is
    proven.

Boundaries:
  - Network downloads and synthetic image generation are not executed.
  - Metric golden values are covered by test_metrics.py.
===============================================================================
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
import requests
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config.config_paths import ProjectPaths
from src.data import data_synthetic
from src.evaluation.evaluation_runner import evaluate_binary_model


class CountingDataset(Dataset):
    def __init__(self) -> None:
        self.calls = 0
        self.inputs = torch.tensor([[-2.0], [2.0]])
        self.labels = torch.tensor([0.0, 1.0])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        self.calls += 1
        return self.inputs[index], self.labels[index]


def test_evaluation_collects_metrics_in_one_loader_pass() -> None:
    dataset = CountingDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.zero_()
    result = evaluate_binary_model(model, loader, device="cpu", criterion=nn.BCEWithLogitsLoss())
    assert dataset.calls == len(dataset)
    assert result.accuracy == 1.0
    assert result.weighted_auc == 1.0
    assert result.loss is not None


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "grey").save(path, format="JPEG")


def _write_complete_proxy(root: Path, *, source_name: str = "00001.jpg") -> None:
    for variant in ("Cover", "JMiPOD", "JUNIWARD", "UERD"):
        _write_jpeg(root / variant / source_name)


def _proxy_archive(*, source_name: str = "00001.jpg") -> bytes:
    variants = {
        "Cover": "Cover",
        "JMiPOD": "synthetic_JMiPOD",
        "JUNIWARD": "synthetic_JUNIWARD",
        "UERD": "synthetic_UERD",
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for directory in variants.values():
            archive.writestr(f"proxy/{directory}/{source_name}", b"fixture-jpeg-bytes")
    return buffer.getvalue()


class _ArchiveResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _ArchiveResponse:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int) -> tuple[bytes, ...]:
        assert chunk_size == 1024 * 1024
        return (self.payload,)


def _mock_proxy_download(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_name: str = "00001.jpg",
) -> list[tuple[str, bool, tuple[float, float]]]:
    calls: list[tuple[str, bool, tuple[float, float]]] = []
    payload = _proxy_archive(source_name=source_name)

    def fake_get(
        url: str,
        *,
        stream: bool,
        timeout: tuple[float, float],
    ) -> _ArchiveResponse:
        calls.append((url, stream, timeout))
        return _ArchiveResponse(payload)

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def _tree_snapshot(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


def _assert_no_download_staging(paths: ProjectPaths) -> None:
    assert not list(paths.data.glob(".pd12m-download-*"))


def test_proxy_validation_requires_jmipod_compatibility_directory(tmp_path: Path) -> None:
    for variant in ("Cover", "JMiPOD", "JUNIWARD", "UERD"):
        _write_jpeg(tmp_path / variant / "00001.jpg")
    assert data_synthetic.validate_proxy_dataset(tmp_path) == 1
    (tmp_path / "JMiPOD" / "00001.jpg").unlink()
    with pytest.raises(ValueError, match="empty"):
        data_synthetic.validate_proxy_dataset(tmp_path)


def test_missing_proxy_target_follows_transactional_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    calls = _mock_proxy_download(monkeypatch)

    report = data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert report.status == "downloaded"
    assert report.saved_images == 4
    assert len(calls) == 1
    assert data_synthetic.validate_proxy_dataset(paths.pd12m) == 1
    assert (paths.pd12m / ".gitkeep").is_file()
    _assert_no_download_staging(paths)


def test_placeholder_only_target_installs_and_retains_gitkeep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    placeholder = paths.pd12m / ".gitkeep"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_bytes(b"tracked-placeholder\n")
    calls = _mock_proxy_download(monkeypatch)

    report = data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert report.status == "downloaded"
    assert len(calls) == 1
    assert data_synthetic.validate_proxy_dataset(paths.pd12m) == 1
    assert placeholder.read_bytes() == b"tracked-placeholder\n"
    _assert_no_download_staging(paths)


def test_existing_complete_proxy_is_reused_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    _write_complete_proxy(paths.pd12m)

    def unexpected_get(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Complete proxy must not trigger a download.")

    monkeypatch.setattr(requests, "get", unexpected_get)
    download_report = data_synthetic.download_pd12m_proxy(paths=paths)
    generation_report = data_synthetic.generate_synthetic_stego(paths=paths)

    assert download_report.status == "existing"
    assert download_report.saved_images == 4
    assert generation_report.status == "existing"


@pytest.mark.parametrize("entry_kind", ["file", "class_directory"])
def test_incomplete_non_placeholder_target_fails_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    paths = ProjectPaths(tmp_path)
    unexpected = paths.pd12m / "notes.bin" if entry_kind == "file" else paths.pd12m / "Cover" / "00001.jpg"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b"user-owned-content")
    before = _tree_snapshot(paths.pd12m)

    def unexpected_get(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Incomplete non-placeholder target must not trigger a download.")

    monkeypatch.setattr(requests, "get", unexpected_get)
    with pytest.raises(ValueError, match="neither a complete proxy nor the tracked placeholder") as exc_info:
        data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert unexpected.parts[-2 if entry_kind == "class_directory" else -1] in str(exc_info.value)
    assert _tree_snapshot(paths.pd12m) == before
    _assert_no_download_staging(paths)


def test_download_failure_preserves_original_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    placeholder = paths.pd12m / ".gitkeep"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_bytes(b"tracked-placeholder\n")
    before = _tree_snapshot(paths.pd12m)

    def failing_get(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise requests.RequestException("simulated download failure")

    monkeypatch.setattr(requests, "get", failing_get)
    with pytest.raises(requests.RequestException, match="simulated download failure"):
        data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert _tree_snapshot(paths.pd12m) == before
    _assert_no_download_staging(paths)


def test_archive_preparation_failure_preserves_original_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    placeholder = paths.pd12m / ".gitkeep"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_bytes(b"tracked-placeholder")
    before = _tree_snapshot(paths.pd12m)

    def invalid_archive_get(
        url: str,
        *,
        stream: bool,
        timeout: tuple[float, float],
    ) -> _ArchiveResponse:
        del url, stream, timeout
        return _ArchiveResponse(b"not-a-zip-archive")

    monkeypatch.setattr(requests, "get", invalid_archive_get)
    with pytest.raises(zipfile.BadZipFile):
        data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert _tree_snapshot(paths.pd12m) == before
    _assert_no_download_staging(paths)


def test_final_installation_failure_restores_placeholder_without_partial_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    placeholder = paths.pd12m / ".gitkeep"
    placeholder.parent.mkdir(parents=True)
    placeholder.write_bytes(b"tracked-placeholder\n")
    before = _tree_snapshot(paths.pd12m)
    _mock_proxy_download(monkeypatch)
    original_replace = Path.replace

    def failing_prepared_replace(source: Path, target: str | Path) -> Path:
        destination = Path(target)
        if source.name == "prepared" and destination == paths.pd12m:
            raise OSError("simulated final installation failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", failing_prepared_replace)
    with pytest.raises(OSError, match="original placeholder was restored"):
        data_synthetic.download_pd12m_proxy(paths=paths, force=False)

    assert _tree_snapshot(paths.pd12m) == before
    assert not any((paths.pd12m / variant).exists() for variant in ("Cover", "JMiPOD", "JUNIWARD", "UERD"))
    _assert_no_download_staging(paths)


def test_force_replacement_behavior_remains_transactional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = ProjectPaths(tmp_path)
    _write_complete_proxy(paths.pd12m, source_name="old.jpg")
    calls = _mock_proxy_download(monkeypatch, source_name="new.jpg")

    report = data_synthetic.download_pd12m_proxy(paths=paths, force=True)

    assert report.status == "downloaded"
    assert len(calls) == 1
    assert data_synthetic.validate_proxy_dataset(paths.pd12m) == 1
    assert all((paths.pd12m / variant / "new.jpg").is_file() for variant in ("Cover", "JMiPOD", "JUNIWARD", "UERD"))
    assert not any((paths.pd12m / variant / "old.jpg").exists() for variant in ("Cover", "JMiPOD", "JUNIWARD", "UERD"))
    assert (paths.pd12m / ".gitkeep").is_file()
    _assert_no_download_staging(paths)


def test_partial_cover_selection_requires_explicit_replacement(tmp_path: Path) -> None:
    paths = ProjectPaths(tmp_path)
    _write_jpeg(paths.pd12m / "Cover" / "00001.jpg")
    with pytest.raises(ValueError, match="1/2"):
        data_synthetic.build_pd12m_reference(paths=paths, cover_count=2)


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="Unsafe ZIP"):
        data_synthetic._safe_extract(archive, tmp_path / "output")
    assert not (tmp_path.parent / "escape.txt").exists()
