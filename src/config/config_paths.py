"""
===============================================================================
config_paths.py
===============================================================================
Portable project and dataset path configuration.

Responsibilities:
  - Resolve paths from the installed source location instead of the current
    directory.
  - Keep both interchangeable datasets under one project data root while
    separating caches, checkpoints, reports, artifacts, and notebook media.
  - Select ALASKA2 or the public PD12M proxy without silently generating
    data.

Design principles:
  - Paths are immutable values. Directory creation is explicit and restricted
    to runtime output locations.

Boundaries:
  - This module does not inspect image contents, download files, or mutate
    the repository.

Notes:
  - The synthetic JMiPOD compatibility class is generated with nsF5. It
    preserves one public four-class workflow without claiming scientific
    equivalence to ALASKA2 JMiPOD.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

__all__ = [
    "CLASS_LABELS",
    "DatasetSelection",
    "DatasetSource",
    "ProjectPaths",
    "default_paths",
    "select_dataset",
]

DatasetSource = Literal["auto", "alaska2", "synthetic"]

CLASS_LABELS: Final[dict[str, int]] = {
    "Cover": 0,
    "JMiPOD": 1,
    "JUNIWARD": 2,
    "UERD": 3,
}


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Represent every repository-relative input and output location.

    Parameters
    ----------
    root
        Absolute or repository-relative project root.

    Notes
    -----
    Directory properties are side-effect free; only ``create_runtime_directories`` writes directories.
    """

    root: Path

    @property
    def data(self) -> Path:
        """Return the project dataset directory."""
        return self.root / "data"

    @property
    def alaska2(self) -> Path:
        """Return the expected ALASKA2 dataset root."""
        return self.data / "ALASKA2"

    @property
    def pd12m(self) -> Path:
        """Return the public synthetic proxy dataset root."""
        return self.data / "PD12M"

    @property
    def cache(self) -> Path:
        """Return the disposable runtime-cache directory."""
        return self.root / "cache"

    @property
    def checkpoints(self) -> Path:
        """Return the generated checkpoint directory."""
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        """Return the generated report directory."""
        return self.root / "reports"

    @property
    def artifacts(self) -> Path:
        """Return the versioned scientific-artifact directory."""
        return self.root / "artifacts"

    @property
    def notebook_assets(self) -> Path:
        """Return versioned images embedded in the academic notebook."""
        return self.root / "assets" / "notebook"

    def create_runtime_directories(self) -> None:
        """Create the cache, checkpoint, and report directories.

        Returns
        -------
        None
            The method creates missing runtime directories in place.

        Notes
        -----
        Dataset and versioned asset directories are never created here.
        """
        for directory in (self.cache, self.checkpoints, self.reports):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """Describe the explicitly resolved dataset and its public class contract.

    Parameters
    ----------
    name
        Stable machine-readable dataset name.
    display_name
        Human-readable dataset description.
    root
        Resolved dataset root containing the class directories.
    class_labels
        Class-to-integer mapping copied for the caller.
    synthetic
        Whether the selected data is the PD12M proxy.
    """

    name: str
    display_name: str
    root: Path
    class_labels: dict[str, int]
    synthetic: bool


def default_paths() -> ProjectPaths:
    """Build project paths from the installed module location.

    Returns
    -------
    ProjectPaths
        Repository path contract independent of the process working directory.
    """
    return ProjectPaths(root=Path(__file__).resolve().parents[2])


def _contains_jpegs(directory: Path) -> bool:
    return directory.is_dir() and any(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"} for path in directory.iterdir()
    )


def _contains_complete_class_layout(root: Path) -> bool:
    return all(_contains_jpegs(root / class_name) for class_name in CLASS_LABELS)


def select_dataset(
    paths: ProjectPaths | None = None,
    *,
    source: DatasetSource = "auto",
) -> DatasetSelection:
    """Resolve ALASKA2 or the PD12M proxy without downloading or generating data.

    Parameters
    ----------
    paths
        Optional project path contract; the installed repository is used by default.
    source
        Explicit source policy: automatic demo selection, real ALASKA2, or synthetic proxy.

    Returns
    -------
    DatasetSelection
        Resolved dataset identity, root, and class-label contract.

    Raises
    ------
    FileNotFoundError
        If the requested source is incomplete or no automatic source is usable.
    ValueError
        If the source policy is invalid.

    Notes
    -----
    An explicit ``source="alaska2"`` request never falls back to the synthetic proxy.
    """
    if source not in {"auto", "alaska2", "synthetic"}:
        raise ValueError("source must be 'auto', 'alaska2', or 'synthetic'.")
    paths = paths or default_paths()
    alaska2_ready = _contains_complete_class_layout(paths.alaska2)
    synthetic_ready = _contains_complete_class_layout(paths.pd12m)
    if source == "alaska2" and not alaska2_ready:
        raise FileNotFoundError(
            f"Explicit ALASKA2 selection requires populated Cover, JMiPOD, JUNIWARD, and UERD "
            f"directories under {paths.alaska2}. No synthetic fallback was attempted."
        )
    if source == "synthetic" and not synthetic_ready:
        raise FileNotFoundError(f"Explicit synthetic selection requires a complete PD12M proxy under {paths.pd12m}.")
    if alaska2_ready and source in {"auto", "alaska2"}:
        return DatasetSelection(
            name="ALASKA2",
            display_name="ALASKA2",
            root=paths.alaska2,
            class_labels=CLASS_LABELS.copy(),
            synthetic=False,
        )
    if synthetic_ready and source in {"auto", "synthetic"}:
        return DatasetSelection(
            name="PD12M",
            display_name="synthetic PD12M proxy",
            root=paths.pd12m,
            class_labels=CLASS_LABELS.copy(),
            synthetic=True,
        )
    raise FileNotFoundError(
        "No usable dataset found. Place ALASKA2 under data/ALASKA2 "
        "or explicitly download/generate the PD12M proxy under data/PD12M."
    )
