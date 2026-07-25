"""Prepare missing project supplements in an active Google Colab kernel.

The module imports only the Python standard library at module load time. It
reuses Colab's scientific stack and never installs Poetry, project dependency
groups, or replacement NumPy/SciPy/scikit-learn/pandas/Torch packages.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

REPOSITORY_URL: Final = "https://github.com/Rinovative/alaska2-steganalysis.git"
REPOSITORY_IDENTITY: Final = "github.com/rinovative/alaska2-steganalysis"
DEFAULT_REPOSITORY_ROOT: Final = Path("/content/alaska2-steganalysis")
PROTECTED_DISTRIBUTIONS: Final = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "torch",
    "torchvision",
)
PREINSTALLED_COLAB_IMPORTS: Final = (
    "numpy",
    "scipy",
    "scipy.sparse",
    "sklearn",
    "sklearn.metrics",
    "pandas",
    "torch",
    "torchvision",
    "IPython.display",
    "ipywidgets",
    "matplotlib.pyplot",
    "PIL.Image",
    "requests",
    "seaborn",
    "tqdm.auto",
)
JPEGIO_BUILD_IMPORTS: Final = ("Cython", "numpy", "setuptools", "wheel")
REQUIRED_BINARY_IMPORTS: Final = (
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "torch",
    "torchvision",
    "jpegio",
)
REQUIRED_PROJECT_EXPORTS: Final = (
    "config",
    "data",
    "datasets",
    "eda",
    "models",
    "presentation",
    "training",
    "transforms",
)


class ColabSetupError(RuntimeError):
    """Base error for actionable Colab setup failures."""


class InvalidRepositoryError(ColabSetupError):
    """Raised when the Colab target is not the expected repository clone."""


class ImportVerificationError(ColabSetupError):
    """Raised when the active same-kernel environment is incomplete."""


@dataclass(frozen=True)
class SupplementalPackage:
    """Describe one audited package that may be installed without dependencies."""

    module: str
    distribution: str


@dataclass(frozen=True)
class BootstrapResult:
    """Summarize one no-restart bootstrap invocation."""

    state: Literal["not-colab", "ready"]
    repository: Path | None = None
    installed: tuple[str, ...] = ()


SUPPLEMENTAL_PACKAGES: Final = (SupplementalPackage(module="jpegio", distribution="jpegio"),)


def is_colab() -> bool:
    """Return whether the active interpreter belongs to Google Colab."""

    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def _canonical_remote(remote: str) -> str:
    value = remote.strip().removesuffix("/").removesuffix(".git")
    if value.startswith("git@github.com:"):
        value = f"github.com/{value.removeprefix('git@github.com:')}"
    elif "://" in value:
        parsed = urlsplit(value)
        value = f"{parsed.hostname or ''}/{parsed.path.lstrip('/')}"
    return value.casefold()


def _validate_repository(repository: Path) -> None:
    if not repository.is_dir() or not (repository / ".git").exists():
        raise InvalidRepositoryError(
            f"{repository} exists, but it is not a Git clone. "
            "Move it aside or choose a fresh Colab runtime; the setup will not overwrite it."
        )

    try:
        inside_work_tree = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
        origin = subprocess.run(
            ["git", "-C", str(repository), "config", "--get", "remote.origin.url"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InvalidRepositoryError(
            f"{repository} is not a readable clone of {REPOSITORY_URL}. The existing directory was left unchanged."
        ) from exc

    if inside_work_tree.stdout.strip() != "true" or _canonical_remote(origin.stdout) != REPOSITORY_IDENTITY:
        raise InvalidRepositoryError(
            f"{repository} is a Git repository, but its origin is not {REPOSITORY_URL}. "
            "The existing directory was left unchanged."
        )


def prepare_repository(repository: Path = DEFAULT_REPOSITORY_ROOT) -> Path:
    """Clone the main branch when absent and safely reuse only the expected clone."""

    if not repository.exists():
        repository.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                "--single-branch",
                REPOSITORY_URL,
                str(repository),
            ],
            check=True,
        )
    _validate_repository(repository)
    return repository


def _require_imports(module_names: tuple[str, ...], *, context: str) -> None:
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise ImportVerificationError(f"{context}: importing {module_name!r} failed: {exc}") from exc


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return False
        raise ImportVerificationError(
            f"{module_name!r} is present but one of its imports is missing: {exc.name!r}."
        ) from exc
    except Exception as exc:
        raise ImportVerificationError(f"Existing module {module_name!r} cannot be imported: {exc}") from exc
    return True


def _protected_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PROTECTED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportVerificationError(
                f"The standard Colab distribution {distribution!r} is missing; no replacement was attempted."
            ) from exc
    return versions


def validate_colab_stack() -> dict[str, str]:
    """Import and snapshot the scientific packages supplied by standard Colab."""

    _require_imports(
        PREINSTALLED_COLAB_IMPORTS,
        context="The standard Colab package stack is incomplete",
    )
    return _protected_versions()


def _locked_requirement(repository: Path, distribution: str) -> str:
    try:
        lock = tomllib.loads((repository / "poetry.lock").read_text(encoding="utf-8"))
        versions = {
            package["version"]
            for package in lock["package"]
            if package.get("name") == distribution and "main" in package.get("groups", ())
        }
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        raise ColabSetupError(f"Cannot read {distribution!r} from poetry.lock.") from exc
    if len(versions) != 1:
        raise ColabSetupError(
            f"Expected exactly one locked main version for {distribution!r}, found {sorted(versions)}."
        )
    return f"{distribution}=={versions.pop()}"


def missing_supplements() -> tuple[SupplementalPackage, ...]:
    """Return only audited notebook supplements that are not importable."""

    return tuple(package for package in SUPPLEMENTAL_PACKAGES if not _module_available(package.module))


def install_missing_supplements(repository: Path) -> tuple[str, ...]:
    """Install missing audited supplements without resolving dependencies."""

    plan = missing_supplements()
    if not plan:
        return ()

    _require_imports(
        JPEGIO_BUILD_IMPORTS,
        context=(
            "jpegio must be built against the already loaded NumPy stack, but a standard Colab build tool is missing"
        ),
    )
    versions_before = _protected_versions()
    installed: list[str] = []
    for package in plan:
        requirement = _locked_requirement(repository, package.distribution)
        print(f"Installing missing Colab supplement without dependencies: {requirement}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                requirement,
            ],
            check=True,
            cwd=repository,
        )
        importlib.invalidate_caches()
        _require_imports(
            (package.module,),
            context=f"The installed Colab supplement {package.distribution!r} is unusable",
        )
        installed.append(package.distribution)

    versions_after = _protected_versions()
    if versions_after != versions_before:
        raise ColabSetupError(
            "A protected Colab package changed unexpectedly despite the dependency-free install: "
            f"before={versions_before}, after={versions_after}."
        )
    return tuple(installed)


def verify_runtime_imports() -> None:
    """Verify all binary and project imports immediately in the same kernel."""

    _require_imports(
        REQUIRED_BINARY_IMPORTS,
        context="Same-kernel binary import verification failed",
    )
    try:
        project = importlib.import_module("src")
        for export in REQUIRED_PROJECT_EXPORTS:
            getattr(project, export)
    except Exception as exc:
        raise ImportVerificationError(f"Same-kernel project import verification failed: {exc}") from exc


def bootstrap_colab(
    repository: Path = DEFAULT_REPOSITORY_ROOT,
    *,
    colab: bool | None = None,
) -> BootstrapResult:
    """Prepare and verify Colab once without replacing its stack or restarting."""

    if not (is_colab() if colab is None else colab):
        return BootstrapResult(state="not-colab")

    prepared_repository = prepare_repository(repository)
    os.chdir(prepared_repository)
    protected_before = validate_colab_stack()
    installed = install_missing_supplements(prepared_repository)
    verify_runtime_imports()
    protected_after = _protected_versions()
    if protected_after != protected_before:
        raise ColabSetupError(
            "The protected Colab package versions changed during setup: "
            f"before={protected_before}, after={protected_after}."
        )

    if installed:
        print(f"Colab setup complete in the active kernel; installed: {', '.join(installed)}.")
    else:
        print("Colab setup already complete; no packages were installed.")
    return BootstrapResult(
        state="ready",
        repository=prepared_repository,
        installed=installed,
    )


def main() -> None:
    """Run the bootstrap when invoked by the notebook setup cell."""

    result = bootstrap_colab()
    if result.state == "not-colab":
        print("Not running in Google Colab; no repository or environment changes were made.")


if __name__ == "__main__":
    main()
