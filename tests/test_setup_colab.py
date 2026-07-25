"""Exercise the no-restart Colab bootstrap without network or real installation."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import setup_colab

ROOT = Path(__file__).parents[1]
PROTECTED_VERSIONS = {
    "numpy": "colab-numpy",
    "scipy": "colab-scipy",
    "scikit-learn": "colab-sklearn",
    "pandas": "colab-pandas",
    "torch": "colab-torch",
    "torchvision": "colab-torchvision",
}


def _completed(command: list[str], *, stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def _fake_git_runner(
    commands: list[list[str]],
    repository: Path,
    *,
    origin: str = setup_colab.REPOSITORY_URL,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs["check"] is True
        if command[:2] == ["git", "clone"]:
            repository.mkdir(parents=True)
            (repository / ".git").mkdir()
            return _completed(command)
        if command[-2:] == ["rev-parse", "--is-inside-work-tree"]:
            return _completed(command, stdout="true\n")
        if command[-3:] == ["config", "--get", "remote.origin.url"]:
            return _completed(command, stdout=f"{origin}\n")
        raise AssertionError(f"Unexpected subprocess call: {command}")

    return run


class _ImportEnvironment:
    def __init__(self, *missing: str) -> None:
        self.missing = set(missing)
        self.imports: list[str] = []

    def __call__(self, module_name: str) -> object:
        self.imports.append(module_name)
        if module_name in self.missing:
            raise ModuleNotFoundError(f"No module named {module_name!r}", name=module_name)
        if module_name == "src":
            return SimpleNamespace(**{name: object() for name in setup_colab.REQUIRED_PROJECT_EXPORTS})
        return SimpleNamespace(__name__=module_name)


def _stable_protected_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_colab, "_protected_versions", lambda: PROTECTED_VERSIONS.copy())


def test_non_colab_environment_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "must-not-be-created"
    original_directory = Path.cwd()

    def unexpected_action(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Non-Colab setup attempted a side effect")

    monkeypatch.setattr(setup_colab, "prepare_repository", unexpected_action)
    monkeypatch.setattr(setup_colab, "validate_colab_stack", unexpected_action)
    monkeypatch.setattr(setup_colab, "install_missing_supplements", unexpected_action)

    result = setup_colab.bootstrap_colab(repository, colab=False)

    assert result.state == "not-colab"
    assert Path.cwd() == original_directory
    assert not repository.exists()


def test_missing_repository_is_shallow_cloned_from_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "content" / "alaska2-steganalysis"
    commands: list[list[str]] = []
    monkeypatch.setattr(setup_colab.subprocess, "run", _fake_git_runner(commands, repository))

    prepared = setup_colab.prepare_repository(repository)

    assert prepared == repository
    assert commands[0] == [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        "main",
        "--single-branch",
        setup_colab.REPOSITORY_URL,
        str(repository),
    ]
    assert len(commands) == 3


def test_valid_existing_repository_is_reused_without_clone_or_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "alaska2-steganalysis"
    (repository / ".git").mkdir(parents=True)
    commands: list[list[str]] = []
    monkeypatch.setattr(setup_colab.subprocess, "run", _fake_git_runner(commands, repository))

    prepared = setup_colab.prepare_repository(repository)

    assert prepared == repository
    assert len(commands) == 2
    assert all("clone" not in command for command in commands)
    assert all("pull" not in command and "reset" not in command for command in commands)


def test_invalid_existing_target_is_left_unchanged_with_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "alaska2-steganalysis"
    repository.mkdir()
    sentinel = repository / "user-file.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    def unexpected_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Invalid target must not trigger Git")

    monkeypatch.setattr(setup_colab.subprocess, "run", unexpected_subprocess)

    with pytest.raises(setup_colab.InvalidRepositoryError, match="will not overwrite"):
        setup_colab.prepare_repository(repository)

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_importable_supplements_are_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    imports = _ImportEnvironment()
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def unexpected_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("An importable supplement must not be installed")

    monkeypatch.setattr(setup_colab.subprocess, "run", unexpected_subprocess)

    assert setup_colab.install_missing_supplements(ROOT) == ()
    assert imports.imports == ["jpegio"]


def test_only_missing_jpegio_is_installed_without_dependencies_or_poetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _ImportEnvironment("jpegio")
    calls: list[tuple[list[str], dict[str, Any]]] = []
    _stable_protected_versions(monkeypatch)
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        imports.missing.discard("jpegio")
        return _completed(command)

    monkeypatch.setattr(setup_colab.subprocess, "run", fake_run)

    installed = setup_colab.install_missing_supplements(ROOT)

    assert installed == ("jpegio",)
    assert len(calls) == 1
    command, options = calls[0]
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--no-deps" in command
    assert "--no-build-isolation" in command
    assert command[-1].startswith("jpegio==")
    assert options == {"check": True, "cwd": ROOT}
    forbidden = {"numpy", "scipy", "scikit-learn", "sklearn", "pandas", "torch", "torchvision", "poetry"}
    assert forbidden.isdisjoint(command)


def test_second_setup_call_performs_no_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _ImportEnvironment("jpegio")
    calls: list[list[str]] = []
    repository = ROOT
    _stable_protected_versions(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_colab, "prepare_repository", lambda _repository: repository)
    monkeypatch.setattr(setup_colab, "validate_colab_stack", lambda: PROTECTED_VERSIONS.copy())
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        imports.missing.discard("jpegio")
        return _completed(command)

    monkeypatch.setattr(setup_colab.subprocess, "run", fake_run)

    first = setup_colab.bootstrap_colab(repository, colab=True)
    second = setup_colab.bootstrap_colab(repository, colab=True)

    assert first.state == second.state == "ready"
    assert first.installed == ("jpegio",)
    assert second.installed == ()
    assert len(calls) == 1


def test_missing_protected_colab_package_is_never_added_to_install_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _ImportEnvironment("numpy")
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def unexpected_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("A protected package must never be installed")

    monkeypatch.setattr(setup_colab.subprocess, "run", unexpected_subprocess)

    with pytest.raises(setup_colab.ImportVerificationError, match="standard Colab package stack"):
        setup_colab.validate_colab_stack()
    assert {package.distribution for package in setup_colab.SUPPLEMENTAL_PACKAGES} == {"jpegio"}


def test_bootstrap_returns_in_same_process_without_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    events: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup_colab, "prepare_repository", lambda _repository: repository)
    monkeypatch.setattr(setup_colab, "validate_colab_stack", lambda: PROTECTED_VERSIONS.copy())
    monkeypatch.setattr(setup_colab, "install_missing_supplements", lambda _repository: ())
    monkeypatch.setattr(setup_colab, "verify_runtime_imports", lambda: events.append("verified"))
    _stable_protected_versions(monkeypatch)

    def unexpected_kill(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Colab setup must not terminate a process")

    monkeypatch.setattr(setup_colab.os, "kill", unexpected_kill)

    result = setup_colab.bootstrap_colab(repository, colab=True)

    assert result.state == "ready"
    assert result.installed == ()
    assert events == ["verified"]


def test_numpy_can_be_loaded_before_setup_and_all_imports_work_after_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports = _ImportEnvironment("jpegio")
    calls: list[list[str]] = []
    _stable_protected_versions(monkeypatch)
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        imports.missing.discard("jpegio")
        return _completed(command)

    monkeypatch.setattr(setup_colab.subprocess, "run", fake_run)

    setup_colab.validate_colab_stack()
    setup_colab.install_missing_supplements(ROOT)
    setup_colab.verify_runtime_imports()

    assert imports.imports[0] == "numpy"
    assert calls
    assert calls[0][-1].startswith("jpegio==")
    assert all(name in imports.imports for name in (*setup_colab.REQUIRED_BINARY_IMPORTS, "src"))


def test_install_failure_is_propagated_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    imports = _ImportEnvironment("jpegio")
    failure = subprocess.CalledProcessError(1, [sys.executable, "-m", "pip", "install"])
    _stable_protected_versions(monkeypatch)
    monkeypatch.setattr(setup_colab.importlib, "import_module", imports)

    def fail_install(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(setup_colab.subprocess, "run", fail_install)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        setup_colab.install_missing_supplements(ROOT)

    assert raised.value is failure
