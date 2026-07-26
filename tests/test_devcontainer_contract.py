"""
===============================================================================
test_devcontainer_contract.py
===============================================================================
Protect the single-interpreter and least-surprise Dev Container contract.

Responsibilities:
  - Keep Python, Poetry, editor tooling, and Codex on the intended container paths.
  - Preserve the standard workspace, GPU, IPC, port, and mount boundaries.
  - Remove the legacy workspace only after explicit safety checks.
  - Keep dependency caches out of persistent image and container layers.
  - Keep the host-local Poetry environment policy separate from the container.

Design principles:
  - Parse structured configuration instead of relying on formatting or key order.
  - Assert security-relevant absences alongside required positive configuration.

Boundaries:
  - These tests validate repository configuration, not a running Docker daemon.
  - Runtime GPU, authentication, and interpreter checks remain post-rebuild steps.
===============================================================================
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
DOCKERFILE = ROOT / ".devcontainer" / "Dockerfile"
LEGACY_WORKSPACE = re.compile(r"/workspace(?!s)")


def _devcontainer() -> dict[str, Any]:
    return json.loads(DEVCONTAINER.read_text(encoding="utf-8"))


def _dockerfile_instructions() -> list[str]:
    instructions: list[str] = []
    current: list[str] = []

    for raw_line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or (stripped.startswith("#") and not current):
            continue
        current.append(stripped.removesuffix("\\").rstrip())
        if not stripped.endswith("\\"):
            instructions.append(" ".join(current))
            current = []

    assert not current, "Dockerfile ends with an unterminated line continuation"
    return instructions


def test_container_uses_one_python_environment_without_host_venv_discovery() -> None:
    config = _devcontainer()
    container_env = config["containerEnv"]
    vscode = config["customizations"]["vscode"]
    settings = vscode["settings"]

    assert container_env == {
        "POETRY_VIRTUALENVS_CREATE": "false",
        "POETRY_VIRTUALENVS_IN_PROJECT": "false",
    }
    assert settings["python.defaultInterpreterPath"] == "/opt/conda/bin/python"
    assert settings["python-envs.workspaceSearchPaths"] == []
    assert settings["python-envs.terminal.autoActivationType"] == "off"

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "PIP_NO_CACHE_DIR=1" in dockerfile
    assert "POETRY_VIRTUALENVS_CREATE=false" in dockerfile
    assert "POETRY_VIRTUALENVS_IN_PROJECT=false" in dockerfile


def test_local_poetry_policy_remains_in_project_and_venv_is_ignored() -> None:
    poetry_config = tomllib.loads((ROOT / "poetry.toml").read_text(encoding="utf-8"))
    assert poetry_config["virtualenvs"]["in-project"] is True
    assert ".venv/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".venv/" in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_official_codex_extension_is_the_only_codex_request() -> None:
    extensions = _devcontainer()["customizations"]["vscode"]["extensions"]
    codex_extensions = [
        extension
        for extension in extensions
        if any(term in extension.casefold() for term in ("codex", "chatgpt", "openai"))
    ]
    assert codex_extensions == ["openai.chatgpt"]
    assert len(extensions) == len(set(extensions))


def test_workspace_gpu_ipc_ports_and_mounts_remain_narrow() -> None:
    config = _devcontainer()
    assert config["workspaceFolder"] == "/workspaces/alaska2-steganalysis"
    assert "containerName" not in config
    assert "workspaceMount" not in config
    assert "mounts" not in config
    assert "forwardPorts" not in config
    assert "portsAttributes" not in config
    assert "appPort" not in config
    assert config["runArgs"] == ["--gpus", "all", "--ipc", "host"]


def test_container_starts_no_jupyter_server() -> None:
    config_text = DEVCONTAINER.read_text(encoding="utf-8").casefold()
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8").casefold()
    forbidden = ("jupyter notebook", "jupyter lab", "--ip=", "--allow-root")
    assert not any(command in config_text for command in forbidden)
    assert not any(command in dockerfile_text for command in forbidden)


def test_dockerfile_removes_legacy_workspace_only_after_safety_checks() -> None:
    instructions = _dockerfile_instructions()
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    workspace_run_indexes = [
        index
        for index, instruction in enumerate(instructions)
        if instruction.startswith("RUN ") and "rmdir /workspace" in instruction
    ]

    assert len(workspace_run_indexes) == 1
    workspace_run_index = workspace_run_indexes[0]
    workspace_run = instructions[workspace_run_index]
    workdirs = [
        (index, instruction) for index, instruction in enumerate(instructions) if instruction.startswith("WORKDIR ")
    ]
    assert [instruction for _, instruction in workdirs] == [
        "WORKDIR /",
        "WORKDIR /workspaces/alaska2-steganalysis",
    ]
    root_workdir_index = workdirs[0][0]
    project_workdir_index = workdirs[1][0]
    assert root_workdir_index < workspace_run_index < project_workdir_index
    assert not any(LEGACY_WORKSPACE.search(instruction) for instruction in instructions[workspace_run_index + 1 :])

    required_checks = (
        "apt-get install -y --no-install-recommends",
        "[ ! -d /workspace ]",
        "[ -L /workspace ]",
        "command -v mountpoint",
        "mountpoint -q /workspace",
        "find /workspace -mindepth 1 -maxdepth 1 -print -quit",
    )
    assert all(check in workspace_run for check in required_checks)
    assert (
        workspace_run.index("apt-get install -y --no-install-recommends")
        < workspace_run.index("util-linux")
        < workspace_run.index("[ ! -d /workspace ]")
        < workspace_run.index("command -v mountpoint")
        < workspace_run.index("mountpoint -q /workspace")
        < workspace_run.index("find /workspace")
        < workspace_run.index("rmdir /workspace")
    )
    assert "rm -rf /workspace" not in dockerfile
    assert "mkdir /workspace" not in dockerfile


def test_docker_build_installs_without_persistent_dependency_caches() -> None:
    install_runs = [
        instruction
        for instruction in _dockerfile_instructions()
        if instruction.startswith("RUN ") and "poetry install" in instruction
    ]

    assert len(install_runs) == 1
    install_run = install_runs[0]
    assert "--with dev,notebook,optimization,generation" in install_run
    assert "--extras cuda" in install_run
    assert "--no-root" in install_run
    assert "--no-cache" in install_run
    cache_cleanup = "rm -rf /root/.cache/pip /root/.cache/pypoetry"
    assert cache_cleanup in install_run
    assert install_run.index("poetry install") < install_run.index(f"&& {cache_cleanup}")


def test_post_create_installs_without_retaining_dependency_caches() -> None:
    post_create = _devcontainer()["postCreateCommand"]
    assert "--with dev,notebook,optimization,generation" in post_create
    assert "--extras cuda" in post_create
    assert "--no-cache" in post_create
    cache_cleanup = "rm -rf /root/.cache/pip /root/.cache/pypoetry"
    assert cache_cleanup in post_create
    assert post_create.index("poetry install") < post_create.index(f"&& {cache_cleanup}")
    assert "||" not in post_create


def test_container_setup_does_not_copy_or_recreate_private_data() -> None:
    setup_text = "\n".join(
        (
            DOCKERFILE.read_text(encoding="utf-8"),
            DEVCONTAINER.read_text(encoding="utf-8"),
        )
    )
    assert "/datasets" not in setup_text
    assert "/artifacts" not in setup_text
    assert "ALASKA2" not in setup_text
    assert "mkdir /workspace" not in setup_text
    copy_instructions = [instruction for instruction in _dockerfile_instructions() if instruction.startswith("COPY ")]
    assert copy_instructions == ["COPY pyproject.toml poetry.lock ./"]

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in dockerignore
    assert "artifacts/" in dockerignore
