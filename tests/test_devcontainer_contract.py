"""
===============================================================================
test_devcontainer_contract.py
===============================================================================
Protect the single-interpreter and least-surprise Dev Container contract.

Responsibilities:
  - Keep Python, Poetry, editor tooling, and Codex on the intended container paths.
  - Preserve the standard workspace, GPU, IPC, port, and mount boundaries.
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
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
DEVCONTAINER = ROOT / ".devcontainer" / "devcontainer.json"
DOCKERFILE = ROOT / ".devcontainer" / "Dockerfile"


def _devcontainer() -> dict[str, Any]:
    return json.loads(DEVCONTAINER.read_text(encoding="utf-8"))


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
