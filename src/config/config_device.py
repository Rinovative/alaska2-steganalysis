"""
===============================================================================
config_device.py
===============================================================================
Validate CUDA availability and resolve explicit PyTorch devices.

Responsibilities:
  - Reject unavailable CUDA devices instead of silently selecting the CPU.
  - Verify the expected CUDA runtime, visible GPU, and a CUDA tensor operation.
  - Confirm that the installed project package is importable in the same interpreter.

Design principles:
  - Accelerator requirements are explicit at every training and evaluation boundary.
  - The preflight performs a minimal deterministic smoke test without loading data.

Boundaries:
  - Container runtime and NVIDIA driver installation remain user responsibilities.
  - Model construction and training belong to the models and training packages.
===============================================================================
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import version as torch_version

__all__ = [
    "EXPECTED_CUDA_RUNTIME",
    "GPUPreflightError",
    "GPUPreflightReport",
    "main",
    "resolve_device",
    "run_gpu_preflight",
]

EXPECTED_CUDA_RUNTIME: Final[str] = "12.1"
_PREFLIGHT_COMMAND: Final[str] = "poetry run alaska2-gpu-preflight"


class GPUPreflightError(RuntimeError):
    """Raised when the required CUDA training environment is unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class GPUPreflightReport:
    """Describe a successfully validated CUDA environment.

    Parameters
    ----------
    torch_version
        Installed PyTorch version string.
    cuda_runtime
        CUDA runtime version reported by PyTorch.
    device_name
        Human-readable name of the first visible CUDA device.
    tensor_result
        Scalar result of the deterministic CUDA tensor smoke operation.
    package_path
        Imported location of the installed ``src`` package.
    """

    torch_version: str
    cuda_runtime: str
    device_name: str
    tensor_result: float
    package_path: Path

    def format_summary(self) -> str:
        """Format the validated environment as a terminal-readable summary.

        Returns
        -------
        str
            Multi-line version, runtime, device, tensor, and package report.
        """
        return "\n".join(
            (
                "GPU preflight passed.",
                f"Torch version: {self.torch_version}",
                f"CUDA runtime: {self.cuda_runtime}",
                f"GPU: {self.device_name}",
                f"CUDA tensor result: {self.tensor_result:.1f}",
                f"src package: {self.package_path}",
            )
        )


def _unavailable_message() -> str:
    return (
        "CUDA was explicitly requested but torch.cuda.is_available() is false. "
        "Reopen the repository in the GPU Dev Container, verify the NVIDIA driver and "
        "Docker GPU passthrough, then rerun: "
        f"{_PREFLIGHT_COMMAND}"
    )


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a PyTorch device without silently downgrading CUDA to CPU.

    Parameters
    ----------
    device
        Explicit PyTorch device name or object.

    Returns
    -------
    torch.device
        Validated device object.

    Raises
    ------
    GPUPreflightError
        If a CUDA device was requested but CUDA is unavailable.
    """
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise GPUPreflightError(_unavailable_message())
    return resolved


def run_gpu_preflight(
    *,
    expected_cuda_runtime: str = EXPECTED_CUDA_RUNTIME,
) -> GPUPreflightReport:
    """Validate the CUDA runtime, first GPU, tensor execution, and project import.

    Parameters
    ----------
    expected_cuda_runtime
        Exact CUDA runtime version expected from the project container.

    Returns
    -------
    GPUPreflightReport
        Structured details from a successful smoke test.

    Raises
    ------
    GPUPreflightError
        If CUDA is unavailable, the runtime differs, no GPU is visible, the tensor
        operation fails, or the installed project package cannot be located.
    """
    device = resolve_device("cuda")
    cuda_runtime = torch_version.cuda
    if cuda_runtime is None:
        raise GPUPreflightError(
            "The installed PyTorch build does not report a CUDA runtime. Reopen the GPU Dev Container."
        )
    if cuda_runtime != expected_cuda_runtime:
        raise GPUPreflightError(
            f"Expected PyTorch CUDA runtime {expected_cuda_runtime}, found {cuda_runtime!r}. "
            "Rebuild or reopen the GPU Dev Container before training."
        )
    try:
        device_name = torch.cuda.get_device_name(device)
        left = torch.tensor([1.0, 2.0], device=device)
        right = torch.tensor([3.0, 4.0], device=device)
        tensor_result = float((left + right).sum().item())
        package = importlib.import_module("src")
        package_file = getattr(package, "__file__", None)
        if package_file is None:
            raise GPUPreflightError("The imported src package has no filesystem location.")
    except (RuntimeError, ImportError, OSError) as error:
        if isinstance(error, GPUPreflightError):
            raise
        raise GPUPreflightError(f"CUDA smoke test failed: {error}") from error
    return GPUPreflightReport(
        torch_version=torch.__version__,
        cuda_runtime=cuda_runtime,
        device_name=device_name,
        tensor_result=tensor_result,
        package_path=Path(str(package_file)).resolve(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line GPU preflight.

    Parameters
    ----------
    argv
        Optional argument sequence. ``None`` reads process arguments.

    Returns
    -------
    int
        Zero on success and one when the environment is not training-ready.
    """
    parser = argparse.ArgumentParser(description="Validate the ALASKA2 CUDA training environment.")
    parser.add_argument(
        "--expected-cuda-runtime",
        default=EXPECTED_CUDA_RUNTIME,
        help="Exact CUDA runtime expected from torch.version.cuda.",
    )
    arguments = parser.parse_args(argv)
    try:
        report = run_gpu_preflight(expected_cuda_runtime=arguments.expected_cuda_runtime)
    except GPUPreflightError as error:
        print(f"GPU preflight failed: {error}", file=sys.stderr)
        return 1
    print(report.format_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
