from __future__ import annotations

import os

# Required before Torch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ.setdefault("MPLBACKEND", "Agg")

import csv
import gc
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as vision_transforms
from torchvision.models import EfficientNet_B0_Weights

REPOSITORY = Path("/workspaces/alaska2-steganalysis").resolve()
DATASET_ROOT = Path("/datasets/ALASKA2").resolve()
RUN_ID = "alaska2_retrain_tiny10_effnet10_seed42_20260721"
ARTIFACT_ROOT = Path("/artifacts/alaska2-retrain")
RUN_ROOT = ARTIFACT_ROOT / RUN_ID
SEED = 42
CLASS_LABELS = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 2, "UERD": 3}
CLASS_NAMES = tuple(CLASS_LABELS)
STARTED_AT = datetime.now(timezone.utc).isoformat()

sys.path.insert(0, str(REPOSITORY))

from src.config.config_device import run_gpu_preflight  # noqa: E402
from src.config.config_runtime import (  # noqa: E402
    ReproducibilityConfig,
    make_generator,
    seed_everything,
    seed_worker,
)
from src.data.data_index import (  # noqa: E402
    add_targets,
    build_file_index,
    discover_jpeg_files,
    validate_complete_groups,
)
from src.data.data_split import DatasetSplits, assert_split_isolation  # noqa: E402
from src.datasets.datasets_images import ImageDataset  # noqa: E402
from src.datasets.datasets_loaders import build_loaders, compute_channel_statistics  # noqa: E402
from src.evaluation.evaluation_metrics import (  # noqa: E402
    confusion_counts,
    predict_binary,
    roc_data,
)
from src.evaluation.evaluation_plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_history,
    plot_roc_curves,
    plot_score_histogram,
)
from src.evaluation.evaluation_runner import evaluate_binary_model  # noqa: E402
from src.models.models_efficientnet import EfficientNetB0YCbCr  # noqa: E402
from src.models.models_freezing import (  # noqa: E402
    apply_frozen_eval as maintained_apply_frozen_eval,
    configure_stage,
    efficientnet_stages,
)
from src.models.models_tinycnn import TinyCNN  # noqa: E402
from src.training import training_loop, training_staged  # noqa: E402
from src.training.training_checkpoint import load_checkpoint  # noqa: E402
from src.transforms.transforms_shuffle import RandomTileShuffle  # noqa: E402
from src.transforms.transforms_spatial import AlignedDeterministicCrop, AlignedRandomCrop  # noqa: E402


MANIFEST = RUN_ROOT / "manifest"
SPLITS = RUN_ROOT / "splits"
LOGS = RUN_ROOT / "logs"
DRIVER = RUN_ROOT / "driver"
MODEL_ROOTS = {
    "tinycnn": RUN_ROOT / "tinycnn",
    "efficientnet_b0": RUN_ROOT / "efficientnet_b0",
}
STATUS_PATH = MANIFEST / "run_state.json"
EVENTS_PATH = LOGS / "events.jsonl"
GPU_LOG_PATH = LOGS / "gpu_monitor.jsonl"
LAUNCH_COMMAND = os.environ.get("ALASKA2_LAUNCH_COMMAND", "not supplied")
RECOVERY_ACTIONS = [
    {
        "attempt": 1,
        "timestamp": "2026-07-21T15:37:48+00:00",
        "failure": "Durable launch command omitted explicit stream redirection/backgrounding; PID 90064 exited during preflight before split creation or optimization, with no checkpoints or result artifacts.",
        "diagnosis": "Launch-level process-lifetime/output handling failure; no evidence of a repository, dataset, CUDA, or model failure.",
        "recovery": "Preserved the incomplete run directory, verified it contained only the driver, preparing state, and first event, then relaunched the same unambiguous run ID with stdin=/dev/null, stdout/stderr redirected to logs/driver.log, and shell backgrounding.",
        "scientific_state_reused": False,
    },
    {
        "attempt": 2,
        "timestamp": "2026-07-21T15:42:29+00:00",
        "failure": "Pre-optimization split assertion found 59,999 training, 7,501 validation, and 7,500 test groups from the maintained float-fraction helper.",
        "diagnosis": "The second sklearn split rounded its floating-point validation size upward; this conflicts with the objective exact reservoir counts.",
        "recovery": "Preserved all preflight evidence and replaced only external reservoir orchestration with the objective one seed-42 shuffled basename ordering sliced exactly 60,000/7,500/7,500; maintained indexing, complete-group validation, DatasetSplits, and isolation checks remain in use.",
        "scientific_state_reused": False,
    }
]

_status_lock = threading.Lock()
_event_lock = threading.Lock()
_status: dict[str, Any] = {
    "run_id": RUN_ID,
    "state": "preparing",
    "pid": os.getpid(),
    "launch_command": LAUNCH_COMMAND,
    "start_timestamp": STARTED_AT,
    "completion_timestamp": None,
    "current_model": None,
    "current_stage": "preflight",
    "current_phase": "preparing",
    "current_epoch": 0,
    "last_valid_checkpoint": None,
    "final_exit_status": None,
}
_instrument_context: dict[str, Any] = {}
_epoch_records: dict[str, list[dict[str, Any]]] = {}
_test_evaluation_counts = {"tinycnn": 0, "efficientnet_b0": 0}
_gpu_system_peak_mib = {"tinycnn": 0, "efficientnet_b0": 0, "preparing": 0}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(**updates: Any) -> None:
    with _status_lock:
        _status.update(updates)
        write_json(STATUS_PATH, _status)


def log_event(event: str, **payload: Any) -> None:
    record = {"timestamp": utc_now(), "event": event, **jsonable(payload)}
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    with _event_lock:
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    print(line, flush=True)


def run_command(arguments: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=REPOSITORY,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_directories() -> None:
    if (RUN_ROOT / "comparison.json").exists() or any(
        (root / "evaluation" / "final_result.json").exists() for root in MODEL_ROOTS.values()
    ):
        raise FileExistsError(f"Refusing to overwrite existing result artifacts under {RUN_ROOT}")
    for directory in (MANIFEST, SPLITS, LOGS, DRIVER):
        directory.mkdir(parents=True, exist_ok=True)
    for root in MODEL_ROOTS.values():
        for name in ("checkpoints", "histories", "predictions", "evaluation", "figures"):
            (root / name).mkdir(parents=True, exist_ok=True)
    update_status()


def check_duplicate_process() -> None:
    output = run_command(["ps", "-eo", "pid=,args="])
    duplicates: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid != os.getpid() and RUN_ID in command and "python" in command:
            duplicates.append(stripped)
    if duplicates:
        raise RuntimeError(f"Duplicate training process detected: {duplicates}")


def maintained_files() -> list[Path]:
    chosen: set[Path] = set()
    for directory in (REPOSITORY / "src", REPOSITORY / "tests"):
        for path in directory.rglob("*.py"):
            if "__pycache__" not in path.parts and not path.name.endswith(".orig"):
                chosen.add(path)
    for directory in (REPOSITORY / ".devcontainer", REPOSITORY / ".github", REPOSITORY / ".vscode"):
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and not path.name.endswith(".orig"):
                    chosen.add(path)
    names = (
        ".dockerignore",
        ".gitattributes",
        ".gitignore",
        "ANN_Projekt_Rino_Albertin_Steganalyse.ipynb",
        "LICENSE",
        "README.md",
        "poetry.lock",
        "poetry.toml",
        "pyproject.toml",
        "pyrightconfig.json",
    )
    for name in names:
        path = REPOSITORY / name
        if path.is_file():
            chosen.add(path)
    return sorted(chosen, key=lambda path: path.relative_to(REPOSITORY).as_posix())


def create_code_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    composite = hashlib.sha256()
    for path in maintained_files():
        relative = path.relative_to(REPOSITORY).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "sha256": digest, "size": size})
        composite.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    destination = MANIFEST / "maintained_file_hashes.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "sha256", "size"))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "algorithm": "sha256(path + NUL + file_sha256 + newline), paths sorted as POSIX",
        "file_count": len(rows),
        "composite_sha256": composite.hexdigest(),
        "manifest_path": str(destination),
        "readme_sha256": next(row["sha256"] for row in rows if row["path"] == "README.md"),
        "notebook_sha256": next(
            row["sha256"] for row in rows if row["path"] == "ANN_Projekt_Rino_Albertin_Steganalyse.ipynb"
        ),
    }
    write_json(MANIFEST / "code_fingerprint.json", result)
    return result


def dataset_snapshot(files_by_class: Mapping[str, Mapping[str, Path]], name: str) -> dict[str, Any]:
    aggregate = hashlib.sha256()
    counts: dict[str, int] = {}
    total_bytes = 0
    for class_name in CLASS_NAMES:
        files = files_by_class[class_name]
        counts[class_name] = len(files)
        for source_id in sorted(files):
            path = files[source_id]
            stat = path.stat()
            total_bytes += stat.st_size
            aggregate.update(
                f"{class_name}\0{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
            )
    result = {
        "dataset_root": str(DATASET_ROOT),
        "classes": list(CLASS_NAMES),
        "filename_counts": counts,
        "total_files": sum(counts.values()),
        "total_bytes": total_bytes,
        "identity_algorithm": "sha256(class + NUL + filename + NUL + size + NUL + mtime_ns + newline)",
        "identity_sha256": aggregate.hexdigest(),
        "captured_at": utc_now(),
    }
    write_json(MANIFEST / f"dataset_{name}.json", result)
    return result


def environment_manifest(gpu_report: Any, weights_path: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(ARTIFACT_ROOT)
    payload = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "gpu_preflight": {
            "torch_version": gpu_report.torch_version,
            "cuda_runtime": gpu_report.cuda_runtime,
            "device_name": gpu_report.device_name,
            "tensor_result": gpu_report.tensor_result,
            "package_path": str(gpu_report.package_path),
        },
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "efficientnet_weights": {
            "enum": str(EfficientNet_B0_Weights.DEFAULT),
            "url": EfficientNet_B0_Weights.DEFAULT.url,
            "cache_path": str(weights_path),
            "cache_sha256": sha256_file(weights_path),
            "cache_size": weights_path.stat().st_size,
        },
        "artifact_disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }
    canonical = json.dumps(jsonable(payload), sort_keys=True).encode("utf-8")
    payload["environment_sha256"] = sha256_bytes(canonical)
    write_json(MANIFEST / "environment.json", payload)
    return payload


def write_membership(name: str, source_ids: Iterable[str], *, preserve_order: bool = False) -> dict[str, Any]:
    raw_values = tuple(str(value) for value in source_ids)
    values = raw_values if preserve_order else tuple(sorted(raw_values))
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate source ID in {name}")
    path = SPLITS / f"{name}.txt"
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    path.write_bytes(payload)
    return {"path": str(path), "group_count": len(values), "sha256": sha256_bytes(payload)}


def subset_frame(frame: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return frame.loc[frame["source_id"].isin(ids)].reset_index(drop=True)


def build_exact_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    update_status(current_stage="indexing", current_phase="preparing")
    index = build_file_index(
        DATASET_ROOT,
        CLASS_LABELS,
        subsample_fraction=1.0,
        seed=SEED,
        on_incomplete="raise",
    )
    numeric = add_targets(index, CLASS_LABELS)
    all_ids = tuple(sorted(str(value) for value in numeric["source_id"].unique()))
    if len(all_ids) != 75_000:
        raise ValueError(f"Expected 75,000 source groups, found {len(all_ids)}")
    source_ordering = list(all_ids)
    random.Random(SEED).shuffle(source_ordering)
    reservoir_ids = {
        "training_reservoir": set(source_ordering[:60_000]),
        "validation_reservoir": set(source_ordering[60_000:67_500]),
        "final_test": set(source_ordering[67_500:75_000]),
    }
    reservoirs = DatasetSplits(
        train=subset_frame(numeric, reservoir_ids["training_reservoir"]),
        validation=subset_frame(numeric, reservoir_ids["validation_reservoir"]),
        test=subset_frame(numeric, reservoir_ids["final_test"]),
    )
    assert_split_isolation(reservoirs)
    expected_counts = {"training_reservoir": 60_000, "validation_reservoir": 7_500, "final_test": 7_500}
    for name, expected in expected_counts.items():
        if len(reservoir_ids[name]) != expected:
            raise ValueError(f"{name} has {len(reservoir_ids[name])} groups, expected {expected}")
    if (
        reservoir_ids["training_reservoir"] & reservoir_ids["validation_reservoir"]
        or reservoir_ids["training_reservoir"] & reservoir_ids["final_test"]
        or reservoir_ids["validation_reservoir"] & reservoir_ids["final_test"]
    ):
        raise ValueError("Reservoir source leakage detected")

    selection_rng = random.Random(SEED)
    training_subset_ids = set(selection_rng.sample(sorted(reservoir_ids["training_reservoir"]), 6_000))
    validation_subset_ids = set(selection_rng.sample(sorted(reservoir_ids["validation_reservoir"]), 750))
    train_frame = subset_frame(reservoirs.train, training_subset_ids)
    validation_frame = subset_frame(reservoirs.validation, validation_subset_ids)
    test_frame = reservoirs.test.reset_index(drop=True)
    for frame in (reservoirs.train, reservoirs.validation, reservoirs.test, train_frame, validation_frame, test_frame):
        validate_complete_groups(frame, expected_classes=CLASS_NAMES)
    if (len(train_frame), len(validation_frame), len(test_frame)) != (24_000, 3_000, 30_000):
        raise ValueError("Exact selected image counts were not produced")
    selected_sets = (training_subset_ids, validation_subset_ids, reservoir_ids["final_test"])
    if selected_sets[0] & selected_sets[1] or selected_sets[0] & selected_sets[2] or selected_sets[1] & selected_sets[2]:
        raise ValueError("Selected source leakage detected")

    memberships = {
        "source_ordering": write_membership("source_ordering", source_ordering, preserve_order=True),
        "training_reservoir": write_membership("training_reservoir", reservoir_ids["training_reservoir"]),
        "validation_reservoir": write_membership("validation_reservoir", reservoir_ids["validation_reservoir"]),
        "final_test": write_membership("final_test", reservoir_ids["final_test"]),
        "shared_training_subset": write_membership("shared_training_subset", training_subset_ids),
        "shared_validation_subset": write_membership("shared_validation_subset", validation_subset_ids),
    }
    evidence = {
        "seed": SEED,
        "algorithm": {
            "source_ordering": "random.Random(42).shuffle applied once to lexicographically sorted complete basenames",
            "reservoirs": "exact slices [0:60000], [60000:67500], [67500:75000] of the persisted seed-42 ordering, wrapped in maintained DatasetSplits and checked by maintained assert_split_isolation",
            "subsets": "a separate random.Random(42) instance; sample 6,000 sorted training-reservoir IDs then 750 sorted validation-reservoir IDs",
            "membership_hash": "SHA-256 of UTF-8 IDs plus LF; reservoirs/subsets sorted, source_ordering in persisted shuffled order",
        },
        "memberships": memberships,
        "image_counts": {
            "training_reservoir": len(reservoirs.train),
            "validation_reservoir": len(reservoirs.validation),
            "final_test": len(test_frame),
            "shared_training_subset": len(train_frame),
            "shared_validation_subset": len(validation_frame),
        },
        "classes_per_group": 4,
        "disjoint": True,
        "complete_groups": True,
        "models_share_training_membership": True,
        "models_share_validation_membership": True,
        "models_share_final_test_membership": True,
    }
    write_json(SPLITS / "split_membership.json", evidence)
    log_event("split_preflight_completed", memberships=memberships)
    return train_frame, validation_frame, test_frame, evidence


def raw_statistics_loader(train_frame: pd.DataFrame, identity_crop: AlignedDeterministicCrop) -> DataLoader:
    raw_dataset = ImageDataset(
        train_frame,
        color_mode="YCbCr",
        target_column="label_bin",
        transform=vision_transforms.ToTensor(),
        identity_crop=identity_crop,
    )
    return DataLoader(
        raw_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=True,
        prefetch_factor=2,
        generator=make_generator(SEED + 10),
    )


def stage_contracts(model: EfficientNetB0YCbCr) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    stages = efficientnet_stages(
        model,
        head_learning_rate=1e-3,
        block_learning_rate=1e-4,
        head_epochs=10,
        block_epochs=8,
    )
    expected_order = (
        "head_stem",
        "feature_8",
        "feature_7",
        "feature_6",
        "feature_5",
        "feature_4",
        "feature_3",
        "feature_2",
        "feature_1",
    )
    if tuple(stage.name for stage in stages) != expected_order:
        raise ValueError(f"Unexpected EfficientNet stage order: {[stage.name for stage in stages]}")
    total = sum(parameter.numel() for parameter in model.parameters())
    contracts: list[dict[str, Any]] = []
    for stage in stages:
        configure_stage(model, stage)
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen_count = total - trainable_count
        contracts.append(
            {
                "name": stage.name,
                "learning_rate": stage.learning_rate,
                "maximum_epochs": stage.epochs,
                "patience": 3,
                "trainable_parameter_count": trainable_count,
                "frozen_parameter_count": frozen_count,
                "trainable_parameter_names": names,
            }
        )
    return stages, contracts


def inspect_stage_contracts_without_rng_effect() -> list[dict[str, Any]]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all()
    try:
        model = EfficientNetB0YCbCr(weights=EfficientNet_B0_Weights.DEFAULT)
        _, contracts = stage_contracts(model)
        del model
        gc.collect()
        return contracts
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(cuda_states)


def begin_context(
    model_name: str,
    stage_name: str,
    learning_rate: float,
    checkpoint_path: Path,
    trainable_count: int,
    frozen_count: int,
) -> None:
    key = f"{model_name}:{stage_name}"
    _instrument_context.clear()
    _instrument_context.update(
        {
            "key": key,
            "model": model_name,
            "stage": stage_name,
            "learning_rate": learning_rate,
            "checkpoint_path": checkpoint_path,
            "trainable_parameter_count": trainable_count,
            "frozen_parameter_count": frozen_count,
            "epoch": 0,
            "epoch_started_perf": None,
            "stage_started_perf": time.perf_counter(),
        }
    )
    _epoch_records[key] = []
    update_status(
        state="running",
        current_model=model_name,
        current_stage=stage_name,
        current_phase="training",
        current_epoch=0,
    )
    log_event(
        "training_stage_started",
        model=model_name,
        stage=stage_name,
        learning_rate=learning_rate,
        trainable_parameter_count=trainable_count,
        frozen_parameter_count=frozen_count,
    )


_original_train_epoch = training_loop._train_epoch
_original_validate_epoch = training_loop._validate_epoch
_original_staged_run_experiment = training_staged.run_experiment


def monitored_train_epoch(*args: Any, **kwargs: Any) -> tuple[float, float]:
    context = _instrument_context
    context["epoch"] = int(context["epoch"]) + 1
    epoch = int(context["epoch"])
    context["epoch_started_perf"] = time.perf_counter()
    update_status(current_phase="training", current_epoch=epoch)
    result = _original_train_epoch(*args, **kwargs)
    if not all(math.isfinite(float(value)) for value in result):
        raise FloatingPointError(f"Non-finite training metrics at {context['key']} epoch {epoch}: {result}")
    context["train_finished_perf"] = time.perf_counter()
    context["train_metrics"] = {"loss": float(result[0]), "accuracy": float(result[1])}
    update_status(current_phase="validation")
    return result


def monitored_validate_epoch(*args: Any, **kwargs: Any) -> Any:
    context = _instrument_context
    validation_started = time.perf_counter()
    result = _original_validate_epoch(*args, **kwargs)
    values = (result.loss, result.accuracy, result.weighted_auc)
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError(
            f"Non-finite validation metrics at {context['key']} epoch {context['epoch']}: {values}"
        )
    finished = time.perf_counter()
    started = float(context["epoch_started_perf"])
    record = {
        "epoch": int(context["epoch"]),
        "started_at": utc_now(),
        "train_seconds": float(context["train_finished_perf"]) - started,
        "validation_seconds": finished - validation_started,
        "epoch_elapsed_seconds": finished - started,
        "stage_elapsed_seconds": finished - float(context["stage_started_perf"]),
        "learning_rate": float(context["learning_rate"]),
        "trainable_parameter_count": int(context["trainable_parameter_count"]),
        "frozen_parameter_count": int(context["frozen_parameter_count"]),
    }
    _epoch_records[str(context["key"])].append(record)
    checkpoint = Path(context["checkpoint_path"])
    update_status(
        current_phase="epoch_completed",
        last_valid_checkpoint=str(checkpoint) if checkpoint.is_file() else _status.get("last_valid_checkpoint"),
    )
    log_event(
        "epoch_completed",
        model=context["model"],
        stage=context["stage"],
        epoch=context["epoch"],
        train_loss=context["train_metrics"]["loss"],
        train_accuracy=context["train_metrics"]["accuracy"],
        validation_loss=result.loss,
        validation_accuracy=result.accuracy,
        validation_weighted_auc=result.weighted_auc,
        epoch_elapsed_seconds=record["epoch_elapsed_seconds"],
    )
    return result


def monitored_apply_frozen_eval(model: nn.Module) -> None:
    maintained_apply_frozen_eval(model)
    violations: list[str] = []
    for name, module in model.named_modules():
        direct = tuple(module.parameters(recurse=False))
        if direct and not any(parameter.requires_grad for parameter in direct) and module.training:
            violations.append(name)
    if violations:
        raise RuntimeError(f"Frozen parameter-bearing modules remained in training mode: {violations[:10]}")


def selected_flags(frame: pd.DataFrame, minimum_improvement: float = 1e-4) -> list[bool]:
    best = float("-inf")
    flags: list[bool] = []
    for score in frame["val_wauc"].astype(float):
        selected = score > best + minimum_improvement
        flags.append(selected)
        if selected:
            best = score
    return flags


def enriched_history(base: pd.DataFrame, key: str) -> pd.DataFrame:
    records = _epoch_records[key]
    if len(base) != len(records):
        raise RuntimeError(f"Instrumentation/history length mismatch for {key}: {len(records)} != {len(base)}")
    result = base.copy()
    metadata = pd.DataFrame.from_records(records)
    for column in metadata.columns:
        if column != "epoch":
            result[column] = metadata[column].to_numpy()
    result["selected_best_state"] = selected_flags(result)
    return result


def compare_checkpoint_to_model(model: nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_state = payload.get("model_state")
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Missing model_state in {checkpoint_path}")
    model_state = model.state_dict()
    if set(model_state) != set(checkpoint_state):
        raise RuntimeError("Selected checkpoint keys do not match restored model")
    mismatches = [
        name
        for name, value in model_state.items()
        if not torch.equal(value.detach().cpu(), checkpoint_state[name].detach().cpu())
    ]
    if mismatches:
        raise RuntimeError(f"Restored model differs from checkpoint: {mismatches[:5]}")
    result = {
        "verified": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": int(payload["epoch"]),
        "validation_weighted_auc": float(payload["validation_weighted_auc"]),
        "validation_accuracy": float(payload["validation_accuracy"]),
        "tensor_count": len(checkpoint_state),
    }
    return result


def save_evaluation_artifacts(
    model_name: str,
    frame: pd.DataFrame,
    evaluation: Any,
    history: pd.DataFrame,
    figure_title: str,
) -> dict[str, Any]:
    root = MODEL_ROOTS[model_name]
    predictions = predict_binary(evaluation.y_probability)
    expected_targets = frame["label_bin"].to_numpy(dtype=np.float32)
    if not np.array_equal(evaluation.y_true.astype(np.float32), expected_targets):
        raise RuntimeError(f"{model_name} evaluation order does not match the common test dataframe")
    prediction_frame = frame[["source_id", "label_name", "path", "label_bin"]].copy()
    prediction_frame["probability_stego"] = evaluation.y_probability
    prediction_frame["prediction_bin"] = predictions
    prediction_path = root / "predictions" / "final_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    confusion = confusion_counts(evaluation.y_true, predictions)
    confusion_path = root / "evaluation" / "confusion_matrix.csv"
    pd.DataFrame(confusion, index=("true_cover", "true_stego"), columns=("pred_cover", "pred_stego")).to_csv(
        confusion_path
    )
    roc = roc_data(evaluation.y_true, evaluation.y_probability)
    roc_path = root / "evaluation" / "roc_data.csv"
    pd.DataFrame(roc).to_csv(roc_path, index=False)
    metric_path = root / "evaluation" / "test_metrics.json"
    write_json(
        metric_path,
        {
            "test_loss": evaluation.loss,
            "test_accuracy": evaluation.accuracy,
            "test_weighted_auc": evaluation.weighted_auc,
            "sample_count": len(evaluation.y_true),
            "confusion_matrix": confusion.tolist(),
            "evaluation_pass_count": _test_evaluation_counts[model_name],
        },
    )

    figures = {
        "history": root / "figures" / "training_history.png",
        "confusion_matrix": root / "figures" / "confusion_matrix.png",
        "roc": root / "figures" / "roc_curve.png",
        "score_histogram": root / "figures" / "score_histogram.png",
    }
    generated = [
        (plot_history(history, title=f"{figure_title} training history"), figures["history"]),
        (plot_confusion_matrix(confusion), figures["confusion_matrix"]),
        (
            plot_roc_curves(
                [{"fpr": roc["fpr"], "tpr": roc["tpr"], "wauc": evaluation.weighted_auc, "label": figure_title}]
            ),
            figures["roc"],
        ),
        (plot_score_histogram(evaluation.y_probability, evaluation.y_true), figures["score_histogram"]),
    ]
    for figure, path in generated:
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
    return {
        "prediction_path": str(prediction_path),
        "test_metric_path": str(metric_path),
        "roc_data_path": str(roc_path),
        "roc_point_count": len(roc["fpr"]),
        "confusion_matrix_path": str(confusion_path),
        "confusion_matrix": confusion.tolist(),
        "figure_paths": {key: str(path) for key, path in figures.items()},
    }


def model_peak_memory() -> dict[str, int]:
    return {
        "torch_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def train_tinycnn(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    identity_crop: AlignedDeterministicCrop,
    common: Mapping[str, Any],
) -> dict[str, Any]:
    root = MODEL_ROOTS["tinycnn"]
    started_at = utc_now()
    started_perf = time.perf_counter()
    train_transform = vision_transforms.Compose(
        [
            AlignedRandomCrop(size=256, block_size=8),
            RandomTileShuffle(tiles_per_axis=8),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize([0.5], [0.5]),
        ]
    )
    evaluation_transform = vision_transforms.Compose(
        [vision_transforms.ToTensor(), vision_transforms.Normalize([0.5], [0.5])]
    )
    train_dataset = ImageDataset(train_frame, color_mode="Y", target_column="label_bin", transform=train_transform)
    validation_dataset = ImageDataset(
        validation_frame,
        color_mode="Y",
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    test_dataset = ImageDataset(
        test_frame,
        color_mode="Y",
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    loaders = build_loaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=48,
        num_workers=2,
        seed=SEED,
        prefetch_factor=2,
        pin_memory=True,
    )
    model = TinyCNN().to("cuda")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1 / 3], device="cuda"))
    optimizer = Adam(model.parameters(), lr=1e-4)
    checkpoint_path = root / "checkpoints" / "tinycnn_best.pt"
    maintained_history_path = root / "histories" / "tinycnn_history.csv"
    total = sum(parameter.numel() for parameter in model.parameters())
    begin_context("tinycnn", "single", 1e-4, checkpoint_path, total, 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = training_loop.run_experiment(
        model,
        loaders.train,
        loaders.validation,
        criterion,
        optimizer,
        num_epochs=50,
        device="cuda",
        run_name="TinyCNN_Y_10pct_seed42",
        checkpoint_path=checkpoint_path,
        history_path=maintained_history_path,
        patience=10,
        progress=False,
    )
    complete_history = enriched_history(result.history, "tinycnn:single")
    complete_history_path = root / "histories" / "tinycnn_complete_history.csv"
    complete_history.to_csv(complete_history_path, index=False)
    verification = compare_checkpoint_to_model(model, checkpoint_path)
    write_json(root / "evaluation" / "checkpoint_verification.json", verification)
    update_status(current_phase="final_test", last_valid_checkpoint=str(checkpoint_path))
    log_event("final_test_evaluation_started", model="tinycnn", test_images=len(test_frame))
    _test_evaluation_counts["tinycnn"] += 1
    evaluation = evaluate_binary_model(model, loaders.test, device="cuda", criterion=criterion)
    log_event(
        "final_test_evaluation_completed",
        model="tinycnn",
        loss=evaluation.loss,
        accuracy=evaluation.accuracy,
        weighted_auc=evaluation.weighted_auc,
        evaluation_pass_count=_test_evaluation_counts["tinycnn"],
    )
    evaluation_artifacts = save_evaluation_artifacts(
        "tinycnn", test_frame, evaluation, complete_history, "TinyCNN"
    )
    runtime = time.perf_counter() - started_perf
    memory = {
        **model_peak_memory(),
        "nvidia_smi_peak_memory_used_mib": _gpu_system_peak_mib["tinycnn"],
    }
    final = {
        **common,
        "model": "TinyCNN",
        "completion_status": "completed",
        "effective_model_configuration": {
            "architecture": "maintained TinyCNN five-convolution luminance baseline",
            "binary_target": "Cover=0 versus any Stego=1",
            "color_mode": "Y",
            "crop_size": [256, 256],
            "training_crop": "maintained 8-pixel-aligned random crop",
            "training_augmentation": "maintained 8x8 RandomTileShuffle",
            "evaluation_crop": "maintained source-identity deterministic aligned crop, seed 42",
            "normalization_mean": [0.5],
            "normalization_std": [0.5],
            "loss": "BCEWithLogitsLoss",
            "pos_weight": 1 / 3,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "maximum_epochs": 50,
            "patience": 10,
            "minimum_improvement": 1e-4,
            "selection_criterion": "validation official ALASKA2 Weighted AUC",
            "amp": False,
            "gradient_accumulation_steps": 1,
            "parameter_count": total,
        },
        "effective_dataloader_configuration": {
            "physical_batch_size": 48,
            "workers": 2,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "independent_generators": {"train": 42, "validation": 43, "test": 44},
        },
        "best_epoch": result.summary.best_epoch,
        "best_validation_accuracy": result.summary.best_val_accuracy,
        "best_validation_weighted_auc": result.summary.best_val_weighted_auc,
        "final_epoch": result.summary.final_epoch,
        "early_stopped": result.summary.early_stopped,
        "final_test_loss": evaluation.loss,
        "final_test_accuracy": evaluation.accuracy,
        "final_test_weighted_auc": evaluation.weighted_auc,
        "test_evaluation_pass_count": _test_evaluation_counts["tinycnn"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_verification": verification,
        "history_paths": [str(maintained_history_path), str(complete_history_path)],
        **evaluation_artifacts,
        "start_timestamp": started_at,
        "completion_timestamp": utc_now(),
        "total_runtime_seconds": runtime,
        "observed_gpu_memory": memory,
    }
    write_json(root / "evaluation" / "final_result.json", final)
    del optimizer, criterion, model, loaders, train_dataset, validation_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return final


def train_efficientnet(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    identity_crop: AlignedDeterministicCrop,
    mean: list[float],
    std: list[float],
    stage_configuration: list[dict[str, Any]],
    common: Mapping[str, Any],
) -> dict[str, Any]:
    root = MODEL_ROOTS["efficientnet_b0"]
    started_at = utc_now()
    started_perf = time.perf_counter()
    train_transform = vision_transforms.Compose(
        [
            AlignedRandomCrop(size=256, block_size=8),
            RandomTileShuffle(tiles_per_axis=8),
            vision_transforms.ToTensor(),
            vision_transforms.Normalize(mean, std),
        ]
    )
    evaluation_transform = vision_transforms.Compose(
        [vision_transforms.ToTensor(), vision_transforms.Normalize(mean, std)]
    )
    train_dataset = ImageDataset(
        train_frame, color_mode="YCbCr", target_column="label_bin", transform=train_transform
    )
    validation_dataset = ImageDataset(
        validation_frame,
        color_mode="YCbCr",
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    test_dataset = ImageDataset(
        test_frame,
        color_mode="YCbCr",
        target_column="label_bin",
        transform=evaluation_transform,
        identity_crop=identity_crop,
    )
    loaders = build_loaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=32,
        num_workers=4,
        seed=SEED,
        prefetch_factor=2,
        pin_memory=True,
    )
    model = EfficientNetB0YCbCr(weights=EfficientNet_B0_Weights.DEFAULT).to("cuda")
    stages, actual_contracts = stage_contracts(model)
    if actual_contracts != stage_configuration:
        raise RuntimeError("Resolved EfficientNet stage configuration changed before optimization")
    contract_by_name = {item["name"]: item for item in actual_contracts}
    stage_by_name = {stage.name: stage for stage in stages}
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1 / 3], device="cuda"))
    checkpoint_directory = root / "checkpoints"
    maintained_history_directory = root / "histories" / "maintained"

    def staged_proxy(
        model_arg: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion_arg: Any,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> Any:
        stage_name = str(kwargs["run_name"])
        contract = contract_by_name[stage_name]
        stage = stage_by_name[stage_name]
        actual_names = [name for name, parameter in model_arg.named_parameters() if parameter.requires_grad]
        if actual_names != contract["trainable_parameter_names"]:
            raise RuntimeError(f"Trainable parameter mismatch for {stage_name}")
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        trainable_ids = {id(parameter) for parameter in model_arg.parameters() if parameter.requires_grad}
        expected_ids = {id(parameter) for module in stage.modules for parameter in module.parameters()}
        if optimizer_ids != trainable_ids or trainable_ids != expected_ids:
            raise RuntimeError(f"Optimizer/trainable module mismatch for {stage_name}")
        begin_context(
            "efficientnet_b0",
            stage_name,
            float(contract["learning_rate"]),
            checkpoint_directory / f"{stage_name}_best.pt",
            int(contract["trainable_parameter_count"]),
            int(contract["frozen_parameter_count"]),
        )
        return _original_staged_run_experiment(
            model_arg,
            train_loader,
            validation_loader,
            criterion_arg,
            optimizer,
            **kwargs,
        )

    training_staged.run_experiment = staged_proxy
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        result = training_staged.run_staged_fine_tuning(
            model,
            stages,
            loaders.train,
            loaders.validation,
            criterion,
            device="cuda",
            patience=3,
            checkpoint_directory=checkpoint_directory,
            history_directory=maintained_history_directory,
            progress=False,
        )
    finally:
        training_staged.run_experiment = _original_staged_run_experiment

    complete_stage_frames: list[pd.DataFrame] = []
    stage_summaries: list[dict[str, Any]] = []
    for stage_index, (stage, stage_result) in enumerate(zip(stages, result.stages, strict=True)):
        key = f"efficientnet_b0:{stage.name}"
        complete = enriched_history(stage_result.history, key)
        complete.insert(0, "stage_index", stage_index)
        complete.insert(1, "stage", stage.name)
        complete_path = root / "histories" / f"{stage.name}_complete_history.csv"
        complete.to_csv(complete_path, index=False)
        complete_stage_frames.append(complete)
        stage_summaries.append(
            {
                **contract_by_name[stage.name],
                "epochs_completed": stage_result.summary.final_epoch,
                "early_stopped": stage_result.summary.early_stopped,
                "best_epoch": stage_result.summary.best_epoch,
                "best_validation_accuracy": stage_result.summary.best_val_accuracy,
                "best_validation_weighted_auc": stage_result.summary.best_val_weighted_auc,
                "final_validation_accuracy": stage_result.summary.final_val_accuracy,
                "final_validation_weighted_auc": stage_result.summary.final_val_weighted_auc,
                "checkpoint_path": str(stage_result.summary.best_checkpoint),
                "maintained_history_path": str(maintained_history_directory / f"{stage.name}_history.csv"),
                "complete_history_path": str(complete_path),
                "runtime_seconds": float(complete["epoch_elapsed_seconds"].sum()),
            }
        )
    complete_history = pd.concat(complete_stage_frames, ignore_index=True)
    complete_history_path = root / "histories" / "efficientnet_complete_history.csv"
    complete_history.to_csv(complete_history_path, index=False)
    selected_checkpoint = checkpoint_directory / f"{result.best_stage}_best.pt"
    verification = compare_checkpoint_to_model(model, selected_checkpoint)
    write_json(root / "evaluation" / "checkpoint_verification.json", verification)
    update_status(current_phase="final_test", last_valid_checkpoint=str(selected_checkpoint))
    log_event("final_test_evaluation_started", model="efficientnet_b0", test_images=len(test_frame))
    _test_evaluation_counts["efficientnet_b0"] += 1
    evaluation = evaluate_binary_model(model, loaders.test, device="cuda", criterion=criterion)
    log_event(
        "final_test_evaluation_completed",
        model="efficientnet_b0",
        loss=evaluation.loss,
        accuracy=evaluation.accuracy,
        weighted_auc=evaluation.weighted_auc,
        evaluation_pass_count=_test_evaluation_counts["efficientnet_b0"],
    )
    evaluation_artifacts = save_evaluation_artifacts(
        "efficientnet_b0", test_frame, evaluation, complete_history, "EfficientNet-B0"
    )
    runtime = time.perf_counter() - started_perf
    memory = {
        **model_peak_memory(),
        "nvidia_smi_peak_memory_used_mib": _gpu_system_peak_mib["efficientnet_b0"],
    }
    selected_summary = next(
        stage_result.summary for stage_result in result.stages if stage_result.summary.run_name == result.best_stage
    )
    final = {
        **common,
        "model": "EfficientNet-B0",
        "completion_status": "completed",
        "channel_statistics": {"mean": mean, "std": std, "source": "shared 6,000-group training subset only"},
        "effective_model_configuration": {
            "architecture": "maintained EfficientNetB0YCbCr with official ImageNet pretrained weights",
            "binary_target": "Cover=0 versus any Stego=1",
            "color_mode": "YCbCr",
            "crop_size": [256, 256],
            "training_crop": "maintained 8-pixel-aligned random crop",
            "training_augmentation": "maintained 8x8 RandomTileShuffle",
            "evaluation_crop": "maintained source-identity deterministic aligned crop, seed 42",
            "loss": "BCEWithLogitsLoss",
            "pos_weight": 1 / 3,
            "optimizer": "Adam",
            "selection_criterion": "validation official ALASKA2 Weighted AUC",
            "stage_contract": "maintained non-cumulative global-best handoff",
            "stages": actual_contracts,
            "amp": False,
            "gradient_accumulation_steps": 1,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "effective_dataloader_configuration": {
            "physical_batch_size": 32,
            "workers": 4,
            "prefetch_factor": 2,
            "pin_memory": True,
            "persistent_workers": True,
            "independent_generators": {"train": 42, "validation": 43, "test": 44},
        },
        "stage_summaries": stage_summaries,
        "best_stage": result.best_stage,
        "best_epoch_within_stage": selected_summary.best_epoch,
        "best_validation_accuracy": selected_summary.best_val_accuracy,
        "best_validation_weighted_auc": result.best_validation_weighted_auc,
        "final_test_loss": evaluation.loss,
        "final_test_accuracy": evaluation.accuracy,
        "final_test_weighted_auc": evaluation.weighted_auc,
        "test_evaluation_pass_count": _test_evaluation_counts["efficientnet_b0"],
        "checkpoint_path": str(selected_checkpoint),
        "checkpoint_verification": verification,
        "history_paths": [
            str(maintained_history_directory),
            *(summary["complete_history_path"] for summary in stage_summaries),
            str(complete_history_path),
        ],
        **evaluation_artifacts,
        "start_timestamp": started_at,
        "completion_timestamp": utc_now(),
        "total_runtime_seconds": runtime,
        "observed_gpu_memory": memory,
    }
    write_json(root / "evaluation" / "final_result.json", final)
    del criterion, model, loaders, train_dataset, validation_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return final


def gpu_monitor(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            output = run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ).strip()
            fields = [field.strip() for field in output.split(",")]
            record = {
                "captured_at": utc_now(),
                "nvidia_timestamp": fields[0],
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "gpu_utilization_percent": int(fields[3]),
                "temperature_c": int(fields[4]),
                "model": _status.get("current_model"),
                "stage": _status.get("current_stage"),
                "epoch": _status.get("current_epoch"),
            }
            key = str(record["model"] or "preparing")
            if key in _gpu_system_peak_mib:
                _gpu_system_peak_mib[key] = max(_gpu_system_peak_mib[key], int(record["memory_used_mib"]))
            with GPU_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as error:
            with GPU_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"captured_at": utc_now(), "error": repr(error)}) + "\n")
        stop.wait(30.0)


def artifact_contract(tiny: Mapping[str, Any], efficient: Mapping[str, Any]) -> dict[str, Any]:
    required_paths = [
        STATUS_PATH,
        MANIFEST / "environment.json",
        MANIFEST / "maintained_file_hashes.csv",
        SPLITS / "split_membership.json",
        Path(str(tiny["checkpoint_path"])),
        Path(str(efficient["checkpoint_path"])),
        Path(str(tiny["prediction_path"])),
        Path(str(efficient["prediction_path"])),
        MODEL_ROOTS["tinycnn"] / "evaluation" / "final_result.json",
        MODEL_ROOTS["efficientnet_b0"] / "evaluation" / "final_result.json",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    result = {
        "passed": not missing,
        "missing_paths": missing,
        "tiny_test_evaluation_passes": _test_evaluation_counts["tinycnn"],
        "efficientnet_test_evaluation_passes": _test_evaluation_counts["efficientnet_b0"],
        "test_exactly_once_each": _test_evaluation_counts == {"tinycnn": 1, "efficientnet_b0": 1},
        "inspected_at": utc_now(),
    }
    if missing or not result["test_exactly_once_each"]:
        raise RuntimeError(f"Final artifact contract failed: {result}")
    write_json(MANIFEST / "artifact_contract.json", result)
    return result


def main() -> int:
    os.chdir(REPOSITORY)
    prepare_directories()
    check_duplicate_process()
    write_json(MANIFEST / "recovery_actions.json", {"actions": RECOVERY_ACTIONS})
    log_event("run_preparing", run_id=RUN_ID, pid=os.getpid())
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(stop_monitor,), daemon=True)
    monitor.start()
    overall_started = time.perf_counter()
    try:
        if not DATASET_ROOT.is_dir():
            raise FileNotFoundError(DATASET_ROOT)
        if shutil.disk_usage(ARTIFACT_ROOT).free < 20 * 1024**3:
            raise OSError("Less than 20 GiB free at artifact root")
        git_head = run_command(["git", "rev-parse", "HEAD"]).strip()
        git_status = run_command(["git", "status", "--porcelain=v2", "--branch", "--untracked-files=all"])
        (MANIFEST / "git_status_start.txt").write_text(git_status, encoding="utf-8")
        (MANIFEST / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")
        code = create_code_manifest()

        seed_everything(ReproducibilityConfig(seed=SEED, deterministic_algorithms=True, warn_only=True))
        gpu_report = run_gpu_preflight()
        weights_name = Path(EfficientNet_B0_Weights.DEFAULT.url).name
        weights_path = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
        if not weights_path.is_file():
            raise FileNotFoundError(f"Official EfficientNet-B0 weights are not cached: {weights_path}")
        environment = environment_manifest(gpu_report, weights_path)

        update_status(current_stage="dataset_filename_preflight")
        files_by_class = {name: discover_jpeg_files(DATASET_ROOT / name) for name in CLASS_NAMES}
        counts = {name: len(files) for name, files in files_by_class.items()}
        if counts != {name: 75_000 for name in CLASS_NAMES}:
            raise ValueError(f"Unexpected dataset counts: {counts}")
        source_sets = [set(files) for files in files_by_class.values()]
        if not all(values == source_sets[0] for values in source_sets[1:]):
            raise ValueError("Dataset basenames no longer correspond across all four classes")
        dataset_start = dataset_snapshot(files_by_class, "start")
        train_frame, validation_frame, test_frame, split_evidence = build_exact_splits()

        identity_crop = AlignedDeterministicCrop(size=256, block_size=8, seed=SEED)
        update_status(current_stage="efficientnet_channel_statistics", current_phase="preparing")
        log_event("channel_statistics_started", training_groups=6_000, training_images=24_000)
        statistics_loader = raw_statistics_loader(train_frame, identity_crop)
        mean, std = compute_channel_statistics(statistics_loader)
        if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(value) and value > 0 for value in std):
            raise FloatingPointError(f"Invalid channel statistics: mean={mean}, std={std}")
        del statistics_loader
        gc.collect()
        log_event("channel_statistics_completed", mean=mean, std=std)

        stage_configuration = inspect_stage_contracts_without_rng_effect()
        resolved_configuration = {
            "run_id": RUN_ID,
            "dataset_root": str(DATASET_ROOT),
            "seed": SEED,
            "binary_target": "Cover versus Stego",
            "dataset_counts": counts,
            "split_evidence": split_evidence,
            "common_counts": {
                "training_groups": 6_000,
                "training_images": 24_000,
                "validation_groups": 750,
                "validation_images": 3_000,
                "final_test_groups": 7_500,
                "final_test_images": 30_000,
            },
            "tinycnn": {
                "batch_size": 48,
                "workers": 2,
                "prefetch_factor": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "crop_size": 256,
                "color_mode": "Y",
                "normalization": {"mean": [0.5], "std": [0.5]},
                "optimizer": "Adam",
                "learning_rate": 1e-4,
                "maximum_epochs": 50,
                "patience": 10,
                "pos_weight": 1 / 3,
                "amp": False,
                "gradient_accumulation_steps": 1,
            },
            "efficientnet_b0": {
                "batch_size": 32,
                "workers": 4,
                "prefetch_factor": 2,
                "pin_memory": True,
                "persistent_workers": True,
                "crop_size": 256,
                "color_mode": "YCbCr",
                "channel_statistics": {"mean": mean, "std": std},
                "weights": str(EfficientNet_B0_Weights.DEFAULT),
                "optimizer": "Adam",
                "pos_weight": 1 / 3,
                "patience": 3,
                "stage_configuration": stage_configuration,
                "amp": False,
                "gradient_accumulation_steps": 1,
            },
            "selection_metric": "official ALASKA2 Weighted AUC on validation only",
            "test_policy": "one complete one-pass evaluation per model after best-state restoration",
            "resolved_at": utc_now(),
        }
        write_json(MANIFEST / "resolved_configuration.json", resolved_configuration)

        common = {
            "run_id": RUN_ID,
            "dataset_root": str(DATASET_ROOT),
            "dataset_identity": dataset_start,
            "seed": SEED,
            "split_membership_hashes": {
                key: value["sha256"] for key, value in split_evidence["memberships"].items()
            },
            "split_and_subset_counts": resolved_configuration["common_counts"],
            "selection_criterion": "validation official ALASKA2 Weighted AUC",
            "code_fingerprint": code,
            "environment_fingerprint": {
                "environment_sha256": environment["environment_sha256"],
                "manifest_path": str(MANIFEST / "environment.json"),
            },
        }

        training_loop._train_epoch = monitored_train_epoch
        training_loop._validate_epoch = monitored_validate_epoch
        training_loop.apply_frozen_eval = monitored_apply_frozen_eval
        update_status(state="running", current_model="tinycnn", current_stage="initializing", current_phase="running")
        tiny = train_tinycnn(train_frame, validation_frame, test_frame, identity_crop, common)
        update_status(current_model="efficientnet_b0", current_stage="initializing", current_phase="running", current_epoch=0)
        efficient = train_efficientnet(
            train_frame,
            validation_frame,
            test_frame,
            identity_crop,
            mean,
            std,
            stage_configuration,
            common,
        )
        training_loop._train_epoch = _original_train_epoch
        training_loop._validate_epoch = _original_validate_epoch
        training_loop.apply_frozen_eval = maintained_apply_frozen_eval

        comparison = [
            {
                "model": "TinyCNN",
                "training_groups": 6_000,
                "training_images": 24_000,
                "best_validation_weighted_auc": tiny["best_validation_weighted_auc"],
                "final_test_accuracy": tiny["final_test_accuracy"],
                "final_test_weighted_auc": tiny["final_test_weighted_auc"],
                "selected_checkpoint": tiny["checkpoint_path"],
                "runtime_seconds": tiny["total_runtime_seconds"],
            },
            {
                "model": "EfficientNet-B0",
                "training_groups": 6_000,
                "training_images": 24_000,
                "best_validation_weighted_auc": efficient["best_validation_weighted_auc"],
                "final_test_accuracy": efficient["final_test_accuracy"],
                "final_test_weighted_auc": efficient["final_test_weighted_auc"],
                "selected_checkpoint": efficient["checkpoint_path"],
                "runtime_seconds": efficient["total_runtime_seconds"],
            },
        ]
        pd.DataFrame(comparison).to_csv(RUN_ROOT / "comparison.csv", index=False)
        write_json(
            RUN_ROOT / "comparison.json",
            {
                "run_id": RUN_ID,
                "note": "Data-matched but not compute-matched comparison",
                "models": comparison,
                "overall_runtime_seconds": time.perf_counter() - overall_started,
            },
        )

        update_status(current_model=None, current_stage="final_verification", current_phase="verifying")
        files_by_class_final = {name: discover_jpeg_files(DATASET_ROOT / name) for name in CLASS_NAMES}
        dataset_final = dataset_snapshot(files_by_class_final, "final")
        if dataset_final["identity_sha256"] != dataset_start["identity_sha256"]:
            raise RuntimeError("Dataset filename/size/mtime identity changed during the run")
        code_final = create_code_manifest()
        if code_final["composite_sha256"] != code["composite_sha256"]:
            raise RuntimeError("Maintained repository file fingerprint changed during the run")
        (MANIFEST / "git_status_final.txt").write_text(
            run_command(["git", "status", "--porcelain=v2", "--branch", "--untracked-files=all"]),
            encoding="utf-8",
        )
        contract = artifact_contract(tiny, efficient)
        final_summary = {
            "run_id": RUN_ID,
            "status": "completed",
            "go_for_readme_notebook_update": True,
            "git_head": git_head,
            "code_fingerprint": code_final,
            "environment_fingerprint": environment["environment_sha256"],
            "dataset_immutable": True,
            "dataset_start_identity_sha256": dataset_start["identity_sha256"],
            "dataset_final_identity_sha256": dataset_final["identity_sha256"],
            "split_evidence": split_evidence,
            "tinycnn_result_path": str(MODEL_ROOTS["tinycnn"] / "evaluation" / "final_result.json"),
            "efficientnet_result_path": str(
                MODEL_ROOTS["efficientnet_b0"] / "evaluation" / "final_result.json"
            ),
            "comparison_paths": [str(RUN_ROOT / "comparison.csv"), str(RUN_ROOT / "comparison.json")],
            "artifact_contract": contract,
            "readme_unchanged": code_final["readme_sha256"] == code["readme_sha256"],
            "notebook_unchanged": code_final["notebook_sha256"] == code["notebook_sha256"],
            "prohibited_actions": {
                "optuna": False,
                "wandb": False,
                "four_class_training": False,
                "resnet": False,
                "srnet": False,
                "xai": False,
                "git_stage": False,
                "git_commit": False,
                "git_push": False,
                "container_rebuild": False,
                "dataset_deletion": False,
            },
            "completion_timestamp": utc_now(),
            "overall_runtime_seconds": time.perf_counter() - overall_started,
            "failures_retries_recoveries": RECOVERY_ACTIONS,
        }
        write_json(RUN_ROOT / "final_summary.json", final_summary)
        update_status(
            state="completed",
            completion_timestamp=final_summary["completion_timestamp"],
            current_model=None,
            current_stage="completed",
            current_phase="completed",
            current_epoch=0,
            final_exit_status=0,
        )
        log_event("run_completed", overall_runtime_seconds=final_summary["overall_runtime_seconds"])
        return 0
    except Exception as error:
        failure_timestamp = utc_now()
        traceback_text = traceback.format_exc()
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "traceback.log").write_text(traceback_text, encoding="utf-8")
        update_status(
            state="failed",
            completion_timestamp=failure_timestamp,
            current_phase="failed",
            final_exit_status=1,
            failure_type=type(error).__name__,
            failure_message=str(error),
            traceback_path=str(LOGS / "traceback.log"),
        )
        log_event(
            "run_failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
            model=_status.get("current_model"),
            stage=_status.get("current_stage"),
            epoch=_status.get("current_epoch"),
        )
        print(traceback_text, file=sys.stderr, flush=True)
        return 1
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
