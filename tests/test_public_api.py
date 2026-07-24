"""
===============================================================================
test_public_api.py
===============================================================================
Verify package exports and protected notebook orchestration contracts.

Responsibilities:
  - Ensure every public package alias matches its explicit __all__
    declaration.
  - Reject legacy imports, stale paths, embedded model classes, and notebook
    outputs.

Design principles:
  - Public surfaces are tested as modules rather than duplicated symbol re-
    exports.
  - Notebook checks preserve its role as orchestration and academic
    presentation.

Boundaries:
  - Notebook execution and rendered visuals are validated separately.
  - Private implementation helpers are intentionally not part of this
    contract.
===============================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

import src


def test_root_exposes_only_domain_packages() -> None:
    assert src.__all__ == [
        "config",
        "data",
        "datasets",
        "eda",
        "evaluation",
        "models",
        "presentation",
        "training",
        "transforms",
    ]
    assert not hasattr(src, "model")
    assert not hasattr(src, "util")


def test_package_exports_are_module_aliases() -> None:
    packages = {
        src.config: ["device", "paths", "runtime"],
        src.data: ["index", "metadata", "preflight", "preparation", "split", "synthetic"],
        src.datasets: ["dct", "images", "loaders"],
        src.eda: ["channels", "dct", "examples", "overview", "style"],
        src.evaluation: ["metrics", "plots", "runner"],
        src.models: ["efficientnet", "freezing", "tinycnn"],
        src.presentation: ["cache", "widgets"],
        src.training: ["artifacts", "checkpoint", "loop", "staged"],
        src.transforms: ["shuffle", "spatial"],
    }
    for package, expected in packages.items():
        assert package.__all__ == expected
        for alias in expected:
            assert getattr(package, alias).__name__.startswith("src.")
    assert "select_eda_population" in src.data.preparation.__all__
    assert "resolve_artifact_paths" in src.training.artifacts.__all__


def test_notebook_uses_maintained_orchestration_apis_and_safe_defaults() -> None:
    notebook_path = Path(__file__).parents[1] / "ANN_Projekt_Rino_Albertin_Steganalyse.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    code_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "class TinyCNN" not in source
    assert "class EfficientNetB0_YCbCr" not in source
    assert "def _compute_mean_std" not in source
    assert "src import eda, util, model" not in source
    assert "evaluate_binary_model" not in source
    assert "RandomGridShuffle" not in source
    assert "images/" not in source
    assert "train_model = True" not in source
    assert "LOAD_EXISTING_ARTIFACTS" not in source
    assert "SupportsFloat" not in source
    assert "SupportsInt" not in source
    assert "TRAIN_TINYCNN = False" in source
    assert "TRAIN_EFFICIENTNET = False" in source
    assert "prepare_dataset(" in source
    assert "select_eda_population(" in source
    assert "split_reservoir_subsets(" not in source
    assert "subsample_fraction" not in source
    assert "build_image_loaders(" in source
    assert source.count("make_evaluation_widget(") == 1
    assert "GENERATE_NEW_SYNTHETIC_PROXY = False" in source
    assert "EXPERIMENT_SEED = 42" in source
    assert "resolve_artifact_paths(" in source
    assert "tiny_checkpoint_path =" not in source
    assert "project_paths.checkpoints / dataset_name.lower()" not in source
    assert "OPTUNA_RUN" not in source
    assert "predictions/final_test_predictions.csv" not in source
    assert "/artifacts" not in code_source
    assert "/datasets" not in code_source
    assert "/workspaces" not in code_source
    assert "/app" not in code_source
    assert "/figures/" not in source


def test_notebook_is_valid_nbformat_and_code_cells_compile() -> None:
    notebook_path = Path(__file__).parents[1] / "ANN_Projekt_Rino_Albertin_Steganalyse.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    transformer = TransformerManager()

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        transformed = transformer.transform_cell(cell.source)
        compile(transformed, f"{notebook_path.name}:cell-{index}", "exec")
