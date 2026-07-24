"""
===============================================================================
models_freezing.py
===============================================================================
Explicit EfficientNet fine-tuning stages.

Responsibilities:
  - Freeze the full network before selecting trainable modules.
  - Define a head/stem stage followed by one reverse backbone block per
    stage.
  - Keep frozen batch-normalization modules in evaluation mode during
    training.

Design principles:
  - Every stage starts from a known frozen state. Stage descriptions hold
    module references and are derived once from the concrete EfficientNet
    adapter.

Boundaries:
  - This module does not create optimizers, train models, or save
    checkpoints.

Notes:
  - The classifier remains trainable in every stage; each feature block is
    selected in exactly one stage.
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .models_efficientnet import EfficientNetB0YCbCr

__all__ = [
    "FineTuneStage",
    "apply_frozen_eval",
    "configure_stage",
    "efficientnet_stages",
    "trainable_parameters",
]


@dataclass(frozen=True, slots=True)
class FineTuneStage:
    """Describe the modules and optimization schedule for one fine-tuning stage.

    Parameters
    ----------
    name
        Unique stage identifier used for artifacts.
    modules
        Modules made trainable for this stage.
    learning_rate
        Positive optimizer learning rate.
    epochs
        Positive maximum epoch count.
    """

    name: str
    modules: tuple[nn.Module, ...]
    learning_rate: float
    epochs: int


def efficientnet_stages(
    model: EfficientNetB0YCbCr,
    *,
    head_learning_rate: float,
    block_learning_rate: float,
    head_epochs: int,
    block_epochs: int,
) -> tuple[FineTuneStage, ...]:
    """Build a head/stem stage followed by reverse feature-block stages.

    Parameters
    ----------
    model
        Concrete EfficientNet-B0 YCbCr adapter.
    head_learning_rate
        Learning rate for the initial classifier and stem stage.
    block_learning_rate
        Learning rate for later feature-block stages.
    head_epochs
        Maximum epochs for the initial stage.
    block_epochs
        Maximum epochs for each later stage.

    Returns
    -------
    tuple[FineTuneStage, ...]
        Ordered non-overlapping stage definitions.

    Raises
    ------
    ValueError
        If a learning rate or epoch count is not positive.
    """
    if min(head_learning_rate, block_learning_rate) <= 0:
        raise ValueError("Learning rates must be positive.")
    if min(head_epochs, block_epochs) <= 0:
        raise ValueError("Stage epoch counts must be positive.")
    features = list(model.backbone.features)
    head = FineTuneStage(
        name="head_stem",
        modules=(model.backbone.classifier, features[0]),
        learning_rate=head_learning_rate,
        epochs=head_epochs,
    )
    block_stages = tuple(
        FineTuneStage(
            name=f"feature_{index}",
            modules=(model.backbone.classifier, features[index]),
            learning_rate=block_learning_rate,
            epochs=block_epochs,
        )
        for index in range(len(features) - 1, 0, -1)
    )
    return (head, *block_stages)


def configure_stage(model: nn.Module, stage: FineTuneStage) -> None:
    """Freeze the complete model and enable only modules declared by one stage.

    Parameters
    ----------
    model
        Model whose parameter trainability is reset.
    stage
        Stage containing the modules to enable.

    Returns
    -------
    None
        Parameter ``requires_grad`` flags are updated in place.

    Raises
    ------
    ValueError
        If the stage selects no trainable parameters.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in stage.modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError(f"Stage {stage.name!r} selected no trainable parameters.")


def apply_frozen_eval(model: nn.Module) -> None:
    """Keep fully frozen modules in evaluation mode after ``model.train()``.

    Parameters
    ----------
    model
        Model containing potentially frozen stateful modules.

    Returns
    -------
    None
        Fully frozen child modules are switched to evaluation mode in place.
    """
    for module in model.modules():
        parameters = tuple(module.parameters(recurse=False))
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect the parameters enabled for the current fine-tuning stage.

    Parameters
    ----------
    model
        Configured model to inspect.

    Returns
    -------
    list[torch.nn.Parameter]
        Non-empty list of trainable parameters.

    Raises
    ------
    ValueError
        If no model parameter is trainable.
    """
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters.")
    return parameters
