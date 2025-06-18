from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.model.model_metrics import weighted_auc


# ───────────────────────────── Helper ────────────────────────────── #
def _move_to_device(
    inputs: torch.Tensor | tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor]:
    """
    Verschiebt Inputs & Labels auf das gewünschte Gerät.

    Unterstützt sowohl Einzel- als auch Tuple-Inputs
    (z. B. (img, dct) aus den Fusion-Datasets).

    Args:
        inputs: Tensor oder Tupel aus Tensoren.
        labels: Zielvariable.
        device: torch.device("cpu") bzw. ("cuda").
    """
    if isinstance(inputs, tuple):
        inputs = tuple(x.to(device) for x in inputs)
    else:
        inputs = inputs.to(device)
    return inputs, labels.to(device)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    *,
    use_tqdm: bool = False,
    desc: str = "",
) -> tuple[float, float]:
    """
    Führt **eine Trainings-Epoche** aus.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0 … 1)
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    iterator = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader

    for inputs, labels in iterator:
        inputs, labels = _move_to_device(inputs, labels, device)
        labels = labels.view(-1, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if use_tqdm:
            iterator.set_postfix(loss=loss.item())

    return running_loss / total, correct / total


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
    """
    Val/Test-Durchlauf ohne Gradienten.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = _move_to_device(inputs, labels, device)
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * labels.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return val_loss / total, correct / total


@torch.no_grad()
def _collect_probs(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Gibt zwei 1-D-Arrays zurück:
        y_true : ground-truth-Label (0/1)
        y_prob : Pr(Stego) aus Sigmoid(logit)
    """
    model.eval()
    y_true, y_prob = [], []

    for inputs, labels in loader:
        inputs, labels = _move_to_device(inputs, labels, device)
        labels = labels.view(-1, 1)
        logits = model(inputs)
        probs = torch.sigmoid(logits).view(-1)
        y_true.append(labels.cpu())
        y_prob.append(probs.cpu())

    return torch.cat(y_true).numpy(), torch.cat(y_prob).numpy()


# ───────────────────────── High-Level API ───────────────────────── #
def run_experiment(
    net: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    *,
    num_epochs: int = 30,
    device: str | torch.device = "cpu",
    show_device: bool = False,
    run_name: str | None = None,
    save_dir: str | Path | None = None,  # Ordner für .pt
    save_csv: str | Path | None = None,  # Ordner für History-CSV
    patience: int = 5,  # Early-Stopping
    min_delta: float = 1e-4,  # minimale wAUC-Steigerung
    use_tqdm: bool = True,
    show_summary: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Komplettes Train-&-Val-Framework **optimiert auf ALASKA-Weighted-AUC**.

    • Speichert bestes Modell nach *val_wauc*
    • Early-Stopping, History-CSV, Checkpoint optional
    • Keine Prints – alle Infos in Rückgabewerten

    Returns
    -------
    hist_df : pd.DataFrame
        epoch, train_loss, train_acc, val_loss, val_acc, val_wauc
    summary : dict
        best_val_wauc, best_epoch, final_*, early_stopped, best_checkpoint
    """
    # ───────── Vorbereitung ─────────
    device_req = torch.device(device)
    device = device_req if device_req.type != "cuda" or torch.cuda.is_available() else torch.device("cpu")

    run_name = run_name or f"{net.__class__.__name__}_{dt.datetime.now():%Y%m%d-%H%M%S}"

    if show_device:
        print(f"[{run_name}]  Device in use: {device}")

    net = net.to(device)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        save_csv = Path(save_csv)
        save_csv.mkdir(parents=True, exist_ok=True)

    if show_summary:
        try:
            from torchinfo import summary

            sample_in, sample_label = next(iter(train_loader))
            sample_in, sample_label = _move_to_device(sample_in, sample_label, device)
            if isinstance(sample_in, torch.Tensor) and sample_in.ndim == 3:
                sample_in = sample_in.unsqueeze(0)
            summary(net, input_data=sample_in, verbose=1)
        except Exception as exc:
            print("⚠️  torchinfo summary failed:", exc)

    history = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_wauc")}

    best_wauc: float = 0.0
    best_ckpt: Optional[Path] = None
    epochs_no_improve = 0

    # ───────── Trainings-Loop ─────────
    for epoch in range(1, num_epochs + 1):
        tl, ta = _train_one_epoch(net, train_loader, criterion, optimizer, device, use_tqdm=use_tqdm, desc=f"Ep {epoch}/{num_epochs}")
        vl, va = _validate(net, val_loader, criterion, device)

        # wAUC berechnen
        y_val, p_val = _collect_probs(net, val_loader, device)
        wauc = weighted_auc(y_val, p_val)

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        history["val_wauc"].append(wauc)

        # Checkpoint + Early-Stopping
        if wauc > best_wauc + min_delta:
            best_wauc = wauc
            epochs_no_improve = 0
            if save_dir:
                best_ckpt = save_dir / f"{run_name}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": net.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_wauc": wauc,
                        "val_loss": vl,
                    },
                    best_ckpt,
                )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break  # Early-Stopping

    # ───────── Resultate ─────────
    hist_df = pd.DataFrame(history)
    if save_csv:
        hist_df.to_csv(save_csv / f"{run_name}_history.csv", index=False)

    summary: Dict = {
        "run": run_name,
        "best_val_wauc": best_wauc,
        "best_epoch": int(hist_df.loc[hist_df["val_wauc"].idxmax(), "epoch"]),
        "final_val_wauc": float(hist_df["val_wauc"].iloc[-1]),
        "final_val_acc": float(hist_df["val_acc"].iloc[-1]),
        "final_train_acc": float(hist_df["train_acc"].iloc[-1]),
        "early_stopped": epochs_no_improve >= patience,
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
    }
    return hist_df, summary
