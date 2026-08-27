"""Generic binary-classification training loop.

Used identically by every notebook (1D/2D/3D/multimodal) via a model-specific
`forward_fn(model, batch, device) -> logits` closure — this keeps the training
procedure (optimizer, early stopping, logging) identical across every ablation,
which is what makes the ablation comparison fair.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def evaluate(model, loader, forward_fn, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        logits = forward_fn(model, batch, device)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch["label"].detach().cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    auroc = roc_auc_score(labels, probs)
    loss = nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(logits), torch.tensor(labels)
    ).item()
    return {"auroc": auroc, "loss": loss}


def train_binary_classifier(
    model, train_loader, val_loader, forward_fn, config, experiment_dir, device,
):
    """Trains with early stopping on val AUROC. Saves train_log.csv, best.ckpt, metrics.json."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"]
    )

    best_val_auroc = -1.0
    epochs_without_improvement = 0
    log_rows = []

    log_path = experiment_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_auroc", "elapsed_sec"])

    t_start = time.time()
    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = forward_fn(model, batch, device)
            labels = batch["label"].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_metrics = evaluate(model, val_loader, forward_fn, device)
        elapsed = time.time() - t_start

        row = [epoch, train_loss, val_metrics["loss"], val_metrics["auroc"], elapsed]
        log_rows.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  val_auroc={val_metrics['auroc']:.4f}")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), experiment_dir / "best.ckpt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["train"]["patience"]:
                print(f"Early stopping at epoch {epoch} (no val AUROC improvement for {config['train']['patience']} epochs)")
                break

    model.load_state_dict(torch.load(experiment_dir / "best.ckpt"))
    return model, {"best_val_auroc": best_val_auroc, "epochs_trained": epoch}
