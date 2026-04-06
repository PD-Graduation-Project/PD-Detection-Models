import os
import inspect
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
import yaml
from .losses_binary import CombinedLoss, _binary_metrics

def _ensure_batch_dim(x: torch.Tensor) -> torch.Tensor:
    """Convert a single sample tensor into batch size 1 when needed."""
    return x.unsqueeze(0) if x.dim() == 3 else x


def _load_drawing_loss_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load drawing loss configuration from YAML file."""
    final_path = Path(config_path) if config_path else Path(__file__).with_name("config.yaml")

    if not final_path.exists():
        raise FileNotFoundError(f"Config file not found: {final_path}")

    with final_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    drawing_losses = cfg.get("drawing_losses")
    if not isinstance(drawing_losses, dict):
        raise ValueError("Missing or invalid 'drawing_losses' section in config file")

    return drawing_losses


def _normalize_drawing_loss_kwargs(loss_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize drawing loss kwargs to the signature expected by losses_binary.CombinedLoss.

    Backward compatibility:
    - Old drawing config used `pos_weight`.
    - losses_binary.CombinedLoss expects `healthy_weight` and `parkinson_weight`.
    """
    normalized = dict(loss_kwargs)

    if "pos_weight" in normalized:
        pos_weight = normalized.pop("pos_weight")
        normalized.setdefault("parkinson_weight", pos_weight)
        normalized.setdefault("healthy_weight", 1.0)

    valid_keys = set(inspect.signature(CombinedLoss.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in normalized.items() if k in valid_keys}


def finetune(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    *,
    load_pretrained: str,
    save_path: Optional[str] = None,
    config_path: Optional[str] = None,
    lr: float = 1e-5,
    weight_decay: float = 1e-5,
    grad_clip_max_norm: float = 1.0,
    steps: int = 1,
    loss_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Finetune a trained drawing model using one feedback sample from production.

    Args:
        model: Trained model to finetune.
        image: One image sample (C,H,W) or already batched (1,C,H,W).
        label: Correct label provided by human feedback.
        load_pretrained: Required checkpoint path loaded before finetuning.
        save_path: Optional path to save updated model checkpoint.
        config_path: Optional config path. Defaults to finetune/config.yaml.
        lr: Learning rate for online finetuning.
        weight_decay: Weight decay for AdamW.
        grad_clip_max_norm: Max norm for gradient clipping.
        steps: Number of update steps on the same sample.
        loss_kwargs: Optional kwargs that override drawing_losses from config.

    Returns:
        Dictionary containing latest loss, probabilities, prediction and metrics.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")

    if not os.path.exists(load_pretrained):
        raise FileNotFoundError(f"Checkpoint not found: {load_pretrained}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    checkpoint = torch.load(load_pretrained, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    cfg_loss_kwargs = _load_drawing_loss_config(config_path=config_path)
    merged_loss_kwargs = {**cfg_loss_kwargs, **(loss_kwargs or {})}
    merged_loss_kwargs = _normalize_drawing_loss_kwargs(merged_loss_kwargs)
    loss_fn = CombinedLoss(**merged_loss_kwargs)

    model.train()

    image = _ensure_batch_dim(image).to(device)
    label = label.view(-1, 1).float().to(device)

    last_loss = None
    last_logits = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(image)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = loss_fn(logits, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
        scaler.step(optimizer)
        scaler.update()

        last_loss = loss.detach()
        last_logits = logits.detach()

    probs = torch.sigmoid(last_logits)
    preds = (probs >= 0.5).long()
    metrics = _binary_metrics(last_logits, label)

    result = {
        "loss": float(last_loss.item()),
        "probability": float(probs.squeeze().item()),
        "predicted_label": int(preds.squeeze().item()),
        "target_label": int(label.squeeze().item()),
        "metrics": metrics,
        "steps": steps,
        "learning_rate": lr,
        "device": device,
    }

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "finetune_result": result,
            },
            save_path,
        )

    return result