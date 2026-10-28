"""Load the same OpenCLIP LoRA checkpoint for online queries and map indexing."""

from pathlib import Path

import open_clip
import torch


def _target_from_state(state):
    visual = any(key.startswith("visual.") and "lora_" in key for key in state)
    text = any((key.startswith("transformer.") or key.startswith("text."))
               and "lora_" in key for key in state)
    if visual and text:
        return "all"
    if visual:
        return "vision"
    if text:
        return "text"
    raise ValueError("Checkpoint contains no visual or text LoRA weights")


def load_lora_state_into_model(model, state):
    """Load adapter tensors while requiring every injected adapter to be present."""
    loaded = model.load_state_dict(state, strict=False)
    missing_adapters = [key for key in loaded.missing_keys if "lora_" in key]
    if missing_adapters or loaded.unexpected_keys:
        raise ValueError(
            f"LoRA checkpoint mismatch: {len(missing_adapters)} missing adapters, "
            f"{len(loaded.unexpected_keys)} unexpected keys; "
            f"first missing={missing_adapters[:3]}, unexpected={loaded.unexpected_keys[:3]}")


def load_openclip_lora_checkpoint(checkpoint_path, device="cpu"):
    """Reconstruct a training checkpoint and reject missing adapter weights.

    Older text-only checkpoints did not record ``lora_target``. Their target
    is inferred from the saved adapter key names. Base-model parameters are
    intentionally absent from ``lora_state`` and come from OpenCLIP pretraining.
    """
    from train_localclip import inject_lora_openclip

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "lora_state" not in checkpoint:
        raise ValueError(f"Not a train_localclip LoRA checkpoint: {checkpoint_path}")
    state = checkpoint["lora_state"]
    inferred_target = _target_from_state(state)
    target = checkpoint.get("lora_target", inferred_target)
    if target != inferred_target:
        raise ValueError(f"LoRA target metadata {target!r} disagrees with weights {inferred_target!r}")

    model_name = checkpoint["model_name"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=checkpoint["pretrained"], device="cpu")
    model = inject_lora_openclip(
        model, target=target, last_n_blocks=checkpoint.get("lora_last_n", 12),
        text_last_n_blocks=checkpoint.get("lora_text_last_n"),
        visual_last_n_blocks=checkpoint.get("lora_visual_last_n"),
        r=checkpoint["lora_r"], alpha=checkpoint["lora_alpha"],
        dropout=checkpoint.get("lora_dropout", 0.0))
    load_lora_state_into_model(model, state)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess, checkpoint
