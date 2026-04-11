"""
Structured channel pruning for model compression.
Removes 30% of neural connections that don't contribute to accuracy.
"""
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


def sensitivity_analysis(model: nn.Module, dataloader, loss_fn, device="cpu", amounts=None):
    """Test each layer's pruning tolerance independently."""
    if amounts is None:
        amounts = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = {}
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        results[name] = {}
        for amount in amounts:
            model.load_state_dict({k: v.clone() for k, v in original_state.items()})
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
            # Quick eval
            model.eval()
            total_loss = 0
            count = 0
            with torch.no_grad():
                for batch in dataloader:
                    if count >= 5:
                        break
                    img_a, img_b, mask = [x.to(device) for x in batch]
                    spk, _ = model(img_a, img_b)
                    total_loss += loss_fn(spk, mask).item()
                    count += 1
            results[name][amount] = total_loss / max(count, 1)

    model.load_state_dict(original_state)
    return results


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    iterative_rounds: int = 3,
    fine_tune_fn=None,
    skip_patterns: list = None,
):
    """
    Structured channel pruning: removes entire filters by L2-norm.

    Args:
        model: The SiameseSNN model
        amount: Total pruning fraction (split across rounds)
        iterative_rounds: Number of prune-finetune rounds
        fine_tune_fn: Callable(model, epochs) for fine-tuning after each round
        skip_patterns: Layer name patterns to skip (e.g. skip connections)
    """
    if skip_patterns is None:
        skip_patterns = ["output_conv", "bottleneck"]

    per_round = 1.0 - (1.0 - amount) ** (1.0 / iterative_rounds)

    for round_num in range(1, iterative_rounds + 1):
        pruned_layers = 0
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if any(pat in name for pat in skip_patterns):
                continue
            if module.weight.shape[0] <= 8:  # Don't prune tiny layers
                continue

            prune.ln_structured(module, name="weight", amount=per_round, n=2, dim=0)
            pruned_layers += 1

        logger.info(f"Pruning round {round_num}/{iterative_rounds}: "
                    f"{per_round:.1%} of {pruned_layers} layers")

        if fine_tune_fn:
            fine_tune_fn(model, epochs=10)

    # Make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")

    # Report compression
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    logger.info(f"Pruning complete: {zero_params/total_params:.1%} parameters zeroed, "
                f"{total_params - zero_params:,} effective parameters")

    return model


def get_model_stats(model: nn.Module) -> dict:
    """Get model size and parameter statistics."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    zeros = sum((p == 0).sum().item() for p in model.parameters())
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6

    return {
        "total_params": total,
        "trainable_params": trainable,
        "zero_params": zeros,
        "sparsity": zeros / total if total > 0 else 0,
        "size_mb": size_mb,
    }
