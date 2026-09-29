#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert Extraction Module

Extract domain-specific experts from a MoE model and build a new dense model
by replacing each MoE layer with the highest-scored expert FFN for the target
domain.

The expert selection relies on the weighted importance scores produced by the
existing ``expert_importance`` module:

    e_score = (f_important - f_unimportant) × p_e

Usage (as library):
    from analysis_specialize.expert_extraction import (
        detect_moe_structure,
        select_experts_by_score,
        build_dense_model,
        save_extracted_model,
    )
"""

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from .token_expert_analysis import ensure_dir, safe_model_name
except ImportError:
    from token_expert_analysis import ensure_dir, safe_model_name


# =============================================================================
# MoE Structure Detection
# =============================================================================

# Attribute names used across popular MoE architectures.
# Order matters: check explicit MoE attributes first, fall back to ``mlp``
# last since ``mlp`` may or may not be MoE depending on the model.
_MOE_LAYER_ATTRS = ("block_sparse_moe", "moe", "feed_forward", "mlp")
_ROUTER_ATTRS = ("gate", "router", "gating_network")
_EXPERT_LIST_ATTRS = ("experts", "local_experts", "mlp_experts")


@dataclass
class MoELayerInfo:
    """Metadata about one MoE layer inside the model."""

    layer_idx: int
    moe_attr: str           # e.g. "mlp", "block_sparse_moe"
    router_attr: str        # e.g. "gate", "router"
    expert_list_attr: str   # e.g. "experts"
    num_experts: int
    expert_module_names: List[str]  # names of child expert modules


@dataclass
class MoEStructure:
    """Summary of the full MoE structure of the model."""

    model_name: str
    num_layers: int
    layers: List[MoELayerInfo]  # only layers that are MoE (some models mix dense + MoE)

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Total transformer layers: {self.num_layers}",
            f"MoE layers detected: {len(self.layers)}",
        ]
        for info in self.layers:
            lines.append(
                f"  Layer {info.layer_idx}: {info.num_experts} experts "
                f"(moe_attr={info.moe_attr}, router={info.router_attr}, "
                f"experts={info.expert_list_attr})"
            )
        return "\n".join(lines)


def _find_transformer_layers(model) -> Optional[nn.ModuleList]:
    """Return the ModuleList of transformer layers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


def detect_moe_structure(model, model_name: str = "") -> MoEStructure:
    """
    Detect the MoE structure of a loaded model.

    Walks through the transformer layers and identifies which ones contain a
    MoE module (with a router + expert list).

    Args:
        model: A loaded ``AutoModelForCausalLM`` instance.
        model_name: Human-readable model name (for reporting only).

    Returns:
        An :class:`MoEStructure` describing the detected MoE layout.

    Raises:
        ValueError: If no transformer layers can be found.
    """
    layers = _find_transformer_layers(model)
    if layers is None:
        raise ValueError(
            "Cannot find transformer layers. "
            "Expected model.model.layers or model.transformer.h"
        )

    moe_layer_infos: List[MoELayerInfo] = []

    for layer_idx, layer in enumerate(layers):
        # Find the MoE sub-module (e.g. layer.mlp)
        moe_module = None
        moe_attr = ""
        for attr in _MOE_LAYER_ATTRS:
            if hasattr(layer, attr):
                candidate = getattr(layer, attr)
                # Verify it has a router – otherwise it's a plain dense MLP
                if any(hasattr(candidate, r) for r in _ROUTER_ATTRS):
                    moe_module = candidate
                    moe_attr = attr
                    break

        if moe_module is None:
            continue

        # Find router attribute name
        router_attr = ""
        for attr in _ROUTER_ATTRS:
            if hasattr(moe_module, attr):
                router_attr = attr
                break

        # Find expert list attribute name
        expert_list_attr = ""
        expert_list = None
        for attr in _EXPERT_LIST_ATTRS:
            if hasattr(moe_module, attr):
                expert_list = getattr(moe_module, attr)
                expert_list_attr = attr
                break

        if expert_list is None:
            continue

        # Number of experts
        if isinstance(expert_list, nn.ModuleList):
            num_experts = len(expert_list)
            expert_names = [f"{expert_list_attr}.{i}" for i in range(num_experts)]
        else:
            continue

        moe_layer_infos.append(
            MoELayerInfo(
                layer_idx=layer_idx,
                moe_attr=moe_attr,
                router_attr=router_attr,
                expert_list_attr=expert_list_attr,
                num_experts=num_experts,
                expert_module_names=expert_names,
            )
        )

    return MoEStructure(
        model_name=model_name or str(type(model).__name__),
        num_layers=len(layers),
        layers=moe_layer_infos,
    )


# =============================================================================
# Expert Selection by Score
# =============================================================================

def load_domain_scores(
    scores_dir: str,
    model_name: str,
    domain: str,
) -> Dict[str, Any]:
    """
    Load the expert importance scores JSON file produced by ``main.py``.

    Args:
        scores_dir: Directory where score files were saved.
        model_name: Model name (used to locate sub-directory).
        domain: Target domain name.

    Returns:
        Parsed JSON data with ``per_layer`` section.
    """
    model_safe = safe_model_name(model_name)
    filename = f"{domain}_trend_results_importance_weighted_scores.json"

    # Try model sub-directory first, then root, then recursive glob
    candidates = [
        Path(scores_dir) / model_safe / filename,
        Path(scores_dir) / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[+] Loaded domain scores from: {candidate}")
            return data

    # Fallback: glob
    matches = sorted(Path(scores_dir).glob(f"**/{filename}"))
    if matches:
        with open(matches[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[+] Loaded domain scores from: {matches[0]}")
        return data

    raise FileNotFoundError(
        f"Expert score file not found for domain '{domain}' under {scores_dir}"
    )


def select_experts_by_score(
    scores_data: Dict[str, Any],
    moe_structure: MoEStructure,
    top_k_per_layer: int = 1,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Select the top-K highest-scored experts per MoE layer.

    Uses the existing ``weighted_score`` from ``expert_importance.py``:
    ``e_score = (f_important - f_unimportant) × p_e``

    Args:
        scores_data: Loaded JSON from ``load_domain_scores``.
        moe_structure: Detected MoE structure.
        top_k_per_layer: Number of experts to select per layer.
            For dense extraction, use ``1`` (best expert per layer).

    Returns:
        Dict mapping ``layer_idx`` → list of ``(expert_idx, score)`` tuples,
        sorted by score descending.
    """
    per_layer = scores_data.get("per_layer", {})
    moe_layer_indices = {info.layer_idx for info in moe_structure.layers}

    selected: Dict[int, List[Tuple[int, float]]] = {}

    for layer_key, experts_dict in per_layer.items():
        layer_idx = int(layer_key)
        if layer_idx not in moe_layer_indices:
            continue

        expert_scores = []
        for expert_key, expert_info in experts_dict.items():
            expert_idx = int(expert_key)
            score = float(expert_info.get("weighted_score", 0.0))
            expert_scores.append((expert_idx, score))

        expert_scores.sort(key=lambda x: x[1], reverse=True)
        selected[layer_idx] = expert_scores[:top_k_per_layer]

    # For MoE layers without scores, fall back to expert 0
    for info in moe_structure.layers:
        if info.layer_idx not in selected:
            print(
                f"  [!] No scores for layer {info.layer_idx}, "
                f"falling back to expert 0"
            )
            selected[info.layer_idx] = [(0, 0.0)]

    return selected


# =============================================================================
# Dense Model Building
# =============================================================================

class _DenseFFNWrapper(nn.Module):
    """
    A thin wrapper that makes a single expert FFN behave like a regular MLP
    layer (i.e. no routing, just a direct forward pass).

    This replaces the entire MoE module (router + expert list) with one expert.
    """

    def __init__(self, expert_module: nn.Module):
        super().__init__()
        self.expert = expert_module

    def forward(self, hidden_states, **kwargs):
        # Expert FFN: hidden_states → hidden_states
        # Some MoE implementations pass extra kwargs (e.g. padding_mask);
        # we just call the expert with hidden_states.
        return self.expert(hidden_states)


def build_dense_model(
    model,
    moe_structure: MoEStructure,
    selected_experts: Dict[int, List[Tuple[int, float]]],
    verbose: bool = True,
) -> nn.Module:
    """
    Build a dense model by replacing each MoE layer with a single expert FFN.

    For each MoE layer, the top-scored expert is extracted and used as a plain
    FFN, removing the router and all other experts.

    Args:
        model: The original MoE model (will be modified **in-place** to save
            memory — pass a ``copy.deepcopy`` if you need the original).
        moe_structure: Detected MoE structure.
        selected_experts: Output of :func:`select_experts_by_score`.
            Only the first expert per layer is used (index 0 in the list).
        verbose: Print progress messages.

    Returns:
        The modified model with MoE layers replaced by dense FFN layers.
    """
    layers = _find_transformer_layers(model)
    if layers is None:
        raise ValueError("Cannot find transformer layers")

    replaced_count = 0

    for info in moe_structure.layers:
        layer = layers[info.layer_idx]
        moe_module = getattr(layer, info.moe_attr)
        expert_list = getattr(moe_module, info.expert_list_attr)

        # Get the selected expert index for this layer
        layer_selection = selected_experts.get(info.layer_idx, [(0, 0.0)])
        expert_idx, expert_score = layer_selection[0]

        if expert_idx >= len(expert_list):
            print(
                f"  [!] Layer {info.layer_idx}: expert {expert_idx} out of range "
                f"(max {len(expert_list) - 1}), using expert 0"
            )
            expert_idx = 0
            expert_score = 0.0

        # Extract the expert module
        expert_module = expert_list[expert_idx]

        # Create dense wrapper
        dense_ffn = _DenseFFNWrapper(expert_module)

        # Replace the entire MoE module with the dense FFN
        setattr(layer, info.moe_attr, dense_ffn)
        replaced_count += 1

        if verbose:
            print(
                f"  Layer {info.layer_idx}: replaced MoE ({info.num_experts} experts) "
                f"→ dense FFN (expert {expert_idx}, score={expert_score:.6f})"
            )

    if verbose:
        print(f"\n[+] Replaced {replaced_count} MoE layer(s) with dense FFN")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[+] Dense model total parameters: {total_params:,}")

    return model


# =============================================================================
# Model Saving & Validation
# =============================================================================

def save_extracted_model(
    model,
    tokenizer,
    output_path: str,
    extraction_info: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> str:
    """
    Save the extracted dense model in HuggingFace format.

    Args:
        model: The dense model to save.
        tokenizer: The tokenizer (saved alongside the model).
        output_path: Directory to save into.
        extraction_info: Optional metadata about the extraction process.
        verbose: Print progress.

    Returns:
        The output directory path.
    """
    ensure_dir(output_path)

    # Save model weights as a PyTorch state dict
    # (We use state_dict instead of save_pretrained because the model
    # architecture has been modified and may not match the original config)
    weights_path = os.path.join(output_path, "pytorch_model.bin")
    torch.save(model.state_dict(), weights_path)

    if verbose:
        print(f"[+] Model weights saved to: {weights_path}")

    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    if verbose:
        print(f"[+] Tokenizer saved to: {output_path}")

    # Save extraction metadata
    if extraction_info is not None:
        meta_path = os.path.join(output_path, "extraction_info.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(extraction_info, f, indent=2)
        if verbose:
            print(f"[+] Extraction info saved to: {meta_path}")

    return output_path


def validate_extracted_model(
    model,
    tokenizer,
    test_prompts: Optional[List[str]] = None,
    max_new_tokens: int = 50,
    verbose: bool = True,
) -> List[Dict[str, str]]:
    """
    Smoke-test the extracted model by generating text from test prompts.

    Args:
        model: The extracted model.
        tokenizer: The tokenizer.
        test_prompts: List of prompts to test. Uses defaults if None.
        max_new_tokens: Maximum tokens to generate per prompt.
        verbose: Print outputs.

    Returns:
        List of dicts with ``prompt`` and ``output`` keys.
    """
    if test_prompts is None:
        test_prompts = [
            "What is 2 + 2?",
            "Solve for x: 3x + 5 = 20",
            "The derivative of x^2 is",
        ]

    model.eval()
    device = next(model.parameters()).device
    results = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt in test_prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        result = {"prompt": prompt, "output": output_text.strip()}
        results.append(result)

        if verbose:
            print(f"\n  Prompt: {prompt}")
            print(f"  Output: {output_text.strip()[:200]}")

    return results
