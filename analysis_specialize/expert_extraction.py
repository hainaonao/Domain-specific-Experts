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
        # Find the MoE sub-module (e.g. layer.mlp, layer.block_sparse_moe)
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

        # Determine number of experts from the expert list.
        # Support nn.ModuleList, nn.ModuleDict, or any sized container.
        # Some models (e.g. OLMoE) wrap experts in a custom nn.Module;
        # we need to dig into children to find the actual expert list.
        num_experts = 0
        expert_names: List[str] = []
        # actual_expert_list: the real indexable container of expert modules
        # (may differ from expert_list if expert_list is a wrapper)
        actual_expert_list = expert_list

        if expert_list is not None:
            if isinstance(expert_list, nn.ModuleList):
                num_experts = len(expert_list)
                expert_names = [f"{expert_list_attr}.{i}" for i in range(num_experts)]
            elif isinstance(expert_list, nn.ModuleDict):
                num_experts = len(expert_list)
                expert_names = [
                    f"{expert_list_attr}.{k}" for k in expert_list.keys()
                ]
            elif isinstance(expert_list, nn.Module):
                # Custom wrapper module — dig into children to find the
                # actual ModuleList of experts.
                found_inner = False
                for child_name, child in expert_list.named_children():
                    if isinstance(child, nn.ModuleList) and len(child) > 1:
                        # Found the real expert list inside the wrapper
                        num_experts = len(child)
                        actual_expert_list = child
                        expert_names = [
                            f"{expert_list_attr}.{child_name}.{i}"
                            for i in range(num_experts)
                        ]
                        found_inner = True
                        break

                if not found_inner:
                    # No inner ModuleList found; count direct children
                    children = list(expert_list.children())
                    if len(children) > 1:
                        num_experts = len(children)
                        expert_names = [
                            f"{expert_list_attr}.{i}"
                            for i in range(num_experts)
                        ]
                    # If only 1 child, don't count it as 1 expert — it's
                    # likely a wrapper. Leave num_experts = 0 for config
                    # fallback below.

            elif hasattr(expert_list, "__len__"):
                num_experts = len(expert_list)
                expert_names = [
                    f"{expert_list_attr}.{i}" for i in range(num_experts)
                ]

        # Cross-check / fallback with model config
        model_config = getattr(model, "config", None)
        config_num_experts = 0
        if model_config is not None:
            for cfg_attr in ("num_experts", "num_local_experts",
                             "n_experts", "moe_num_experts"):
                if hasattr(model_config, cfg_attr):
                    config_num_experts = int(getattr(model_config, cfg_attr))
                    break

        if num_experts == 0 and config_num_experts > 0:
            # Detection failed but config knows the expert count
            num_experts = config_num_experts
            expert_list_attr = expert_list_attr or "(from config)"
            expert_names = [f"expert.{i}" for i in range(num_experts)]
        elif num_experts > 0 and config_num_experts > 0 and num_experts != config_num_experts:
            # Mismatch: trust config over detection
            print(
                f"  [!] Layer {layer_idx}: detected {num_experts} experts "
                f"but config says {config_num_experts}. Using config value."
            )
            num_experts = config_num_experts
            expert_names = [f"{expert_list_attr}.{i}" for i in range(num_experts)]

        if num_experts == 0:
            # Debug: show what we found so the user can diagnose
            print(
                f"  [!] Layer {layer_idx}: found MoE module "
                f"(attr={moe_attr!r}, router={router_attr!r}) but could not "
                f"determine expert count. Children: "
                f"{[name for name, _ in moe_module.named_children()]}"
            )
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

def _resolve_expert_list(source_module) -> Optional[nn.ModuleList]:
    """
    Find the actual indexable expert ``nn.ModuleList`` from a MoE module.

    Strategy (in order):
    1. Check known attribute names (``experts``, ``local_experts``, etc.)
    2. If the attribute is already a ``ModuleList`` → return it
    3. If the attribute is a wrapper ``Module`` → dig into its children
    4. Fall back to scanning **all** named children for the largest
       ``ModuleList`` (works for any architecture)

    Args:
        source_module: The MoE module (e.g. ``layer.mlp``) or an expert
            list attribute.

    Returns:
        The ``nn.ModuleList`` of expert FFN modules, or ``None``.
    """
    if source_module is None:
        return None

    # Already a ModuleList — use directly
    if isinstance(source_module, nn.ModuleList):
        return source_module if len(source_module) > 1 else None

    if not isinstance(source_module, nn.Module):
        return None

    # Strategy 1: try known attribute names on the module
    for attr in _EXPERT_LIST_ATTRS:
        child = getattr(source_module, attr, None)
        if child is None:
            continue
        if isinstance(child, nn.ModuleList) and len(child) > 1:
            return child
        # Might be a wrapper Module containing the real ModuleList
        if isinstance(child, nn.Module):
            for _n, grandchild in child.named_children():
                if isinstance(grandchild, nn.ModuleList) and len(grandchild) > 1:
                    return grandchild

    # Strategy 2: scan ALL direct children for the largest ModuleList
    best_list = None
    best_len = 0
    for _name, child in source_module.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > best_len:
            best_list = child
            best_len = len(child)

    if best_list is not None and best_len > 1:
        return best_list

    # Strategy 3: recursive scan (one level deeper)
    for _name, child in source_module.named_children():
        if isinstance(child, nn.Module) and not isinstance(child, nn.ModuleList):
            for _n2, grandchild in child.named_children():
                if isinstance(grandchild, nn.ModuleList) and len(grandchild) > best_len:
                    best_list = grandchild
                    best_len = len(grandchild)

    if best_list is not None and best_len > 1:
        return best_list

    return None

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
        return self.expert(hidden_states)


class _FusedExpertFFN(nn.Module):
    """
    A standalone FFN extracted from a fused expert module.

    Fused experts (e.g. OlmoeExperts) store all expert weights in stacked 3D
    tensors like ``gate_up_proj[num_experts, ...]``.  This class holds the
    sliced weights for a single expert and performs the SwiGLU forward pass.
    """

    def __init__(self, gate_proj_weight, up_proj_weight, down_proj_weight,
                 act_fn=None):
        super().__init__()
        self.gate_proj = nn.Linear(
            gate_proj_weight.shape[1], gate_proj_weight.shape[0], bias=False
        )
        self.up_proj = nn.Linear(
            up_proj_weight.shape[1], up_proj_weight.shape[0], bias=False
        )
        self.down_proj = nn.Linear(
            down_proj_weight.shape[1], down_proj_weight.shape[0], bias=False
        )
        self.gate_proj.weight = nn.Parameter(gate_proj_weight.clone())
        self.up_proj.weight = nn.Parameter(up_proj_weight.clone())
        self.down_proj.weight = nn.Parameter(down_proj_weight.clone())
        self.act_fn = act_fn or nn.SiLU()

    def forward(self, hidden_states, **kwargs):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def _get_expert_weight(module, attr_name, expert_idx):
    """Get weight for a specific expert from a 3D parameter or Linear."""
    param = getattr(module, attr_name, None)
    if param is None:
        return None
    if isinstance(param, (nn.Parameter, torch.Tensor)):
        if param.dim() == 3 and expert_idx < param.shape[0]:
            return param[expert_idx].detach()
    if isinstance(param, nn.Linear):
        w = param.weight
        if w.dim() == 3 and expert_idx < w.shape[0]:
            return w[expert_idx].detach()
    return None


def _extract_expert_from_fused(fused_module, expert_idx, num_experts):
    """
    Extract a single expert from a fused expert module.

    Handles patterns:
    1. Separate gate_proj, up_proj, down_proj (3D)
    2. Combined gate_up_proj + down_proj
    3. Any 3D parameters with dim0 == num_experts
    """
    # Pattern 1: separate projections
    gate_w = _get_expert_weight(fused_module, "gate_proj", expert_idx)
    up_w = _get_expert_weight(fused_module, "up_proj", expert_idx)
    down_w = _get_expert_weight(fused_module, "down_proj", expert_idx)

    if gate_w is not None and up_w is not None and down_w is not None:
        return _FusedExpertFFN(gate_w, up_w, down_w)

    # Pattern 2: combined gate_up_proj
    gate_up_w = _get_expert_weight(fused_module, "gate_up_proj", expert_idx)
    down_w = down_w if down_w is not None else _get_expert_weight(fused_module, "down_proj", expert_idx)

    if gate_up_w is not None and down_w is not None:
        mid = gate_up_w.shape[0] // 2
        return _FusedExpertFFN(gate_up_w[:mid], gate_up_w[mid:], down_w)

    # Pattern 3: scan all 3D parameters with dim0 == num_experts
    param_3d = {}
    for name, param in fused_module.named_parameters(recurse=False):
        if param.dim() == 3 and param.shape[0] == num_experts:
            param_3d[name] = param[expert_idx].detach()

    if len(param_3d) >= 2:
        sorted_params = sorted(param_3d.items())
        if len(sorted_params) == 2:
            combined_w = sorted_params[0][1]
            d_w = sorted_params[1][1]
            mid = combined_w.shape[0] // 2
            return _FusedExpertFFN(combined_w[:mid], combined_w[mid:], d_w)
        elif len(sorted_params) >= 3:
            return _FusedExpertFFN(
                sorted_params[0][1], sorted_params[1][1], sorted_params[2][1]
            )

    return None


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

        # Get the selected expert index for this layer
        layer_selection = selected_experts.get(info.layer_idx, [(0, 0.0)])
        expert_idx, expert_score = layer_selection[0]

        # ---- Approach 1: ModuleList-based experts ----
        resolved_list = _resolve_expert_list(moe_module)
        expert_module = None

        if resolved_list is not None and len(resolved_list) > 0:
            if expert_idx >= len(resolved_list):
                print(
                    f"  [!] Layer {info.layer_idx}: expert {expert_idx} "
                    f"out of range (max {len(resolved_list) - 1}), using 0"
                )
                expert_idx, expert_score = 0, 0.0
            expert_module = resolved_list[expert_idx]

        # ---- Approach 2: Fused expert module (e.g. OlmoeExperts) ----
        if expert_module is None:
            for attr in _EXPERT_LIST_ATTRS:
                candidate = getattr(moe_module, attr, None)
                if (candidate is not None
                        and isinstance(candidate, nn.Module)
                        and not isinstance(candidate, nn.ModuleList)):
                    expert_module = _extract_expert_from_fused(
                        candidate, expert_idx, info.num_experts
                    )
                    if expert_module is not None:
                        if verbose:
                            print(
                                f"  [i] Layer {info.layer_idx}: extracted "
                                f"expert {expert_idx} from fused "
                                f"{type(candidate).__name__}"
                            )
                        break

        if expert_module is None:
            children_info = [
                f"{n}({type(c).__name__})"
                for n, c in moe_module.named_children()
            ]
            print(
                f"  [!] Layer {info.layer_idx}: could not extract expert "
                f"{expert_idx}. moe children: {children_info}"
            )
            for attr in _EXPERT_LIST_ATTRS:
                candidate = getattr(moe_module, attr, None)
                if candidate is not None and isinstance(candidate, nn.Module):
                    params = [
                        f"{n}: {list(p.shape)}"
                        for n, p in candidate.named_parameters(recurse=False)
                    ]
                    print(f"      {attr} params: {params[:6]}")
            continue

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
