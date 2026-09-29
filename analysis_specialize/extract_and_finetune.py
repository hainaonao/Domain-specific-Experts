#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert Extraction & Finetuning CLI

End-to-end pipeline:
1. Load expert importance scores (from a previous ``main.py`` run)
2. Load the original MoE model
3. Select top-scored experts per layer for the target domain
4. Build a dense model by replacing MoE layers with single expert FFNs
5. Validate and save the extracted model
6. (Optional) Finetune the extracted model with LoRA or full finetuning

Usage:
    # Extract only
    python -m analysis_specialize.extract_and_finetune \\
        --config configs/extract_config.yaml

    # Extract + finetune
    python -m analysis_specialize.extract_and_finetune \\
        --config configs/extract_config.yaml \\
        --finetune --train_data data/math_train.jsonl
"""

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml

try:
    from .main import load_model_and_tokenizer
    from .token_expert_analysis import ensure_dir, safe_model_name
    from .expert_extraction import (
        build_dense_model,
        detect_moe_structure,
        load_domain_scores,
        save_extracted_model,
        select_experts_by_score,
        validate_extracted_model,
    )
except ImportError:
    from main import load_model_and_tokenizer
    from token_expert_analysis import ensure_dir, safe_model_name
    from expert_extraction import (
        build_dense_model,
        detect_moe_structure,
        load_domain_scores,
        save_extracted_model,
        select_experts_by_score,
        validate_extracted_model,
    )


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for expert extraction and optional finetuning."""

    # Model
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B"
    device: str = "auto"

    # Expert scores (output from main.py)
    expert_scores_path: str = "output/sample_run"
    target_domain: str = "mathematics"

    # Extraction
    extraction_mode: str = "dense"  # currently only "dense" supported
    top_k_experts_per_layer: int = 1  # for dense mode, use 1

    # Validation
    validate: bool = True
    test_prompts: Optional[List[str]] = None

    # Output
    output_dir: str = "output/extracted_models"

    # Finetuning (optional)
    finetune: bool = False
    train_data: Optional[str] = None
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_sequence_length: int = 512
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    verbose: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExtractionConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# =============================================================================
# Finetuning
# =============================================================================

def finetune_model(
    model,
    tokenizer,
    config: ExtractionConfig,
) -> str:
    """
    Finetune the extracted dense model.

    Supports both LoRA and full finetuning. Requires the ``peft`` package
    when ``use_lora=True``.

    Args:
        model: The extracted dense model.
        tokenizer: The tokenizer.
        config: Extraction configuration with finetune settings.

    Returns:
        Path to the saved finetuned model.
    """
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )

    if config.train_data is None:
        raise ValueError("train_data must be specified for finetuning")

    # Load training data
    print(f"\n[Finetune] Loading training data from: {config.train_data}")
    train_texts = _load_training_texts(config.train_data)
    if not train_texts:
        raise ValueError(f"No training examples found in {config.train_data}")
    print(f"[Finetune] Loaded {len(train_texts)} training examples")

    # Tokenize
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=config.max_sequence_length,
        padding="max_length",
        return_tensors="pt",
    )

    # Create dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings.input_ids[idx],
                "attention_mask": self.encodings.attention_mask[idx],
                "labels": self.encodings.input_ids[idx].clone(),
            }

    dataset = TextDataset(encodings)

    # Apply LoRA if requested
    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print("[Finetune] LoRA applied")
        except ImportError:
            print(
                "[!] peft package not installed. "
                "Install with: pip install peft\n"
                "Falling back to full finetuning."
            )

    # Training arguments
    model_safe = safe_model_name(config.model_name)
    ft_output_dir = os.path.join(
        config.output_dir, model_safe,
        f"{config.target_domain}_dense_finetuned",
    )

    training_args = TrainingArguments(
        output_dir=ft_output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"\n[Finetune] Starting training for {config.num_epochs} epoch(s)...")
    trainer.train()

    # Save
    trainer.save_model(ft_output_dir)
    tokenizer.save_pretrained(ft_output_dir)
    print(f"\n[Finetune] Model saved to: {ft_output_dir}")

    return ft_output_dir


def _load_training_texts(data_path: str) -> List[str]:
    """
    Load training texts from a JSON/JSONL file.

    Supports formats:
    - JSONL with ``question`` + ``options`` fields (MCQA format)
    - JSONL with a ``text`` field
    - Plain text file (one example per line)
    """
    from pathlib import Path

    path = Path(data_path)
    texts = []

    if path.suffix.lower() in (".jsonl", ".json"):
        records = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]

        for record in records:
            if isinstance(record, str):
                texts.append(record)
                continue
            if not isinstance(record, dict):
                continue

            # Try MCQA format
            question = record.get("question") or record.get("prompt")
            options = record.get("options") or record.get("choices")
            if question and options:
                if isinstance(options, list):
                    prompt = question + "\n" + "\n".join(
                        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
                    )
                    answer = record.get("answer", "")
                    texts.append(f"{prompt}\nAnswer: {answer}")
                    continue

            # Try text field
            text = record.get("text") or record.get("content") or record.get("input")
            if text:
                texts.append(str(text))

    elif path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    return texts


# =============================================================================
# Main Pipeline
# =============================================================================

def run_extraction(config: ExtractionConfig) -> Dict[str, Any]:
    """
    Run the full expert extraction pipeline.

    Args:
        config: Extraction configuration.

    Returns:
        Results dictionary with extraction details.
    """
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    if config.verbose:
        print("=" * 70)
        print("EXPERT EXTRACTION — DENSE MODEL")
        print("=" * 70)
        print(f"Model:           {config.model_name}")
        print(f"Target domain:   {config.target_domain}")
        print(f"Scores path:     {config.expert_scores_path}")
        print(f"Extraction mode: {config.extraction_mode}")
        print(f"Device:          {device}")
        print("=" * 70)

    # Step 1: Load model
    if config.verbose:
        print("\n[1/5] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model_name, device)

    # Step 2: Detect MoE structure
    if config.verbose:
        print("\n[2/5] Detecting MoE structure...")
    moe_structure = detect_moe_structure(model, config.model_name)
    if config.verbose:
        print(moe_structure.summary())

    if not moe_structure.layers:
        raise ValueError(
            "No MoE layers detected. This model may not be a MoE model, "
            "or its architecture is not supported."
        )

    # Step 3: Load expert scores and select experts
    if config.verbose:
        print(f"\n[3/5] Loading expert scores for domain '{config.target_domain}'...")
    scores_data = load_domain_scores(
        scores_dir=config.expert_scores_path,
        model_name=config.model_name,
        domain=config.target_domain,
    )
    selected_experts = select_experts_by_score(
        scores_data=scores_data,
        moe_structure=moe_structure,
        top_k_per_layer=config.top_k_experts_per_layer,
    )

    if config.verbose:
        print("\nSelected experts per layer:")
        for layer_idx in sorted(selected_experts.keys()):
            for expert_idx, score in selected_experts[layer_idx]:
                print(f"  Layer {layer_idx}: Expert {expert_idx} (score={score:.6f})")

    # Step 4: Build dense model
    if config.verbose:
        print(f"\n[4/5] Building dense model...")
    model = build_dense_model(
        model=model,
        moe_structure=moe_structure,
        selected_experts=selected_experts,
        verbose=config.verbose,
    )

    # Step 5: Validate
    validation_results = []
    if config.validate:
        if config.verbose:
            print("\n[5/5] Validating extracted model...")
        validation_results = validate_extracted_model(
            model=model,
            tokenizer=tokenizer,
            test_prompts=config.test_prompts,
            verbose=config.verbose,
        )
    elif config.verbose:
        print("\n[5/5] Validation skipped")

    # Save extracted model
    model_safe = safe_model_name(config.model_name)
    save_path = os.path.join(
        config.output_dir, model_safe,
        f"{config.target_domain}_dense",
    )

    extraction_info = {
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "moe_structure": {
            "model_name": moe_structure.model_name,
            "num_layers": moe_structure.num_layers,
            "num_moe_layers": len(moe_structure.layers),
            "moe_layers": [
                {
                    "layer_idx": info.layer_idx,
                    "num_experts": info.num_experts,
                    "moe_attr": info.moe_attr,
                }
                for info in moe_structure.layers
            ],
        },
        "selected_experts": {
            str(layer_idx): [
                {"expert_idx": eidx, "score": score}
                for eidx, score in experts
            ]
            for layer_idx, experts in selected_experts.items()
        },
        "validation": validation_results,
    }

    save_extracted_model(
        model=model,
        tokenizer=tokenizer,
        output_path=save_path,
        extraction_info=extraction_info,
        verbose=config.verbose,
    )

    if config.verbose:
        print(f"\n[+] Dense model saved to: {save_path}")

    # Optional: Finetune
    ft_output_dir = None
    if config.finetune:
        if config.verbose:
            print("\n" + "=" * 70)
            print("FINETUNING EXTRACTED MODEL")
            print("=" * 70)
        ft_output_dir = finetune_model(model, tokenizer, config)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "extraction_path": save_path,
        "finetune_path": ft_output_dir,
        "extraction_info": extraction_info,
    }

    if config.verbose:
        print("\n" + "=" * 70)
        print("DONE")
        print(f"  Extracted model: {save_path}")
        if ft_output_dir:
            print(f"  Finetuned model: {ft_output_dir}")
        print("=" * 70)

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract domain-specific experts from a MoE model to build "
            "a dense model, with optional finetuning"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    # Extraction
    parser.add_argument(
        "--expert_scores_path", type=str, default="output/sample_run",
        help="Directory with expert score files from main.py",
    )
    parser.add_argument(
        "--domain", type=str, default="mathematics",
        help="Target domain for expert selection",
    )
    parser.add_argument(
        "--top_k", type=int, default=1,
        help="Top-K experts per layer (1 for dense)",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="output/extracted_models",
    )
    parser.add_argument(
        "--no_validate", action="store_true",
        help="Skip model validation",
    )

    # Finetuning
    parser.add_argument(
        "--finetune", action="store_true",
        help="Finetune the extracted model",
    )
    parser.add_argument(
        "--train_data", type=str, default=None,
        help="Training data file for finetuning",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3,
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Use full finetuning instead of LoRA",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
    )
    parser.add_argument(
        "--quiet", action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.config:
        config = ExtractionConfig.from_yaml(args.config)
        # CLI overrides for finetune flags
        if args.finetune:
            config.finetune = True
        if args.train_data:
            config.train_data = args.train_data
    else:
        config = ExtractionConfig(
            model_name=args.model,
            device=args.device,
            expert_scores_path=args.expert_scores_path,
            target_domain=args.domain,
            top_k_experts_per_layer=args.top_k,
            validate=not args.no_validate,
            output_dir=args.output_dir,
            finetune=args.finetune,
            train_data=args.train_data,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            use_lora=not args.no_lora,
            lora_rank=args.lora_rank,
            verbose=not args.quiet,
        )

    run_extraction(config)


if __name__ == "__main__":
    main()
