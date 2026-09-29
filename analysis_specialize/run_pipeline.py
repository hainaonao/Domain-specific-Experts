#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Pipeline: Expert Discovery → Dense Model Extraction

Run the complete pipeline with a single command for any MoE model and domain:

    # Minimal usage — just specify model, domain, and data
    python -m analysis_specialize.run_pipeline \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --domains mathematics \\
        --data_path data/sample_mmlu.jsonl

    # Multiple domains at once
    python -m analysis_specialize.run_pipeline \\
        --model Qwen/Qwen1.5-MoE-A2.7B \\
        --domains biology physics mathematics \\
        --data_path data/my_dataset.jsonl

    # With finetuning
    python -m analysis_specialize.run_pipeline \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --domains mathematics \\
        --data_path data/sample_mmlu.jsonl \\
        --finetune --train_data data/math_train.jsonl

    # Skip discovery if scores already exist
    python -m analysis_specialize.run_pipeline \\
        --model allenai/OLMoE-1B-7B-0924-Instruct \\
        --domains mathematics \\
        --skip_discovery \\
        --output_dir output/my_run

The pipeline:
  Step 1: Expert Discovery — compute token importance + expert activation scores
  Step 2: Dense Extraction — replace MoE layers with top-scored expert FFNs
  Step 3: (Optional) Finetune — LoRA or full finetuning on the dense model
"""

import argparse
import sys
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: expert discovery → dense extraction "
            "(→ optional finetune) for any MoE model and domain"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m analysis_specialize.run_pipeline \\
      --model allenai/OLMoE-1B-7B-0924-Instruct \\
      --domains mathematics \\
      --data_path data/sample_mmlu.jsonl

  # Multiple domains → creates one dense model per domain
  python -m analysis_specialize.run_pipeline \\
      --model Qwen/Qwen1.5-MoE-A2.7B \\
      --domains biology physics chemistry \\
      --data_path data/dataset.jsonl

  # Skip discovery (reuse existing scores)
  python -m analysis_specialize.run_pipeline \\
      --model allenai/OLMoE-1B-7B-0924-Instruct \\
      --domains mathematics \\
      --skip_discovery
        """,
    )

    # === Required ===
    parser.add_argument(
        "--model", type=str, required=True,
        help="MoE model name or path (e.g. allenai/OLMoE-1B-7B-0924-Instruct)",
    )
    parser.add_argument(
        "--domains", type=str, nargs="+", required=True,
        help="Target domain(s) to extract (e.g. mathematics biology physics)",
    )

    # === Data ===
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to JSON/JSONL dataset with domain/question/options fields",
    )

    # === Pipeline control ===
    parser.add_argument(
        "--skip_discovery", action="store_true",
        help="Skip Step 1 (use existing expert scores from output_dir)",
    )
    parser.add_argument(
        "--skip_extraction", action="store_true",
        help="Skip Step 2 (only run discovery, don't extract dense model)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/pipeline_run",
        help="Base output directory for all artifacts",
    )

    # === Discovery settings ===
    discovery = parser.add_argument_group("Discovery (Step 1)")
    discovery.add_argument(
        "--sample_percentage", type=float, default=100.0,
        help="Percentage of data to sample per domain",
    )
    discovery.add_argument(
        "--max_samples", type=int, default=100,
        help="Max samples per domain",
    )
    discovery.add_argument(
        "--importance_threshold", type=float, default=15.0,
        help="Percentile threshold for token importance",
    )
    discovery.add_argument(
        "--max_length", type=int, default=512,
        help="Maximum sequence length",
    )
    discovery.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    # === Extraction settings ===
    extraction = parser.add_argument_group("Extraction (Step 2)")
    extraction.add_argument(
        "--top_k", type=int, default=1,
        help="Top-K experts per layer (1 for dense model)",
    )
    extraction.add_argument(
        "--no_validate", action="store_true",
        help="Skip validation of extracted model",
    )

    # === Finetune settings ===
    finetune = parser.add_argument_group("Finetune (Step 3, optional)")
    finetune.add_argument(
        "--finetune", action="store_true",
        help="Finetune the extracted dense model",
    )
    finetune.add_argument(
        "--train_data", type=str, default=None,
        help="Training data file for finetuning",
    )
    finetune.add_argument(
        "--num_epochs", type=int, default=3,
    )
    finetune.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate",
    )
    finetune.add_argument(
        "--batch_size", type=int, default=4,
    )
    finetune.add_argument(
        "--no_lora", action="store_true",
        help="Use full finetuning instead of LoRA",
    )
    finetune.add_argument(
        "--lora_rank", type=int, default=16,
    )

    # === Output control ===
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if not args.skip_discovery and args.data_path is None:
        print(
            "Error: --data_path is required for expert discovery.\n"
            "  Use --skip_discovery if you already have score files in --output_dir.\n"
            "  Example: python -m analysis_specialize.run_pipeline \\\n"
            f"    --model {args.model} --domains {' '.join(args.domains)} \\\n"
            "    --data_path data/your_data.jsonl"
        )
        sys.exit(1)

    # =====================================================================
    # Step 1: Expert Discovery
    # =====================================================================
    if not args.skip_discovery:
        from .main import Config, run_expert_discovery

        if verbose:
            print("\n" + "█" * 70)
            print("  STEP 1/2: EXPERT DISCOVERY")
            print("█" * 70)

        discovery_config = Config(
            model_name=args.model,
            device=args.device,
            domains=args.domains,
            data_path=args.data_path,
            sample_percentage=args.sample_percentage,
            max_samples_per_domain=args.max_samples,
            random_seed=args.seed,
            importance_threshold=args.importance_threshold,
            max_sequence_length=args.max_length,
            output_dir=args.output_dir,
            save_expert_scores=True,
            save_token_classifications=True,
            verbose=verbose,
        )

        discovery_results = run_expert_discovery(discovery_config)

        if not discovery_results:
            print("[!] Expert discovery produced no results. Exiting.")
            sys.exit(1)

        if verbose:
            print("\n[✓] Expert discovery complete")
    else:
        if verbose:
            print("\n[→] Skipping discovery (using existing scores)")

    # =====================================================================
    # Step 2: Dense Extraction (one per domain)
    # =====================================================================
    if not args.skip_extraction:
        from .extract_and_finetune import ExtractionConfig, run_extraction

        if verbose:
            print("\n" + "█" * 70)
            print("  STEP 2/2: DENSE MODEL EXTRACTION")
            print("█" * 70)

        for domain in args.domains:
            if verbose:
                print(f"\n{'─' * 50}")
                print(f"  Extracting dense model for domain: {domain}")
                print(f"{'─' * 50}")

            extract_config = ExtractionConfig(
                model_name=args.model,
                device=args.device,
                expert_scores_path=args.output_dir,
                target_domain=domain,
                extraction_mode="dense",
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
                verbose=verbose,
            )

            try:
                results = run_extraction(extract_config)
                if verbose:
                    print(f"\n[✓] {domain}: dense model → {results['extraction_path']}")
                    if results.get("finetune_path"):
                        print(f"[✓] {domain}: finetuned  → {results['finetune_path']}")
            except FileNotFoundError as e:
                print(f"\n[✗] {domain}: {e}")
                print(f"    Make sure expert discovery ran for domain '{domain}'")
                continue
            except Exception as e:
                print(f"\n[✗] {domain}: extraction failed — {e}")
                continue
    else:
        if verbose:
            print("\n[→] Skipping extraction")

    if verbose:
        print("\n" + "█" * 70)
        print("  PIPELINE COMPLETE")
        print("█" * 70)


if __name__ == "__main__":
    main()
