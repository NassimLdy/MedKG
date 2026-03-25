"""
TD5 - Part 3: Train Knowledge Graph Embedding Models (TransE, DistMult)
Usage: python src/kge/train_kge.py [--data-dir data/kge/] [--output-dir results/]

Uses PyKEEN to train TransE and DistMult on the medical KB splits and
evaluates both with MRR, Hits@1, Hits@3, Hits@10 (filtered).
"""

import os
import sys
import json
import argparse

# ---------------------------------------------------------------------------
# PyKEEN import guard
# ---------------------------------------------------------------------------
try:
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
except ImportError:
    print("ERROR: pykeen is not installed.")
    print("Install it with:  pip install pykeen")
    print("You may also need:  pip install torch")
    sys.exit(1)

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def load_factory(path: str, entity_to_id=None, relation_to_id=None):
    """Load a TriplesFactory from a TSV file."""
    if not os.path.isfile(path):
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    if entity_to_id is not None and relation_to_id is not None:
        return TriplesFactory.from_path(
            path,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )
    return TriplesFactory.from_path(path)


def extract_metrics(results) -> dict:
    """Extract key metrics from a pykeen PipelineResult."""
    metric_results = results.metric_results
    mrr = metric_results.get_metric("mean_reciprocal_rank")
    h1 = metric_results.get_metric("hits_at_1")
    h3 = metric_results.get_metric("hits_at_3")
    h10 = metric_results.get_metric("hits_at_10")
    return {
        "MRR": round(float(mrr), 4),
        "Hits@1": round(float(h1), 4),
        "Hits@3": round(float(h3), 4),
        "Hits@10": round(float(h10), 4),
    }


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 50       # reduced to fit in RAM (75k entities)
NUM_EPOCHS = 100
BATCH_SIZE = 512         # larger batch = fewer allocs = less peak memory
LEARNING_RATE = 0.01
NUM_NEGS_PER_POS = 5    # reduced to save memory during negative sampling

MODEL_CONFIGS = {
    "TransE": {
        "model": "TransE",
        "model_kwargs": {"embedding_dim": EMBEDDING_DIM},
        "loss": "MarginRankingLoss",
        "loss_kwargs": {"margin": 1.0},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": LEARNING_RATE},
        "negative_sampler": "basic",
        "negative_sampler_kwargs": {"num_negs_per_pos": NUM_NEGS_PER_POS},
    },
    "DistMult": {
        "model": "DistMult",
        "model_kwargs": {"embedding_dim": EMBEDDING_DIM},
        "loss": "BCEWithLogitsLoss",
        "loss_kwargs": {},
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": LEARNING_RATE},
        "negative_sampler": "basic",
        "negative_sampler_kwargs": {"num_negs_per_pos": NUM_NEGS_PER_POS},
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TransE and DistMult on the medical KG."
    )
    parser.add_argument(
        "--data-dir",
        default="data/kge/",
        help="Directory containing train.txt, valid.txt, test.txt (default: data/kge/)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory to save trained models and results (default: results/)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/\\")
    output_dir = args.output_dir.rstrip("/\\")

    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.getcwd(), data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.txt")
    valid_path = os.path.join(data_dir, "valid.txt")
    test_path = os.path.join(data_dir, "test.txt")

    # ------------------------------------------------------------------
    # Load triples factories
    # ------------------------------------------------------------------
    print_section("Loading Triples Factories")
    print(f"  Data directory: {data_dir}")

    training_factory = load_factory(train_path)
    validation_factory = load_factory(
        valid_path,
        entity_to_id=training_factory.entity_to_id,
        relation_to_id=training_factory.relation_to_id,
    )
    testing_factory = load_factory(
        test_path,
        entity_to_id=training_factory.entity_to_id,
        relation_to_id=training_factory.relation_to_id,
    )

    print(f"  Train triples   : {training_factory.num_triples}")
    print(f"  Valid triples   : {validation_factory.num_triples}")
    print(f"  Test triples    : {testing_factory.num_triples}")
    print(f"  Entities        : {training_factory.num_entities}")
    print(f"  Relations       : {training_factory.num_relations}")

    # ------------------------------------------------------------------
    # Train models
    # ------------------------------------------------------------------
    all_results = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        print_section(f"Training {model_name}")
        print(f"  embedding_dim  = {EMBEDDING_DIM}")
        print(f"  num_epochs     = {NUM_EPOCHS}")
        print(f"  batch_size     = {BATCH_SIZE}")
        print(f"  learning_rate  = {LEARNING_RATE}")
        print(f"  neg_per_pos    = {NUM_NEGS_PER_POS}")
        print(f"  loss           = {cfg['loss']}")

        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Build pipeline kwargs — loss_kwargs only if non-empty
        loss_kw = cfg["loss_kwargs"] if cfg["loss_kwargs"] else None

        try:
            result = pipeline(
                training=training_factory,
                validation=validation_factory,
                testing=testing_factory,
                model=cfg["model"],
                model_kwargs=cfg["model_kwargs"],
                loss=cfg["loss"],
                loss_kwargs=loss_kw,
                optimizer=cfg["optimizer"],
                optimizer_kwargs=cfg["optimizer_kwargs"],
                training_loop="sLCWA",
                negative_sampler=cfg["negative_sampler"],
                negative_sampler_kwargs=cfg["negative_sampler_kwargs"],
                training_kwargs={
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                },
                evaluator="RankBasedEvaluator",
                evaluator_kwargs={"filtered": True, "batch_size": 32},
                random_seed=42,
                device="cpu",
            )
        except Exception as exc:
            print(f"  ERROR during training {model_name}: {exc}")
            all_results[model_name] = {"error": str(exc)}
            continue

        # Save model
        result.save_to_directory(model_output_dir)
        print(f"  Model saved to: {model_output_dir}")

        # Extract metrics
        metrics = extract_metrics(result)
        all_results[model_name] = metrics

        print(f"\n  Evaluation Results ({model_name}):")
        print(f"    MRR     : {metrics['MRR']:.4f}")
        print(f"    Hits@1  : {metrics['Hits@1']:.4f}")
        print(f"    Hits@3  : {metrics['Hits@3']:.4f}")
        print(f"    Hits@10 : {metrics['Hits@10']:.4f}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print_section("Model Comparison Table")
    header = f"{'Model':<12} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>9}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{model_name:<12}  ERROR: {metrics['error']}")
        else:
            print(
                f"{model_name:<12} {metrics['MRR']:>8.4f} "
                f"{metrics['Hits@1']:>8.4f} "
                f"{metrics['Hits@3']:>8.4f} "
                f"{metrics['Hits@10']:>9.4f}"
            )

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results_json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "hyperparameters": {
                    "embedding_dim": EMBEDDING_DIM,
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_negs_per_pos": NUM_NEGS_PER_POS,
                },
                "results": all_results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Results saved to: {results_json_path}")

    # ------------------------------------------------------------------
    # Training data info for head/tail prediction reference
    # ------------------------------------------------------------------
    print_section("Entity/Relation Counts (for analysis)")
    print(f"  Entities  : {training_factory.num_entities}")
    print(f"  Relations : {training_factory.num_relations}")

    # ------------------------------------------------------------------
    # Section 5.2 — KB Size Sensitivity
    # ------------------------------------------------------------------
    print_section("5.2 KB Size Sensitivity (TransE only)")
    print("  Subsampling training set at 20k, 50k, and full size...")

    import random
    import numpy as np

    full_triples = training_factory.mapped_triples.numpy()
    total = len(full_triples)
    size_labels = []
    for target in [20_000, 50_000, total]:
        if target > total:
            continue
        label = f"{target // 1000}k" if target < total else "full"
        size_labels.append((label, target))

    sensitivity_results = {}
    for label, n in size_labels:
        print(f"\n  --- Size: {label} ({n} triples) ---")
        idx = np.random.choice(total, size=min(n, total), replace=False)
        sub_triples = full_triples[idx]
        sub_factory = TriplesFactory(
            mapped_triples=torch.tensor(sub_triples),
            entity_to_id=training_factory.entity_to_id,
            relation_to_id=training_factory.relation_to_id,
        )
        try:
            sub_result = pipeline(
                training=sub_factory,
                validation=validation_factory,
                testing=testing_factory,
                model="TransE",
                model_kwargs={"embedding_dim": EMBEDDING_DIM},
                loss="MarginRankingLoss",
                loss_kwargs={"margin": 1.0},
                optimizer="Adam",
                optimizer_kwargs={"lr": LEARNING_RATE},
                training_loop="sLCWA",
                negative_sampler="basic",
                negative_sampler_kwargs={"num_negs_per_pos": NUM_NEGS_PER_POS},
                training_kwargs={"num_epochs": 50, "batch_size": BATCH_SIZE},
                evaluator="RankBasedEvaluator",
                evaluator_kwargs={"filtered": True, "batch_size": 32},
                random_seed=42,
                device="cpu",
            )
            m = extract_metrics(sub_result)
            sensitivity_results[label] = m
            print(f"    MRR={m['MRR']:.4f}  Hits@1={m['Hits@1']:.4f}  Hits@10={m['Hits@10']:.4f}")
        except Exception as exc:
            print(f"    ERROR: {exc}")
            sensitivity_results[label] = {"error": str(exc)}

    print_section("Size Sensitivity Summary")
    header2 = f"{'Size':<8} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>9}"
    print(header2)
    print("-" * len(header2))
    for label, m in sensitivity_results.items():
        if "error" in m:
            print(f"{label:<8}  ERROR")
        else:
            print(f"{label:<8} {m['MRR']:>8.4f} {m['Hits@1']:>8.4f} {m['Hits@10']:>9.4f}")

    # Save sensitivity results
    all_results["size_sensitivity"] = sensitivity_results
    with open(results_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "hyperparameters": {
                    "embedding_dim": EMBEDDING_DIM,
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "num_negs_per_pos": NUM_NEGS_PER_POS,
                },
                "results": all_results,
            },
            fh,
            indent=2,
        )
    print(f"\n  Results (with sensitivity) saved to: {results_json_path}")
    print("\nDone. Use analyze_kge.py for embedding analysis.")


if __name__ == "__main__":
    main()
