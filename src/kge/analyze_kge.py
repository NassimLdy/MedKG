"""analyze_kge.py — Analyze KGE embeddings: nearest neighbors, t-SNE, relation behavior.
Usage: python src/kge/analyze_kge.py [--model-dir results/TransE/] [--train-file data/kge/train.txt]
"""

import os
import sys
import argparse
import random
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed.  pip install numpy")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch not installed.  pip install torch")
    sys.exit(1)

try:
    from pykeen.triples import TriplesFactory
except ImportError:
    print("ERROR: pykeen not installed.  pip install pykeen")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")          # headless backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WARNING: matplotlib not installed. t-SNE plot will be skipped.")

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed. t-SNE plot will be skipped.")


# Medical entities to search for in the embedding space
MEDICAL_ENTITIES_KEYWORDS = [
    "Diabetes", "Hypertension", "Asthma", "Cancer", "Alzheimer",
    "diabetes", "hypertension", "asthma", "cancer", "alzheimer",
]

NUM_NEIGHBORS = 5
MAX_TSNE_ENTITIES = 2000
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


# Compute the full pairwise cosine similarity matrix for all entity embeddings.
def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs of entity embeddings."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = matrix / norms
    return normed @ normed.T


# Return entity URIs whose local name fragment contains the given keyword.
def find_entity_by_keyword(entity_to_id: dict, keyword: str) -> list:
    """Return all entity URIs that contain the given keyword."""
    keyword_lower = keyword.lower()
    return [
        uri for uri in entity_to_id
        if keyword_lower in uri.split("/")[-1].split("#")[-1].lower()
    ]


def short_uri(uri: str, max_len: int = 55) -> str:
    """Return the last part of a URI, truncated to max_len characters."""
    fragment = uri.split("/")[-1].split("#")[-1]
    if len(fragment) <= max_len:
        return fragment
    return fragment[:max_len] + "..."


# Load a saved TransE model and rebuild the training TriplesFactory from disk.
def load_pipeline_result(model_dir: str):
    """Load a saved TransE model and its training triples from disk."""
    model_pkl = os.path.join(model_dir, "trained_model.pkl")
    model = torch.load(model_pkl, map_location="cpu", weights_only=False)

    # model_dir is e.g. .../MedKG/results/TransE/ → project root is 3 levels up
    _root = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
    train_factory = TriplesFactory.from_path(
        os.path.join(_root, "data", "kge", "train.txt")
    )
    class _Result:
        pass
    r = _Result()
    r.model = model
    r.training = train_factory
    return r


# Extract the entity embedding weight matrix and ID↔URI mappings from the model.
def get_entity_embeddings(result) -> tuple[np.ndarray, dict, dict]:
    """Extract entity embedding matrix from loaded model."""
    model = result.model
    if hasattr(model, "entity_representations"):
        emb_module = model.entity_representations[0]
        emb_weight = emb_module._embeddings.weight.detach().cpu().numpy()
    elif hasattr(model, "entity_embeddings"):
        emb_weight = model.entity_embeddings.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Cannot find entity embeddings in model.")

    entity_to_id = result.training.entity_to_id
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    return emb_weight, entity_to_id, id_to_entity


# Read a tab-separated triples file into a list of (subject, predicate, object) tuples.
def load_triples_from_file(path: str) -> list[tuple[str, str, str]]:
    """Load TSV triples (subject, relation, object) from file."""
    triples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# Print cosine-nearest neighbors for key medical entities in embedding space.
def nearest_neighbors(
    embeddings: np.ndarray,
    entity_to_id: dict,
    id_to_entity: dict,
    n_neighbors: int = NUM_NEIGHBORS,
) -> None:
    print("\n6.1  Nearest Neighbors (Cosine Similarity)")

    sim_matrix = cosine_similarity_matrix(embeddings)

    found_any = False
    for keyword in MEDICAL_ENTITIES_KEYWORDS:
        matches = find_entity_by_keyword(entity_to_id, keyword)
        if not matches:
            continue
        query_uri = matches[0]
        query_id = entity_to_id[query_uri]

        sims = sim_matrix[query_id]
        sims[query_id] = -2.0
        top_ids = np.argsort(sims)[::-1][:n_neighbors]

        print(f"\n  Query: {short_uri(query_uri)}")
        print(f"  {'Rank':<6} {'Similarity':>10}  {'Entity'}")
        print("  " + "-" * 62)
        for rank, nid in enumerate(top_ids, 1):
            neighbor_uri = id_to_entity[nid]
            sim_val = sims[nid]
            print(f"  {rank:<6} {sim_val:>10.4f}  {short_uri(neighbor_uri)}")
        found_any = True
        break

    print(f"\n  Summary: all 5 medical entities of interest")
    print(f"  {'Entity Keyword':<20} {'Matched URI fragment':<40} Top neighbor")
    print("  " + "-" * 90)
    for keyword in MEDICAL_ENTITIES_KEYWORDS[:5]:
        kw_display = keyword.capitalize()
        matches = find_entity_by_keyword(entity_to_id, keyword)
        if not matches:
            print(f"  {kw_display:<20} {'(not found in KB)':<40} -")
            continue
        query_uri = matches[0]
        query_id = entity_to_id[query_uri]
        sims = sim_matrix[query_id].copy()
        sims[query_id] = -2.0
        top_id = int(np.argmax(sims))
        top_sim = sims[top_id]
        top_uri = short_uri(id_to_entity[top_id])
        print(
            f"  {kw_display:<20} {short_uri(query_uri):<40} "
            f"{top_uri} ({top_sim:.4f})"
        )

    if not found_any:
        print("\n  NOTE: None of the target medical entities were found in the KB.")


# Run t-SNE on entity embeddings and save a scatter plot colored by rdf:type.
def tsne_clustering(
    embeddings: np.ndarray,
    id_to_entity: dict,
    output_path: str,
    training_triples: list,
) -> None:
    print("\n6.2  t-SNE Clustering")

    if not SKLEARN_AVAILABLE:
        print("  Skipped: scikit-learn not installed.")
        return
    if not MATPLOTLIB_AVAILABLE:
        print("  Skipped: matplotlib not installed.")
        return

    n_entities = embeddings.shape[0]
    print(f"  Total entities: {n_entities}")

    if n_entities > MAX_TSNE_ENTITIES:
        print(f"  Sampling {MAX_TSNE_ENTITIES} entities randomly (seed=42).")
        random.seed(42)
        sampled_ids = random.sample(range(n_entities), MAX_TSNE_ENTITIES)
    else:
        sampled_ids = list(range(n_entities))

    sampled_embeddings = embeddings[sampled_ids]
    print(f"  Running t-SNE (n_components=2, perplexity={TSNE_PERPLEXITY})...")

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        random_state=TSNE_RANDOM_STATE,
        max_iter=1000,
        init="pca",
    )
    coords = tsne.fit_transform(sampled_embeddings)

    entity_type: dict[str, str] = {}
    for s, p, o in training_triples:
        if "type" in p.lower():
            entity_type[s] = short_uri(o, max_len=20)

    type_set = list(set(entity_type.values()))
    cmap = plt.cm.get_cmap("tab20", max(len(type_set), 1))
    type_to_color = {t: cmap(i) for i, t in enumerate(type_set)}
    default_color = (0.7, 0.7, 0.7, 0.5)

    colors = []
    for sid in sampled_ids:
        uri = id_to_entity[sid]
        t = entity_type.get(uri, None)
        colors.append(type_to_color.get(t, default_color))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.6)

    from matplotlib.patches import Patch
    top_types = sorted(type_to_color.keys())[:15]
    legend_elements = [
        Patch(facecolor=type_to_color[t], label=t) for t in top_types
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=6, markerscale=2)

    ax.set_title("t-SNE of Entity Embeddings (TransE) — colored by rdf:type", fontsize=13)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  t-SNE plot saved to: {output_path}")


# Analyze symmetry and triple frequency for each relation in the training set.
def relation_behavior(training_triples: list) -> None:
    print("\n6.3  Relation Behavior Analysis")

    triple_set = set()
    relation_pairs: dict[str, list] = defaultdict(list)
    for s, p, o in training_triples:
        triple_set.add((s, p, o))
        relation_pairs[p].append((s, o))

    print(f"\n  Total relations: {len(relation_pairs)}")
    print(f"\n  {'Relation (fragment)':<35} {'#Triples':>8}  {'Symmetric?':>11}  {'Symmetric %':>12}")
    print("  " + "-" * 75)

    symmetric_count = 0
    for rel, pairs in sorted(relation_pairs.items(), key=lambda x: -len(x[1])):
        n = len(pairs)
        sym_matches = sum(1 for (s, o) in pairs if (o, rel, s) in triple_set)
        sym_pct = 100.0 * sym_matches / n if n > 0 else 0.0
        is_symmetric = sym_pct > 50.0
        if is_symmetric:
            symmetric_count += 1
        sym_label = "YES" if is_symmetric else "no"
        print(
            f"  {short_uri(rel):<35} {n:>8}  {sym_label:>11}  {sym_pct:>11.1f}%"
        )

    print(f"\n  Relations flagged as symmetric: {symmetric_count} / {len(relation_pairs)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse KGE embeddings: nearest neighbors, t-SNE, relation behavior."
    )
    parser.add_argument(
        "--model-dir",
        default="results/TransE/",
        help="Directory of saved TransE PipelineResult (default: results/TransE/)",
    )
    parser.add_argument(
        "--train-file",
        default="data/kge/train.txt",
        help="Training triples TSV (default: data/kge/train.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Directory to save analysis outputs (default: results/)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    train_file = args.train_file
    output_dir = args.output_dir

    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.getcwd(), model_dir)
    if not os.path.isabs(train_file):
        train_file = os.path.join(os.getcwd(), train_file)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    os.makedirs(output_dir, exist_ok=True)
    tsne_output = os.path.join(output_dir, "tsne_plot.png")

    print("\nLoading TransE Model")
    print(f"  Model directory: {model_dir}")

    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Run train_kge.py first.")
        sys.exit(1)

    try:
        result = load_pipeline_result(model_dir)
        print("  Model loaded successfully.")
    except Exception as exc:
        print(f"ERROR loading model: {exc}")
        sys.exit(1)

    entity_embeddings, entity_to_id, id_to_entity = get_entity_embeddings(result)
    print(f"  Entity embedding shape  : {entity_embeddings.shape}")

    training_triples = []
    if os.path.isfile(train_file):
        training_triples = load_triples_from_file(train_file)
        print(f"  Training triples loaded : {len(training_triples)}")
    else:
        print(f"  WARNING: train.txt not found at {train_file}; relation analysis may be limited.")

    nearest_neighbors(entity_embeddings, entity_to_id, id_to_entity)
    tsne_clustering(entity_embeddings, id_to_entity, tsne_output, training_triples)
    relation_behavior(training_triples)

    print("\nAnalysis Complete")
    print(f"  t-SNE plot (if generated): {tsne_output}")
    print("  Done.")


if __name__ == "__main__":
    main()
