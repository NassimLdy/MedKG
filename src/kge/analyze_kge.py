"""
TD5 - Part 4: KGE Embedding Analysis
Usage: python src/kge/analyze_kge.py [--model-dir results/TransE/]
                                      [--train-file data/kge/train.txt]
                                      [--output-dir results/]

Sections:
  6.1 Nearest Neighbors    — cosine similarity in embedding space
  6.2 Clustering (t-SNE)   — 2-D visualisation of entity embeddings
  6.3 Relation Behavior    — symmetry / composition analysis
  7.  Critical Reflection  — printed commentary
  8.  SWRL vs Embedding    — medical SWRL rule vs learned relation vectors
"""

import os
import sys
import argparse
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Imports with graceful error messages
# ---------------------------------------------------------------------------
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
    from pykeen.pipeline import PipelineResult
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


# ---------------------------------------------------------------------------
# Medical entities of interest (labels that may appear as URI fragments)
# ---------------------------------------------------------------------------
MEDICAL_ENTITIES_KEYWORDS = [
    "Diabetes", "Hypertension", "Asthma", "Cancer", "Alzheimer",
    "diabetes", "hypertension", "asthma", "cancer", "alzheimer",
]

NUM_NEIGHBORS = 5
MAX_TSNE_ENTITIES = 2000
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 66
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for rows of matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = matrix / norms
    return normed @ normed.T


def find_entity_by_keyword(entity_to_id: dict, keyword: str) -> list:
    """Return all entity URIs whose fragment/path contains keyword."""
    keyword_lower = keyword.lower()
    return [
        uri for uri in entity_to_id
        if keyword_lower in uri.split("/")[-1].split("#")[-1].lower()
    ]


def short_uri(uri: str, max_len: int = 55) -> str:
    """Shorten a URI for display."""
    fragment = uri.split("/")[-1].split("#")[-1]
    if len(fragment) <= max_len:
        return fragment
    return fragment[:max_len] + "..."


def load_pipeline_result(model_dir: str):
    """Load a saved model from directory using torch.load on trained_model.pkl."""
    import json
    model_pkl = os.path.join(model_dir, "trained_model.pkl")
    training_dir = os.path.join(model_dir, "training_triples")

    model = torch.load(model_pkl, map_location="cpu", weights_only=False)

    # Load entity/relation mappings from the training triples factory
    tf = TriplesFactory.from_path(os.path.join(training_dir, "new_to_old_ids.tsv.gz"),
                                   ) if False else None
    # Use metadata.json to get mapping files
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load the training triples factory directly
    # model_dir is e.g. .../MedKG/results/TransE/ → root is 3 levels up
    _root = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))
    train_factory = TriplesFactory.from_path(
        os.path.join(_root, "data", "kge", "train.txt")
    )
    # Attach to a simple namespace for compatibility
    class _Result:
        pass
    r = _Result()
    r.model = model
    r.training = train_factory
    return r


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


def get_relation_embeddings(result) -> tuple[np.ndarray, dict, dict]:
    """Extract relation embedding matrix from loaded model."""
    model = result.model
    if hasattr(model, "relation_representations"):
        emb_module = model.relation_representations[0]
        emb_weight = emb_module._embeddings.weight.detach().cpu().numpy()
    elif hasattr(model, "relation_embeddings"):
        emb_weight = model.relation_embeddings.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Cannot find relation embeddings in model.")

    relation_to_id = result.training.relation_to_id
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    return emb_weight, relation_to_id, id_to_relation


def load_triples_from_file(path: str) -> list[tuple[str, str, str]]:
    """Load TSV triples (subject, relation, object) from file."""
    triples = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


# ---------------------------------------------------------------------------
# Section 6.1 — Nearest Neighbors
# ---------------------------------------------------------------------------

def nearest_neighbors(
    embeddings: np.ndarray,
    entity_to_id: dict,
    id_to_entity: dict,
    n_neighbors: int = NUM_NEIGHBORS,
) -> None:
    print_section("6.1  Nearest Neighbors (Cosine Similarity)")

    sim_matrix = cosine_similarity_matrix(embeddings)

    found_any = False
    for keyword in MEDICAL_ENTITIES_KEYWORDS:
        matches = find_entity_by_keyword(entity_to_id, keyword)
        if not matches:
            continue
        # Use the first match
        query_uri = matches[0]
        query_id = entity_to_id[query_uri]

        sims = sim_matrix[query_id]
        # Exclude self
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
        break   # Show one detailed example; loop would print all 5

    # Show all 5 keywords briefly
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
        print("  Showing random entity example instead.")
        rand_id = random.randint(0, len(id_to_entity) - 1)
        rand_uri = id_to_entity[rand_id]
        sims = sim_matrix[rand_id].copy()
        sims[rand_id] = -2.0
        top_ids = np.argsort(sims)[::-1][:n_neighbors]
        print(f"\n  Random entity: {short_uri(rand_uri)}")
        for rank, nid in enumerate(top_ids, 1):
            print(f"  {rank}. {short_uri(id_to_entity[nid])}  (sim={sims[nid]:.4f})")


# ---------------------------------------------------------------------------
# Section 6.2 — t-SNE Clustering
# ---------------------------------------------------------------------------

def tsne_clustering(
    embeddings: np.ndarray,
    id_to_entity: dict,
    output_path: str,
    training_triples: list,
) -> None:
    print_section("6.2  t-SNE Clustering")

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

    # Build simple color map by rdf:type (from triples)
    # entity -> type URI fragment
    entity_type: dict[str, str] = {}
    for s, p, o in training_triples:
        if "type" in p.lower():
            entity_type[s] = short_uri(o, max_len=20)

    # Assign colors
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
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.6)

    # Legend (top types only)
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


# ---------------------------------------------------------------------------
# Section 6.3 — Relation Behavior
# ---------------------------------------------------------------------------

def relation_behavior(training_triples: list) -> None:
    print_section("6.3  Relation Behavior Analysis")

    # Build triple sets
    triple_set = set()
    relation_pairs: dict[str, list] = defaultdict(list)
    for s, p, o in training_triples:
        triple_set.add((s, p, o))
        relation_pairs[p].append((s, o))

    # Relation co-occurrence for composition detection
    # Build: entity -> outgoing relations
    entity_out: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for s, p, o in training_triples:
        entity_out[s][p].add(o)

    print(f"\n  Total relations: {len(relation_pairs)}")
    print(f"\n  {'Relation (fragment)':<35} {'#Triples':>8}  {'Symmetric?':>11}  {'Symmetric %':>12}")
    print("  " + "-" * 75)

    symmetric_count = 0
    for rel, pairs in sorted(relation_pairs.items(), key=lambda x: -len(x[1])):
        n = len(pairs)
        # Check symmetry: count how many (s,o) have inverse (o,s) in same rel
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

    # Simple composition check: R1 ; R2 ≈ R3
    print("\n  Composition analysis (keyword-based path check):")
    composition_candidates = [
        ("parent", "parent", "grandparent"),
        ("hasSymptom", "causes", "leadsTo"),
        ("treats", "targets", "cures"),
    ]
    for r1_kw, r2_kw, r3_kw in composition_candidates:
        r1_rels = [r for r in relation_pairs if r1_kw.lower() in r.lower()]
        r2_rels = [r for r in relation_pairs if r2_kw.lower() in r.lower()]
        r3_rels = [r for r in relation_pairs if r3_kw.lower() in r.lower()]
        found = bool(r1_rels and r2_rels and r3_rels)
        print(
            f"  {r1_kw} ; {r2_kw} -> {r3_kw}: "
            f"{'Candidate found in KB' if found else 'Not present in KB'}"
        )


# ---------------------------------------------------------------------------
# Section 7 — Critical Reflection
# ---------------------------------------------------------------------------

def critical_reflection() -> None:
    print_section("7.  Critical Reflection")

    points = [
        (
            "Impact of predicate alignment quality",
            "When heterogeneous knowledge sources are merged, predicate URIs rarely "
            "align naturally. A predicate in one ontology may encode the same semantic "
            "as a different URI in another. Poor alignment creates spurious duplicate "
            "relations that dilute embedding signal: the model sees 'treats' and "
            "'hasTreatment' as independent dimensions when they encode the same "
            "concept, degrading both MRR and Hits@k. Careful owl:equivalentProperty "
            "declarations or a post-hoc predicate normalisation step are essential."
        ),
        (
            "Impact of noisy expansion",
            "The expanded KB was produced by joining across multiple sources. "
            "Automatic expansion via SPARQL can introduce false triples, especially "
            "when intermediate entities are shared by multiple disease contexts. "
            "These noisy triples degrade the signal-to-noise ratio during training; "
            "the negative sampler may 'sample' what are actually true triples (open-"
            "world issue), and corrupted positives shift embedding centroids."
        ),
        (
            "Ontology choice effects",
            "The expressiveness of the source ontologies determines which logical "
            "constraints guide embedding geometry. A rich OWL ontology (with disjoint "
            "classes, functional properties, cardinality restrictions) supplies more "
            "implicit constraints that could regularise the geometry. A flat schema "
            "yields purely data-driven embeddings with no semantic anchoring."
        ),
        (
            "Open-world assumption vs embedding closed-world",
            "OWL/SWRL reasoning operates under the Open-World Assumption (OWA): a "
            "statement not in the KB is not necessarily false. KGE models, however, "
            "implicitly assume a partial Closed-World Assumption: they treat absent "
            "triples as negatives during training. This mismatch means embeddings can "
            "penalise true-but-unobserved facts, biasing the learned geometry away "
            "from the ontological truth."
        ),
    ]

    for i, (title, body) in enumerate(points, 1):
        print(f"\n  {i}. {title}")
        # Word-wrap at 70 chars
        words = body.split()
        line, indent = "     ", "     "
        for word in words:
            if len(line) + len(word) + 1 > 75:
                print(line)
                line = indent + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)


# ---------------------------------------------------------------------------
# Section 8 — SWRL vs Embedding Comparison
# ---------------------------------------------------------------------------

def swrl_vs_embedding(
    relation_embeddings: np.ndarray,
    relation_to_id: dict,
    id_to_relation: dict,
) -> None:
    print_section("8.  SWRL Rule vs Embedding Comparison")

    rule = "Disease(?d) ^ hasSymptom(?d, ?s) -> affectedBy(?s, ?d)"
    print(f"\n  Medical SWRL rule:")
    print(f"    {rule}")
    print()
    print("  This rule states that 'affectedBy' is the inverse of 'hasSymptom'.")
    print("  Under TransE, if hasSymptom is modelled as vector r_s, then its")
    print("  inverse should satisfy: r_s ~= -r_affectedBy.")

    # Find matching relations
    sym_rels = {kw: None for kw in ["hasSymptom", "symptom", "affectedBy", "affected"]}
    for uri in relation_to_id:
        frag = uri.split("/")[-1].split("#")[-1].lower()
        for kw in sym_rels:
            if kw.lower() in frag and sym_rels[kw] is None:
                sym_rels[kw] = uri

    has_symptom_uri = sym_rels.get("hasSymptom") or sym_rels.get("symptom")
    affected_by_uri = sym_rels.get("affectedBy") or sym_rels.get("affected")

    if has_symptom_uri and affected_by_uri:
        r_s = relation_embeddings[relation_to_id[has_symptom_uri]]
        r_a = relation_embeddings[relation_to_id[affected_by_uri]]

        # Cosine similarity between r_s and -r_a
        def cos_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        sim_direct = cos_sim(r_s, r_a)
        sim_inverse = cos_sim(r_s, -r_a)

        print(f"\n  hasSymptom URI   : {short_uri(has_symptom_uri)}")
        print(f"  affectedBy URI   : {short_uri(affected_by_uri)}")
        print(f"\n  cos(r_hasSymptom, r_affectedBy)   = {sim_direct:+.4f}")
        print(f"  cos(r_hasSymptom, -r_affectedBy)  = {sim_inverse:+.4f}")

        if sim_inverse > 0.7:
            verdict = "CONFIRMED - embeddings learned the inverse relationship."
        elif sim_inverse > 0.3:
            verdict = "PARTIAL - weak inverse signal captured."
        else:
            verdict = "NOT CONFIRMED - embeddings did not clearly capture the inverse."

        print(f"\n  Verdict: {verdict}")
        print()
        print("  Interpretation:")
        print("  SWRL can derive affectedBy facts deductively from any Disease-")
        print("  hasSymptom triple, even for unseen entities, given the rule.")
        print("  TransE can only approximate this if the geometry cooperates.")
        print("  SWRL is strictly stronger for logical axioms; embeddings are")
        print("  stronger for soft, probabilistic patterns across large KGs.")
    else:
        print("\n  hasSymptom / affectedBy relations not found in this KB.")
        print("  Available relations (sample):")
        for rid, uri in sorted(id_to_relation.items())[:10]:
            print(f"    {short_uri(uri)}")

        print("\n  Theoretical comparison:")
        print("  SWRL rule guarantees complete derivation of affectedBy from")
        print("  hasSymptom under OWA. TransE can only approximate the inverse")
        print("  if r_hasSymptom ~= -r_affectedBy in the learned vector space.")
        print("  For sparse KGs, SWRL rules are more reliable; for large noisy")
        print("  KGs, embeddings generalise better to unseen entity pairs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print_section("Loading TransE Model")
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

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    entity_embeddings, entity_to_id, id_to_entity = get_entity_embeddings(result)
    relation_embeddings, relation_to_id, id_to_relation = get_relation_embeddings(result)

    print(f"  Entity embedding shape  : {entity_embeddings.shape}")
    print(f"  Relation embedding shape: {relation_embeddings.shape}")

    # ------------------------------------------------------------------
    # Load training triples
    # ------------------------------------------------------------------
    training_triples = []
    if os.path.isfile(train_file):
        training_triples = load_triples_from_file(train_file)
        print(f"  Training triples loaded : {len(training_triples)}")
    else:
        print(f"  WARNING: train.txt not found at {train_file}; relation analysis may be limited.")

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    nearest_neighbors(entity_embeddings, entity_to_id, id_to_entity)
    tsne_clustering(entity_embeddings, id_to_entity, tsne_output, training_triples)
    relation_behavior(training_triples)
    critical_reflection()
    swrl_vs_embedding(relation_embeddings, relation_to_id, id_to_relation)

    print_section("Analysis Complete")
    print(f"  t-SNE plot (if generated): {tsne_output}")
    print("  Done.")


if __name__ == "__main__":
    main()
