"""prepare_data.py — Prepare KGE train/valid/test splits from the expanded KB N-Triples."""

import os
import sys
import random
import argparse
from collections import defaultdict

BLOCKED_PREDICATES = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2002/07/owl#sameAs",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
    "http://www.w3.org/2002/07/owl#equivalentClass",
    "http://www.w3.org/2002/07/owl#equivalentProperty",
}


# Parse one N-Triples line into (s, p, o) URI strings, returning None for literals or comments.
def parse_nt_line(line: str):
    """Parse one N-Triples line. Returns (s, p, o) URIs or None for literals/comments."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    if line.endswith(" ."):
        line = line[:-2].strip()
    elif line.endswith("."):
        line = line[:-1].strip()

    parts = line.split(" ", 2)
    if len(parts) < 3:
        return None

    subj, pred, obj = parts[0], parts[1], parts[2].strip()

    if not (subj.startswith("<") and subj.endswith(">")):
        return None
    if not (pred.startswith("<") and pred.endswith(">")):
        return None
    if not (obj.startswith("<") and obj.endswith(">")):
        return None

    subj_uri = subj[1:-1]
    pred_uri = pred[1:-1]
    obj_uri = obj[1:-1]

    return subj_uri, pred_uri, obj_uri


# Load and filter entity-entity triples from an N-Triples file, skipping blocked predicates.
def load_triples(nt_file: str):
    """Load and filter triples from an N-Triples file."""
    triples = []
    skipped_literal = 0
    skipped_predicate = 0
    total_lines = 0

    with open(nt_file, "r", encoding="utf-8") as fh:
        for line in fh:
            total_lines += 1
            parsed = parse_nt_line(line)
            if parsed is None:
                skipped_literal += 1
                continue
            subj, pred, obj = parsed
            if pred in BLOCKED_PREDICATES:
                skipped_predicate += 1
                continue
            triples.append((subj, pred, obj))

    print(f"  Total lines read      : {total_lines}")
    print(f"  Triples with literals : {skipped_literal} (skipped)")
    print(f"  Blocked predicates    : {skipped_predicate} (skipped)")
    print(f"  Entity-entity triples : {len(triples)}")
    return triples


# Split triples 80/10/10, moving triples with unseen entities into the training set.
def split_triples(triples, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """Split into train/valid/test. Entities in valid/test must appear in train."""
    random.seed(seed)
    random.shuffle(triples)

    n = len(triples)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = triples[:n_train]
    valid = triples[n_train: n_train + n_valid]
    test = triples[n_train + n_valid:]

    train_entities = set()
    for s, p, o in train:
        train_entities.add(s)
        train_entities.add(o)

    safe_valid, overflow_valid = [], []
    for triple in valid:
        s, p, o = triple
        if s in train_entities and o in train_entities:
            safe_valid.append(triple)
        else:
            overflow_valid.append(triple)
            train_entities.add(s)
            train_entities.add(o)

    safe_test, overflow_test = [], []
    for triple in test:
        s, p, o = triple
        if s in train_entities and o in train_entities:
            safe_test.append(triple)
        else:
            overflow_test.append(triple)
            train_entities.add(s)
            train_entities.add(o)

    train = train + overflow_valid + overflow_test
    valid = safe_valid
    test = safe_test

    return train, valid, test


def write_tsv(triples, filepath: str) -> None:
    """Write triples as TSV: subject\trelation\tobject"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, "w", encoding="utf-8") as fh:
        for s, p, o in triples:
            fh.write(f"{s}\t{p}\t{o}\n")


def write_list(items, filepath: str) -> None:
    """Write a list of strings, one per line."""
    with open(filepath, "w", encoding="utf-8") as fh:
        for item in sorted(items):
            fh.write(item + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare KGE train/valid/test splits from expanded KB N-Triples."
    )
    parser.add_argument(
        "--input",
        default="kg_artifacts/medical_kb_expanded.nt",
        help="Path to the input N-Triples file (default: kg_artifacts/medical_kb_expanded.nt)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/kge/",
        help="Directory to write output files (default: data/kge/)",
    )
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir.rstrip("/\\")

    # Convert relative paths to absolute
    if not os.path.isabs(input_file):
        input_file = os.path.join(os.getcwd(), input_file)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading N-Triples")
    print(f"  Input file: {input_file}")

    if not os.path.isfile(input_file):
        print(f"\nERROR: Input file not found: {input_file}")
        print("Make sure kg_artifacts/medical_kb_expanded.nt exists.")
        print("You can run the expansion script first: python src/kg/expand_kb.py")
        sys.exit(1)

    triples = load_triples(input_file)

    if len(triples) == 0:
        print("ERROR: No valid entity-entity triples found. Check the input file.")
        sys.exit(1)

    triples = list(set(triples))
    print(f"  After deduplication  : {len(triples)}")

    entities = set()
    relations = set()
    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    print("\nSplitting Triples (80/10/10)")
    train, valid, test = split_triples(triples, train_ratio=0.8, valid_ratio=0.1)

    print("\nWriting Output Files")

    train_file = os.path.join(output_dir, "train.txt")
    valid_file = os.path.join(output_dir, "valid.txt")
    test_file = os.path.join(output_dir, "test.txt")
    entities_file = os.path.join(output_dir, "entities.txt")
    relations_file = os.path.join(output_dir, "relations.txt")

    write_tsv(train, train_file)
    write_tsv(valid, valid_file)
    write_tsv(test, test_file)
    write_list(entities, entities_file)
    write_list(relations, relations_file)

    print(f"  train.txt    -> {train_file}")
    print(f"  valid.txt    -> {valid_file}")
    print(f"  test.txt     -> {test_file}")
    print(f"  entities.txt -> {entities_file}")
    print(f"  relations.txt-> {relations_file}")

    print("\nStatistics")
    print(f"  Total triples (after dedup) : {len(triples)}")
    print(f"  Train                       : {len(train)}  ({100*len(train)/len(triples):.1f}%)")
    print(f"  Valid                       : {len(valid)}  ({100*len(valid)/len(triples):.1f}%)")
    print(f"  Test                        : {len(test)}   ({100*len(test)/len(triples):.1f}%)")
    print(f"  Unique entities             : {len(entities)}")
    print(f"  Unique relations            : {len(relations)}")

    rel_freq: dict[str, int] = defaultdict(int)
    for _, p, _ in triples:
        rel_freq[p] += 1

    print(f"\n  Top-10 relations by frequency:")
    for rel, cnt in sorted(rel_freq.items(), key=lambda x: -x[1])[:10]:
        short = rel.split("/")[-1].split("#")[-1]
        print(f"    {cnt:>6}  {short}")


if __name__ == "__main__":
    main()
