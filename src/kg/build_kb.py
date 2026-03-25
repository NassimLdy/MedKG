"""
build_kb.py — Step 1: Build the Initial Knowledge Base
=======================================================

Reads the NER and relation CSV files and builds an RDF graph in Turtle format.

Usage:
    python src/kg/build_kb.py

Input:
    data/extracted_knowledge.csv   — entities found by NER
    data/candidate_triples.csv     — (subject, relation, object) triples

Output:
    kg_artifacts/medical_kb_initial.ttl    — RDF Knowledge Base file

Steps:
    1. Read entities; keep only medical types (DISEASE, SYMPTOM, etc.).
    2. Create a URI for each entity (e.g. med:diabetes).
    3. Add rdf:type, rdfs:label, and med:fromSource triples.
    4. Read triples; skip co-occurrence rows (relation ends with '*').
    5. Add relation triples in the med: namespace.
    6. Save the graph and print statistics.
"""

import os
import re
import csv
import sys
from pathlib import Path

try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
    from rdflib.namespace import NamespaceManager
except ImportError:
    sys.exit("Error: rdflib is required. Install it with: pip install rdflib")

# ==============================================================================
# Configuration
# ==============================================================================

# Project directory paths
SCRIPT_DIR   = Path(__file__).resolve().parent          # src/kg/
SRC_DIR      = SCRIPT_DIR.parent                        # src/
ROOT_DIR     = SRC_DIR.parent                           # MedKG/

INPUT_ENTITIES = ROOT_DIR / "data" / "extracted_knowledge.csv"
INPUT_TRIPLES  = ROOT_DIR / "data" / "candidate_triples.csv"
OUTPUT_KB      = ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl"
ONTOLOGY_FILE  = ROOT_DIR / "kg_artifacts" / "ontology.ttl"

# RDF namespace definitions
MED_NS   = "http://medkg.local/"
MED      = Namespace(MED_NS)
WIKIDATA = Namespace("http://www.wikidata.org/entity/")

# Entity types to include (others are ignored)
KEPT_LABELS = {
    "DISEASE":           MED.Disease,
    "SYMPTOM":           MED.Symptom,
    "TREATMENT":         MED.Treatment,
    "MEDICATION":        MED.Medication,
    "MEDICAL_SPECIALTY": MED.MedicalSpecialty,
}

# Map relation names to their med: predicate URIs
RELATION_MAP = {
    "hasSymptom":   MED.hasSymptom,
    "hasTreatment": MED.hasTreatment,
    "hasMedication":MED.hasMedication,
    "treatedBy":    MED.treatedBy,
}

# ==============================================================================
# Utility functions
# ==============================================================================

def slugify(text: str) -> str:
    """
    Convert an entity label to a URI-safe slug.
    Example: "Type 2 diabetes" → "type_2_diabetes"
    """
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^\w]", "", text)           # keep word chars (a-z,0-9,_)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text if text else "unknown"


def make_entity_uri(entity: str) -> URIRef:
    """Build the med: URI for an entity (e.g. med:diabetes)."""
    return MED[slugify(entity)]


def load_ontology(graph: Graph) -> None:
    """Load class and property definitions from ontology.ttl into the graph."""
    if ONTOLOGY_FILE.exists():
        try:
            graph.parse(str(ONTOLOGY_FILE), format="turtle")
            print(f"  [ontology] Loaded {ONTOLOGY_FILE.name}")
        except Exception as exc:
            print(f"  [ontology] Warning: could not load ontology.ttl: {exc}")
    else:
        print("  [ontology] ontology.ttl not found; proceeding without it.")


# ==============================================================================
# Step 1 — Build entities from extracted_knowledge.csv
# ==============================================================================

def build_entities(graph: Graph) -> dict[str, URIRef]:
    """
    Read extracted_knowledge.csv and add entities to the RDF graph.
    For each entity, adds: rdf:type, rdfs:label, and med:fromSource triples.
    Returns a dict mapping entity slug → URIRef.
    """
    entity_uris: dict[str, URIRef] = {}
    rows_read   = 0
    rows_kept   = 0
    rows_skipped = 0

    if not INPUT_ENTITIES.exists():
        sys.exit(f"Error: Input file not found: {INPUT_ENTITIES}")

    with open(INPUT_ENTITIES, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_read += 1
            entity = (row.get("entity") or "").strip()
            label  = (row.get("label")  or "").strip()
            source = (row.get("source_url") or "").strip()

            if not entity or label not in KEPT_LABELS:
                rows_skipped += 1
                continue

            uri = make_entity_uri(entity)
            rdf_class = KEPT_LABELS[label]

            # Add the three core triples for this entity
            graph.add((uri, RDF.type,      rdf_class))
            graph.add((uri, RDFS.label,    Literal(entity, lang="en")))
            if source:
                graph.add((uri, MED.fromSource, URIRef(source)))

            entity_uris[slugify(entity)] = uri
            rows_kept += 1

    print(f"  [entities] Rows read: {rows_read}  |  Kept: {rows_kept}  |  Skipped: {rows_skipped}")
    return entity_uris


# ==============================================================================
# Step 2 — Build relations from candidate_triples.csv
# ==============================================================================

def build_relations(graph: Graph, entity_uris: dict[str, URIRef]) -> int:
    """
    Read candidate_triples.csv and add relation triples to the graph.
    Skips rows with empty fields, co-occurrence triples (*), or unknown relations.
    Returns the number of triples added.
    """
    rows_read     = 0
    triples_added = 0
    skipped_cooc  = 0
    skipped_empty = 0
    skipped_map   = 0

    if not INPUT_TRIPLES.exists():
        sys.exit(f"Error: Input file not found: {INPUT_TRIPLES}")

    with open(INPUT_TRIPLES, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_read += 1
            subject  = (row.get("subject")  or "").strip()
            relation = (row.get("relation") or "").strip()
            obj      = (row.get("object")   or "").strip()

            if not subject or not obj:
                skipped_empty += 1
                continue

            # Skip co-occurrence triples (marked with *)
            if relation.endswith("*"):
                skipped_cooc += 1
                continue

            # Skip unknown relation names
            pred_uri = RELATION_MAP.get(relation)
            if pred_uri is None:
                skipped_map += 1
                continue

            # Build URIs for subject and object
            subj_uri = make_entity_uri(subject)
            obj_uri  = make_entity_uri(obj)

            graph.add((subj_uri, pred_uri, obj_uri))
            triples_added += 1

    print(
        f"  [relations] Rows read: {rows_read}  |  Added: {triples_added}  "
        f"|  Skipped co-occ: {skipped_cooc}  |  Skipped empty: {skipped_empty}  "
        f"|  Skipped unknown rel: {skipped_map}"
    )
    return triples_added


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """Run all steps to build the initial Knowledge Base."""
    print("=" * 70)
    print("Step 1 — Initial Knowledge Base Construction")
    print("=" * 70)

    OUTPUT_KB.parent.mkdir(parents=True, exist_ok=True)

    # Create the RDF graph and register namespace prefixes
    g = Graph()
    g.bind("med",  MED)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)
    g.bind("wd",   WIKIDATA)

    print("\n[1/3] Loading ontology schema...")
    load_ontology(g)

    print("\n[2/3] Processing entities (extracted_knowledge.csv)...")
    entity_uris = build_entities(g)

    print("\n[3/3] Processing relations (candidate_triples.csv)...")
    n_relations = build_relations(g, entity_uris)

    # Save the graph to a Turtle file
    g.serialize(destination=str(OUTPUT_KB), format="turtle")

    # Print statistics
    total_triples  = len(g)
    entity_set     = set(g.subjects(RDF.type, None))  # subjects with rdf:type
    relation_preds = {
        MED.hasSymptom, MED.hasTreatment, MED.hasMedication, MED.treatedBy
    }
    relation_triples = sum(1 for _ in g.triples((None, None, None))
                           if _[1] in relation_preds)

    print("\n" + "=" * 70)
    print("Output Statistics")
    print("=" * 70)
    print(f"  Output file     : {OUTPUT_KB}")
    print(f"  Total triples   : {total_triples}")
    print(f"  Unique entities : {len(entity_set)}")
    print(f"  Entities from CSV: {len(entity_uris)}")
    print(f"  Relation triples: {relation_triples}")
    print(f"  Relations added : {n_relations}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
