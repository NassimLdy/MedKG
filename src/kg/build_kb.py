"""build_kb.py — Build the initial RDF Knowledge Base from NER and relation CSV files.
Usage: python src/kg/build_kb.py
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

SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
ROOT_DIR     = SRC_DIR.parent

INPUT_ENTITIES = ROOT_DIR / "data" / "extracted_knowledge.csv"
INPUT_TRIPLES  = ROOT_DIR / "data" / "candidate_triples.csv"
OUTPUT_KB      = ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl"
ONTOLOGY_FILE  = ROOT_DIR / "kg_artifacts" / "ontology.ttl"

MED_NS   = "http://medkg.local/"
MED      = Namespace(MED_NS)
WIKIDATA = Namespace("http://www.wikidata.org/entity/")

KEPT_LABELS = {
    "DISEASE":           MED.Disease,
    "SYMPTOM":           MED.Symptom,
    "TREATMENT":         MED.Treatment,
    "MEDICATION":        MED.Medication,
    "MEDICAL_SPECIALTY": MED.MedicalSpecialty,
}

RELATION_MAP = {
    "hasSymptom":   MED.hasSymptom,
    "hasTreatment": MED.hasTreatment,
    "hasMedication":MED.hasMedication,
    "treatedBy":    MED.treatedBy,
}

# Normalize an entity label to a URI-safe lowercase slug.
def slugify(text: str) -> str:
    """Convert a label to a URI-safe slug. Example: "Type 2 diabetes" → "type_2_diabetes"."""
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^\w]", "", text)           # keep word chars (a-z,0-9,_)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")
    return text if text else "unknown"


def make_entity_uri(entity: str) -> URIRef:
    """Build the med: URI for an entity."""
    return MED[slugify(entity)]


# Parse ontology.ttl and merge its class and property definitions into the graph.
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


# Read extracted_knowledge.csv and add entity triples to the RDF graph.
def build_entities(graph: Graph) -> dict[str, URIRef]:
    """Read extracted_knowledge.csv and add entity triples to the graph."""
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

            graph.add((uri, RDF.type,      rdf_class))
            graph.add((uri, RDFS.label,    Literal(entity, lang="en")))
            if source:
                graph.add((uri, MED.fromSource, URIRef(source)))

            entity_uris[slugify(entity)] = uri
            rows_kept += 1

    print(f"  [entities] Rows read: {rows_read}  |  Kept: {rows_kept}  |  Skipped: {rows_skipped}")
    return entity_uris


# Read candidate_triples.csv and add relation triples to the RDF graph.
def build_relations(graph: Graph, entity_uris: dict[str, URIRef]) -> int:
    """Read candidate_triples.csv and add relation triples to the graph."""
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

            if relation.endswith("*"):  # skip co-occurrence rows
                skipped_cooc += 1
                continue

            pred_uri = RELATION_MAP.get(relation)
            if pred_uri is None:
                skipped_map += 1
                continue

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


def main() -> None:
    """Build the initial Knowledge Base."""
    print("Step 1 — Initial Knowledge Base Construction")
    OUTPUT_KB.parent.mkdir(parents=True, exist_ok=True)
    g = Graph()
    g.bind("med",  MED)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)
    g.bind("wd",   WIKIDATA)

    print("\n[1/3] Loading ontology...")
    load_ontology(g)
    print("\n[2/3] Processing entities...")
    entity_uris = build_entities(g)
    print("\n[3/3] Processing relations...")
    n_relations = build_relations(g, entity_uris)

    g.serialize(destination=str(OUTPUT_KB), format="turtle")

    total_triples  = len(g)
    entity_set     = set(g.subjects(RDF.type, None))
    relation_preds = {
        MED.hasSymptom, MED.hasTreatment, MED.hasMedication, MED.treatedBy
    }
    relation_triples = sum(1 for _ in g.triples((None, None, None))
                           if _[1] in relation_preds)

    print(f"\n  Output: {OUTPUT_KB}")
    print(f"  Total triples: {total_triples}  |  Entities: {len(entity_set)}  |  Relations: {relation_triples}")
    print("Done.")


if __name__ == "__main__":
    main()
