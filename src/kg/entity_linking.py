"""entity_linking.py — Link KB entities to Wikidata via owl:sameAs.
Output: alignment.ttl and entity_mapping.csv
Usage: python src/kg/entity_linking.py
"""

import csv
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Error: requests is required. Install with: pip install requests")

try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
except ImportError:
    sys.exit("Error: rdflib is required. Install with: pip install rdflib")

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent
ROOT_DIR   = SRC_DIR.parent

INPUT_KB         = ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl"
OUTPUT_ALIGNMENT = ROOT_DIR / "kg_artifacts" / "alignment.ttl"
OUTPUT_MAPPING   = ROOT_DIR / "kg_artifacts" / "entity_mapping.csv"
ONTOLOGY_FILE    = ROOT_DIR / "kg_artifacts" / "ontology.ttl"

MED  = Namespace("http://medkg.local/")
WD   = Namespace("http://www.wikidata.org/entity/")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
API_RATE_LIMIT  = 0.5   # wait 0.5 seconds between API calls
MIN_CONFIDENCE  = 0.6   # skip links below this score

MEDICAL_CLASSES = {
    MED.Disease,
    MED.Symptom,
    MED.Treatment,
    MED.Medication,
    MED.MedicalSpecialty,
}

# Query the Wikidata search API for an entity label and return raw results.
def search_wikidata(label: str, retries: int = 2) -> list[dict]:
    """Search Wikidata for a label. Returns up to 3 results, or [] on failure."""
    params = {
        "action":   "wbsearchentities",
        "search":   label,
        "language": "en",
        "format":   "json",
        "limit":    "3",
    }
    headers = {"User-Agent": "MedKGBot/1.0 (educational project)"}

    for attempt in range(retries + 1):
        try:
            resp = requests.get(WIKIDATA_API, params=params,
                                headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("search", [])
        except requests.exceptions.HTTPError as exc:
            print(f"    [API] HTTP error for '{label}': {exc}  "
                  f"(attempt {attempt + 1}/{retries + 1})")
        except requests.exceptions.ConnectionError as exc:
            print(f"    [API] Connection error for '{label}': {exc}  "
                  f"(attempt {attempt + 1}/{retries + 1})")
        except requests.exceptions.Timeout:
            print(f"    [API] Timeout for '{label}'  "
                  f"(attempt {attempt + 1}/{retries + 1})")
        except ValueError as exc:
            print(f"    [API] JSON parse error for '{label}': {exc}")
            return []

        if attempt < retries:
            time.sleep(1.0)

    return []


# Score how well a Wikidata search result matches the query label.
def compute_confidence(query_label: str, result: dict) -> float:
    """Score match quality: 1.0 = exact, 0.8 = partial, 0.6 = any result."""
    wd_label = result.get("label", "").lower().strip()
    wd_desc  = result.get("description", "").lower().strip()
    q_lower  = query_label.lower().strip()

    if wd_label == q_lower:
        return 1.0
    if q_lower in wd_desc or wd_label in q_lower:
        return 0.8
    return 0.6


# For each KB entity, find a Wikidata QID and add owl:sameAs triples to the alignment graph.
def link_entities(kb_graph: Graph, align_graph: Graph) -> list[dict]:
    """Search Wikidata for each entity. Adds owl:sameAs triples to align_graph."""
    mapping_records: list[dict] = []

    entities_to_link: list[tuple[URIRef, str, URIRef]] = []

    for class_uri in MEDICAL_CLASSES:
        for subj in kb_graph.subjects(RDF.type, class_uri):
            label_val = None
            for lbl in kb_graph.objects(subj, RDFS.label):
                if isinstance(lbl, Literal):
                    if lbl.language == "en" or lbl.language is None:
                        label_val = str(lbl)
                        break
            if label_val is None:
                label_val = str(subj).replace(str(MED), "").replace("_", " ")

            entities_to_link.append((subj, label_val, class_uri))

    seen_uris: set[str] = set()
    unique_entities: list[tuple[URIRef, str, URIRef]] = []
    for item in entities_to_link:
        uri_str = str(item[0])
        if uri_str not in seen_uris:
            seen_uris.add(uri_str)
            unique_entities.append(item)

    total   = len(unique_entities)
    linked  = 0
    not_found = 0

    print(f"\n  Total unique medical entities to link: {total}")

    for idx, (uri, label, class_uri) in enumerate(unique_entities, start=1):
        if idx % 20 == 0 or idx == 1:
            print(f"  Progress: {idx}/{total} entities processed ...")

        results = search_wikidata(label)
        time.sleep(API_RATE_LIMIT)

        if results:
            top = results[0]
            confidence = compute_confidence(label, top)

            if confidence >= MIN_CONFIDENCE:
                qid      = top["id"]
                wd_uri   = WD[qid]
                align_graph.add((uri, OWL.sameAs, wd_uri))

                mapping_records.append({
                    "private_entity": str(uri),
                    "external_uri":   str(wd_uri),
                    "confidence":     round(confidence, 2),
                })
                linked += 1
                continue

        not_found += 1
        align_graph.add((uri, RDF.type, OWL.Class))
        align_graph.add((uri, RDFS.subClassOf, class_uri))
        mapping_records.append({
            "private_entity": str(uri),
            "external_uri":   "",
            "confidence":     0.0,
        })

    print(f"\n  Linked   : {linked}/{total}")
    print(f"  Not found: {not_found}/{total}")
    return mapping_records


# Declare med: predicates equivalent to their Wikidata property counterparts via owl:equivalentProperty.
def add_predicate_alignments(align_graph: Graph) -> None:
    """Link med: predicates to Wikidata equivalents via owl:equivalentProperty."""
    alignments = [
        (MED.hasSymptom,    WDT.P780),
        (MED.hasTreatment,  WDT.P924),
        (MED.hasMedication, WDT.P2176),
        (MED.treatedBy,     WDT.P1995),
    ]
    for med_prop, wdt_prop in alignments:
        align_graph.add((med_prop, OWL.equivalentProperty, wdt_prop))
        align_graph.add((wdt_prop, OWL.equivalentProperty, med_prop))

    print(f"  Added {len(alignments) * 2} predicate alignment triples.")


def main() -> None:
    """Run the entity linking pipeline."""
    print("Step 2 — Entity Linking to Wikidata")
    if not INPUT_KB.exists():
        sys.exit(f"Error: Input KB not found: {INPUT_KB}\n"
                 "Run build_kb.py first.")

    print(f"\n[1/4] Loading initial KB: {INPUT_KB.name} ...")
    kb_graph = Graph()
    kb_graph.parse(str(INPUT_KB), format="turtle")
    print(f"  Loaded {len(kb_graph)} triples.")

    align_graph = Graph()
    align_graph.bind("med",  MED)
    align_graph.bind("wd",   WD)
    align_graph.bind("wdt",  WDT)
    align_graph.bind("owl",  OWL)
    align_graph.bind("rdfs", RDFS)
    align_graph.bind("rdf",  RDF)

    print("\n[2/4] Adding predicate alignments ...")
    add_predicate_alignments(align_graph)
    print("\n[3/4] Linking entities via Wikidata API ...")
    mapping_records = link_entities(kb_graph, align_graph)
    print("\n[4/4] Writing output files ...")
    OUTPUT_ALIGNMENT.parent.mkdir(parents=True, exist_ok=True)
    align_graph.serialize(destination=str(OUTPUT_ALIGNMENT), format="turtle")
    print(f"  Wrote {OUTPUT_ALIGNMENT.name} ({len(align_graph)} triples)")
    with open(OUTPUT_MAPPING, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["private_entity", "external_uri", "confidence"]
        )
        writer.writeheader()
        writer.writerows(mapping_records)
    print(f"  Wrote {OUTPUT_MAPPING.name} ({len(mapping_records)} rows)")
    linked_count   = sum(1 for r in mapping_records if r["external_uri"])
    unlinked_count = sum(1 for r in mapping_records if not r["external_uri"])

    print(f"\n  Entities: {len(mapping_records)}  |  Linked: {linked_count}  |  Not found: {unlinked_count}")
    print("Done.")


if __name__ == "__main__":
    main()
