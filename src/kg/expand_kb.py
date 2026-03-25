"""
expand_kb.py — Step 3: Expand the Knowledge Base via Wikidata
=============================================================

Runs SPARQL queries on Wikidata to add more triples to the knowledge base.
Target: 50,000–200,000 triples total.

Usage:
    python src/kg/expand_kb.py

Input:
    kg_artifacts/medical_kb_initial.ttl   — initial KB
    kg_artifacts/alignment.ttl            — owl:sameAs links to Wikidata

Output:
    kg_artifacts/medical_kb_expanded.nt   — expanded KB (N-Triples format)
    kg_artifacts/stats.json               — statistics

Steps:
    1. Load the initial KB and alignment.
    2. Find all Wikidata QIDs from owl:sameAs links.
    3. For each QID: run a 1-hop SPARQL query (get direct properties).
    4. For new entities found in step 3: run 2-hop queries (capped at 50).
    5. Clean: remove non-English text and bad URIs.
    6. Save the result as N-Triples and write stats.json.

Politeness: 1 second delay between SPARQL calls. Retry once on error.
"""

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("Error: requests is required. Install with: pip install requests")

try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD, BNode
except ImportError:
    sys.exit("Error: rdflib is required. Install with: pip install rdflib")

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent   # src/kg/
SRC_DIR    = SCRIPT_DIR.parent                 # src/
ROOT_DIR   = SRC_DIR.parent                    # MedKG/

INPUT_KB        = ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl"
INPUT_ALIGNMENT = ROOT_DIR / "kg_artifacts" / "alignment.ttl"
OUTPUT_EXPANDED = ROOT_DIR / "kg_artifacts" / "medical_kb_expanded.nt"
OUTPUT_STATS    = ROOT_DIR / "kg_artifacts" / "stats.json"

# Namespaces
MED = Namespace("http://medkg.local/")
WD  = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

# Wikidata SPARQL endpoint
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
SPARQL_DELAY    = 1.0   # seconds to wait between queries
MAX_2HOP_ENTITIES = 50  # max entities for 2-hop expansion
SPARQL_LIMIT    = 500   # max results per SPARQL query

# HTTP headers for the Wikidata API
HEADERS = {
    "User-Agent": "MedKGBot/1.0 (educational project)",
    "Accept":     "application/sparql-results+json",
}

# ==============================================================================
# Predicate whitelist — only medical/useful Wikidata predicates are retained
# ==============================================================================

WHITELIST_PIDS = {
    "P780",   # symptoms
    "P2176",  # drug used for treatment
    "P924",   # possible treatment
    "P1995",  # health specialty
    "P279",   # subclass of
    "P31",    # instance of
    "P361",   # part of
    "P527",   # has part
    "P1050",  # medical condition
    "P2175",  # medical condition treated
    "P636",   # route of administration
    "P769",   # significant drug interaction
    "P2293",  # genetic association
    "P828",   # has cause
    "P1419",  # anatomy
    "P486",   # MeSH descriptor ID
    "P652",   # OMIM ID (for completeness)
    "P2888",  # exact match
    "P18",    # image (excluded in cleaning)
}

WHITELIST_URIS = {
    f"http://www.wikidata.org/prop/direct/{pid}" for pid in WHITELIST_PIDS
}

# ==============================================================================
# SPARQL query helpers
# ==============================================================================

def build_sparql_query(qid: str, limit: int = SPARQL_LIMIT) -> str:
    """Build a SPARQL query to get all direct properties of a Wikidata entity."""
    return f"""
SELECT ?p ?o WHERE {{
  wd:{qid} ?p ?o .
  FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
}}
LIMIT {limit}
""".strip()


def execute_sparql(query: str, retries: int = 1) -> list[dict]:
    """
    Send a SPARQL query to Wikidata and return the result rows.
    Returns an empty list if the query fails after retries.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", {}).get("bindings", [])
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            print(f"    [SPARQL] HTTP {status} error (attempt {attempt + 1}): {exc}")
            if status == 429:
                # Rate limited — wait longer
                time.sleep(5.0)
            elif status == 503:
                time.sleep(3.0)
        except requests.exceptions.ConnectionError as exc:
            print(f"    [SPARQL] Connection error (attempt {attempt + 1}): {exc}")
            time.sleep(2.0)
        except requests.exceptions.Timeout:
            print(f"    [SPARQL] Timeout (attempt {attempt + 1})")
            time.sleep(2.0)
        except ValueError as exc:
            print(f"    [SPARQL] JSON parse error: {exc}")
            return []

        if attempt < retries:
            time.sleep(SPARQL_DELAY)

    return []


# ==============================================================================
# URI / Literal validation helpers
# ==============================================================================

def is_valid_uri(uri_str: str) -> bool:
    """Return True if the string is a valid URI (not a blank node)."""
    if not uri_str or uri_str.startswith("_:"):
        return False
    try:
        parsed = urlparse(uri_str)
        return bool(parsed.scheme and (parsed.netloc or parsed.path))
    except Exception:
        return False


def is_acceptable_literal(obj) -> bool:
    """Return True for non-string literals or English string literals."""
    if not isinstance(obj, Literal):
        return True
    lang = obj.language
    if lang is None:
        return True
    return lang.lower().startswith("en")


def is_whitelisted_predicate(pred_uri: str) -> bool:
    """Return True if we want to keep triples with this predicate."""
    return pred_uri in WHITELIST_URIS


# ==============================================================================
# Expansion core
# ==============================================================================

def sparql_bindings_to_triples(
    qid: str,
    bindings: list[dict],
    expanded_graph: Graph,
) -> set[str]:
    """
    Convert SPARQL results into rdflib triples and add them to expanded_graph.
    Only keeps whitelisted predicates and English literals.
    Returns new Wikidata QIDs found (for 2-hop expansion).
    """
    subject_uri = WD[qid]
    new_qids: set[str] = set()

    for binding in bindings:
        p_val = binding.get("p", {}).get("value", "")
        o_val = binding.get("o", {}).get("value", "")
        o_type = binding.get("o", {}).get("type", "")
        o_lang = binding.get("o", {}).get("xml:lang", None)
        o_dtype = binding.get("o", {}).get("datatype", None)

        # Filter predicate
        if not is_whitelisted_predicate(p_val):
            continue

        # Validate predicate URI
        if not is_valid_uri(p_val):
            continue

        pred_uri = URIRef(p_val)

        # Build object node
        if o_type == "uri":
            if not is_valid_uri(o_val):
                continue
            obj_node = URIRef(o_val)
            # Save QIDs of new entities for 2-hop expansion
            if o_val.startswith("http://www.wikidata.org/entity/Q"):
                qid_candidate = o_val.rsplit("/", 1)[-1]
                new_qids.add(qid_candidate)
        elif o_type in ("literal", "typed-literal"):
            # Filter non-English strings
            if o_lang and not o_lang.lower().startswith("en"):
                continue
            if o_dtype:
                try:
                    obj_node = Literal(o_val, datatype=URIRef(o_dtype))
                except Exception:
                    obj_node = Literal(o_val)
            elif o_lang:
                obj_node = Literal(o_val, lang=o_lang)
            else:
                obj_node = Literal(o_val)
        else:
            continue  # blank nodes, etc.

        expanded_graph.add((subject_uri, pred_uri, obj_node))

    return new_qids


def run_expansion(aligned_qids: list[str], expanded_graph: Graph) -> set[str]:
    """
    Run 1-hop SPARQL queries for all aligned QIDs.
    Returns new QIDs found during expansion (for 2-hop queries).
    """
    all_new_qids: set[str] = set()
    total = len(aligned_qids)

    for idx, qid in enumerate(aligned_qids, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"  [1-hop] {idx}/{total} — querying wd:{qid} ...")

        query    = build_sparql_query(qid)
        bindings = execute_sparql(query)
        new_qids = sparql_bindings_to_triples(qid, bindings, expanded_graph)
        all_new_qids.update(new_qids)
        all_new_qids.discard(qid)  # don't re-expand what we already did

        time.sleep(SPARQL_DELAY)

    return all_new_qids


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """Run all steps to expand the Knowledge Base."""
    print("=" * 70)
    print("Step 3 — Knowledge Base Expansion via Wikidata SPARQL")
    print("=" * 70)

    # --- Load inputs ----------------------------------------------------------
    for fpath in (INPUT_KB, INPUT_ALIGNMENT):
        if not fpath.exists():
            sys.exit(f"Error: Required input not found: {fpath}\n"
                     "Run previous pipeline steps first.")

    print(f"\n[1/5] Loading initial KB ({INPUT_KB.name}) ...")
    kb_graph = Graph()
    kb_graph.parse(str(INPUT_KB), format="turtle")
    print(f"  {len(kb_graph)} triples loaded.")

    print(f"\n[2/5] Loading alignment ({INPUT_ALIGNMENT.name}) ...")
    align_graph = Graph()
    align_graph.parse(str(INPUT_ALIGNMENT), format="turtle")
    print(f"  {len(align_graph)} alignment triples loaded.")

    # --- Extract aligned QIDs -------------------------------------------------
    WD_PREFIX = "http://www.wikidata.org/entity/"
    aligned_qids: list[str] = []
    for _, _, obj in align_graph.triples((None, OWL.sameAs, None)):
        obj_str = str(obj)
        if obj_str.startswith(WD_PREFIX + "Q"):
            qid = obj_str[len(WD_PREFIX):]
            aligned_qids.append(qid)

    aligned_qids = list(dict.fromkeys(aligned_qids))  # remove duplicates
    print(f"\n[3/5] Found {len(aligned_qids)} aligned Wikidata QIDs.")

    if not aligned_qids:
        print("  Warning: No aligned entities found. "
              "The expanded KB will contain only the original triples.")

    # Start with KB + alignment triples, then add Wikidata data
    print("\n[4/5] Running SPARQL expansion ...")
    expanded_graph = Graph()
    expanded_graph.bind("med", MED)
    expanded_graph.bind("wd",  WD)
    expanded_graph.bind("wdt", WDT)

    for triple in kb_graph:
        expanded_graph.add(triple)
    print(f"  Copied {len(kb_graph)} KB triples.")

    for triple in align_graph:
        expanded_graph.add(triple)
    print(f"  Copied alignment triples. Graph now has {len(expanded_graph)} triples.")

    # 1-hop: query each aligned entity
    print(f"\n  --- 1-hop expansion ({len(aligned_qids)} entities) ---")
    before_1hop = len(expanded_graph)
    all_new_qids = run_expansion(aligned_qids, expanded_graph)
    after_1hop = len(expanded_graph)
    print(f"  1-hop added {after_1hop - before_1hop} triples. "
          f"Discovered {len(all_new_qids)} new QIDs for 2-hop.")

    # 2-hop: query new entities discovered in step 3 (capped)
    already_expanded = set(aligned_qids)
    hop2_candidates = sorted(all_new_qids - already_expanded)[:MAX_2HOP_ENTITIES]

    print(f"\n  --- 2-hop expansion ({len(hop2_candidates)} entities, "
          f"cap={MAX_2HOP_ENTITIES}) ---")
    before_2hop = len(expanded_graph)

    for idx, qid in enumerate(hop2_candidates, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"  [2-hop] {idx}/{len(hop2_candidates)} — querying wd:{qid} ...")
        query    = build_sparql_query(qid, limit=200)
        bindings = execute_sparql(query)
        sparql_bindings_to_triples(qid, bindings, expanded_graph)
        time.sleep(SPARQL_DELAY)

    after_2hop = len(expanded_graph)
    print(f"  2-hop added {after_2hop - before_2hop} triples.")

    # Remove blank nodes, bad URIs, and non-English text
    print("\n  --- Cleaning triples ---")
    to_remove: list[tuple] = []
    for s, p, o in expanded_graph:
        if isinstance(s, BNode) or isinstance(p, BNode):
            to_remove.append((s, p, o))
            continue
        if isinstance(s, URIRef) and not is_valid_uri(str(s)):
            to_remove.append((s, p, o))
            continue
        if isinstance(o, Literal) and not is_acceptable_literal(o):
            to_remove.append((s, p, o))

    for triple in to_remove:
        expanded_graph.remove(triple)

    print(f"  Removed {len(to_remove)} unacceptable triples.")
    print(f"  Final graph size: {len(expanded_graph)} triples.")

    # --- Serialize output -----------------------------------------------------
    OUTPUT_EXPANDED.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[5/5] Serializing to {OUTPUT_EXPANDED.name} ...")
    expanded_graph.serialize(destination=str(OUTPUT_EXPANDED), format="nt")
    print(f"  Written: {OUTPUT_EXPANDED}")

    # --- Compute statistics ---------------------------------------------------
    total_triples   = len(expanded_graph)
    unique_subj = {s for s, _, _ in expanded_graph if isinstance(s, URIRef)}
    unique_obj  = {o for _, _, o in expanded_graph if isinstance(o, URIRef)}
    total_entities_precise = len(unique_subj | unique_obj)
    unique_preds = {p for _, p, _ in expanded_graph if isinstance(p, URIRef)}
    total_relations = len(unique_preds)

    stats = {
        "total_triples":   total_triples,
        "total_entities":  total_entities_precise,
        "total_relations": total_relations,
        "aligned_qids":    len(aligned_qids),
        "hop2_entities":   len(hop2_candidates),
        "triples_removed_cleaning": len(to_remove),
    }

    with open(OUTPUT_STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"  Written: {OUTPUT_STATS}")

    # --- Final summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Expansion Statistics")
    print("=" * 70)
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<35} : {v:,}")
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
