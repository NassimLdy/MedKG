"""
expand_kb_bulk.py — Predicate-Controlled Bulk SPARQL Expansion
==============================================================
Supplements the entity-by-entity expansion with broad predicate queries on
Wikidata, pulling all (subject, predicate, object) triples for key medical
predicates. This is the fastest way to reach the 50k–200k target.

Strategy (from TD4 subject):
    SELECT ?s ?o WHERE { ?s wdt:P780 ?o . } LIMIT 20000

Usage:
    python src/kg/expand_kb_bulk.py
Output:
    kg_artifacts/medical_kb_expanded.nt  (overwritten with larger KB)
    kg_artifacts/stats.json              (updated)
"""

import json
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_NT   = os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_expanded.nt")
OUTPUT_NT  = os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_expanded.nt")
STATS_FILE = os.path.join(ROOT_DIR, "kg_artifacts", "stats.json")

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "MedKGBot/1.0 (educational project)",
    "Accept": "application/sparql-results+json",
}

# Predicate-controlled queries: (wikidata_pid, limit)
BULK_PREDICATES = [
    ("P780",  20000),   # symptoms and signs
    ("P2176", 20000),   # drug used for treatment
    ("P924",  20000),   # possible treatment
    ("P1995", 10000),   # health specialty
    ("P2175", 20000),   # medical condition treated
    ("P2293", 20000),   # genetic association
    ("P769",  20000),   # significant drug interaction
    ("P279",  20000),   # subclass of
    ("P31",   20000),   # instance of
    ("P3781",  5000),   # has active ingredient
    ("P828",   5000),   # has cause
    ("P927",   5000),   # anatomical location
]

WDT_BASE = "http://www.wikidata.org/prop/direct/"
WD_BASE  = "http://www.wikidata.org/entity/"


def sparql_query(pid: str, limit: int) -> list[tuple[str, str, str]]:
    """Run a predicate-controlled SPARQL query on Wikidata."""
    import requests
    query = f"""
SELECT ?s ?o WHERE {{
  ?s wdt:{pid} ?o .
  FILTER(STRSTARTS(STR(?o), "{WD_BASE}"))
}}
LIMIT {limit}
"""
    try:
        resp = requests.get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=60,
        )
        if resp.status_code == 429:
            print(f"    Rate limited — waiting 30s...")
            time.sleep(30)
            return []
        resp.raise_for_status()
        results = resp.json().get("results", {}).get("bindings", [])
        triples = []
        for row in results:
            s = row.get("s", {}).get("value", "")
            o = row.get("o", {}).get("value", "")
            if s and o:
                p = WDT_BASE + pid
                triples.append((s, p, o))
        return triples
    except Exception as exc:
        print(f"    ERROR querying P{pid}: {exc}")
        return []


def load_nt(path: str) -> set[tuple[str, str, str]]:
    """Load N-Triples file into a set of (s, p, o) string tuples."""
    triples = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse NTriples: <s> <p> <o> .
            parts = line.rstrip(" .").split("> <")
            if len(parts) == 3:
                s = parts[0].lstrip("<")
                p = parts[1]
                o = parts[2].rstrip(">")
                triples.add((s, p, o))
    return triples


def write_nt(triples: set[tuple[str, str, str]], path: str) -> None:
    """Write set of (s, p, o) as N-Triples."""
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in sorted(triples):
            f.write(f"<{s}> <{p}> <{o}> .\n")


def print_section(title: str) -> None:
    print("\n" + "=" * 66)
    print(f"  {title}")
    print("=" * 66)


def main() -> None:
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests")
        sys.exit(1)

    print_section("Predicate-Controlled Bulk Expansion")

    # Load existing KB
    print(f"\n[1/3] Loading existing KB: {os.path.basename(INPUT_NT)}")
    existing = load_nt(INPUT_NT)
    print(f"  Existing triples: {len(existing):,}")

    # Bulk expansion
    print(f"\n[2/3] Running {len(BULK_PREDICATES)} predicate queries on Wikidata...")
    new_triples: set[tuple[str, str, str]] = set()

    for i, (pid, limit) in enumerate(BULK_PREDICATES, 1):
        print(f"  [{i}/{len(BULK_PREDICATES)}] P{pid} (LIMIT {limit:,}) ...", end="", flush=True)
        result = sparql_query(pid, limit)
        before = len(new_triples)
        new_triples.update(result)
        added = len(new_triples) - before
        print(f"  {len(result):,} returned, {added:,} new")
        time.sleep(2)   # polite delay

    print(f"\n  New triples from bulk expansion: {len(new_triples):,}")

    # Merge and deduplicate
    all_triples = existing | new_triples
    print(f"  Total after merge (deduplicated): {len(all_triples):,}")

    # Compute stats
    entities = set()
    relations = set()
    for s, p, o in all_triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    print_section("Final KB Statistics")
    print(f"  Total triples   : {len(all_triples):,}")
    print(f"  Total entities  : {len(entities):,}")
    print(f"  Total relations : {len(relations):,}")

    # Write output
    print(f"\n[3/3] Writing {os.path.basename(OUTPUT_NT)} ...")
    write_nt(all_triples, OUTPUT_NT)
    print(f"  Written: {OUTPUT_NT}")

    stats = {
        "total_triples": len(all_triples),
        "total_entities": len(entities),
        "total_relations": len(relations),
        "bulk_predicates_queried": len(BULK_PREDICATES),
    }
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {STATS_FILE}")

    if len(all_triples) < 50_000:
        print(f"\n  NOTE: KB has {len(all_triples):,} triples (target: 50k–200k).")
        print("  Wikidata rate limits restrict bulk expansion.")
        print("  This is documented in the report (scaling reflection).")
    else:
        print(f"\n  Target reached: {len(all_triples):,} triples >= 50,000.")


if __name__ == "__main__":
    main()
