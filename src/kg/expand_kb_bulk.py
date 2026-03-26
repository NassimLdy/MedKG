"""expand_kb_bulk.py — Bulk KB expansion by querying Wikidata for each medical predicate.
Usage: python src/kg/expand_kb_bulk.py
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

BULK_PREDICATES = [
    ("P780",  20000),
    ("P2176", 20000),
    ("P924",  20000),
    ("P1995", 10000),
    ("P2175", 20000),
    ("P2293", 20000),
    ("P769",  20000),
    ("P279",  20000),
    ("P31",   20000),
    ("P3781",  5000),
    ("P828",   5000),
    ("P927",   5000),
]

WDT_BASE = "http://www.wikidata.org/prop/direct/"
WD_BASE  = "http://www.wikidata.org/entity/"


# Fetch all entity triples for a given Wikidata predicate in a single bulk query.
def sparql_query(pid: str, limit: int) -> list[tuple[str, str, str]]:
    """Query Wikidata for all triples with the given predicate."""
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
            print("    Rate limited — waiting 30s...")
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


# Load an N-Triples file into a set of (subject, predicate, object) string tuples.
def load_nt(path: str) -> set[tuple[str, str, str]]:
    """Load an N-Triples file into a set of (s, p, o) string tuples."""
    triples = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip(" .").split("> <")
            if len(parts) == 3:
                s = parts[0].lstrip("<")
                p = parts[1]
                o = parts[2].rstrip(">")
                triples.add((s, p, o))
    return triples


# Write a set of (s, p, o) string tuples to an N-Triples file.
def write_nt(triples: set[tuple[str, str, str]], path: str) -> None:
    """Write (s, p, o) tuples as N-Triples."""
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in sorted(triples):
            f.write(f"<{s}> <{p}> <{o}> .\n")


def main() -> None:
    try:
        import requests
    except ImportError:
        print("ERROR: requests not installed. Run: pip install requests")
        sys.exit(1)

    print("Bulk Expansion via Wikidata")
    print(f"\n[1/3] Loading existing KB: {os.path.basename(INPUT_NT)}")
    existing = load_nt(INPUT_NT)
    print(f"  Existing triples: {len(existing):,}")

    print(f"\n[2/3] Running {len(BULK_PREDICATES)} predicate queries on Wikidata...")
    new_triples: set[tuple[str, str, str]] = set()

    for i, (pid, limit) in enumerate(BULK_PREDICATES, 1):
        print(f"  [{i}/{len(BULK_PREDICATES)}] P{pid} (LIMIT {limit:,}) ...", end="", flush=True)
        result = sparql_query(pid, limit)
        before = len(new_triples)
        new_triples.update(result)
        added = len(new_triples) - before
        print(f"  {len(result):,} returned, {added:,} new")
        time.sleep(2)

    print(f"\n  New triples from bulk expansion: {len(new_triples):,}")

    all_triples = existing | new_triples
    print(f"  Total after merge: {len(all_triples):,}")
    entities = set()
    relations = set()
    for s, p, o in all_triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    print(f"\n  Triples: {len(all_triples):,}  |  Entities: {len(entities):,}  |  Relations: {len(relations):,}")
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
