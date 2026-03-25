"""
filter_kb_for_kge.py — Keep only core med: entities and their neighbors
========================================================================
75k entities is too big for KGE training on CPU.
This script keeps only triples that involve the core med: entities
and their direct Wikidata neighbors. The result is ~5-10k entities.

Usage:
    python src/kge/filter_kb_for_kge.py
Output:
    kg_artifacts/medical_kb_filtered.nt
"""

import os, re

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_NT   = os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_expanded.nt")
OUTPUT_NT  = os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_filtered.nt")
MED_NS     = "http://medkg.local/"


def parse_nt_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # Match <s> <p> <o> .  (object can be IRI or literal)
    m = re.match(r'<([^>]+)>\s+<([^>]+)>\s+(.*)\s+\.$', line)
    if not m:
        return None
    s, p, o_raw = m.group(1), m.group(2), m.group(3).strip()
    # Only keep entity-entity triples (both IRI)
    if o_raw.startswith("<") and o_raw.endswith(">"):
        o = o_raw[1:-1]
        return s, p, o
    return None


def main():
    print("=" * 60)
    print("  KB Filter: med: core + 1-hop Wikidata neighbors")
    print("=" * 60)

    # Step 1: read all triples and find the core med: entities
    print(f"\n[1/3] Reading {os.path.basename(INPUT_NT)} ...")
    all_triples = []
    med_entities = set()

    with open(INPUT_NT, encoding="utf-8") as f:
        for line in f:
            t = parse_nt_line(line)
            if t:
                s, p, o = t
                all_triples.append(t)
                if s.startswith(MED_NS):
                    med_entities.add(s)
                if o.startswith(MED_NS):
                    med_entities.add(o)

    print(f"  Total triples read  : {len(all_triples):,}")
    print(f"  Core med: entities  : {len(med_entities):,}")

    # Also keep Wikidata QIDs that are linked to med: entities via owl:sameAs
    OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"
    wikidata_qids = set()
    for s, p, o in all_triples:
        if p == OWL_SAME_AS and s in med_entities:
            wikidata_qids.add(o)
        if p == OWL_SAME_AS and o in med_entities:
            wikidata_qids.add(s)
    core = med_entities | wikidata_qids
    print(f"  Aligned Wikidata QIDs: {len(wikidata_qids):,}")
    print(f"  Total core entities  : {len(core):,}")

    # Step 2: keep triples where the subject or object is a core entity
    hop1 = [(s, p, o) for s, p, o in all_triples
            if s in core or o in core]
    hop1_entities = {s for s, p, o in hop1} | {o for s, p, o in hop1}
    print(f"\n[2/3] 1-hop expansion ...")
    print(f"  Triples kept        : {len(hop1):,}")
    print(f"  Entities in 1-hop   : {len(hop1_entities):,}")

    # Step 3: write the filtered triples
    print(f"\n[3/3] Writing {os.path.basename(OUTPUT_NT)} ...")
    with open(OUTPUT_NT, "w", encoding="utf-8") as f:
        for s, p, o in hop1:
            f.write(f"<{s}> <{p}> <{o}> .\n")

    print(f"  Done: {OUTPUT_NT}")
    print(f"\n  Summary:")
    print(f"    Triples  : {len(hop1):,}")
    print(f"    Entities : {len(hop1_entities):,}")


if __name__ == "__main__":
    main()
