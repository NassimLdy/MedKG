"""swrl_reasoning.py — SWRL reasoning with OWLReady2 on family.owl.
Usage: python src/reason/swrl_reasoning.py
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OWL_FILE = os.path.join(SCRIPT_DIR, "family.owl")

try:
    from owlready2 import get_ontology, sync_reasoner_hermit, sync_reasoner_pellet
except ImportError:
    print("ERROR: owlready2 is not installed.")
    print("Install it with:  pip install owlready2")
    sys.exit(1)


def get_age(individual) -> int | None:
    """Return the hasAge value for an individual, or None."""
    vals = individual.hasAge
    if vals:
        return vals[0] if isinstance(vals, list) else vals
    return None


def main() -> None:
    print("\nLoading family.owl")
    if not os.path.isfile(OWL_FILE):
        print(f"ERROR: Cannot find {OWL_FILE}")
        sys.exit(1)

    onto = get_ontology(f"file://{OWL_FILE}").load()
    print(f"Ontology loaded: {onto.base_iri}")
    print(f"Classes         : {[c.name for c in onto.classes()]}")
    print(f"Data properties : {[p.name for p in onto.data_properties()]}")
    print(f"Object properties: {[p.name for p in onto.object_properties()]}")

    print("\nIndividuals BEFORE Reasoning")

    individuals = list(onto.individuals())
    print(f"Total individuals: {len(individuals)}\n")

    print(f"{'Name':<12} {'Classes':<35} {'Age'}")
    print("-" * 65)
    for ind in individuals:
        classes = [c.name for c in ind.is_a]
        age = get_age(ind)
        print(f"{ind.name:<12} {str(classes):<35} {age}")

    # Before reasoning — who is already OldPerson?
    OldPerson_before = [i for i in individuals if onto.OldPerson in i.is_a]
    Person_before = list(onto.Person.instances())
    print(f"\nPerson instances (before): {[i.name for i in Person_before]}")
    print(f"OldPerson instances (before): {[i.name for i in OldPerson_before]}")

    print("\nRunning Reasoner (HermiT -> Pellet fallback)")

    reasoner_used = None
    with onto:
        try:
            print("Attempting HermiT reasoner...")
            sync_reasoner_hermit(infer_property_values=True)
            reasoner_used = "HermiT"
            print("HermiT reasoning complete.")
        except Exception as hermit_err:
            print(f"HermiT failed: {hermit_err}")
            try:
                print("Attempting Pellet reasoner...")
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
                reasoner_used = "Pellet"
                print("Pellet reasoning complete.")
            except Exception as pellet_err:
                print(f"Pellet also failed: {pellet_err}")
                print("\nFalling back to manual SWRL rule evaluation...")
                reasoner_used = "manual"
                _apply_rule_manually(onto)

    print(f"\nReasoner used: {reasoner_used}")

    print("\nIndividuals AFTER Reasoning")

    individuals_after = list(onto.individuals())
    print(f"{'Name':<12} {'Classes':<45} {'Age'}")
    print("-" * 70)
    for ind in individuals_after:
        classes = [c.name for c in ind.is_a]
        age = get_age(ind)
        print(f"{ind.name:<12} {str(classes):<45} {age}")

    print("\nSummary")

    Person_after = list(onto.Person.instances())
    OldPerson_after = list(onto.OldPerson.instances())

    print(f"All Person instances ({len(Person_after)}):")
    for p in sorted(Person_after, key=lambda x: x.name):
        age = get_age(p)
        print(f"  - {p.name} (age {age})")

    print(f"\nInferred OldPerson instances ({len(OldPerson_after)}):")
    for op in sorted(OldPerson_after, key=lambda x: x.name):
        age = get_age(op)
        print(f"  - {op.name} (age {age})")

    newly_inferred = set(OldPerson_after) - set(OldPerson_before)
    if newly_inferred:
        print(f"\nNewly inferred as OldPerson by rule (age > 60):")
        for ni in sorted(newly_inferred, key=lambda x: x.name):
            print(f"  * {ni.name} (age {get_age(ni)})")
    else:
        print("\nNo new OldPerson inferences (check if SWRL support is active in your reasoner).")

    print("\nSWRL Rule Applied")
    print("  Person(?p) ^ hasAge(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> OldPerson(?p)")
    print()
    print("  Explanation:")
    print("  For every individual ?p that is a Person, if ?p has an age ?a")
    print("  and ?a is strictly greater than 60, then ?p is inferred to be")
    print("  an OldPerson.")


# Assert OldPerson for all Persons with age > 60 when no OWL reasoner is available.
def _apply_rule_manually(onto) -> None:
    """Apply the SWRL rule by hand when no reasoner works."""
    print("Manually applying rule: Person(?p) ^ hasAge(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> OldPerson(?p)")
    count = 0
    for ind in list(onto.individuals()):
        if onto.Person in ind.is_a or any(
            issubclass(c, onto.Person) for c in ind.is_a if hasattr(c, "__mro__")
        ):
            age = get_age(ind)
            if age is not None and age > 60:
                if onto.OldPerson not in ind.is_a:
                    ind.is_a.append(onto.OldPerson)
                    print(f"  -> Asserting {ind.name} as OldPerson (age={age})")
                    count += 1
    print(f"Manual rule application: {count} new OldPerson assertions.")


# Apply the Disease-hasSymptom → affectedBy SWRL rule on the medical KB using rdflib.
def run_medical_swrl() -> None:
    """Apply Disease(?d) ^ hasSymptom(?d, ?s) -> affectedBy(?s, ?d) on the medical KB."""
    print("\nSWRL Rule on Medical KB")

    RULE = "Disease(?d) ^ hasSymptom(?d, ?s) -> affectedBy(?s, ?d)"
    print(f"\n  Rule: {RULE}")
    print()

    ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    candidates = [
        os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_expanded.nt"),
        os.path.join(ROOT_DIR, "kg_artifacts", "medical_kb_initial.ttl"),
    ]
    kb_path = None
    for c in candidates:
        if os.path.isfile(c):
            kb_path = c
            break

    if kb_path is None:
        print("  Medical KB not found. Run src/kg/run_td4.py first.")
        print("  (Expected: kg_artifacts/medical_kb_initial.ttl)")
        return

    try:
        from rdflib import Graph, Namespace
    except ImportError:
        print("  rdflib not installed. Run: pip install rdflib")
        return

    fmt = "nt" if kb_path.endswith(".nt") else "turtle"
    g = Graph()
    g.parse(kb_path, format=fmt)
    print(f"  Loaded {len(g)} triples from {os.path.basename(kb_path)}")

    MED = Namespace("http://medkg.local/")
    hasSymptom = MED.hasSymptom
    affectedBy = MED.affectedBy

    new_triples = set()
    for d, _, s in g.triples((None, hasSymptom, None)):
        new_triples.add((s, affectedBy, d))

    print(f"\n  Applying rule: {len(new_triples)} new 'affectedBy' triples inferred")
    print("\n  Sample inferences (up to 10):")
    for i, (s, p, d) in enumerate(list(new_triples)[:10]):
        s_label = str(s).split("/")[-1].replace("_", " ")
        d_label = str(d).split("/")[-1].replace("_", " ")
        print(f"    ({s_label}) --affectedBy--> ({d_label})")

    print(f"\n  SWRL rule inference complete: {len(new_triples)} triples derived.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWRL Reasoning")
    parser.add_argument("--medical", action="store_true",
                        help="Also run the medical KB SWRL rule")
    args = parser.parse_args()
    main()
    if args.medical:
        run_medical_swrl()
