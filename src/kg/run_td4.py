"""run_td4.py — Master runner for the TD4 pipeline (build KB → link → expand).
Usage: python src/kg/run_td4.py [--skip-build] [--skip-link] [--skip-expand]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent
ROOT_DIR   = SRC_DIR.parent

OUTPUT_FILES = {
    "ontology.ttl":             ROOT_DIR / "kg_artifacts" / "ontology.ttl",
    "medical_kb_initial.ttl":   ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl",
    "alignment.ttl":            ROOT_DIR / "kg_artifacts" / "alignment.ttl",
    "entity_mapping.csv":       ROOT_DIR / "kg_artifacts" / "entity_mapping.csv",
    "medical_kb_expanded.nt":   ROOT_DIR / "kg_artifacts" / "medical_kb_expanded.nt",
    "stats.json":               ROOT_DIR / "kg_artifacts" / "stats.json",
}


# Dynamically import a Python file as a module by path.
def load_module_from_file(name: str, file_path: Path):
    """Load a Python file as a module, or exit on failure."""
    if not file_path.exists():
        sys.exit(f"Error: Script not found: {file_path}")

    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        sys.exit(f"Error: Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load and run a pipeline step module, reporting elapsed time and success.
def run_step(step_num: int, description: str, module_path: Path) -> bool:
    """Run main() of a pipeline step. Returns True on success."""
    print(f"\nSTEP {step_num}: {description}")

    start = time.time()
    try:
        module = load_module_from_file(f"step{step_num}", module_path)
        if not hasattr(module, "main"):
            print(f"  Error: {module_path.name} has no main() function.")
            return False
        module.main()
        elapsed = time.time() - start
        print(f"  Step {step_num} done in {elapsed:.1f}s.")
        return True
    except SystemExit as exc:
        print(f"  Step {step_num} exited: {exc}")
        return False
    except Exception as exc:
        print(f"  Step {step_num} failed with error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False


def format_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def print_summary(results: dict[str, bool]) -> None:
    """Print step results and output file sizes."""
    print("\nTD4 — Summary")
    print("  Step Results:")
    for step_name, success in results.items():
        status = "OK" if success else "SKIPPED / FAILED"
        print(f"    {step_name:<35} {status}")

    print("\n  Output Files:")
    for fname, fpath in OUTPUT_FILES.items():
        if fpath.exists():
            size_str = format_size(fpath.stat().st_size)
            print(f"    {fname:<35} {size_str:>10}   OK")
        else:
            print(f"    {fname:<35} {'—':>10}   MISSING")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TD4 pipeline: build KB, link to Wikidata, expand."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip Step 1: Initial KB construction (requires medical_kb_initial.ttl)",
    )
    parser.add_argument(
        "--skip-link",
        action="store_true",
        help="Skip Step 2: Entity linking (requires alignment.ttl)",
    )
    parser.add_argument(
        "--skip-expand",
        action="store_true",
        help="Skip Step 3: KB expansion via Wikidata SPARQL",
    )
    return parser.parse_args()


def main() -> None:
    """Run all TD4 pipeline steps."""
    args = parse_args()
    print("TD4 — Medical Knowledge Graph: Build, Link, Expand")
    print(f"  Root: {ROOT_DIR}")
    print()

    if not args.skip_build:
        data_entities = ROOT_DIR / "data" / "extracted_knowledge.csv"
        data_triples  = ROOT_DIR / "data" / "candidate_triples.csv"
        missing = [f for f in (data_entities, data_triples) if not f.exists()]
        if missing:
            for f in missing:
                print(f"  Error: Data source file not found: {f}")
            sys.exit(1)

    step_results: dict[str, bool] = {}
    overall_start = time.time()

    # Step 1
    if args.skip_build:
        print("[SKIP] Step 1: Initial KB Construction")
        step_results["Step 1: Build KB"] = True
        if not (ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl").exists():
            print("  Warning: medical_kb_initial.ttl not found.")
    else:
        success = run_step(1, "Initial KB Construction", SCRIPT_DIR / "build_kb.py")
        step_results["Step 1: Build KB"] = success
        if not success:
            print_summary(step_results)
            sys.exit(1)

    # Step 2
    if args.skip_link:
        print("\n[SKIP] Step 2: Entity Linking")
        step_results["Step 2: Entity Linking"] = True
        if not (ROOT_DIR / "kg_artifacts" / "alignment.ttl").exists():
            print("  Warning: alignment.ttl not found.")
    else:
        success = run_step(2, "Entity Linking to Wikidata", SCRIPT_DIR / "entity_linking.py")
        step_results["Step 2: Entity Linking"] = success
        if not success:
            print_summary(step_results)
            sys.exit(1)

    # Step 3
    if args.skip_expand:
        print("\n[SKIP] Step 3: KB Expansion")
        step_results["Step 3: KB Expansion"] = True
    else:
        success = run_step(3, "KB Expansion via Wikidata SPARQL", SCRIPT_DIR / "expand_kb.py")
        step_results["Step 3: KB Expansion"] = success

    total_elapsed = time.time() - overall_start
    print(f"\n  Total time: {total_elapsed:.1f}s")
    print_summary(step_results)


if __name__ == "__main__":
    main()
