"""
run_td4.py — Master Runner for TD4 Pipeline
============================================

Executes all three TD4 steps in sequence:
    1. Build initial KB from IE data         (build_kb.py logic)
    2. Link entities to Wikidata             (entity_linking.py logic)
    3. Expand KB via Wikidata SPARQL         (expand_kb.py logic)

Each step can be skipped via a command-line flag.

Usage:
    python src/kg/run_td4.py
    python src/kg/run_td4.py --skip-build
    python src/kg/run_td4.py --skip-link
    python src/kg/run_td4.py --skip-expand
    python src/kg/run_td4.py --skip-build --skip-link

Flags:
    --skip-build    Skip Step 1 (build_kb.py); requires medical_kb_initial.ttl
    --skip-link     Skip Step 2 (entity_linking.py); requires alignment.ttl
    --skip-expand   Skip Step 3 (expand_kb.py)

After all steps, prints a summary table of all output files and their sizes.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

# ==============================================================================
# Configuration — expected output files
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent   # src/kg/
SRC_DIR    = SCRIPT_DIR.parent                 # src/
ROOT_DIR   = SRC_DIR.parent                    # MedKG/

OUTPUT_FILES = {
    "ontology.ttl":             ROOT_DIR / "kg_artifacts" / "ontology.ttl",
    "medical_kb_initial.ttl":   ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl",
    "alignment.ttl":            ROOT_DIR / "kg_artifacts" / "alignment.ttl",
    "entity_mapping.csv":       ROOT_DIR / "kg_artifacts" / "entity_mapping.csv",
    "medical_kb_expanded.nt":   ROOT_DIR / "kg_artifacts" / "medical_kb_expanded.nt",
    "stats.json":               ROOT_DIR / "kg_artifacts" / "stats.json",
}


# ==============================================================================
# Step loading helpers
# ==============================================================================

def load_module_from_file(name: str, file_path: Path):
    """
    Dynamically import a Python module from *file_path* with module *name*.
    Returns the module object. Raises SystemExit on failure.
    """
    if not file_path.exists():
        sys.exit(f"Error: Script not found: {file_path}")

    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        sys.exit(f"Error: Cannot load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_step(step_num: int, description: str, module_path: Path) -> bool:
    """
    Execute the ``main()`` function of the module at *module_path*.

    Returns True on success, False on failure.
    Prints a header and footer around the step.
    """
    print("\n" + "#" * 70)
    print(f"# STEP {step_num}: {description}")
    print("#" * 70)

    start = time.time()
    try:
        module = load_module_from_file(f"step{step_num}", module_path)
        if not hasattr(module, "main"):
            print(f"  Error: {module_path.name} has no main() function.")
            return False
        module.main()
        elapsed = time.time() - start
        print(f"\n  Step {step_num} completed in {elapsed:.1f}s.")
        return True
    except SystemExit as exc:
        print(f"  Step {step_num} exited: {exc}")
        return False
    except Exception as exc:
        print(f"  Step {step_num} failed with error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Summary printer
# ==============================================================================

def format_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def print_summary(results: dict[str, bool]) -> None:
    """Print a formatted table of step results and output file sizes."""
    print("\n" + "=" * 70)
    print("TD4 Pipeline — Final Summary")
    print("=" * 70)

    # Step results
    print("\n  Step Results:")
    for step_name, success in results.items():
        status = "OK" if success else "SKIPPED / FAILED"
        print(f"    {step_name:<35} {status}")

    # Output files
    print("\n  Output Files:")
    print(f"    {'File':<35} {'Size':>10}   {'Status'}")
    print("    " + "-" * 55)
    for fname, fpath in OUTPUT_FILES.items():
        if fpath.exists():
            size_str = format_size(fpath.stat().st_size)
            print(f"    {fname:<35} {size_str:>10}   PRESENT")
        else:
            print(f"    {fname:<35} {'—':>10}   MISSING")

    print("=" * 70)


# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="TD4 Master Runner — Medical Knowledge Graph Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    """Orchestrate the full TD4 pipeline."""
    args = parse_args()

    print("=" * 70)
    print(" TD4 — Medical Knowledge Graph: Build, Link, Expand")
    print("=" * 70)
    print(f"\n  Working directory : {ROOT_DIR}")
    print(f"  KG artifacts dir  : {ROOT_DIR / 'kg_artifacts'}")
    print(f"  Data dir          : {ROOT_DIR / 'data'}")
    print()

    # Check that the IE source files exist (needed by build_kb)
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

    # ------------------------------------------------------------------
    # Step 1 — Build initial KB
    # ------------------------------------------------------------------
    if args.skip_build:
        print("[SKIP] Step 1: Initial KB Construction")
        step_results["Step 1: Build KB"] = True
        if not (ROOT_DIR / "kg_artifacts" / "medical_kb_initial.ttl").exists():
            print("  Warning: medical_kb_initial.ttl not found — "
                  "subsequent steps may fail.")
    else:
        success = run_step(
            1, "Initial KB Construction",
            SCRIPT_DIR / "build_kb.py"
        )
        step_results["Step 1: Build KB"] = success
        if not success:
            print("\n  Step 1 failed. Aborting pipeline.")
            print_summary(step_results)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 — Entity Linking
    # ------------------------------------------------------------------
    if args.skip_link:
        print("\n[SKIP] Step 2: Entity Linking")
        step_results["Step 2: Entity Linking"] = True
        if not (ROOT_DIR / "kg_artifacts" / "alignment.ttl").exists():
            print("  Warning: alignment.ttl not found — "
                  "expansion step may find no aligned entities.")
    else:
        success = run_step(
            2, "Entity Linking to Wikidata",
            SCRIPT_DIR / "entity_linking.py"
        )
        step_results["Step 2: Entity Linking"] = success
        if not success:
            print("\n  Step 2 failed. Aborting pipeline.")
            print_summary(step_results)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3 — KB Expansion
    # ------------------------------------------------------------------
    if args.skip_expand:
        print("\n[SKIP] Step 3: KB Expansion")
        step_results["Step 3: KB Expansion"] = True
    else:
        success = run_step(
            3, "KB Expansion via Wikidata SPARQL",
            SCRIPT_DIR / "expand_kb.py"
        )
        step_results["Step 3: KB Expansion"] = success
        if not success:
            print("\n  Step 3 failed (see errors above).")
            step_results["Step 3: KB Expansion"] = False

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    print(f"\n  Total pipeline time: {total_elapsed:.1f}s")
    print_summary(step_results)


if __name__ == "__main__":
    main()
