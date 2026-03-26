"""run_td5.py — Master runner for TD5: SWRL reasoning, data prep, KGE training, analysis.
Usage: python src/kge/run_td5.py [--skip-swrl] [--skip-prepare] [--skip-train] [--skip-analyze]
"""

import argparse
import subprocess
import sys
import os
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)


# Run a Python script as a subprocess and return True if it exited successfully.
def run_script(script_path: str, extra_args: list[str] | None = None) -> bool:
    """Run a Python script. Returns True on success."""
    cmd = [sys.executable, script_path] + (extra_args or [])
    print(f"\n  Running: {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - start
    script_name = os.path.basename(script_path)
    if result.returncode != 0:
        print(f"\n  [FAILED] {script_name} exited with code {result.returncode}  ({elapsed:.1f}s)")
        return False
    print(f"\n  [OK] {script_name} completed in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TD5 master runner - executes all four steps."
    )
    parser.add_argument(
        "--skip-swrl", action="store_true",
        help="Skip Step 1: SWRL reasoning"
    )
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Skip Step 2: data preparation"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip Step 3: KGE training"
    )
    parser.add_argument(
        "--skip-analyze", action="store_true",
        help="Skip Step 4: embedding analysis"
    )
    parser.add_argument(
        "--input",
        default="kg_artifacts/medical_kb_expanded.nt",
        help="Path to the expanded N-Triples file (passed to prepare_data.py)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/kge/",
        help="Directory with train/valid/test splits (passed to train_kge.py)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/",
        help="Output directory for models and results"
    )
    args = parser.parse_args()

    print("TD5 — Knowledge Reasoning with SWRL & KGE")
    print(f"  Python      : {sys.executable}")
    print(f"  Root dir    : {ROOT_DIR}")
    print(f"  Input KB    : {args.input}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Output dir  : {args.output_dir}")

    results: dict[str, str] = {}

    swrl_script = os.path.join(SRC_DIR, "reason", "swrl_reasoning.py")
    if not args.skip_swrl:
        print("\nSTEP 1: SWRL Reasoning (family.owl + OWLReady2)")
        ok = run_script(swrl_script)
        results["Step 1 - SWRL"] = "OK" if ok else "FAILED"
    else:
        print("\n  [SKIPPED] Step 1: SWRL reasoning")
        results["Step 1 - SWRL"] = "SKIPPED"

    prepare_script = os.path.join(SCRIPT_DIR, "prepare_data.py")
    if not args.skip_prepare:
        print("\nSTEP 2: Data Preparation (N-Triples -> train/valid/test)")
        ok = run_script(
            prepare_script,
            ["--input", args.input, "--output-dir", args.data_dir],
        )
        results["Step 2 - Prepare"] = "OK" if ok else "FAILED"
    else:
        print("\n  [SKIPPED] Step 2: data preparation")
        results["Step 2 - Prepare"] = "SKIPPED"

    train_script = os.path.join(SCRIPT_DIR, "train_kge.py")
    if not args.skip_train:
        print("\nSTEP 3: KGE Training (TransE + DistMult via PyKEEN)")
        ok = run_script(
            train_script,
            ["--data-dir", args.data_dir, "--output-dir", args.output_dir],
        )
        results["Step 3 - Train"] = "OK" if ok else "FAILED"
    else:
        print("\n  [SKIPPED] Step 3: KGE training")
        results["Step 3 - Train"] = "SKIPPED"

    analyze_script = os.path.join(SCRIPT_DIR, "analyze_kge.py")
    if not args.skip_analyze:
        print("\nSTEP 4: Embedding Analysis (nearest neighbors, t-SNE, relations)")
        transe_dir = os.path.join(args.output_dir.rstrip("/\\"), "TransE/")
        ok = run_script(
            analyze_script,
            [
                "--model-dir", transe_dir,
                "--train-file", os.path.join(args.data_dir.rstrip("/\\"), "train.txt"),
                "--output-dir", args.output_dir,
            ],
        )
        results["Step 4 - Analyze"] = "OK" if ok else "FAILED"
    else:
        print("\n  [SKIPPED] Step 4: embedding analysis")
        results["Step 4 - Analyze"] = "SKIPPED"

    print("\nTD5 — Run Summary")
    all_ok = True
    for step, status in results.items():
        icon = "[OK]" if status == "OK" else ("[SKIP]" if status == "SKIPPED" else "[FAIL]")
        print(f"  {icon:<8} {step}")
        if status == "FAILED":
            all_ok = False

    if all_ok:
        print("\n  All steps completed successfully.")
    else:
        print("\n  Some steps failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
