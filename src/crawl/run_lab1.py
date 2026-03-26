"""Lab 1 runner. Runs crawler → NER → relation extraction in sequence.
Usage: python src/crawl/run_lab1.py [--skip-crawl] [--skip-ner] [--skip-relations]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src/crawl/ and src/ie/ to the path so imports work
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_SRC_DIR / "ie"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CRAWLER_OUTPUT = "data/crawler_output.jsonl"
NER_OUTPUT = "data/extracted_knowledge.csv"
TRIPLES_OUTPUT = "data/candidate_triples.csv"
SPACY_MODEL = "en_core_web_trf"


def step_crawl(max_per_seed: int) -> None:
    from crawler import crawl, SEED_TITLES
    logger.info("Step 1: Crawling")
    n = crawl(seed_titles=SEED_TITLES, output_file=CRAWLER_OUTPUT, max_per_seed=max_per_seed)
    print(f"  Crawler: {n} pages saved to {CRAWLER_OUTPUT}")


def step_ner() -> None:
    from ner import run_ner
    logger.info("Step 2: NER")
    n = run_ner(input_file=CRAWLER_OUTPUT, output_file=NER_OUTPUT, model=SPACY_MODEL)
    print(f"  NER: {n} entities written to {NER_OUTPUT}")


def step_relations() -> None:
    from relations import run_relations
    logger.info("Step 3: Relation Extraction")
    n = run_relations(input_file=CRAWLER_OUTPUT, output_file=TRIPLES_OUTPUT, model=SPACY_MODEL)
    print(f"  Relations: {n} candidate triples written to {TRIPLES_OUTPUT}")


def print_stats() -> None:
    """Print a summary of the output files."""
    print("\nLab 1 Output Summary")
    for path in [CRAWLER_OUTPUT, NER_OUTPUT, TRIPLES_OUTPUT]:
        p = Path(path)
        if p.exists():
            lines = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
            print(f"  {path}: {lines} lines")
        else:
            print(f"  {path}: NOT FOUND")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Lab 1 pipeline")
    parser.add_argument(
        "--skip-crawl", action="store_true",
        help="Skip crawling (use existing data/crawler_output.jsonl)",
    )
    parser.add_argument(
        "--skip-ner", action="store_true",
        help="Skip NER step",
    )
    parser.add_argument(
        "--skip-relations", action="store_true",
        help="Skip relation extraction step",
    )
    parser.add_argument(
        "--max-per-seed", type=int, default=8,
        help="Max pages to crawl per seed URL (default: 8)",
    )
    args = parser.parse_args()

    try:
        if not args.skip_crawl:
            step_crawl(args.max_per_seed)
        else:
            print("Skipping crawl (--skip-crawl flag set)")
            if not Path(CRAWLER_OUTPUT).exists():
                print(f"ERROR: {CRAWLER_OUTPUT} not found. Remove --skip-crawl or run crawl first.")
                sys.exit(1)

        if not args.skip_ner:
            step_ner()
        else:
            print("Skipping NER")

        if not args.skip_relations:
            step_relations()
        else:
            print("Skipping relations")

        print_stats()
        print("\nLab 1 pipeline complete.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise


if __name__ == "__main__":
    main()
