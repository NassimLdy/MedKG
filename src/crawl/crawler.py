"""
Lab 1 – Phase 1: Web Crawler for Medical Domain (Wikipedia API)
===============================================================
Uses the Wikipedia MediaWiki API (/w/api.php) to fetch article text.
The API is explicitly allowed by Wikipedia's robots.txt for all agents.
No trafilatura needed: the API returns clean plain text directly.

Usage:
    python src/crawl/crawler.py
    python src/crawl/crawler.py --max-per-seed 10 --output data/crawler_output.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED_TITLES = [
    "Diabetes",
    "Hypertension",
    "Asthma",
    "Cancer",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Stroke",
    "Major depressive disorder",
    "COVID-19",
    "Heart failure",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
BOT_USER_AGENT = "MedKGBot/1.0 (educational project; uses Wikipedia API)"
HEADERS = {"User-Agent": BOT_USER_AGENT}

MIN_WORDS = 500
CRAWL_DELAY = 1.0          # polite delay between API calls (seconds)
DEFAULT_MAX_PER_SEED = 8   # seed page + up to N-1 linked pages per seed

SKIP_PREFIXES = (
    "Wikipedia:", "Talk:", "User:", "Help:", "Portal:",
    "File:", "Category:", "Template:", "Draft:", "Special:",
)

MEDICAL_KEYWORDS = {
    "disease", "disorder", "syndrome", "infection", "cancer", "tumor",
    "diabetes", "hypertension", "heart", "lung", "brain", "liver",
    "kidney", "treatment", "therapy", "medication", "drug", "surgery",
    "symptom", "diagnosis", "patient", "clinical", "medical", "health",
    "virus", "bacteria", "inflammation", "chronic", "acute", "condition",
    "pathology", "etiology", "prognosis", "pharmaceutical", "antibiotic",
}


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def fetch_article(session: requests.Session, title: str) -> dict | None:
    """
    Fetch plain-text content for *title* via the MediaWiki API.
    Returns a dict with keys: url, title, text, word_count, timestamp.
    Returns None if the article is missing or too short.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": "true",
        "exsectionformat": "plain",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    }
    try:
        resp = session.get(WIKIPEDIA_API, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("API error for '%s': %s", title, exc)
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        logger.debug("Page missing: %s", title)
        return None

    text = page.get("extract", "")
    if not text:
        return None

    word_count = len(text.split())
    if word_count < MIN_WORDS:
        logger.debug("Too short (%d words): %s", word_count, title)
        return None

    url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")

    return {
        "url": url,
        "title": page.get("title", title),
        "text": text,
        "word_count": word_count,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def fetch_links(session: requests.Session, title: str, limit: int = 100) -> list[str]:
    """
    Fetch internal Wikipedia links from *title* via the API.
    Returns a list of linked article titles filtered to medical ones.
    """
    params = {
        "action": "query",
        "prop": "links",
        "titles": title,
        "pllimit": limit,
        "plnamespace": 0,
        "format": "json",
        "formatversion": "2",
    }
    try:
        resp = session.get(WIKIPEDIA_API, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Link fetch error for '%s': %s", title, exc)
        return []

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return []

    links = []
    for link in pages[0].get("links", []):
        link_title = link.get("title", "")
        if any(link_title.startswith(p) for p in SKIP_PREFIXES):
            continue
        if is_medical_title(link_title):
            links.append(link_title)
    return links


def is_medical_title(title: str) -> bool:
    """Heuristic: does the article title look medically relevant?"""
    words = title.lower().replace("_", " ")
    return any(kw in words for kw in MEDICAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Core crawl logic
# ---------------------------------------------------------------------------

def crawl(
    seed_titles: list[str] = SEED_TITLES,
    output_file: str = "data/crawler_output.jsonl",
    max_per_seed: int = DEFAULT_MAX_PER_SEED,
) -> int:
    """
    Crawl Wikipedia medical articles via the API starting from *seed_titles*.
    For each seed, also fetches up to *max_per_seed* linked medical articles.
    Returns the number of pages saved.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    total_saved = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    with open(out_path, "w", encoding="utf-8") as fout:
        for seed_title in seed_titles:
            logger.info("=== Seed: %s ===", seed_title)
            queue = [seed_title]
            seed_count = 0

            while queue and seed_count < max_per_seed:
                title = queue.pop(0)

                if title in visited:
                    continue
                visited.add(title)

                record = fetch_article(session, title)
                time.sleep(CRAWL_DELAY)

                if record is None:
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                total_saved += 1
                seed_count += 1
                logger.info(
                    "[%d] Saved: %s (%d words)",
                    total_saved, record["title"], record["word_count"],
                )

                # Enqueue medical links found on this page
                if seed_count < max_per_seed:
                    links = fetch_links(session, record["title"])
                    time.sleep(CRAWL_DELAY)
                    new = [t for t in links if t not in visited]
                    queue.extend(new[:50])
                    logger.debug("Enqueued %d candidate links from %s", len(new[:50]), title)

    logger.info("Crawl complete — %d pages saved to %s", total_saved, output_file)
    return total_saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Wikipedia Crawler (API) – Lab 1")
    parser.add_argument(
        "--output", default="data/crawler_output.jsonl",
        help="Path to output JSONL file (default: data/crawler_output.jsonl)",
    )
    parser.add_argument(
        "--max-per-seed", type=int, default=DEFAULT_MAX_PER_SEED,
        help=f"Max pages to save per seed title (default: {DEFAULT_MAX_PER_SEED})",
    )
    args = parser.parse_args()

    try:
        n = crawl(
            seed_titles=SEED_TITLES,
            output_file=args.output,
            max_per_seed=args.max_per_seed,
        )
        print(f"\nDone. {n} pages saved to {args.output}")
    except KeyboardInterrupt:
        print("\nCrawl interrupted by user.")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        raise


if __name__ == "__main__":
    main()
