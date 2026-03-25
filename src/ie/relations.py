"""
Lab 1 – Phase 2b: Relation Extraction
======================================
Reads crawler_output.jsonl and finds (subject, relation, object) triples.
Two entities in the same sentence are linked by the verb between them.

Target relations: hasSymptom | hasTreatment | hasMedication | treatedBy

Output: data/candidate_triples.csv with columns:
  subject | relation | object | subject_label | object_label
  sentence | source_url

Usage:
    python src/ie/relations.py
    python src/ie/relations.py --input data/crawler_output.jsonl
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import spacy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verb → relation mapping
# ---------------------------------------------------------------------------

# When a verb connects a disease to another entity, we map it to a relation.

SYMPTOM_VERBS = {
    "cause", "present", "include", "manifest", "produce", "trigger",
    "characterize", "associate", "feature", "involve", "result",
    "lead", "induce", "exhibit", "show", "display",
}
TREATMENT_VERBS = {
    "treat", "manage", "cure", "alleviate", "relieve", "address",
    "control", "improve", "require", "use", "employ", "undergo",
    "recommend", "involve",
}
MEDICATION_VERBS = {
    "prescribe", "administer", "use", "receive", "take", "require",
    "include", "treat", "involve",
}
SPECIALTY_VERBS = {
    "manage", "treat", "specialize", "handle", "diagnose", "monitor",
    "oversee",
}

# Only diseases can be the subject of a relation
DISEASE_LABELS = {"DISEASE"}

# Expected object type for each relation
RELATION_OBJECT_LABELS = {
    "hasSymptom": {"SYMPTOM"},
    "hasTreatment": {"TREATMENT"},
    "hasMedication": {"MEDICATION"},
    "treatedBy": {"MEDICAL_SPECIALTY"},
}

# All medical entity labels (same as in ner.py)
MEDICAL_LABELS = {"DISEASE", "SYMPTOM", "TREATMENT", "MEDICATION", "MEDICAL_SPECIALTY"}


# ---------------------------------------------------------------------------
# Load a spaCy model with the same medical EntityRuler as ner.py
# ---------------------------------------------------------------------------

def _build_ruler(nlp):
    """Add medical term rules from ner.py to the spaCy pipeline."""
    try:
        import sys
        import os
        ie_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if ie_dir not in sys.path:
            sys.path.insert(0, ie_dir)
        from ner import build_medical_ruler
        return build_medical_ruler(nlp)
    except ImportError:
        # If ner.py cannot be imported, continue without medical patterns
        logger.warning("Could not import ner.py – medical entity patterns disabled")
        return nlp


# ---------------------------------------------------------------------------
# Relation helpers
# ---------------------------------------------------------------------------

def _lemma(token) -> str:
    return token.lemma_.lower()


def verb_to_relation(verb_lemma: str, subj_label: str, obj_label: str) -> str | None:
    """
    Return the relation name for a (subject, verb, object) combination.
    Returns None if no matching relation exists.
    """
    if subj_label not in DISEASE_LABELS:
        return None

    if obj_label in {"SYMPTOM"} and verb_lemma in SYMPTOM_VERBS:
        return "hasSymptom"
    if obj_label in {"TREATMENT"} and verb_lemma in TREATMENT_VERBS:
        return "hasTreatment"
    if obj_label in {"MEDICATION"} and verb_lemma in MEDICATION_VERBS:
        return "hasMedication"
    if obj_label in {"MEDICAL_SPECIALTY"} and verb_lemma in SPECIALTY_VERBS:
        return "treatedBy"

    return None


def _get_ent_label(token, ent_map: dict) -> str | None:
    """Return the entity label for this token, or None if not an entity."""
    return ent_map.get(token.i)


# ---------------------------------------------------------------------------
# Per-sentence extraction
# ---------------------------------------------------------------------------

def extract_from_sentence(sent, ent_map: dict) -> list[dict]:
    """
    Find relation triples in one sentence.

    Step 1: Look for verbs that connect two medical entities (dependency path).
    Step 2: If two entities appear in the same sentence, add a weaker triple
            marked with * (co-occurrence, not confirmed by a verb).
    """
    triples: list[dict] = []
    sentence_text = sent.text.strip().replace("\n", " ")

    # Build a list of entities present in this sentence
    sent_ents = [(tok, _get_ent_label(tok, ent_map)) for tok in sent
                 if _get_ent_label(tok, ent_map) is not None]

    if len(sent_ents) < 2:
        return triples

    # --- Step 1: Verb-based triples (dependency path) ---
    for token in sent:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        verb_lemma = _lemma(token)

        # Collect subjects and objects attached to this verb
        subjects = []
        objects = []

        for child in token.children:
            child_label = _get_ent_label(child, ent_map)
            if child.dep_ in {"nsubj", "nsubjpass"} and child_label:
                subjects.append((child, child_label))
            if child.dep_ in {"dobj", "attr", "pobj", "nmod"} and child_label:
                objects.append((child, child_label))

        for subj_tok, subj_label in subjects:
            for obj_tok, obj_label in objects:
                relation = verb_to_relation(verb_lemma, subj_label, obj_label)
                if relation:
                    triples.append({
                        "subject": subj_tok.text,
                        "relation": relation,
                        "object": obj_tok.text,
                        "subject_label": subj_label,
                        "object_label": obj_label,
                        "sentence": sentence_text[:300],
                    })

    # --- Step 2: Co-occurrence fallback ---
    # If a disease and another medical entity appear in the same sentence,
    # add a weaker triple marked with * (lower confidence).
    disease_ents = [(t, l) for t, l in sent_ents if l == "DISEASE"]
    for dis_tok, _ in disease_ents:
        for obj_tok, obj_label in sent_ents:
            if dis_tok.i == obj_tok.i:
                continue
            if obj_label == "SYMPTOM":
                rel = "hasSymptom"
            elif obj_label == "TREATMENT":
                rel = "hasTreatment"
            elif obj_label == "MEDICATION":
                rel = "hasMedication"
            elif obj_label == "MEDICAL_SPECIALTY":
                rel = "treatedBy"
            else:
                continue

            # Do not add if already found in step 1
            already = any(
                t["subject"] == dis_tok.text and t["relation"] == rel
                and t["object"] == obj_tok.text
                for t in triples
            )
            if not already:
                triples.append({
                    "subject": dis_tok.text,
                    "relation": rel + "*",   # * means co-occurrence (not confirmed by verb)
                    "object": obj_tok.text,
                    "subject_label": "DISEASE",
                    "object_label": obj_label,
                    "sentence": sentence_text[:300],
                })

    return triples


# ---------------------------------------------------------------------------
# Full document processing
# ---------------------------------------------------------------------------

def process_document(nlp, text: str, url: str) -> list[dict]:
    """Extract all relation triples from one article."""
    records: list[dict] = []

    max_len = 900_000
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        try:
            doc = nlp(chunk)
        except Exception as exc:
            logger.warning("spaCy error: %s", exc)
            continue

        # Build a map: token index → entity label (for fast lookup)
        ent_map: dict[int, str] = {}
        for ent in doc.ents:
            for tok in ent:
                ent_map[tok.i] = ent.label_

        for sent in doc.sents:
            try:
                triples = extract_from_sentence(sent, ent_map)
                for t in triples:
                    t["source_url"] = url
                records.extend(triples)
            except Exception as exc:
                logger.debug("Error in sentence: %s", exc)

    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_relations(
    input_file: str = "data/crawler_output.jsonl",
    output_file: str = "data/candidate_triples.csv",
    model: str = "en_core_web_trf",
) -> int:
    """
    Extract relation triples from all articles and save to output_file.
    Returns the total number of triples written.
    """
    in_path = Path(input_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    logger.info("Loading spaCy model: %s", model)
    try:
        nlp = spacy.load(model)
    except OSError:
        logger.error("Model '%s' not found. Run: python -m spacy download %s", model, model)
        raise

    nlp = _build_ruler(nlp)

    fieldnames = [
        "subject", "relation", "object",
        "subject_label", "object_label",
        "sentence", "source_url",
    ]
    total = 0

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        with open(in_path, "r", encoding="utf-8") as fin:
            for i, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc_data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line %d: %s", i + 1, exc)
                    continue

                url = doc_data.get("url", "")
                title = doc_data.get("title", "")
                text = doc_data.get("text", "")

                logger.info("Extracting relations [%d] %s ...", i + 1, title)
                records = process_document(nlp, text, url)
                writer.writerows(records)
                total += len(records)
                logger.info("  → %d candidate triples", len(records))

    logger.info("Relation extraction complete — %d triples written to %s", total, output_file)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Relation Extraction – Lab 1")
    parser.add_argument(
        "--input", default="data/crawler_output.jsonl",
        help="JSONL crawler output file",
    )
    parser.add_argument(
        "--output", default="data/candidate_triples.csv",
        help="Output CSV for candidate triples",
    )
    parser.add_argument(
        "--model", default="en_core_web_trf",
        help="spaCy model name (default: en_core_web_trf)",
    )
    args = parser.parse_args()

    try:
        n = run_relations(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
        )
        print(f"\nDone. {n} candidate triples written to {args.output}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        raise


if __name__ == "__main__":
    main()
