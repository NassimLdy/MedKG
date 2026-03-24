"""
Lab 1 – Phase 2b: Relation Extraction via Dependency Parsing
=============================================================
Reads crawler_output.jsonl, finds entity pairs in the same sentence,
uses spaCy dependency tags (nsubj / dobj / prep) to extract a connecting
verb and outputs candidate (subject, relation, object) triples.

Focuses on the four target relations:
  hasSymptom   |  hasTreatment  |  hasMedication  |  treatedBy

Outputs data/candidate_triples.csv with columns:
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
# Verb → relation mapping  (lower-case lemma → canonical relation URI)
# ---------------------------------------------------------------------------

# These verbs, when connecting a DISEASE-like entity to another entity,
# suggest a specific relation.  Covers a wide range of Wikipedia phrasing.

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

# Labels that can be subjects (disease-like)
DISEASE_LABELS = {"DISEASE"}

# Labels that can be objects for each relation
RELATION_OBJECT_LABELS = {
    "hasSymptom": {"SYMPTOM"},
    "hasTreatment": {"TREATMENT"},
    "hasMedication": {"MEDICATION"},
    "treatedBy": {"MEDICAL_SPECIALTY"},
}

# All medical entity labels (must mirror ner.py vocabulary)
MEDICAL_LABELS = {"DISEASE", "SYMPTOM", "TREATMENT", "MEDICATION", "MEDICAL_SPECIALTY"}


# ---------------------------------------------------------------------------
# Load a spaCy model with the same medical EntityRuler as ner.py
# ---------------------------------------------------------------------------

def _build_ruler(nlp):
    """Import and reuse the ruler builder from ner.py."""
    try:
        import sys
        import os
        # Add src/ie to path so ner can be imported
        ie_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        if ie_dir not in sys.path:
            sys.path.insert(0, ie_dir)
        from ner import build_medical_ruler
        return build_medical_ruler(nlp)
    except ImportError:
        # Fallback: add ruler without patterns (basic NER still runs)
        logger.warning("Could not import ner.py – medical entity patterns disabled")
        return nlp


# ---------------------------------------------------------------------------
# Relation helpers
# ---------------------------------------------------------------------------

def _lemma(token) -> str:
    return token.lemma_.lower()


def verb_to_relation(verb_lemma: str, subj_label: str, obj_label: str) -> str | None:
    """
    Given a verb lemma and entity labels, return the canonical relation name
    or None if no match.
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
    """Return the NER label for the span that *token* is part of."""
    return ent_map.get(token.i)


# ---------------------------------------------------------------------------
# Per-sentence extraction
# ---------------------------------------------------------------------------

def extract_from_sentence(sent, ent_map: dict) -> list[dict]:
    """
    Extract (subject, relation, object) triples from a single sentence.

    Strategy:
    1. For every ROOT verb in the sentence, look for nsubj and dobj / pobj.
    2. Check if nsubj and object are medical entities with a valid relation.
    3. Also search for co-occurrence pairs even without a clear dep path
       (using sentence-level co-occurrence as fallback).
    """
    triples: list[dict] = []
    sentence_text = sent.text.strip().replace("\n", " ")

    # Build a list of entities present in this sentence
    sent_ents = [(tok, _get_ent_label(tok, ent_map)) for tok in sent
                 if _get_ent_label(tok, ent_map) is not None]

    if len(sent_ents) < 2:
        return triples

    # --- Strategy 1: Dependency-path triples ---
    for token in sent:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        verb_lemma = _lemma(token)

        # Collect subjects and objects attached to this verb
        subjects = []
        objects = []

        for child in token.children:
            child_label = _get_ent_label(child, ent_map)
            if child_label is None:
                # Check if child is part of a multi-token entity head
                # by looking at children of child
                pass
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

    # --- Strategy 2: Co-occurrence fallback ---
    # If a DISEASE and a SYMPTOM/TREATMENT/MEDICATION appear in the same sentence,
    # emit a tentative triple (lower confidence, marked with *).
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

            # Avoid duplicate with strategy 1
            already = any(
                t["subject"] == dis_tok.text and t["relation"] == rel
                and t["object"] == obj_tok.text
                for t in triples
            )
            if not already:
                triples.append({
                    "subject": dis_tok.text,
                    "relation": rel + "*",   # * = co-occurrence, not dep-parsed
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
    """Run relation extraction on the full text of one crawled page."""
    records: list[dict] = []

    max_len = 900_000
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        try:
            doc = nlp(chunk)
        except Exception as exc:
            logger.warning("spaCy error: %s", exc)
            continue

        # Build token-index → NER label map for fast lookup
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
    Process all documents in *input_file* and write triples to *output_file*.
    Returns total number of triples written.
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
