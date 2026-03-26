"""Lab 1 – Named Entity Recognition. Finds medical entities in crawled articles.
Output: extracted_knowledge.csv (entity, label, context, source_url, source_title)
Usage: python src/ie/ner.py [--input PATH] [--output PATH] [--model MODEL]
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import spacy
from spacy.language import Language

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DISEASES = [
    "diabetes", "type 1 diabetes", "type 2 diabetes", "gestational diabetes",
    "hypertension", "high blood pressure", "asthma", "cancer",
    "alzheimer's disease", "alzheimer disease", "parkinson's disease",
    "parkinson disease", "stroke", "ischemic stroke", "hemorrhagic stroke",
    "major depressive disorder", "depression", "COVID-19", "covid 19",
    "SARS-CoV-2", "heart failure", "congestive heart failure",
    "coronary artery disease", "myocardial infarction", "heart attack",
    "atrial fibrillation", "hepatitis", "hepatitis B", "hepatitis C",
    "tuberculosis", "malaria", "HIV", "AIDS", "HIV/AIDS",
    "rheumatoid arthritis", "osteoarthritis", "arthritis",
    "osteoporosis", "crohn's disease", "crohn disease", "ulcerative colitis",
    "multiple sclerosis", "epilepsy", "seizure disorder",
    "obesity", "sepsis", "pneumonia", "influenza", "flu",
    "chronic obstructive pulmonary disease", "COPD", "emphysema",
    "leukemia", "lymphoma", "melanoma", "breast cancer", "lung cancer",
    "prostate cancer", "colon cancer", "colorectal cancer",
    "chronic kidney disease", "kidney failure", "renal failure",
    "liver disease", "cirrhosis", "fatty liver disease",
    "schizophrenia", "bipolar disorder", "anxiety disorder",
    "attention deficit hyperactivity disorder", "ADHD",
    "autism spectrum disorder", "autism",
    "anemia", "sickle cell anemia", "thalassemia",
    "hyperthyroidism", "hypothyroidism", "thyroid disease",
    "glaucoma", "macular degeneration", "cataracts",
    "psoriasis", "eczema", "dermatitis",
    "migraine", "cluster headache",
    "gout", "lupus", "systemic lupus erythematosus",
    "fibromyalgia", "chronic fatigue syndrome",
]

SYMPTOMS = [
    "fever", "headache", "fatigue", "nausea", "vomiting",
    "cough", "dyspnea", "shortness of breath",
    "chest pain", "chest tightness", "palpitations",
    "dizziness", "vertigo", "syncope",
    "abdominal pain", "stomach pain", "cramping",
    "diarrhea", "constipation", "bloating",
    "inflammation", "swelling", "edema",
    "rash", "hives", "itching", "pruritus",
    "muscle weakness", "muscle pain", "myalgia",
    "joint pain", "arthralgia",
    "numbness", "tingling", "paresthesia",
    "blurred vision", "vision loss",
    "weight loss", "weight gain",
    "night sweats", "chills",
    "jaundice", "yellowing",
    "seizure", "convulsion",
    "tremor", "rigidity",
    "memory loss", "confusion", "cognitive decline",
    "anxiety", "irritability", "mood swings",
    "insomnia", "sleep disturbance",
    "loss of appetite", "anorexia",
    "polyuria", "polydipsia", "polyphagia",
    "hemoptysis", "coughing blood",
    "hematuria", "blood in urine",
    "dysuria", "painful urination",
    "hypertension", "high blood pressure",
    "tachycardia", "bradycardia",
    "pallor", "cyanosis",
]

TREATMENTS = [
    "chemotherapy", "radiation therapy", "radiotherapy",
    "surgery", "surgical resection", "amputation",
    "immunotherapy", "targeted therapy",
    "dialysis", "hemodialysis", "peritoneal dialysis",
    "physical therapy", "physiotherapy",
    "cognitive behavioral therapy", "CBT", "psychotherapy",
    "organ transplant", "kidney transplant", "liver transplant",
    "heart transplant", "bone marrow transplant",
    "stem cell therapy", "stem cell transplant",
    "angioplasty", "bypass surgery", "coronary bypass",
    "stenting", "stent placement",
    "blood transfusion",
    "oxygen therapy", "mechanical ventilation",
    "electrotherapy", "laser therapy",
    "hormone therapy", "hormone replacement therapy",
    "rehabilitation", "occupational therapy",
    "dietary therapy", "nutritional therapy",
    "phototherapy", "light therapy",
    "deep brain stimulation",
    "electroconvulsive therapy", "ECT",
]

MEDICATIONS = [
    "insulin", "metformin", "glipizide", "sitagliptin",
    "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "naproxen",
    "amoxicillin", "penicillin", "ampicillin", "azithromycin",
    "ciprofloxacin", "doxycycline", "metronidazole",
    "atorvastatin", "simvastatin", "rosuvastatin",
    "metoprolol", "atenolol", "propranolol",
    "lisinopril", "enalapril", "ramipril",
    "amlodipine", "nifedipine", "verapamil",
    "warfarin", "heparin", "apixaban", "rivaroxaban",
    "prednisone", "dexamethasone", "hydrocortisone",
    "sertraline", "fluoxetine", "paroxetine",
    "citalopram", "escitalopram", "venlafaxine",
    "haloperidol", "risperidone", "olanzapine",
    "levodopa", "carbidopa",
    "donepezil", "memantine",
    "albuterol", "salbutamol", "salmeterol",
    "montelukast", "tiotropium",
    "omeprazole", "pantoprazole", "lansoprazole",
    "furosemide", "spironolactone",
    "levothyroxine", "methimazole",
    "hydroxychloroquine", "methotrexate",
    "adalimumab", "infliximab", "etanercept",
    "trastuzumab", "bevacizumab", "pembrolizumab",
    "antibiotic", "antibiotics",
    "antidepressant", "antidepressants",
    "antihypertensive", "antihypertensives",
    "anticoagulant", "anticoagulants",
    "antiviral", "antivirals",
    "antifungal", "antifungals",
    "beta-blocker", "beta blockers",
    "ACE inhibitor", "ACE inhibitors",
    "statin", "statins",
    "diuretic", "diuretics",
    "corticosteroid", "corticosteroids",
    "opioid", "opioids",
]

MEDICAL_SPECIALTIES = [
    "cardiology", "cardiologist",
    "neurology", "neurologist",
    "oncology", "oncologist",
    "psychiatry", "psychiatrist",
    "endocrinology", "endocrinologist",
    "pulmonology", "pulmonologist",
    "gastroenterology", "gastroenterologist",
    "nephrology", "nephrologist",
    "orthopedics", "orthopedist", "orthopaedics",
    "dermatology", "dermatologist",
    "ophthalmology", "ophthalmologist",
    "urology", "urologist",
    "gynecology", "gynecologist",
    "pediatrics", "pediatrician",
    "geriatrics", "geriatrician",
    "hematology", "hematologist",
    "rheumatology", "rheumatologist",
    "infectious disease specialist",
    "emergency medicine", "emergency physician",
    "radiology", "radiologist",
    "pathology", "pathologist",
    "immunology", "immunologist",
    "allergy and immunology",
    "internal medicine", "internist",
    "family medicine", "family physician",
    "surgery", "surgeon",
    "neurosurgery", "neurosurgeon",
]

MEDICAL_VOCAB: dict[str, list[str]] = {
    "DISEASE": DISEASES,
    "SYMPTOM": SYMPTOMS,
    "TREATMENT": TREATMENTS,
    "MEDICATION": MEDICATIONS,
    "MEDICAL_SPECIALTY": MEDICAL_SPECIALTIES,
}

KEEP_SPACY_LABELS = {"PERSON", "ORG", "GPE", "DATE"}
ALL_LABELS = set(MEDICAL_VOCAB.keys()) | KEEP_SPACY_LABELS


# Convert a list of medical terms into spaCy EntityRuler pattern dicts.
def make_patterns(label: str, terms: list[str]) -> list[dict]:
    """Convert term list into spaCy EntityRuler pattern dicts."""
    patterns = []
    for term in terms:
        tokens = term.split()
        token_patterns = [{"LOWER": t.lower()} for t in tokens]
        patterns.append({"label": label, "pattern": token_patterns})
    return patterns


# Add domain-specific medical entity patterns to the spaCy NLP pipeline.
def build_medical_ruler(nlp: Language) -> Language:
    """Add medical term rules to spaCy (runs before the built-in NER)."""
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
    all_patterns: list[dict] = []
    for label, terms in MEDICAL_VOCAB.items():
        all_patterns.extend(make_patterns(label, terms))
    ruler.add_patterns(all_patterns)
    logger.info("EntityRuler loaded with %d patterns", len(all_patterns))
    return nlp


# Run NER on one article and return de-duplicated entity records.
def process_text(nlp: Language, text: str, url: str, title: str) -> list[dict]:
    """Find all medical entities in one article. Returns a list of entity records."""
    records: list[dict] = []
    seen: set[tuple[str, str]] = set()
    max_len = 900_000
    chunks = [text[i : i + max_len] for i in range(0, len(text), max_len)]

    for chunk in chunks:
        try:
            doc = nlp(chunk)
        except Exception as exc:
            logger.warning("spaCy error on chunk from %s: %s", url, exc)
            continue

        for ent in doc.ents:
            if ent.label_ not in ALL_LABELS:
                continue

            ent_text = ent.text.strip()
            if len(ent_text) < 2 or ent_text.isdigit():
                continue

            key = (ent_text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)

            try:
                context = ent.sent.text.strip().replace("\n", " ")
            except Exception:
                context = ""

            records.append({
                "entity": ent_text,
                "label": ent.label_,
                "context": context[:300],
                "source_url": url,
                "source_title": title,
            })

    return records


# Run the full NER pipeline over all crawled articles and write results to CSV.
def run_ner(
    input_file: str = "data/crawler_output.jsonl",
    output_file: str = "data/extracted_knowledge.csv",
    model: str = "en_core_web_trf",
) -> int:
    """Run NER on all articles and save results. Returns total entities found."""
    in_path = Path(input_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    logger.info("Loading spaCy model: %s", model)
    try:
        nlp = spacy.load(model)
    except OSError:
        logger.error(
            "Model '%s' not found. Run: python -m spacy download %s", model, model
        )
        raise

    nlp = build_medical_ruler(nlp)
    logger.info("Pipeline components: %s", nlp.pipe_names)

    fieldnames = ["entity", "label", "context", "source_url", "source_title"]
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

                logger.info("Processing [%d] %s ...", i + 1, title)

                records = process_text(nlp, text, url, title)
                writer.writerows(records)
                total += len(records)
                logger.info("  → %d entities extracted", len(records))

    logger.info("NER complete — %d entity records written to %s", total, output_file)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="NER Pipeline – Lab 1")
    parser.add_argument(
        "--input", default="data/crawler_output.jsonl",
        help="JSONL file produced by the crawler",
    )
    parser.add_argument(
        "--output", default="data/extracted_knowledge.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--model", default="en_core_web_trf",
        help="spaCy model name (default: en_core_web_trf)",
    )
    args = parser.parse_args()

    try:
        n = run_ner(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
        )
        print(f"\nDone. {n} entity records written to {args.output}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        raise


if __name__ == "__main__":
    main()
