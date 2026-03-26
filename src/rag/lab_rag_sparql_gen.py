"""lab_rag_sparql_gen.py — Medical KG chatbot using SPARQL-generation RAG with Ollama.
Usage: python src/rag/lab_rag_sparql_gen.py [--eval] [--graph PATH] [--model MODEL]
"""

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Optional

import requests
from rdflib import Graph, Namespace, URIRef
try:
    from rdflib.plugins.sparql.exceptions import ResultException
except ImportError:
    ResultException = Exception

# Graph file paths — use the expanded KB if it exists, else fall back to the initial KB

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_BASE_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

_CANDIDATE_GRAPHS = [
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "medical_kb_expanded.nt"),
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "medical_kb_initial.ttl"),
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "ontology.ttl"),  # fallback for demo
]

def _find_default_graph() -> str:
    """Return the first graph file that exists. If none exist, return the first path."""
    for path in _CANDIDATE_GRAPHS:
        if os.path.isfile(path):
            return path
    return _CANDIDATE_GRAPHS[0]

TTL_FILE = _find_default_graph()

# Ollama settings

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"

# How many items to include in the schema summary for the prompt

MAX_PREDICATES = 80   # maximum distinct predicates to list in the schema summary
MAX_CLASSES    = 40   # maximum distinct classes to list
SAMPLE_TRIPLES = 20   # number of example triples to embed in the prompt

# Medical namespace used throughout the KB

MED = Namespace("http://medkg.local/")


# POST a prompt to the Ollama API and return the model's text response.
def ask_local_llm(prompt: str, model: str = MODEL, timeout: int = 300) -> str:
    """Send a prompt to Ollama and return the answer, or a string starting with 'ERROR:'."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,   # wait for the full response before returning
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "ERROR: Cannot connect to Ollama at http://localhost:11434.\n"
            "Make sure Ollama is running:  ollama serve\n"
            "Then pull the model:          ollama pull " + model
        )
    except requests.exceptions.Timeout:
        return f"ERROR: Ollama request timed out after {timeout}s."
    except requests.exceptions.HTTPError as exc:
        return f"ERROR: HTTP error from Ollama: {exc}"
    except (KeyError, json.JSONDecodeError) as exc:
        return f"ERROR: Unexpected response format from Ollama: {exc}"


# Load an RDF file from disk into an rdflib Graph.
def load_graph(path: str) -> Graph:
    """Load an RDF file (.ttl, .nt, .rdf, .owl, .n3) into a graph. Exits on error."""
    if not os.path.isfile(path):
        print(f"[ERROR] Graph file not found: {path}")
        print("  - Check that the MedKG pipeline has been run and the file was produced.")
        print(f"  - Or pass a custom path with:  --graph <path>")
        sys.exit(1)

    ext = os.path.splitext(path)[1].lower()
    fmt_map = {
        ".ttl": "turtle",
        ".turtle": "turtle",
        ".nt":  "nt",
        ".rdf": "xml",
        ".owl": "xml",
        ".xml": "xml",
        ".n3":  "n3",
        ".trig":"trig",
        ".jsonld": "json-ld",
    }
    fmt = fmt_map.get(ext, "turtle")

    print(f"[INFO] Loading graph from: {path}  (format: {fmt})")
    g = Graph()
    try:
        g.parse(path, format=fmt)
    except Exception as exc:
        print(f"[ERROR] Failed to parse graph: {exc}")
        sys.exit(1)

    print(f"[INFO] Graph loaded: {len(g):,} triples")
    return g


# Serialize the graph schema (prefixes, classes, predicates, examples) as a string for the LLM prompt.
def build_schema_summary(g: Graph) -> str:
    """Build a text schema description (prefixes, classes, predicates, examples) for the LLM prompt."""
    lines = []

    lines.append("=== Namespace Prefixes ===")
    lines.append("PREFIX med:  <http://medkg.local/>")
    lines.append("PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
    lines.append("PREFIX owl:  <http://www.w3.org/2002/07/owl#>")
    lines.append("PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>")
    lines.append("PREFIX wdt:  <http://www.wikidata.org/prop/direct/>")
    lines.append("PREFIX wd:   <http://www.wikidata.org/entity/>")
    lines.append("")

    lines.append("=== Medical Domain Context ===")
    lines.append("Base namespace: http://medkg.local/")
    lines.append("")
    lines.append("Classes:")
    lines.append("  med:Disease           - a disease or medical condition (e.g. Diabetes, Cancer)")
    lines.append("  med:Symptom           - a symptom (e.g. fatigue, polyuria, blurred_vision)")
    lines.append("  med:Treatment         - a treatment or therapeutic procedure")
    lines.append("  med:Medication        - a drug or pharmacological agent")
    lines.append("  med:MedicalSpecialty  - a branch of medicine (e.g. neurology, oncology)")
    lines.append("")
    lines.append("Key object properties (predicates):")
    lines.append("  med:hasSymptom    (Disease -> Symptom)       [equiv wdt:P780]")
    lines.append("  med:hasTreatment  (Disease -> Treatment)     [equiv wdt:P924]")
    lines.append("  med:hasMedication (Disease -> Medication)    [equiv wdt:P2176]")
    lines.append("  med:treatedBy     (Disease -> MedicalSpecialty) [equiv wdt:P1995]")
    lines.append("  med:fromSource    (any -> URL provenance)")
    lines.append("  rdfs:label        - human-readable name of any entity")
    lines.append("")

    class_query = """
        SELECT DISTINCT ?cls WHERE {
            ?s rdf:type ?cls .
            FILTER(STRSTARTS(STR(?cls), "http://"))
        }
        LIMIT %d
    """ % MAX_CLASSES
    try:
        classes = sorted(str(row.cls) for row in g.query(class_query))
        if classes:
            lines.append("=== Classes found in graph ===")
            for cls in classes:
                lines.append(f"  <{cls}>")
            lines.append("")
    except Exception:
        pass  # non-critical - schema still useful without this section

    pred_query = """
        SELECT DISTINCT ?p WHERE { ?s ?p ?o . }
        LIMIT %d
    """ % MAX_PREDICATES
    try:
        predicates = sorted(str(row.p) for row in g.query(pred_query))
        if predicates:
            lines.append("=== Predicates found in graph ===")
            for p in predicates:
                lines.append(f"  <{p}>")
            lines.append("")
    except Exception:
        pass

    sample_query = """
        SELECT ?s ?p ?o WHERE {
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), "http://medkg.local/"))
        }
        LIMIT %d
    """ % SAMPLE_TRIPLES
    try:
        rows = list(g.query(sample_query))
        if rows:
            lines.append("=== Sample triples (med: predicates only) ===")
            for row in rows:
                s = _shorten_uri(str(row.s))
                p = _shorten_uri(str(row.p))
                o = _shorten_uri(str(row.o))
                lines.append(f"  {s}  {p}  {o} .")
            lines.append("")
    except Exception:
        pass

    return "\n".join(lines)


def _shorten_uri(uri: str) -> str:
    """Replace known base URIs with their declared prefix abbreviation."""
    replacements = [
        ("http://medkg.local/",                                "med:"),
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#",       "rdf:"),
        ("http://www.w3.org/2000/01/rdf-schema#",             "rdfs:"),
        ("http://www.w3.org/2002/07/owl#",                    "owl:"),
        ("http://www.w3.org/2001/XMLSchema#",                 "xsd:"),
        ("http://www.wikidata.org/prop/direct/",              "wdt:"),
        ("http://www.wikidata.org/entity/",                   "wd:"),
    ]
    for full, short in replacements:
        if uri.startswith(full):
            return short + uri[len(full):]
    return f"<{uri}>"


SPARQL_INSTRUCTIONS = (
    "You generate SPARQL queries. Output ONLY the SPARQL query. No explanation. No markdown.\n\n"
    "Prefixes:\n"
    "  PREFIX med: <http://medkg.local/>\n"
    "  PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
    "  PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n\n"
    "Important: med: entities link to Wikidata via owl:sameAs.\n"
    "Use wdt:P780 (symptom), wdt:P2176 (drug), wdt:P2175 (treatment), wdt:P1995 (specialty).\n\n"
    "EXAMPLE — \"What are the symptoms of Diabetes?\":\n"
    "PREFIX med: <http://medkg.local/>\n"
    "PREFIX owl: <http://www.w3.org/2002/07/owl#>\n"
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    "SELECT ?symptom WHERE {{\n"
    "  ?disease owl:sameAs ?wd .\n"
    "  ?wd wdt:P780 ?symptom .\n"
    "  FILTER(CONTAINS(LCASE(STR(?disease)), \"diabetes\"))\n"
    "}} LIMIT 20\n\n"
    "Question: {question}\n"
    "SPARQL:"
)


# Ask the LLM to translate a natural-language question into a SPARQL query.
def generate_sparql(question: str, schema: str, model: str = MODEL) -> str:
    """
    Ask the LLM to turn a question into a SPARQL query.
    The result may have errors; run_sparql will try to fix them.
    """
    prompt = SPARQL_INSTRUCTIONS.format(question=question)
    return ask_local_llm(prompt, model=model)


_STD_PREFIXES = (
    "PREFIX med:  <http://medkg.local/>\n"
    "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX wdt:  <http://www.wikidata.org/prop/direct/>\n"
    "PREFIX wd:   <http://www.wikidata.org/entity/>\n"
)


# Extract and clean the SPARQL query from an LLM's raw text response.
def _extract_sparql_block(text: str) -> str:
    """Extract a SPARQL query from an LLM response. Adds standard prefixes if missing."""
    import re
    text = text.strip()

    fence = re.search(r"```(?:sparql|sql|SPARQL)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        sparql = fence.group(1).strip()
    else:
        kw = re.search(r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", text, re.IGNORECASE)
        sparql = text[kw.start():].strip() if kw else text

    if "PREFIX" not in sparql.upper():
        sparql = _STD_PREFIXES + sparql

    return sparql


# Execute a SPARQL query on the graph, with LLM self-repair and keyword fallback on failure.
def run_sparql(
    g: Graph,
    sparql_text: str,
    question: str,
    schema: str,
    model: str = MODEL,
    enable_repair: bool = True,
) -> tuple[list[dict], str]:
    """
    Run a SPARQL query on the graph.
    If it fails and enable_repair is True, ask the LLM to fix it and try again.
    If repair also fails, fall back to keyword search.
    Returns (rows, final_query).
    """
    sparql_clean = _extract_sparql_block(sparql_text)

    # Try a hard-coded template first — it is always valid
    template = _template_sparql(question)
    if template:
        try:
            result = g.query(template)
            rows = [
                {str(var): str(val) for var, val in zip(result.vars, row) if val is not None}
                for row in result
            ]
            if rows:
                print(f"[INFO] Template SPARQL succeeded ({len(rows)} results).")
                return rows, template
        except Exception:
            pass  # template failed (shouldn't happen) — continue to LLM query

    try:
        result = g.query(sparql_clean)
        rows = [
            {str(var): str(val) for var, val in zip(result.vars, row)}
            for row in result
        ]
        if rows:
            return rows, sparql_clean
        # LLM query ran but returned nothing — fall through to repair / fallback
        print("[INFO] LLM SPARQL returned no results, trying repair ...")
        raise Exception("empty result set")

    except Exception as first_error:
        if not enable_repair:
            print(f"[SPARQL ERROR] {first_error}")
            return [], sparql_clean

        print(f"[SPARQL ERROR] First attempt failed: {first_error}")
        print("[INFO] Attempting self-repair ...")

        repair_prompt = f"""
The following SPARQL query produced an error when run against a medical
knowledge graph (namespace http://medkg.local/).

Original question: {question}

Faulty query:
{sparql_clean}

Error message:
{first_error}

Schema:
{schema}

Please output a corrected SPARQL query that:
1. Fixes the syntax or predicate error described above.
2. Uses only the prefixes and predicates listed in the schema.
3. Contains ONLY the raw SPARQL - no explanations, no markdown.

Corrected SPARQL:
""".strip()

        repaired_text = ask_local_llm(repair_prompt, model=model)
        repaired_clean = _extract_sparql_block(repaired_text)

        try:
            result = g.query(repaired_clean)
            rows = [
                {str(var): str(val) for var, val in zip(result.vars, row)}
                for row in result
            ]
            print("[INFO] Self-repair succeeded.")
            return rows, repaired_clean

        except Exception as second_error:
            print(f"[SPARQL ERROR] Repair attempt also failed: {second_error}")
            print("[INFO] SPARQL failed — falling back to keyword search ...")
            fallback_rows = keyword_fallback(g, question)
            if fallback_rows:
                print(f"[INFO] Keyword fallback returned {len(fallback_rows)} results.")
            return fallback_rows, repaired_clean


# Match a question to a hand-crafted SPARQL pattern; returns None if no pattern matches.
def _template_sparql(question: str) -> Optional[str]:
    """Generate a SPARQL query from hand-crafted patterns. Returns None if no pattern matches."""
    import re
    q = question.lower().strip().rstrip("?.")

    # Each entry: (regex pattern, Wikidata predicate, result type)
    _PATTERNS = [
        (r"symptoms?\s+of\s+(.+)",                                    "P780",  "symptom"),
        (r"what\s+(?:are\s+)?(?:the\s+)?symptoms?\s+(?:of|for)\s+(.+)", "P780", "symptom"),
        (r"medications?\s+(?:are\s+)?(?:used\s+(?:to\s+)?)?(?:for|to\s+treat|that\s+treat)\s+(.+)",
            "P2176", "medication"),
        (r"what\s+medications?\s+(?:are\s+)?(?:used\s+(?:to\s+)?)?(?:for|to\s+treat|treat)\s+(.+)",
            "P2176", "medication"),
        (r"(?:drug|drugs|medicine|medicines?)\s+(?:for|used\s+(?:for|to\s+treat))\s+(.+)",
            "P2176", "medication"),
        (r"treatments?\s+(?:(?:are\s+)?available\s+for|for|of)\s+(.+)", "P2176", "treatment"),
        (r"what\s+treats?\s+(.+)",                                      "P2176", "treatment"),
        (r"which\s+(?:medical\s+)?specialty\s+(?:handles?|treats?)\s+(.+)", "P1995", "specialty"),
        (r"(?:medical\s+)?specialty\s+(?:for|that\s+handles?)\s+(.+)",      "P1995", "specialty"),
        (r"(?:diseases?|conditions?)\s+(?:related\s+to|that\s+have)\s+(.+)\s+as",
            "P780", "related"),
    ]
    for pattern, wdt_prop, result_var in _PATTERNS:
        m = re.search(pattern, q)
        if m:
            entity = m.group(1).strip()
            entity = re.sub(r"\s+(?:disease|condition|disorder|syndrome)$", "", entity).strip()
            return (
                "PREFIX med:  <http://medkg.local/>\n"
                "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
                "PREFIX wdt:  <http://www.wikidata.org/prop/direct/>\n"
                f"SELECT DISTINCT ?result ?name WHERE {{\n"
                f"  ?disease owl:sameAs ?wd .\n"
                f"  ?wd wdt:{wdt_prop} ?result .\n"
                # Reverse lookup: if med:polyuria owl:sameAs ?result, get the readable name
                f"  OPTIONAL {{ ?medEnt owl:sameAs ?result .\n"
                f"    BIND(REPLACE(STR(?medEnt), \"http://medkg.local/\", \"\") AS ?name) }}\n"
                f'  FILTER(CONTAINS(LCASE(STR(?disease)), "{entity}"))\n'
                f"}} LIMIT 20"
            )
    return None


# Search the graph by keyword when SPARQL generation fails, returning matching triples.
def keyword_fallback(g: Graph, question: str) -> list[dict]:
    """
    Search the graph by keyword when SPARQL fails.
    Finds med: entities whose URI contains a word from the question.
    Returns a list of dicts with keys 'entity', 'relation', 'value'.
    """
    import re

    stopwords = {
        "what", "are", "the", "of", "which", "is", "a", "an", "have", "has",
        "do", "does", "how", "when", "where", "who", "why", "that", "this",
        "used", "treat", "treating", "available", "for", "to", "in", "by",
        "with", "related", "condition", "disease", "symptom", "treatment",
        "medication", "specialty", "medical", "handle", "handles",
    }
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", question)]
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    if not keywords:
        return []

    MED_NS  = "http://medkg.local/"
    WDT_NS  = "http://www.wikidata.org/prop/direct/"
    USEFUL_PREDS = {
        MED_NS + "hasSymptom", MED_NS + "hasTreatment",
        MED_NS + "hasMedication", MED_NS + "treatedBy",
        WDT_NS + "P780", WDT_NS + "P924", WDT_NS + "P2176", WDT_NS + "P1995",
    }
    results = []

    for kw in keywords[:3]:
        matching = [
            s for s in g.subjects()
            if str(s).startswith(MED_NS) and kw in str(s).lower()
        ]
        for subj in matching[:10]:
            for p, o in g.predicate_objects(subj):
                if str(p) in USEFUL_PREDS:
                    results.append({
                        "entity":   str(subj).replace(MED_NS, ""),
                        "relation": str(p).replace(MED_NS, "").replace(WDT_NS, "wdt:"),
                        "value":    str(o).replace(MED_NS, ""),
                    })
                    if len(results) >= 25:
                        return results
        if results:
            return results
        for subj in matching[:5]:
            for p, o in g.predicate_objects(subj):
                if str(p).startswith(MED_NS):
                    results.append({
                        "entity":   str(subj).replace(MED_NS, ""),
                        "relation": str(p).replace(MED_NS, ""),
                        "value":    str(o).replace(MED_NS, ""),
                    })
                    if len(results) >= 25:
                        return results
        if results:
            return results

    return results


def answer_no_rag(question: str, model: str = MODEL) -> str:
    """Answer using the LLM alone (no graph) — baseline for comparison."""
    prompt = (
        "You are a helpful medical knowledge assistant.\n"
        "Answer the following question using your own knowledge. "
        "Be concise (2-4 sentences). Do not look anything up.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return ask_local_llm(prompt, model=model)


# Resolve Wikidata QIDs to English labels via the wbgetentities API.
def resolve_qids(qids: list[str]) -> dict[str, str]:
    """Batch-fetch English labels for a list of QIDs. Returns {QID: label}."""
    if not qids:
        return {}
    labels: dict[str, str] = {}
    for i in range(0, len(qids), 50):
        batch = qids[i : i + 50]
        try:
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": "MedKGBot/1.0 (educational project)"},
                timeout=10,
            )
            resp.raise_for_status()
            for qid, entity in resp.json().get("entities", {}).items():
                en_label = entity.get("labels", {}).get("en", {}).get("value")
                if en_label:
                    labels[qid] = en_label
        except Exception:
            pass  # fail silently — unresolved QIDs are simply omitted
    return labels


# Convert raw graph result rows into a readable English answer string.
def generate_answer(question: str, rows: list[dict], model: str = MODEL) -> str:
    """Turn raw graph results into a clean English answer."""
    import re

    if not rows:
        return "No relevant information was found in the knowledge graph."

    qid_re = re.compile(r"entity/(Q\d+)")

    # Collect all QIDs and resolve them in one batch before processing rows.
    all_qids: list[str] = []
    for row in rows[:20]:
        for v in row.values():
            m = qid_re.search(v)
            if m and m.group(1) not in all_qids:
                all_qids.append(m.group(1))
    qid_labels = resolve_qids(all_qids)

    readable: list[str] = []
    relation_types: list[str] = []

    for row in rows[:20]:
        best: Optional[str] = None

        for key, v in row.items():
            if key not in relation_types:
                relation_types.append(key)

            hp   = re.search(r"obo/HP_(\d+)",     v)
            symp = re.search(r"obo/SYMP_(\d+)",   v)
            med  = re.search(r"medkg\.local/(.+)", v)
            qid  = qid_re.search(v)

            if hp:
                cand: Optional[str] = f"HP:{hp.group(1)}"
            elif symp:
                cand = f"SYMP:{symp.group(1)}"
            elif med:
                cand = med.group(1).replace("_", " ")
            elif qid:
                cand = qid_labels.get(qid.group(1))  # resolved label, or None if lookup failed
            elif not v.startswith("http") and v.strip() and v != "None":
                cand = v.replace("_", " ").strip()
            else:
                cand = None

            if cand and len(cand) > 1 and best is None:
                best = cand

        if best and best not in readable:
            readable.append(best)

    _raw_rel = (relation_types[0] if relation_types else "result").replace("_", " ")
    rel_label = _raw_rel.replace("Qid", "").strip() or "result"

    if readable:
        items_str = ", ".join(v.replace("_", " ") for v in readable[:10])
        _rel_map = {
            "symptom":    "symptoms include",
            "name":       "items include",
            "medication": "medications include",
            "treatment":  "treatments include",
            "specialty":  "medical specialty is",
            "related":    "related conditions include",
            "value":      "related items include",
        }
        q_lower = question.lower()
        connector = "results include"
        if "symptom" in q_lower:
            connector = "symptoms include"
        elif "medication" in q_lower or "drug" in q_lower:
            connector = "medications include"
        elif "treatment" in q_lower:
            connector = "treatments include"
        elif "specialty" in q_lower:
            connector = "medical specialty is"
        elif "related" in q_lower or "condition" in q_lower:
            connector = "related conditions include"
        else:
            for key, phrase in _rel_map.items():
                if key in rel_label.lower():
                    connector = phrase
                    break
        return f"Based on the knowledge graph, {connector}: {items_str}."
    else:
        n = len(rows)
        return (
            f"Based on the knowledge graph, {n} {rel_label}(s) were found "
            f"for this query, but their Wikidata labels could not be resolved."
        )


def _fmt_results(rows: list[dict]) -> str:
    """Format result rows as a readable string."""
    if not rows:
        return "(no results)"

    all_vars: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_vars:
                all_vars.append(k)

    def clean(val: str) -> str:
        return (
            val.replace("http://medkg.local/", "")
               .replace("http://www.wikidata.org/entity/", "wd:")
               .replace("http://purl.obolibrary.org/obo/", "obo:")
        )

    lines = []
    for row in rows:
        parts = [f"{v}={clean(row.get(v, ''))}" for v in all_vars]
        lines.append("  " + ", ".join(parts))
    return "\n".join(lines)


def check_ollama(model: str = MODEL) -> bool:
    """Check that Ollama is running and the model is available. Return True if ready."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        available = [m.get("name", "") for m in tags]
        print(f"[OK] Ollama is running.  Available models: {available or '(none pulled yet)'}")
        if model not in available and not any(model in a for a in available):
            print(f"[WARN] Model '{model}' not found locally.  Pull it with: ollama pull {model}")
            return False
        return True
    except requests.exceptions.ConnectionError:
        print("[FAIL] Cannot connect to Ollama at http://localhost:11434")
        print("  Start it with:   ollama serve")
        return False
    except Exception as exc:
        print(f"[FAIL] Unexpected error checking Ollama: {exc}")
        return False


EVAL_QUESTIONS = [
    "What are the symptoms of Diabetes?",
    "What medications are used to treat Hypertension?",
    "Which diseases have Asthma as a related condition?",
    "What treatments are available for Cancer?",
    "Which medical specialty handles Alzheimer's disease?",
]


# Run 5 benchmark questions and print baseline vs SPARQL-RAG answers side-by-side.
def run_evaluation(g: Graph, schema: str, model: str, enable_repair: bool) -> None:
    """Run 5 test questions and show baseline vs RAG answers side by side."""
    print("\nEVALUATION - 5 Medical Questions (Baseline vs SPARQL-RAG)")
    print(f"Model  : {model}")
    print(f"Graph  : {len(g):,} triples")
    print(f"Repair : {'enabled' if enable_repair else 'disabled'}")
    print()

    for idx, question in enumerate(EVAL_QUESTIONS, 1):
        print("-" * 72)
        print(f"[Q{idx}] {question}")

        print("  >> Baseline (no RAG) - querying LLM parametric memory ...")
        t0 = time.time()
        baseline_answer = answer_no_rag(question, model=model)
        baseline_time = time.time() - t0
        print(f"  BASELINE ({baseline_time:.1f}s):")
        for line in textwrap.wrap(baseline_answer, width=68):
            print(f"    {line}")

        print()

        print("  >> SPARQL-RAG - querying knowledge graph and generating answer ...")
        t1 = time.time()
        sparql_query = generate_sparql(question, schema, model=model)
        sparql_clean = _extract_sparql_block(sparql_query)
        rows, final_sparql = run_sparql(
            g, sparql_clean, question, schema,
            model=model, enable_repair=enable_repair,
        )
        rag_answer = generate_answer(question, rows, model=model)
        rag_time = time.time() - t1

        print(f"  SPARQL-RAG ({rag_time:.1f}s):")
        for line in textwrap.wrap(rag_answer, width=68):
            print(f"    {line}")

        print()

    print("\nEvaluation complete.")


# Run an interactive question-answering loop against the knowledge graph.
def interactive_loop(g: Graph, schema: str, model: str, enable_repair: bool) -> None:
    """Run an interactive chat loop. Type 'quit' to stop."""
    print("\nMedical Knowledge Graph - SPARQL-RAG Chatbot")
    print(f"Model : {model}")
    print(f"Graph : {len(g):,} triples loaded")
    print(f"Repair: {'enabled' if enable_repair else 'disabled'}")
    print()
    print("Ask any medical question about diseases, symptoms, treatments,")
    print("medications, or medical specialties.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        print("-" * 72)

        print("[1/2] Querying LLM without graph (baseline) ...")
        baseline = answer_no_rag(question, model=model)
        print(f"\nBaseline answer:\n")
        for line in textwrap.wrap(baseline, width=70):
            print(f"  {line}")

        print()

        print("[2/2] Querying knowledge graph ...")
        sparql_raw = generate_sparql(question, schema, model=model)
        sparql_clean = _extract_sparql_block(sparql_raw)

        rows, final_sparql = run_sparql(
            g, sparql_clean, question, schema,
            model=model, enable_repair=enable_repair,
        )

        rag_answer = generate_answer(question, rows, model=model)
        print(f"\nRAG answer (from knowledge graph):\n")
        for line in textwrap.wrap(rag_answer, width=70):
            print(f"  {line}")

        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Medical Knowledge Graph chatbot using SPARQL-generation RAG "
            "with a local LLM via Ollama."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python src/rag/lab_rag_sparql_gen.py
              python src/rag/lab_rag_sparql_gen.py --eval
              python src/rag/lab_rag_sparql_gen.py --model deepseek-r1:1.5b
              python src/rag/lab_rag_sparql_gen.py --graph kg_artifacts/medical_kb_expanded.nt
              python src/rag/lab_rag_sparql_gen.py --ollama-check
        """),
    )
    parser.add_argument(
        "--graph",
        default=TTL_FILE,
        metavar="PATH",
        help=f"Path to the RDF knowledge graph file. Default: {TTL_FILE}",
    )
    parser.add_argument(
        "--model",
        default=MODEL,
        metavar="MODEL",
        help=f"Ollama model tag to use. Default: {MODEL}",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the predefined evaluation suite of 5 medical questions and exit.",
    )
    parser.add_argument(
        "--ollama-check",
        action="store_true",
        dest="ollama_check",
        help="Verify that Ollama is running and the model is available, then exit.",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        dest="no_repair",
        help="Disable the SPARQL self-repair loop (useful for ablation studies).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    enable_repair = not args.no_repair

    if args.ollama_check:
        ok = check_ollama(args.model)
        sys.exit(0 if ok else 1)

    g = load_graph(args.graph)

    print("[INFO] Building schema summary ...")
    schema = build_schema_summary(g)
    print(f"[INFO] Schema summary: {len(schema)} characters")

    if args.eval:
        run_evaluation(g, schema, model=args.model, enable_repair=enable_repair)
        sys.exit(0)

    interactive_loop(g, schema, model=args.model, enable_repair=enable_repair)


if __name__ == "__main__":
    main()
