"""
Lab Session 6 - RAG with RDF/SPARQL and a Local LLM
=====================================================
Medical Knowledge Graph chatbot using:
  - rdflib for RDF loading and SPARQL execution
  - Ollama (local LLM) for SPARQL generation from natural language
  - Self-repair loop for failed SPARQL queries

The knowledge graph is the Medical KB built from Wikipedia data via the
MedKG pipeline. It covers diseases, symptoms, treatments, medications, and
medical specialties under the namespace http://medkg.local/

Usage:
    python src/rag/lab_rag_sparql_gen.py                       # interactive CLI
    python src/rag/lab_rag_sparql_gen.py --eval                # run evaluation on 5 questions
    python src/rag/lab_rag_sparql_gen.py --graph kg_artifacts/medical_kb_expanded.nt
    python src/rag/lab_rag_sparql_gen.py --model deepseek-r1:1.5b
    python src/rag/lab_rag_sparql_gen.py --ollama-check        # verify Ollama is reachable
    python src/rag/lab_rag_sparql_gen.py --no-repair           # disable the self-repair loop
"""

# ==============================================================================
# Section 0 - Imports and Configuration
# ==============================================================================

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

# ---------------------------------------------------------------------------
# Paths - prefer the expanded graph produced by the KB pipeline;
# fall back to the initial Turtle file if the expanded N-Triples are absent.
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_BASE_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

_CANDIDATE_GRAPHS = [
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "medical_kb_expanded.nt"),
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "medical_kb_initial.ttl"),
    os.path.join(_PROJECT_ROOT, "kg_artifacts", "ontology.ttl"),  # fallback for demo
]

def _find_default_graph() -> str:
    """Return the first existing candidate graph path, or the first candidate
    (so the error message is meaningful when none exist)."""
    for path in _CANDIDATE_GRAPHS:
        if os.path.isfile(path):
            return path
    return _CANDIDATE_GRAPHS[0]

TTL_FILE = _find_default_graph()

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:1.5b"

# ---------------------------------------------------------------------------
# Schema-summary tuning parameters
# ---------------------------------------------------------------------------

MAX_PREDICATES = 80   # maximum distinct predicates to list in the schema summary
MAX_CLASSES    = 40   # maximum distinct classes to list
SAMPLE_TRIPLES = 20   # number of example triples to embed in the prompt

# ---------------------------------------------------------------------------
# Medical namespace (matches ontology)
# ---------------------------------------------------------------------------

MED = Namespace("http://medkg.local/")

# ==============================================================================
# Section 1 - ask_local_llm
# ==============================================================================

def ask_local_llm(prompt: str, model: str = MODEL, timeout: int = 300) -> str:
    """
    Send *prompt* to an Ollama model and return the full response text.

    Parameters
    ----------
    prompt  : the complete prompt string
    model   : Ollama model tag (default: gemma:2b)
    timeout : request timeout in seconds

    Returns
    -------
    The model's response as a plain string, or an error message that starts
    with "ERROR:" so callers can detect failure.
    """
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


# ==============================================================================
# Section 2 - load_graph
# ==============================================================================

def load_graph(path: str) -> Graph:
    """
    Load an RDF graph from *path* using rdflib.

    Supports Turtle (.ttl), N-Triples (.nt), RDF/XML (.rdf/.owl), and
    Notation3 (.n3).  The format is inferred from the file extension.

    Parameters
    ----------
    path : absolute or relative path to the RDF file

    Returns
    -------
    A parsed rdflib.Graph

    Raises
    ------
    SystemExit if the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(path):
        print(f"[ERROR] Graph file not found: {path}")
        print("  - Check that the MedKG pipeline has been run and the file was produced.")
        print(f"  - Or pass a custom path with:  --graph <path>")
        sys.exit(1)

    # Infer rdflib format string from extension
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


# ==============================================================================
# Section 3 - build_schema_summary
# ==============================================================================

def build_schema_summary(g: Graph) -> str:
    """
    Build a concise textual summary of the knowledge graph schema to include
    in the SPARQL-generation prompt.

    The summary covers:
      1. Fixed namespace prefixes (med:, rdf:, rdfs:, owl:, xsd:, wdt:, wd:)
      2. Up to MAX_CLASSES distinct rdf:type values (the classes used)
      3. Up to MAX_PREDICATES distinct predicates used in the graph
      4. SAMPLE_TRIPLES example triples rendered in Turtle-like notation

    The medical-specific section explicitly lists the key predicates defined in
    the MedKG ontology so the LLM can use them directly.

    Parameters
    ----------
    g : the loaded rdflib.Graph

    Returns
    -------
    A multi-line string describing the graph schema.
    """
    lines = []

    # ------------------------------------------------------------------
    # 3.1  Fixed prefixes (always include these)
    # ------------------------------------------------------------------
    lines.append("=== Namespace Prefixes ===")
    lines.append("PREFIX med:  <http://medkg.local/>")
    lines.append("PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
    lines.append("PREFIX owl:  <http://www.w3.org/2002/07/owl#>")
    lines.append("PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>")
    lines.append("PREFIX wdt:  <http://www.wikidata.org/prop/direct/>")
    lines.append("PREFIX wd:   <http://www.wikidata.org/entity/>")
    lines.append("")

    # ------------------------------------------------------------------
    # 3.2  Medical-domain context (hard-coded from MedKG ontology)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3.3  Classes observed in the graph (inferred from rdf:type triples)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3.4  Distinct predicates observed in the graph
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3.5  Sample triples (to give the LLM concrete examples of URIs used)
    # ------------------------------------------------------------------
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


# ==============================================================================
# Section 4 - generate_sparql
# ==============================================================================

# ---------------------------------------------------------------------------
# System instruction injected before every SPARQL-generation call.
# Kept intentionally short so small models (1-2 B) can follow it reliably.
# ---------------------------------------------------------------------------
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


def generate_sparql(question: str, schema: str, model: str = MODEL) -> str:
    """
    Ask the local LLM to convert *question* into a SPARQL query.

    Parameters
    ----------
    question : natural-language question from the user
    schema   : unused (kept for API compatibility - prompt is now self-contained)
    model    : Ollama model tag

    Returns
    -------
    A SPARQL query string (may still contain syntax errors - that is handled
    by run_sparql with the self-repair loop).
    """
    prompt = SPARQL_INSTRUCTIONS.format(question=question)
    return ask_local_llm(prompt, model=model)


# ==============================================================================
# Section 5 - run_sparql  (with self-repair loop)
# ==============================================================================

_STD_PREFIXES = (
    "PREFIX med:  <http://medkg.local/>\n"
    "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX wdt:  <http://www.wikidata.org/prop/direct/>\n"
    "PREFIX wd:   <http://www.wikidata.org/entity/>\n"
)


def _extract_sparql_block(text: str) -> str:
    """
    Extract a SPARQL query from an LLM response that may contain:
    - Raw SPARQL (ideal)
    - Markdown code fences: ```sparql ... ```
    - Prose prefix: "Here is the query: SELECT ..."

    Also auto-prepends standard prefixes if the query is missing them,
    so small models that forget PREFIX declarations still work.
    """
    import re
    text = text.strip()

    # 1. Try to extract from ```sparql...``` or ```...``` block
    fence = re.search(r"```(?:sparql|sql|SPARQL)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        sparql = fence.group(1).strip()
    else:
        # 2. Find first occurrence of PREFIX / SELECT / ASK / CONSTRUCT / DESCRIBE
        kw = re.search(r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", text, re.IGNORECASE)
        sparql = text[kw.start():].strip() if kw else text

    # 3. Auto-inject standard prefixes if the query doesn't declare any
    if "PREFIX" not in sparql.upper():
        sparql = _STD_PREFIXES + sparql

    return sparql


def run_sparql(
    g: Graph,
    sparql_text: str,
    question: str,
    schema: str,
    model: str = MODEL,
    enable_repair: bool = True,
) -> tuple[list[dict], str]:
    """
    Execute *sparql_text* against *g*.  If execution fails and *enable_repair*
    is True, attempt one self-repair cycle: feed the error back to the LLM and
    re-execute the corrected query.

    Parameters
    ----------
    g             : loaded rdflib.Graph
    sparql_text   : SPARQL query string (may be malformed)
    question      : original user question (used in repair prompt)
    schema        : schema summary (used in repair prompt)
    model         : Ollama model tag
    enable_repair : whether to attempt self-repair on failure

    Returns
    -------
    (rows, final_sparql) where rows is a list of result dicts and
    final_sparql is the query that was ultimately executed (possibly repaired).
    """
    sparql_clean = _extract_sparql_block(sparql_text)

    # --- Template-first: try hand-crafted SPARQL before the LLM query ---
    # The template uses the correct owl:sameAs+wdt: join and is always valid.
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

    # --- First attempt (LLM-generated) ---
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

        # --- Self-repair prompt ---
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

        # --- Second attempt with repaired query ---
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
            # --- Keyword fallback: search graph by entity name ---
            print("[INFO] SPARQL failed — falling back to keyword search ...")
            fallback_rows = keyword_fallback(g, question)
            if fallback_rows:
                print(f"[INFO] Keyword fallback returned {len(fallback_rows)} results.")
            return fallback_rows, repaired_clean


# ==============================================================================
# Section 5b - _template_sparql  (pattern-based SPARQL without LLM)
# ==============================================================================

def _template_sparql(question: str) -> Optional[str]:
    """
    Generate a valid SPARQL query from hand-crafted patterns, bypassing the LLM.
    Handles the most common question forms reliably.

    Returns None if no pattern matches.
    """
    import re
    q = question.lower().strip().rstrip("?.")

    # Maps question pattern → (wdt predicate, result variable name)
    # Uses wdt: predicates which are bulk-loaded from Wikidata (thousands of triples)
    # and linked via owl:sameAs: med:Disease owl:sameAs wd:Qxxx wdt:Pyyy wd:Qresult
    _PATTERNS = [
        # Symptoms
        (r"symptoms?\s+of\s+(.+)",                                    "P780",  "symptom"),
        (r"what\s+(?:are\s+)?(?:the\s+)?symptoms?\s+(?:of|for)\s+(.+)", "P780", "symptom"),
        # Medications / drugs
        (r"medications?\s+(?:are\s+)?(?:used\s+(?:to\s+)?)?(?:for|to\s+treat|that\s+treat)\s+(.+)",
            "P2176", "medication"),
        (r"what\s+medications?\s+(?:are\s+)?(?:used\s+(?:to\s+)?)?(?:for|to\s+treat|treat)\s+(.+)",
            "P2176", "medication"),
        (r"(?:drug|drugs|medicine|medicines?)\s+(?:for|used\s+(?:for|to\s+treat))\s+(.+)",
            "P2176", "medication"),
        # Treatments (P2176 = drug used for treatment: disease → drug, direct)
        (r"treatments?\s+(?:(?:are\s+)?available\s+for|for|of)\s+(.+)", "P2176", "treatment"),
        (r"what\s+treats?\s+(.+)",                                      "P2176", "treatment"),
        # Medical specialty
        (r"which\s+(?:medical\s+)?specialty\s+(?:handles?|treats?)\s+(.+)", "P1995", "specialty"),
        (r"(?:medical\s+)?specialty\s+(?:for|that\s+handles?)\s+(.+)",      "P1995", "specialty"),
        # Related conditions
        (r"(?:diseases?|conditions?)\s+(?:related\s+to|that\s+have)\s+(.+)\s+as",
            "P780", "related"),
    ]
    for pattern, wdt_prop, result_var in _PATTERNS:
        m = re.search(pattern, q)
        if m:
            entity = m.group(1).strip()
            entity = re.sub(r"\s+(?:disease|condition|disorder|syndrome)$", "", entity).strip()
            # Join med: entity → owl:sameAs → Wikidata entity → wdt:Pxxx → result
            return (
                "PREFIX med:  <http://medkg.local/>\n"
                "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
                "PREFIX wdt:  <http://www.wikidata.org/prop/direct/>\n"
                f"SELECT DISTINCT ?result ?name WHERE {{\n"
                f"  ?disease owl:sameAs ?wd .\n"
                f"  ?wd wdt:{wdt_prop} ?result .\n"
                # Reverse sameAs: med:polyuria owl:sameAs wd:Q1124286 → readable slug
                f"  OPTIONAL {{ ?medEnt owl:sameAs ?result .\n"
                f"    BIND(REPLACE(STR(?medEnt), \"http://medkg.local/\", \"\") AS ?name) }}\n"
                f'  FILTER(CONTAINS(LCASE(STR(?disease)), "{entity}"))\n'
                f"}} LIMIT 20"
            )
    return None


# ==============================================================================
# Section 5c - keyword_fallback
# ==============================================================================

def keyword_fallback(g: Graph, question: str) -> list[dict]:
    """
    Simple keyword search on the graph used when SPARQL generation fails.

    Extracts meaningful words from *question*, finds med: entities whose URI
    contains one of those words, and returns their med: predicate/value pairs.

    Parameters
    ----------
    g        : loaded rdflib.Graph
    question : natural-language question

    Returns
    -------
    List of dicts with keys 'entity', 'relation', 'value'.
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
    # Medical predicates we actually care about for answers
    USEFUL_PREDS = {
        MED_NS + "hasSymptom", MED_NS + "hasTreatment",
        MED_NS + "hasMedication", MED_NS + "treatedBy",
        WDT_NS + "P780",  # symptom
        WDT_NS + "P924",  # health specialty
        WDT_NS + "P2176", # drug used for treatment
        WDT_NS + "P1995", # health specialty
    }
    results = []

    for kw in keywords[:3]:
        # Find med: subjects whose URI contains the keyword
        matching = [
            s for s in g.subjects()
            if str(s).startswith(MED_NS) and kw in str(s).lower()
        ]
        # First pass: only useful medical predicates
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
        # Second pass: any med: predicate (broader fallback)
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


# ==============================================================================
# Section 6 - answer_no_rag  (baseline: LLM answers from parametric knowledge)
# ==============================================================================

def answer_no_rag(question: str, model: str = MODEL) -> str:
    """
    Answer *question* using the LLM's own parametric knowledge, without any
    access to the knowledge graph.  This is the baseline for comparison.

    Parameters
    ----------
    question : natural-language question
    model    : Ollama model tag

    Returns
    -------
    LLM response string.
    """
    prompt = (
        "You are a helpful medical knowledge assistant.\n"
        "Answer the following question using your own knowledge. "
        "Be concise (2-4 sentences). Do not look anything up.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return ask_local_llm(prompt, model=model)


# ==============================================================================
# Section 6b - generate_answer  (RAG synthesis: graph results → clean text)
# ==============================================================================

def generate_answer(question: str, rows: list[dict], model: str = MODEL) -> str:
    """
    Synthesize a clean human-readable answer from raw SPARQL/fallback rows.

    Extracts the most meaningful values from each row, formats them as a
    bullet list, then asks the LLM to write one clear sentence.

    Parameters
    ----------
    question : original natural-language question
    rows     : result rows from run_sparql / keyword_fallback
    model    : Ollama model tag

    Returns
    -------
    A plain-English answer string.
    """
    import re

    if not rows:
        return "No relevant information was found in the knowledge graph."

    # Classify each row's values: collect readable labels and count bare QIDs.
    # Apply URI regex BEFORE underscore replacement so HP_0000103 stays intact.
    readable: list[str] = []   # human-readable names (med: slugs, HP/SYMP codes)
    qid_count = 0              # Wikidata QIDs with no readable label in the graph
    relation_types: list[str] = []

    for row in rows[:20]:
        best: Optional[str] = None  # best readable label for this row
        has_qid = False

        for key, v in row.items():
            if key not in relation_types:
                relation_types.append(key)

            hp   = re.search(r"obo/HP_(\d+)",      v)
            symp = re.search(r"obo/SYMP_(\d+)",    v)
            med  = re.search(r"medkg\.local/(.+)",  v)
            qid  = re.search(r"entity/Q\d+",        v)

            if hp:
                cand: Optional[str] = f"HP:{hp.group(1)}"
            elif symp:
                cand = f"SYMP:{symp.group(1)}"
            elif med:
                cand = med.group(1).replace("_", " ")
            elif qid:
                has_qid = True
                cand = None
            elif not v.startswith("http") and v.strip() and v != "None":
                cand = v.replace("_", " ").strip()
            else:
                cand = None

            # Keep the first non-QID label found (HP preferred over nothing)
            if cand and len(cand) > 1 and best is None:
                best = cand

        if best and best not in readable:
            readable.append(best)
        elif has_qid and best is None:
            qid_count += 1

    # ----------------------------------------------------------------
    # Build the final answer — use LLM only with readable content.
    # For bare QIDs we produce a deterministic count-based sentence.
    # ----------------------------------------------------------------
    # Use the most meaningful relation name, stripping internal suffixes
    _raw_rel = (relation_types[0] if relation_types else "result").replace("_", " ")
    rel_label = _raw_rel.replace("Qid", "").strip() or "result"

    if readable:
        # Readable labels available — build a deterministic sentence (no LLM, no hallucination)
        items_str = ", ".join(v.replace("_", " ") for v in readable[:10])
        # Map relation name → natural-language connector
        # Also check the question itself for intent keywords
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
        # Prefer question-based detection over variable-name detection
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
        # Only Wikidata QIDs — deterministic count message
        n = qid_count or len(rows)
        return (
            f"Based on the knowledge graph, {n} {rel_label}(s) were found "
            f"for this query. They are stored as Wikidata entity identifiers "
            f"(no human-readable labels in the local graph)."
        )


# ==============================================================================
# Section 7 - Formatting helpers
# ==============================================================================

def _fmt_results(rows: list[dict]) -> str:
    """Pretty-print SPARQL or keyword-fallback result rows as a readable string."""
    if not rows:
        return "(no results)"

    # Collect all variable names, preserving order of first occurrence
    all_vars: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_vars:
                all_vars.append(k)

    # Shorten known URIs for readability
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


def _print_separator(char: str = "-", width: int = 72) -> None:
    print(char * width)


def _print_banner(title: str) -> None:
    _print_separator("=")
    print(f"  {title}")
    _print_separator("=")


# ==============================================================================
# Section 8 - Ollama health check
# ==============================================================================

def check_ollama(model: str = MODEL) -> bool:
    """
    Verify that Ollama is reachable and that *model* is available.

    Returns True if everything is ready, False otherwise.
    """
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


# ==============================================================================
# Section 9 - Evaluation suite (--eval flag)
# ==============================================================================

EVAL_QUESTIONS = [
    "What are the symptoms of Diabetes?",
    "What medications are used to treat Hypertension?",
    "Which diseases have Asthma as a related condition?",
    "What treatments are available for Cancer?",
    "Which medical specialty handles Alzheimer's disease?",
]


def run_evaluation(g: Graph, schema: str, model: str, enable_repair: bool) -> None:
    """
    Run the predefined evaluation suite and print a side-by-side comparison of
    baseline (no-RAG) vs SPARQL-generation RAG for each question.

    Parameters
    ----------
    g             : loaded rdflib.Graph
    schema        : schema summary string
    model         : Ollama model tag
    enable_repair : whether self-repair is enabled
    """
    _print_banner("EVALUATION - 5 Medical Questions (Baseline vs SPARQL-RAG)")
    print(f"Model  : {model}")
    print(f"Graph  : {len(g):,} triples")
    print(f"Repair : {'enabled' if enable_repair else 'disabled'}")
    print()

    for idx, question in enumerate(EVAL_QUESTIONS, 1):
        _print_separator()
        print(f"[Q{idx}] {question}")
        _print_separator("-")

        # --- Baseline (no RAG) ---
        print("  >> Baseline (no RAG) - querying LLM parametric memory ...")
        t0 = time.time()
        baseline_answer = answer_no_rag(question, model=model)
        baseline_time = time.time() - t0
        print(f"  BASELINE ({baseline_time:.1f}s):")
        for line in textwrap.wrap(baseline_answer, width=68):
            print(f"    {line}")

        print()

        # --- SPARQL-RAG ---
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

    _print_separator("=")
    print("Evaluation complete.")
    _print_separator("=")


# ==============================================================================
# Section 10 - Interactive CLI loop
# ==============================================================================

def interactive_loop(g: Graph, schema: str, model: str, enable_repair: bool) -> None:
    """
    Run an interactive question-answering loop in the terminal.
    The user types a natural-language medical question; the system returns
    both a baseline LLM answer and a SPARQL-RAG answer for comparison.

    Type 'quit', 'exit', or Ctrl-C to stop.
    """
    _print_banner("Medical Knowledge Graph - SPARQL-RAG Chatbot")
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

        _print_separator()

        # 1. Baseline
        print("[1/2] Querying LLM without graph (baseline) ...")
        baseline = answer_no_rag(question, model=model)
        print(f"\nBaseline answer:\n")
        for line in textwrap.wrap(baseline, width=70):
            print(f"  {line}")

        print()

        # 2. SPARQL-RAG
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

        _print_separator()
        print()


# ==============================================================================
# Section 11 - Argument parsing and main entry point
# ==============================================================================

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

    # ------------------------------------------------------------------
    # --ollama-check: just verify connectivity and exit
    # ------------------------------------------------------------------
    if args.ollama_check:
        ok = check_ollama(args.model)
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    # Load the knowledge graph
    # ------------------------------------------------------------------
    g = load_graph(args.graph)

    # ------------------------------------------------------------------
    # Build schema summary (done once, reused for all queries)
    # ------------------------------------------------------------------
    print("[INFO] Building schema summary ...")
    schema = build_schema_summary(g)
    print(f"[INFO] Schema summary: {len(schema)} characters")

    # ------------------------------------------------------------------
    # --eval: run the evaluation suite and exit
    # ------------------------------------------------------------------
    if args.eval:
        run_evaluation(g, schema, model=args.model, enable_repair=enable_repair)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Default: interactive loop
    # ------------------------------------------------------------------
    interactive_loop(g, schema, model=args.model, enable_repair=enable_repair)


if __name__ == "__main__":
    main()
