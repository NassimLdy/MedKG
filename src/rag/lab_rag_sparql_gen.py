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
# ---------------------------------------------------------------------------
SPARQL_INSTRUCTIONS = """
You are a SPARQL expert working with a medical knowledge graph.
The graph is stored under the namespace http://medkg.local/ and describes
diseases, their symptoms, treatments, medications, and medical specialties.

Your task: convert the user's natural-language question into a single, valid
SPARQL SELECT query that answers it using the schema below.

Rules:
1. Output ONLY the raw SPARQL query - no prose, no markdown fences, no explanation.
2. Always declare all PREFIX lines at the top of the query.
3. Use only the predicates and classes that appear in the schema.
4. Prefer med:hasSymptom, med:hasTreatment, med:hasMedication, med:treatedBy
   for medical relationships.
5. Use FILTER(CONTAINS(LCASE(STR(?x)), "keyword")) when matching names as
   string patterns (entity URIs are usually snake_cased, e.g. med:Diabetes).
6. Select meaningful variable names and add LIMIT 50 to avoid huge results.
7. If the question cannot be answered from this graph, output:
       SELECT ?nothing WHERE {{ FILTER(false) }}

Schema:
{schema}
""".strip()


def generate_sparql(question: str, schema: str, model: str = MODEL) -> str:
    """
    Ask the local LLM to convert *question* into a SPARQL query.

    Parameters
    ----------
    question : natural-language question from the user
    schema   : schema summary produced by build_schema_summary()
    model    : Ollama model tag

    Returns
    -------
    A SPARQL query string (may still contain syntax errors - that is handled
    by run_sparql with the self-repair loop).
    """
    instructions = SPARQL_INSTRUCTIONS.format(schema=schema)
    prompt = f"{instructions}\n\nQuestion: {question}\n\nSPARQL:"
    return ask_local_llm(prompt, model=model)


# ==============================================================================
# Section 5 - run_sparql  (with self-repair loop)
# ==============================================================================

def _extract_sparql_block(text: str) -> str:
    """
    Extract a SPARQL query from an LLM response that may contain:
    - Raw SPARQL (ideal)
    - Markdown code fences: ```sparql ... ```
    - Prose prefix: "Here is the query: SELECT ..."
    """
    import re
    text = text.strip()

    # 1. Try to extract from ```sparql...``` or ```...``` block
    fence = re.search(r"```(?:sparql|sql|SPARQL)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # 2. Find first occurrence of SELECT / ASK / CONSTRUCT / DESCRIBE
    kw = re.search(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", text, re.IGNORECASE)
    if kw:
        return text[kw.start():].strip()

    return text


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

    # --- First attempt ---
    try:
        result = g.query(sparql_clean)
        rows = [
            {str(var): str(val) for var, val in zip(result.vars, row)}
            for row in result
        ]
        return rows, sparql_clean

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
            return [], repaired_clean


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
# Section 7 - Formatting helpers
# ==============================================================================

def _fmt_results(rows: list[dict]) -> str:
    """Pretty-print SPARQL result rows as a readable string."""
    if not rows:
        return "(no results)"
    # Collect all variable names, preserving order of first occurrence
    all_vars: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_vars:
                all_vars.append(k)

    # Shorten med: URIs for readability
    def clean(val: str) -> str:
        if val.startswith("http://medkg.local/"):
            return val.replace("http://medkg.local/", "")
        return val

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
        print("  >> SPARQL-RAG - generating SPARQL and querying knowledge graph ...")
        t1 = time.time()
        sparql_query = generate_sparql(question, schema, model=model)
        sparql_clean = _extract_sparql_block(sparql_query)
        rows, final_sparql = run_sparql(
            g, sparql_clean, question, schema,
            model=model, enable_repair=enable_repair,
        )
        rag_time = time.time() - t1

        print(f"  SPARQL-RAG ({rag_time:.1f}s):")
        print(f"  Generated query:\n    {sparql_clean[:200].replace(chr(10), ' ')}")
        print(f"  Results from KB:")
        for line in _fmt_results(rows).splitlines():
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
        print("[2/2] Generating SPARQL and querying knowledge graph ...")
        sparql_raw = generate_sparql(question, schema, model=model)
        sparql_clean = _extract_sparql_block(sparql_raw)

        print(f"\nGenerated SPARQL:\n  {sparql_clean[:300].replace(chr(10), ' ')}")

        rows, final_sparql = run_sparql(
            g, sparql_clean, question, schema,
            model=model, enable_repair=enable_repair,
        )

        print(f"\nSPARQL-RAG results from knowledge graph:\n")
        print(_fmt_results(rows))

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
