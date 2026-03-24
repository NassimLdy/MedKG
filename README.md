# MedKG — Medical Knowledge Graph Pipeline

MedKG is an end-to-end pipeline that builds a Medical Knowledge Graph from
Wikipedia, enriches it via Wikidata alignment, reasons over it with SWRL rules,
trains Knowledge Graph Embedding (KGE) models, and exposes it through a
Retrieval-Augmented Generation (RAG) interface backed by a local LLM.

The project integrates four lab sessions covering web crawling, named entity
recognition, RDF/OWL knowledge base construction, ontology reasoning, embedding
training, and SPARQL-RAG chatbot development.

---

## Project Structure

```
MedKG/
├── src/
│   ├── crawl/
│   │   ├── crawler.py          # Wikipedia API crawler (10 medical seed topics)
│   │   └── run_lab1.py         # Runner: crawl + NER + relation extraction
│   ├── ie/
│   │   ├── ner.py              # spaCy NER with medical EntityRuler
│   │   └── relations.py        # Dependency-based relation extraction
│   ├── kg/
│   │   ├── build_kb.py         # Build initial RDF KB from CSV data
│   │   ├── entity_linking.py   # Link entities to Wikidata (owl:sameAs)
│   │   ├── expand_kb.py        # Expand KB via Wikidata SPARQL (1-hop + 2-hop)
│   │   └── run_td4.py          # Runner: build + link + expand
│   ├── reason/
│   │   ├── family.owl          # OWL ontology with SWRL rule for reasoning demo
│   │   └── swrl_reasoning.py   # OWLReady2 SWRL reasoning (HermiT/Pellet)
│   ├── kge/
│   │   ├── prepare_data.py     # Prepare train/valid/test KGE splits
│   │   ├── train_kge.py        # Train TransE and DistMult via PyKEEN
│   │   ├── analyze_kge.py      # Nearest neighbors, t-SNE, relation analysis
│   │   └── run_td5.py          # Runner: SWRL + prepare + train + analyze
│   └── rag/
│       └── lab_rag_sparql_gen.py  # RAG chatbot: NL -> SPARQL -> answers
├── data/
│   ├── samples/                # Sample outputs committed to the repo
│   │   ├── crawler_output.jsonl
│   │   ├── extracted_knowledge.csv
│   │   └── candidate_triples.csv
│   └── README.md
├── kg_artifacts/
│   └── ontology.ttl            # OWL ontology (classes, properties, alignments)
├── reports/
│   └── final_report.md         # Combined 6-section academic report
├── notebooks/                  # (empty) Jupyter notebooks directory
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Create a virtual environment and install dependencies

> **Important:** PyKEEN requires **Python 3.10**. Python 3.11+ also works.
> Python 3.14 is **not** supported (numpy/torch incompatibility).

```bash
# Windows — use py launcher to select Python 3.10
py -3.10 -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3.10 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the spaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3. Set up Ollama (for the RAG module)

Install Ollama from https://ollama.com, then pull the default model:

```bash
ollama pull gemma:2b
ollama serve
```

You can use any model supported by Ollama (e.g. `deepseek-r1:1.5b`, `mistral`).

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | **3.10** (required for KGE) | 3.10–3.12 |
| RAM | 4 GB | 8 GB |
| GPU | Optional | For KGE training |
| Disk | 2 GB | 5 GB (for expanded KB) |

---

## How to Run Each Module

All commands should be run from the **MedKG root directory**.

### Module 1 — Crawl + NER + Relation Extraction

```bash
python src/crawl/run_lab1.py
```

Flags:
- `--skip-crawl` — skip the crawl step (reuse existing `data/crawler_output.jsonl`)
- `--skip-ner` — skip the NER step
- `--skip-relations` — skip relation extraction
- `--max-per-seed N` — crawl N pages per seed topic (default: 8)

Outputs: `data/crawler_output.jsonl`, `data/extracted_knowledge.csv`, `data/candidate_triples.csv`

### Module 2 — KB Construction, Entity Linking, Expansion

```bash
python src/kg/run_td4.py
```

Flags:
- `--skip-build` — skip KB construction
- `--skip-link` — skip entity linking
- `--skip-expand` — skip Wikidata expansion

Outputs: `kg_artifacts/medical_kb_initial.ttl`, `kg_artifacts/alignment.ttl`,
`kg_artifacts/entity_mapping.csv`, `kg_artifacts/medical_kb_expanded.nt`,
`kg_artifacts/stats.json`

### Module 3 — SWRL Reasoning

```bash
python src/reason/swrl_reasoning.py
```

Loads `src/reason/family.owl` and applies OWL reasoning to infer OldPerson
instances from the embedded SWRL rule.

### Module 4 — Knowledge Graph Embeddings

Run each step individually:

```bash
python src/kge/prepare_data.py
python src/kge/train_kge.py
python src/kge/analyze_kge.py
```

Or run all steps together:

```bash
python src/kge/run_td5.py
```

Outputs: `data/kge/` (splits), `results/TransE/`, `results/DistMult/`,
`results/tsne_plot.png`, `results/evaluation_results.json`

### Module 5 — RAG over RDF/SPARQL

**Interactive demo:**
```bash
python src/rag/lab_rag_sparql_gen.py
```

**Evaluation (5 predefined questions, baseline vs SPARQL-RAG):**
```bash
python src/rag/lab_rag_sparql_gen.py --eval
```

**Check Ollama connectivity:**
```bash
python src/rag/lab_rag_sparql_gen.py --ollama-check
```

**Use a different model:**
```bash
python src/rag/lab_rag_sparql_gen.py --model deepseek-r1:1.5b
```

---

## RAG Demo — Step by Step

1. Ensure the Knowledge Base has been built (Module 2 above).
2. Start Ollama: `ollama serve`
3. Pull a model if not yet done: `ollama pull gemma:2b`
4. Run the interactive chatbot:
   ```bash
   python src/rag/lab_rag_sparql_gen.py
   ```
5. Type a medical question at the prompt, e.g.:
   - `What are the symptoms of Diabetes?`
   - `What medications are used to treat Hypertension?`
   - `Which medical specialty handles Alzheimer's disease?`
6. The system outputs both a baseline (LLM-only) answer and a SPARQL-RAG answer
   retrieved directly from the Knowledge Graph.
7. Type `quit` to exit.

[Screenshot of RAG demo - see reports/]

---

## Pipeline Diagram

```
Wikipedia API
     |
     v
[Crawl]  ──────────────────────────────> crawler_output.jsonl
     |
     v
[NER + Relations]  ─────────────────────> extracted_knowledge.csv
                                          candidate_triples.csv
     |
     v
[Build KB]  ────────────────────────────> medical_kb_initial.ttl
     |
     v
[Entity Linking (Wikidata)]  ───────────> alignment.ttl
     |
     v
[Expand KB (SPARQL 1-hop+2-hop)]  ──────> medical_kb_expanded.nt
     |
     +──────────────────────┐
     |                      |
     v                      v
[SWRL Reasoning]       [KGE: TransE + DistMult]
(family.owl,           (PyKEEN, 100 epochs,
 OldPerson rule)        MRR, Hits@k, t-SNE)
                             |
                             v
                    [RAG: NL -> SPARQL -> Answer]
                    (Ollama, self-repair loop,
                     baseline comparison)
```

---

## License

MIT License — see `LICENSE` for details.
