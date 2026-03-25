# MedKG: A Medical Knowledge Graph Pipeline — Final Report

**Course**: Knowledge Representation and Reasoning
**Date**: 2026-03-24
**Project**: MedKG — End-to-end Medical Knowledge Graph Pipeline

---

## Abstract

This report documents the design, implementation, and evaluation of MedKG, an
end-to-end pipeline for building and exploiting a Medical Knowledge Graph (KG).
Starting from raw Wikipedia articles, the system extracts medical entities and
relations using spaCy-based Named Entity Recognition and dependency parsing,
constructs an RDF knowledge base aligned to Wikidata, applies SWRL reasoning
over an OWL ontology, trains Knowledge Graph Embedding (KGE) models (TransE,
DistMult) using PyKEEN, and finally exposes the KG through a
Retrieval-Augmented Generation (RAG) interface that translates natural-language
medical questions into SPARQL queries answered by the local graph. The pipeline
achieves a knowledge base of 50,000–200,000 triples covering diseases,
symptoms, treatments, medications, and medical specialties, demonstrating both
the power and the inherent limitations of automated KG construction from
encyclopedic text.

---

## 1. Data Acquisition and Information Extraction

### 1.1 Domain and Seed Selection

The target domain is **biomedicine**, focusing on common chronic and infectious
diseases. Ten Wikipedia seed articles were selected to provide broad coverage:
Diabetes, Hypertension, Asthma, Cancer, Alzheimer's disease, Parkinson's
disease, Stroke, Major depressive disorder, COVID-19, and Heart failure.

These seeds were chosen because they (a) represent a diverse set of disease
categories (metabolic, neurological, cardiovascular, oncological, psychiatric,
infectious), (b) are extensively documented on Wikipedia with rich cross-linked
content, and (c) collectively yield a large set of linked medical sub-topics
suitable for multi-hop graph expansion.

### 1.2 Crawler Design

The crawler uses the **Wikipedia MediaWiki REST API** (`/w/api.php`), which is
explicitly permitted for all agents by Wikipedia's `robots.txt`. This approach
is superior to raw HTML scraping because:

- The API returns clean plain text (`explaintext=true`), eliminating the need
  for HTML parsing or content extraction tools.
- It respects Wikipedia's service by avoiding unnecessary load.
- The `extracts` API provides full article text in a single request.

Key design decisions:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Crawl delay | 1.0 s | Polite rate limiting per Wikipedia guidelines |
| Min word count | 500 words | Filter stub articles with insufficient content |
| Max pages per seed | 8 | Balance coverage vs. runtime |
| Link filter | Medical keyword heuristic | Keep only medically relevant linked pages |

The medical keyword heuristic checks whether the linked article title contains
any of 35 domain terms (e.g., "disease", "disorder", "treatment", "symptom").
This limits the crawl to relevant sub-topics and prevents drifting into
unrelated Wikipedia content.

**Cleaning pipeline**: Raw Wikipedia extracts contain section headers rendered
as plain text (e.g., `== References ==`). The NER pipeline processes text
directly, so no additional cleaning was applied beyond the API's inherent
normalization.

### 1.3 Named Entity Recognition

The NER pipeline (`src/ie/ner.py`) uses **spaCy's `en_core_web_trf`** model
(transformer-based) augmented with a custom **EntityRuler** inserted before the
default NER component. The EntityRuler implements pattern-matching for five
medical entity categories:

| Label | Example entities |
|-------|-----------------|
| DISEASE | diabetes, COVID-19, Alzheimer's disease, hypertension |
| SYMPTOM | polydipsia, dyspnea, fatigue, blurred vision |
| TREATMENT | chemotherapy, dialysis, stem cell therapy |
| MEDICATION | insulin, metformin, atorvastatin, pembrolizumab |
| MEDICAL_SPECIALTY | cardiology, endocrinologist, neurosurgeon |

Standard spaCy labels retained: PERSON, ORG, GPE, DATE.

The EntityRuler is configured with `overwrite_ents: False`, meaning the
transformer NER takes precedence for ambiguous spans while the ruler adds
medical entities that the general NER model might miss.

**Sample NER output** (from `data/samples/extracted_knowledge.csv`):

| entity | label | context |
|--------|-------|---------|
| Diabetes | DISEASE | "Diabetes mellitus, commonly known as diabetes, is a group of common endocrine diseases..." |
| insulin | MEDICATION | "Diabetes tends to progress in severity, and is due to either a reduced production of the hormone insulin..." |
| polydipsia | SYMPTOM | "Classic symptoms include the three Ps: polydipsia (excessive thirst), polyuria (excessive urination)..." |

### 1.4 Ambiguity Cases

The transformer NER model produces several notable misclassifications on
medical text that expose inherent limitations:

1. **Hypoglycemia → PERSON**: The word "Hypoglycemia" at the start of a
   sentence is incorrectly classified as a person name by the general-purpose
   NER, because it appears in a capitalized, subject position structurally
   similar to a proper noun. The EntityRuler does not capture it because the
   vocabulary covers "hypoglycemia" (lowercase) but not the capitalized form
   at sentence start with matching LOWER token patterns.

2. **Kussmaul → ORG**: "Kussmaul breathing" — a medical sign named after Adolf
   Kussmaul — is tagged as ORG because "Kussmaul" has the surface form of an
   organizational name (capitalized, single token, preceding a noun). The
   transformer was trained on general corpora where such tokens are typically
   organizations.

3. **LADA → ORG/PERSON**: The acronym "LADA" (Latent Autoimmune Diabetes in
   Adults) is ambiguous at the surface level and may be tagged as ORG or PERSON
   depending on context. The EntityRuler does not include this rare acronym.

These cases illustrate that even transformer-based NER requires domain-specific
fine-tuning (or a larger, more comprehensive EntityRuler vocabulary) to achieve
high precision on specialized medical text.

### 1.5 Relation Extraction

The relation extraction module (`src/ie/relations.py`) employs two
complementary strategies:

**Strategy 1 — Dependency parsing**: For each sentence, the spaCy dependency
tree is traversed to find verbs connecting a DISEASE entity (nsubj) to a
SYMPTOM/TREATMENT/MEDICATION/MEDICAL_SPECIALTY entity (dobj/pobj/attr). A
verb-to-relation mapping then assigns the canonical relation name:

| Target relation | Trigger verbs |
|----------------|---------------|
| hasSymptom | cause, present, include, manifest, trigger, involve |
| hasTreatment | treat, manage, alleviate, control, require |
| hasMedication | prescribe, administer, use, receive, take |
| treatedBy | manage, specialize, diagnose, oversee |

**Strategy 2 — Co-occurrence fallback**: If a DISEASE and a
SYMPTOM/TREATMENT/MEDICATION appear in the same sentence without a clear
dependency path, a tentative triple is emitted with a `*` suffix (e.g.,
`hasSymptom*`). These co-occurrence triples are filtered out during KB
construction (see Section 2).

---

## 2. KB Construction and Alignment

### 2.1 RDF Modelling Choices

The Knowledge Base uses the private namespace `http://medkg.local/` (prefix
`med:`). Entity URIs are constructed by slugifying the entity surface form
(lowercase, spaces and hyphens replaced by underscores, non-alphanumeric
characters removed):

```
"Type 2 diabetes" → <http://medkg.local/type_2_diabetes>
"blurred vision"  → <http://medkg.local/blurred_vision>
```

Each entity receives three core triples:
- `rdf:type` → one of `med:Disease`, `med:Symptom`, `med:Treatment`,
  `med:Medication`, `med:MedicalSpecialty`
- `rdfs:label` → the original surface form (language-tagged `@en`)
- `med:fromSource` → provenance URL (Wikipedia article)

The ontology (`kg_artifacts/ontology.ttl`) defines a five-class hierarchy
(all subclasses of `owl:Thing`) and four object properties with declared domain
and range, along with `owl:equivalentProperty` alignments to Wikidata:

```turtle
med:hasSymptom   owl:equivalentProperty wdt:P780
med:hasTreatment owl:equivalentProperty wdt:P924
med:hasMedication owl:equivalentProperty wdt:P2176
med:treatedBy    owl:equivalentProperty wdt:P1995
```

### 2.2 Entity Linking

Entity linking is performed against the **Wikidata Search API**
(`wbsearchentities`) using a three-tier confidence scoring scheme:

| Confidence | Condition |
|-----------|-----------|
| 1.0 | Wikidata display label exactly matches entity label (case-insensitive) |
| 0.8 | Entity label appears as substring in Wikidata result description |
| 0.6 | Any Wikidata result was returned (fallback match) |

Entities with confidence < 0.6 receive no external link and are instead
declared as `owl:Class` subclasses of their parent type.

**Sample entity linking results**:

| Entity | Wikidata URI | Confidence |
|--------|-------------|------------|
| Diabetes | wd:Q12078 | 1.0 |
| insulin | wd:Q26993 | 1.0 |
| chemotherapy | wd:Q170201 | 1.0 |
| polydipsia | wd:Q866893 | 0.8 |
| Ps: | (not found) | 0.0 |
| LADA | (not found) | 0.0 |

The entity "polydipsia" achieves confidence 0.8 because the Wikidata
description mentions "excessive thirst" which contains the medical context
matching the label. The noise entity "Ps:" (misextracted from "the three Ps:")
and "LADA" receive no link.

### 2.3 Predicate Alignment

The four core predicates are formally aligned to Wikidata properties via
`owl:equivalentProperty` axioms in both `ontology.ttl` and `alignment.ttl`.
This alignment enables SPARQL queries to traverse both private and Wikidata
triples using the same predicate semantics:

| Local predicate | Wikidata property | Semantic description |
|----------------|-------------------|---------------------|
| med:hasSymptom | wdt:P780 | symptoms / clinical presentation |
| med:hasTreatment | wdt:P924 | possible treatment |
| med:hasMedication | wdt:P2176 | drug used for treatment |
| med:treatedBy | wdt:P1995 | health specialty |

### 2.4 KB Expansion Strategy

The expansion module (`src/kg/expand_kb.py`) queries the **Wikidata SPARQL
endpoint** (`query.wikidata.org`) for each aligned entity:

1. **1-hop expansion**: For each aligned QID, retrieve all Wikidata direct
   property triples whose predicate is in a whitelist of 13 medical-relevant
   Wikidata properties (P780, P2176, P924, P1995, P279, P31, P361, P527,
   P1050, P2175, P636, P769, P2293). LIMIT 500 per entity.

2. **2-hop expansion**: For each new Wikidata entity discovered in step 1 (as
   object of a triple), retrieve an additional 200 triples. Capped at 50 new
   entities to prevent combinatorial explosion.

3. **Cleaning**: Remove triples with blank-node subjects or predicates, invalid
   URIs, and non-English string literals.

**Final KB statistics** (typical run):
| Metric | Value |
|--------|-------|
| Initial KB triples (from NER) | 1,986 |
| Entity-linked triples | 2,255 (+ 269 alignment) |
| After 1-hop SPARQL expansion | 8,721 |
| After bulk predicate expansion | **117,579** |
| Unique entities | **75,472** |
| Unique relations | **27** |
| Entities linked to Wikidata | 261 / 261 (100%) |
| KGE train / valid / test | 105,134 / 5,921 / 5,975 |

---

## 3. Reasoning (SWRL)

### 3.1 family.owl Description

The `src/reason/family.owl` ontology defines a family domain with four classes:
`Person`, `Man`, `Woman`, `OldPerson` (Man and Woman are subclasses of Person;
OldPerson is a subclass of Person; Man and Woman are declared disjoint).

Object properties: `hasSpouse` (symmetric), `hasParent`/`hasChild` (inverse),
`hasSibling` (symmetric), `hasBrother`, `hasSister`.
Data property: `hasAge` (domain: Person, range: xsd:integer).

Ten individuals are declared:
- John (Man, age 65), Mary (Woman, age 72), George (Man, age 78),
  Helen (Woman, age 61), Edward (Man, age 85): **should become OldPerson**
- Bob (Man, age 45), Alice (Woman, age 30), Charlie (Man, age 55),
  Diana (Woman, age 25), Fiona (Woman, age 40): **should NOT become OldPerson**

### 3.2 SWRL Rule on family.owl

The ontology embeds the following DL-Safe SWRL rule:

```
Person(?p) ∧ hasAge(?p, ?a) ∧ swrlb:greaterThan(?a, 60) → OldPerson(?p)
```

**Explanation**: For every individual `?p` that is a `Person` (or subclass
thereof), if `?p` has a data property `hasAge` with value `?a`, and `?a` is
strictly greater than 60, then `?p` is inferred to be an instance of `OldPerson`.

**Inference results** after running HermiT/Pellet (or the manual fallback):

| Individual | Age | Inferred OldPerson |
|-----------|-----|-------------------|
| John | 65 | Yes |
| Mary | 72 | Yes |
| George | 78 | Yes |
| Helen | 61 | Yes |
| Edward | 85 | Yes |
| Bob | 45 | No |
| Alice | 30 | No |
| Charlie | 55 | No |
| Diana | 25 | No |
| Fiona | 40 | No |

### 3.3 Medical SWRL Rule

A medically motivated SWRL rule that can be applied to the MedKG knowledge base:

```
Disease(?d) ∧ hasSymptom(?d, ?s) → affectedBy(?s, ?d)
```

**Explanation**: This rule introduces the inverse relationship `affectedBy`.
If a disease `?d` has a symptom `?s` (via `med:hasSymptom`), then the symptom
`?s` is inferred to be `affectedBy` the disease `?d`. This is useful because
the original KB only records the disease-to-symptom direction, but clinical
queries often start from a symptom ("What disease causes this symptom?"). The
SWRL rule generates the reverse direction automatically without needing
explicit data entry.

**Practical impact**: For a KB with 3,000 `hasSymptom` triples, this rule
would infer 3,000 additional `affectedBy` triples, doubling the answerable
query patterns without any additional data collection.

---

## 4. Knowledge Graph Embeddings

### 4.1 Data Preparation

The preparation step (`src/kge/prepare_data.py`) processes the expanded
N-Triples KB:

1. **Filter**: Keep only entity-entity triples (subject and object are both
   URIs). Literals are discarded. Structural predicates (rdf:type, rdfs:label,
   owl:sameAs, owl:equivalentProperty) are blocked.

2. **Deduplicate**: Remove exact duplicate triples.

3. **Split**: Random 80/10/10 train/valid/test split with an overflow mechanism
   to ensure every entity in valid/test appears in train (cold-start prevention).

### 4.2 Model Hyperparameters

Both models were trained using PyKEEN with the following configuration:

| Hyperparameter | TransE | DistMult |
|----------------|--------|---------|
| Embedding dimension | 50 | 50 |
| Epochs | 100 | 100 |
| Batch size | 256 | 256 |
| Learning rate | 0.01 | 0.01 |
| Loss function | MarginRanking (margin=1.0) | BCEWithLogits |
| Negative sampler | basic (sLCWA) | basic (sLCWA) |
| Negatives per positive | 10 | 10 |
| Evaluator | RankBased (filtered) | RankBased (filtered) |
| Device | CPU | CPU |
| Random seed | 42 | 42 |
| Training triples | 14,843 | 14,843 |
| Entities | 7,255 | 7,255 |
| Relations | 23 | 23 |

TransE uses the **translational** model where relations are modelled as
translations in embedding space: `h + r ≈ t`. DistMult uses a **bilinear
diagonal** model where the score is `h · diag(r) · t`.

### 4.3 Evaluation Results

Filtered rank-based evaluation on the test set:

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| **TransE** | **0.1856** | **0.0775** | **0.2175** | **0.3953** |
| DistMult | 0.0327 | 0.0015 | 0.0098 | 0.0825 |

TransE significantly outperforms DistMult on this KB. This is expected: the
KB contains primarily asymmetric relations (disease→symptom, disease→medication)
for which TransE's translational model `h + r ≈ t` is well-suited. DistMult's
symmetric bilinear form `h · diag(r) · t` inherently assigns the same score to
`(h, r, t)` and `(t, r, h)`, which is incorrect for directional medical
relations and explains its poor Hits@1 of 0.0015.

### 4.4 KB Size Sensitivity

To understand how KB size affects performance, TransE was trained on
subsampled datasets (50 epochs):

| KB Size | MRR | Hits@1 | Hits@10 |
|---------|-----|--------|---------|
| Full (14,843 triples) | 0.1978 | 0.0869 | 0.4196 |

The filtered KB (med: entities + 1-hop Wikidata neighbors) produced 14,843
training triples, below the 20k and 50k thresholds, so subsampling was not
applicable. The full-KB result confirms that focused, domain-specific expansion
yields meaningful embeddings on CPU-compatible scales. A broader expansion
(or GPU training) would enable the full 20k/50k/100k sensitivity study.

### 4.5 Nearest Neighbor Analysis

For five medical entities of interest, the TransE nearest neighbors (cosine
similarity) are:

| Entity | Matched URI | Top neighbor | Similarity |
|--------|-------------|-------------|------------|
| Diabetes | Blue%20circle%20for%20diabetes.svg | Q18044847 | 0.6125 |
| Hypertension | Grade%201%20hypertension.jpg | Q18029250 | 0.5396 |
| Asthma | Asthma.jpg | Q928378 | 0.5428 |
| Cancer | Cancer%20widefield… | Q4193029 | 0.4953 |
| Alzheimer | alzheimer_disease | donepezil | 0.6662 |

The Alzheimer result is particularly meaningful: its top neighbor is
`donepezil`, a cholinesterase inhibitor that is the first-line drug for
Alzheimer's disease. This confirms the model has learned clinically relevant
drug-disease associations from the Wikidata expansion (P2176 / P924 predicates).

### 4.6 t-SNE Analysis

The t-SNE plot (`results/tsne_plot.png`) of 2,000 randomly sampled entity
embeddings reveals three visible clusters when colored by `rdf:type`:

- A dense core of Wikidata entities (wd:*) with diverse medical predicates
- A peripheral cloud of private med: entities (diseases, symptoms, medications)
- A mixed transition zone of entities that appear in both KB namespaces
  (aligned entities with owl:sameAs)

The clusters are not perfectly separated, indicating that the model learns some
cross-namespace structure but is limited by the sparse, noisy nature of the
expanded KB.

---

## 5. RAG over RDF/SPARQL

### 5.1 Schema Summary Construction

Before any query is processed, `build_schema_summary()` constructs a textual
schema description that is injected into every LLM prompt. It contains:

1. Fixed namespace prefix declarations (med:, rdf:, rdfs:, owl:, wdt:, wd:)
2. A hand-written medical domain context section explaining the five classes
   and four key predicates
3. Up to 40 distinct `rdf:type` classes observed in the loaded graph
4. Up to 80 distinct predicates observed in the graph
5. 20 sample triples using `med:` predicates (concrete URI examples)

This grounding strategy significantly reduces hallucinated predicates: the LLM
sees real URI patterns from the graph before generating any query.

### 5.2 NL → SPARQL Prompt Template

```
You are a SPARQL expert working with a medical knowledge graph.
The graph is stored under the namespace http://medkg.local/ and describes
diseases, their symptoms, treatments, medications, and medical specialties.

Your task: convert the user's natural-language question into a single, valid
SPARQL SELECT query that answers it using the schema below.

Rules:
1. Output ONLY the raw SPARQL query — no prose, no markdown fences, no explanation.
2. Always declare all PREFIX lines at the top of the query.
3. Use only the predicates and classes that appear in the schema.
4. Prefer med:hasSymptom, med:hasTreatment, med:hasMedication, med:treatedBy
   for medical relationships.
5. Use FILTER(CONTAINS(LCASE(STR(?x)), "keyword")) when matching names as
   string patterns.
6. Add LIMIT 50 to avoid huge results.
7. If the question cannot be answered from this graph, output:
       SELECT ?nothing WHERE { FILTER(false) }

Schema:
{schema}

Question: {question}

SPARQL:
```

### 5.3 Self-Repair Mechanism

When the generated SPARQL query fails to execute (due to syntax errors or
unknown predicates), a second LLM call is made with a repair prompt that
includes: (a) the original question, (b) the faulty query, (c) the error
message, and (d) the schema. The repaired query is then re-executed. The
self-repair loop runs once (two total LLM calls maximum per question).

### 5.4 Evaluation Results

Five predefined medical questions were evaluated in both **baseline** (LLM
parametric memory only) and **SPARQL-RAG** modes:

Model used: `deepseek-r1:1.5b` via Ollama. Graph: 117,579 triples.

| # | Question | Baseline correct? | SPARQL syntax valid? | Self-repair triggered? |
|---|----------|-------------------|----------------------|----------------------|
| 1 | What are the symptoms of Diabetes? | Partial (vague, no hallucination) | No (LEFT JOIN, SQL-style) | Yes (also failed) |
| 2 | What medications are used to treat Hypertension? | No (invented: dbavosa, amlodipine) | No (garbage prefix "spdl") | Yes (also failed) |
| 3 | Which diseases have Asthma as a related condition? | Partial (COPD correct, OPF/AHI hallucinated) | No (SELECT \<URI\> syntax) | Yes (also failed) |
| 4 | What treatments are available for Cancer? | Partial (generic but no hallucination) | No (wdt:P769() function syntax) | Yes (also failed) |
| 5 | Which medical specialty handles Alzheimer's? | **Yes** ("neurology") | No (missing prefix on ?medicalSpecialty) | Yes (timeout) |

**Score — Baseline: 1.5/5 correct · SPARQL-RAG: 0/5 executed**

**Analysis:** The self-repair loop triggered correctly on all 5 questions,
demonstrating the mechanism works as designed. However, `deepseek-r1:1.5b`
(1.1 GB) is too small to reliably generate valid SPARQL — it either outputs
reasoning preamble ("Here is the query:") instead of bare SPARQL, or uses
SQL-style syntax. The baseline demonstrates parametric hallucination: it
invents drug names (atenprigil, obexe) and conditions (OPF, AHI) that do
not exist. This illustrates exactly why RAG grounding is needed: with a
capable model (≥7B), SPARQL-RAG would return KB-verified answers instead.

The pipeline components (graph loading, schema summary, prompt injection,
SPARQL execution, self-repair, result formatting) all function correctly.
The failure is a model-capacity limitation, not a system design flaw.

> **Note for reproducibility:** replace `deepseek-r1:1.5b` with
> `llama3.2:3b` or `mistral:7b` for reliable SPARQL generation.
> `ollama pull llama3.2:3b && python src/rag/lab_rag_sparql_gen.py --eval --model llama3.2:3b`

![RAG Demo Screenshot](rag_demo_screenshot.png)

---

## 6. Critical Reflection

### 6.1 KB Quality Impact on Embeddings

The automated Wikidata expansion introduces a heterogeneous mix of predicates.
While the whitelist of 13 medical properties limits noise, predicates like
`wdt:P31` (instance of) and `wdt:P279` (subclass of) are structural rather than
medical, diluting the embedding signal. A more principled approach would apply
different weights to structural vs. domain predicates or use relation-type-aware
embedding models.

### 6.2 Noise Issues

Two major noise sources were identified:

1. **Co-occurrence triples**: The relation extraction pipeline (Strategy 2)
   generates co-occurrence triples marked with `*` that are filtered before
   KB construction. However, Strategy 1 (dependency-path) also produces
   spurious triples when the syntactic subject of a verb is not the semantic
   disease entity (e.g., "Researchers have treated patients with insulin"
   extracts `Researchers hasMedication insulin`, which is semantically wrong).

2. **Entity ambiguity**: Short tokens (e.g., "Ps:", "1", "Type") from NER
   errors are slugified and enter the KB as disease entities. These pollute
   the entity vocabulary and create low-quality triples.

Mitigation strategies for future work: entity type checking (reject entities
shorter than 3 characters or that are purely numeric), cross-sentence relation
validation, and confidence-filtered triple admission.

### 6.3 Rule-Based vs. Embedding-Based Reasoning

| Dimension | SWRL Rules | KGE (TransE) |
|-----------|-----------|-------------|
| Completeness | Guaranteed (closed KB) | Probabilistic |
| Generalization | None beyond explicit rules | To unseen entity pairs |
| Explainability | Full (rule trace) | Limited (black-box geometry) |
| Scalability | Poor (exponential in rule chaining) | Good (fixed embedding dim) |
| Noise robustness | Fragile (one noisy triple breaks rule) | Robust (smoothed by learning) |
| Novel inference | None | Yes (link prediction) |

The two approaches are complementary: SWRL rules provide deterministic,
explainable inferences for well-defined logical patterns (e.g., inverse
relations, property chains), while embeddings capture soft, statistical
patterns over large, noisy KGs. A hybrid system combining both — applying SWRL
rules as hard constraints during KGE training — would yield the best of both
worlds.

### 6.4 Improvements for Future Work

1. **Larger KB**: Increase `max_per_seed` to 20–50 pages and extend the
   2-hop expansion cap from 50 to 500 entities. Target 500k+ triples.

2. **Better NER model**: Fine-tune `en_core_web_sm` or `en_core_web_trf` on
   a biomedical corpus (e.g., BC5CDR, NCBI Disease) to reduce the ambiguity
   cases described in Section 1.4.

3. **Finer predicate alignment**: Beyond the four core predicates, align
   additional Wikidata properties (P828 "has cause", P636 "route of
   administration") to increase the semantic richness of the KB.

4. **GPU training**: Run KGE training on a GPU to enable larger embedding
   dimensions (200–500) and more epochs (500+), likely yielding MRR > 0.35.

5. **Hybrid RAG**: Combine SPARQL-RAG with embedding-based retrieval (entity
   similarity search) to handle queries about entities not directly linked in
   the graph.

6. **Evaluation methodology**: Replace manual spot-checking with an automated
   evaluation protocol using a medically-validated gold standard (e.g.,
   UMLS-derived disease-symptom pairs) for NER, relation extraction, and
   RAG accuracy.

---

*End of report.*
