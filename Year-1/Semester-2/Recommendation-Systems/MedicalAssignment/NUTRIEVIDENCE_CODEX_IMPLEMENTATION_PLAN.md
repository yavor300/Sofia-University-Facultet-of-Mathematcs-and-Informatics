# NutriEvidence Agent — Implementation Plan for Codex

This document is a step-by-step implementation plan for building the **NutriEvidence Agent** project.

The project is an educational biomedical literature recommender system. It retrieves PubMed articles, ranks them using semantic and graph-based similarity, recommends relevant papers, and optionally uses an LLM to generate explanations and evidence summaries.

> Important: This system is for educational and research support only. It must not provide diagnosis, treatment, or personalized medical advice.

---

## 1. Project Goal

Build an end-to-end MVP with the following capabilities:

1. Retrieve biomedical articles from PubMed.
2. Store/cache article metadata locally.
3. Preprocess article text.
4. Generate semantic embeddings for article title + abstract.
5. Recommend articles using semantic similarity.
6. Build a simple Article–MeSH knowledge graph.
7. Train node2vec graph embeddings.
8. Recommend similar articles using graph embeddings.
9. Compare semantic, graph-based, and baseline recommendations.
10. Provide a Streamlit UI.
11. Optionally use an LLM for query planning, evidence extraction, summaries, and recommendation explanations.

---

## 2. Recommended MVP Scope

Implement the project in three layers.

### Layer 1 — Core Recommender MVP

This is the minimum version that must work first.

- PubMed retrieval
- Cached article dataset
- Text preprocessing
- Sentence-transformer embeddings
- Semantic similarity recommender
- Streamlit UI

### Layer 2 — Knowledge Graph Extension

This satisfies the Knowledge Base / graph embedding requirement.

- Extract MeSH terms from PubMed metadata
- Build Article–MeSH graph with NetworkX
- Train node2vec
- Recommend similar articles by graph embedding similarity
- Add MeSH-overlap baseline

### Layer 3 — LLM Agentic Layer

This makes the project modern and agentic.

- Query Planner Agent
- Evidence Extraction Agent
- Answer Generation Agent
- Recommendation Explanation Agent
- Safety Checker Agent

Implement Layer 1 first, then Layer 2, then Layer 3.

---

## 3. Technology Stack

Use the following stack for the first implementation:

```text
Python 3.10+
Streamlit
pandas
numpy
scikit-learn
sentence-transformers
Biopython
NetworkX
node2vec
gensim
python-dotenv
```

Optional:

```text
OpenAI API or local Ollama model
LangChain / LangGraph later, not required for MVP
```

Do not use Elasticsearch in the MVP. Keep vector search in memory using `numpy` and `scikit-learn`.

---

## 4. Target Repository Structure

Create this structure:

```text
nutri-evidence-agent/
│
├── README.md
├── requirements.txt
├── .env.example
├── app.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pubmed_client.py
│   │   └── cache.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── article_preprocessor.py
│   │
│   ├── recommenders/
│   │   ├── __init__.py
│   │   ├── semantic_recommender.py
│   │   ├── mesh_overlap_recommender.py
│   │   ├── graph_recommender.py
│   │   └── hybrid_recommender.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   └── node2vec_trainer.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_planner.py
│   │   ├── evidence_extractor.py
│   │   ├── answer_generator.py
│   │   ├── recommendation_explainer.py
│   │   └── safety_checker.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── text_utils.py
│
├── data/
│   ├── sample_queries.json
│   ├── pubmed_articles.json
│   ├── evaluation_annotations.csv
│   └── artifacts/
│       ├── article_embeddings.npy
│       ├── article_embedding_index.json
│       ├── article_mesh_graph.gpickle
│       └── node2vec_embeddings.kv
│
├── notebooks/
│   ├── 01_pubmed_retrieval.ipynb
│   ├── 02_semantic_recommender.ipynb
│   ├── 03_graph_recommender.ipynb
│   └── 04_evaluation.ipynb
│
└── docs/
    ├── architecture.md
    └── example_outputs.md
```

---

## 5. Environment Setup

### 5.1 Create `requirements.txt`

Add:

```txt
biopython
pandas
numpy
scikit-learn
sentence-transformers
streamlit
networkx
node2vec
gensim
python-dotenv
tqdm
```

Optional LLM dependencies:

```txt
openai
```

### 5.2 Create `.env.example`

```env
NCBI_EMAIL=your_email@example.com
NCBI_API_KEY=
OPENAI_API_KEY=
USE_LLM=false
```

Notes:

- `NCBI_EMAIL` is required by NCBI Entrez.
- `NCBI_API_KEY` is optional.
- `OPENAI_API_KEY` is optional and only needed if the LLM layer is enabled.
- The project must work without LLM by using rule-based fallbacks.

---

## 6. Data Model

All article records should use this schema:

```json
{
  "pmid": "12345678",
  "title": "Article title",
  "abstract": "Article abstract",
  "year": 2023,
  "journal": "Journal name",
  "authors": ["Author A", "Author B"],
  "publication_types": ["Journal Article", "Review"],
  "mesh_terms": ["Cerebral Palsy", "Nutritional Status", "Child"],
  "doi": "10.xxxx/yyyy",
  "source_query": "nutrition risk cerebral palsy children"
}
```

Store all articles in:

```text
data/pubmed_articles.json
```

---

## 7. Step-by-Step Implementation

---

# Phase 1 — Project Bootstrap

## Step 1.1 — Create project structure

Create all directories and empty Python modules from the structure above.

Acceptance criteria:

- The repository contains the expected folders.
- All Python packages include `__init__.py`.
- `python -m compileall src` succeeds.

---

## Step 1.2 — Implement configuration utility

File:

```text
src/utils/config.py
```

Implement:

```python
from dataclasses import dataclass
import os
from dotenv import load_dotenv

@dataclass
class Settings:
    ncbi_email: str
    ncbi_api_key: str | None
    openai_api_key: str | None
    use_llm: bool

def load_settings() -> Settings:
    ...
```

Behavior:

- Load `.env`.
- Read `NCBI_EMAIL`, `NCBI_API_KEY`, `OPENAI_API_KEY`, `USE_LLM`.
- Convert `USE_LLM` to boolean.
- Raise a clear error if `NCBI_EMAIL` is missing when PubMed retrieval is used.

Acceptance criteria:

- Settings are loaded correctly.
- Missing optional keys do not break the app.
- Missing `NCBI_EMAIL` only breaks PubMed live retrieval, not cached mode.

---

# Phase 2 — PubMed Retrieval

## Step 2.1 — Implement PubMed client

File:

```text
src/retrieval/pubmed_client.py
```

Use Biopython Entrez.

Implement class:

```python
class PubMedClient:
    def __init__(self, email: str, api_key: str | None = None):
        ...

    def search(self, query: str, max_results: int = 50) -> list[str]:
        ...

    def fetch_details(self, pmids: list[str]) -> list[dict]:
        ...

    def search_and_fetch(self, query: str, max_results: int = 50) -> list[dict]:
        ...
```

Required extracted fields:

- `pmid`
- `title`
- `abstract`
- `year`
- `journal`
- `authors`
- `publication_types`
- `mesh_terms`
- `doi`
- `source_query`

Implementation notes:

- Use `Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")`.
- Use `Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="xml")`.
- Safely handle missing abstract.
- Safely handle missing MeSH terms.
- Safely handle missing year.
- Deduplicate articles by PMID.
- Do not crash on malformed records.

Acceptance criteria:

- Searching `"gut microbiome Parkinson disease"` returns a list of PMIDs.
- Fetching PMIDs returns normalized dictionaries.
- Articles without abstracts are handled safely.

---

## Step 2.2 — Implement cache utilities

File:

```text
src/retrieval/cache.py
```

Implement:

```python
def load_articles(path: str) -> list[dict]:
    ...

def save_articles(articles: list[dict], path: str) -> None:
    ...

def merge_articles(existing: list[dict], new_articles: list[dict]) -> list[dict]:
    ...
```

Behavior:

- JSON read/write with UTF-8.
- Merge by `pmid`.
- Preserve existing articles if duplicate appears.
- Create parent directories if missing.

Acceptance criteria:

- Re-running retrieval does not duplicate articles.
- `data/pubmed_articles.json` is valid JSON.

---

## Step 2.3 — Create sample query file

File:

```text
data/sample_queries.json
```

Content:

```json
[
  "nutrition risk cerebral palsy children",
  "body composition cerebral palsy children",
  "gut microbiome Parkinson disease",
  "gut microbiome Alzheimer disease",
  "Mediterranean diet pregnancy child neurodevelopment",
  "maternal diet child cognition",
  "urbanization dietary patterns"
]
```

---

## Step 2.4 — Create dataset builder script

Optional but useful file:

```text
scripts/build_pubmed_dataset.py
```

Implement a script that:

1. Loads sample queries.
2. For each query, retrieves up to 50 articles.
3. Adds `source_query`.
4. Merges with existing cache.
5. Saves to `data/pubmed_articles.json`.

Acceptance criteria:

- Running the script creates or updates `data/pubmed_articles.json`.
- Dataset contains at least 200 articles when online retrieval works.
- The app can still work with a manually prepared cached file.

---

# Phase 3 — Preprocessing

## Step 3.1 — Implement article preprocessor

File:

```text
src/preprocessing/article_preprocessor.py
```

Implement:

```python
def normalize_text(text: str) -> str:
    ...

def build_document_text(article: dict) -> str:
    ...

def filter_valid_articles(articles: list[dict]) -> list[dict]:
    ...

def articles_to_dataframe(articles: list[dict]):
    ...
```

Behavior:

- `build_document_text` returns `"title. abstract"`.
- Remove articles with missing title and missing abstract.
- Keep articles with title even if abstract is missing.
- Normalize whitespace.
- Ensure `mesh_terms` is always a list.

Acceptance criteria:

- Preprocessed articles have `document_text`.
- No recommender crashes because of missing fields.

---

# Phase 4 — Semantic Recommender

## Step 4.1 — Implement semantic recommender

File:

```text
src/recommenders/semantic_recommender.py
```

Use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Implement:

```python
class SemanticRecommender:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        ...

    def fit(self, articles: list[dict]) -> None:
        ...

    def recommend_by_query(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        ...

    def save_artifacts(self, embeddings_path: str, index_path: str) -> None:
        ...

    def load_artifacts(self, articles: list[dict], embeddings_path: str, index_path: str) -> None:
        ...
```

Each recommendation result should include:

```json
{
  "pmid": "...",
  "title": "...",
  "abstract": "...",
  "year": 2023,
  "journal": "...",
  "mesh_terms": ["..."],
  "score": 0.87,
  "method": "semantic"
}
```

Implementation notes:

- Use normalized embeddings.
- Use cosine similarity.
- Exclude seed article from recommendations in `recommend_by_article`.
- Save embeddings as `.npy`.
- Save article index as JSON.

Acceptance criteria:

- Given a query, returns Top 5 relevant articles.
- Given a PMID, returns Top 5 similar articles.
- Results include scores and method name.
- App does not recompute embeddings unnecessarily if artifacts exist.

---

# Phase 5 — MeSH Overlap Baseline

## Step 5.1 — Implement MeSH overlap recommender

File:

```text
src/recommenders/mesh_overlap_recommender.py
```

Implement:

```python
def jaccard_similarity(a: set[str], b: set[str]) -> float:
    ...

class MeshOverlapRecommender:
    def __init__(self):
        ...

    def fit(self, articles: list[dict]) -> None:
        ...

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        ...
```

Behavior:

- Compute Jaccard similarity over MeSH terms.
- Exclude seed article.
- Return sorted recommendations.
- If article has no MeSH terms, return empty list or low-scored results.

Acceptance criteria:

- Baseline works without embeddings.
- Results include shared MeSH terms if possible.

---

# Phase 6 — Knowledge Graph and node2vec

## Step 6.1 — Build Article–MeSH graph

File:

```text
src/graph/graph_builder.py
```

Implement:

```python
import networkx as nx

class ArticleMeshGraphBuilder:
    def build(self, articles: list[dict]) -> nx.Graph:
        ...

    def save(self, graph: nx.Graph, path: str) -> None:
        ...

    def load(self, path: str) -> nx.Graph:
        ...
```

Graph schema:

Article node:

```text
article:<PMID>
```

Node attributes:

```python
{
  "node_type": "article",
  "pmid": "...",
  "title": "...",
  "year": 2023,
  "journal": "..."
}
```

MeSH node:

```text
mesh:<normalized_mesh_label>
```

Node attributes:

```python
{
  "node_type": "mesh_term",
  "label": "Cerebral Palsy"
}
```

Edge:

```python
{
  "edge_type": "has_mesh_term"
}
```

Acceptance criteria:

- Graph contains article nodes and MeSH nodes.
- Every article with MeSH terms has edges to MeSH nodes.
- Graph can be saved and loaded.
- Isolated articles are allowed but should be ignored by node2vec recommender if no embedding exists.

---

## Step 6.2 — Train node2vec

File:

```text
src/graph/node2vec_trainer.py
```

Implement:

```python
class Node2VecTrainer:
    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 20,
        num_walks: int = 100,
        workers: int = 2,
        window: int = 10,
        min_count: int = 1
    ):
        ...

    def train(self, graph):
        ...

    def save(self, model, path: str) -> None:
        ...

    def load(self, path: str):
        ...
```

Acceptance criteria:

- Trains a node2vec model on the Article–MeSH graph.
- Saves embeddings to `data/artifacts/node2vec_embeddings.kv`.
- Loading saved embeddings works.

---

## Step 6.3 — Implement graph recommender

File:

```text
src/recommenders/graph_recommender.py
```

Implement:

```python
class GraphRecommender:
    def __init__(self):
        ...

    def fit(self, articles: list[dict], graph, node2vec_model) -> None:
        ...

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        ...

    def recommend_from_liked_articles(self, pmids: list[str], top_k: int = 5) -> list[dict]:
        ...
```

Behavior:

- Use only article node embeddings.
- Convert PMID to `article:<PMID>`.
- Compute cosine similarity between graph embeddings.
- For multiple liked articles, compute average vector.
- Return article metadata, similarity score, method name, and shared MeSH terms.

Acceptance criteria:

- Given a valid PMID with node2vec embedding, returns Top 5 graph-based recommendations.
- If PMID has no embedding, return clear empty result instead of crashing.
- Multiple liked articles recommendation works.

---

# Phase 7 — Hybrid Recommender

## Step 7.1 — Implement hybrid recommender

File:

```text
src/recommenders/hybrid_recommender.py
```

Implement:

```python
class HybridRecommender:
    def __init__(
        self,
        semantic_recommender,
        graph_recommender=None,
        semantic_weight: float = 0.6,
        graph_weight: float = 0.4
    ):
        ...

    def recommend_by_article(self, pmid: str, top_k: int = 5) -> list[dict]:
        ...
```

Behavior:

- Call semantic and graph recommenders.
- Merge results by PMID.
- Normalize method scores if needed.
- Compute:

```text
final_score = semantic_weight * semantic_score + graph_weight * graph_score
```

- If graph recommender is unavailable, fall back to semantic only.
- Return method `"hybrid"`.

Acceptance criteria:

- Hybrid recommendations work even if graph model is missing.
- Results include component scores:
  - `semantic_score`
  - `graph_score`
  - `final_score`

---

# Phase 8 — LLM Agentic Layer

The project must work without LLM. Implement simple rule-based fallbacks first. Add LLM calls only if `USE_LLM=true`.

---

## Step 8.1 — Query Planner Agent

File:

```text
src/agents/query_planner.py
```

Implement:

```python
class QueryPlannerAgent:
    def __init__(self, use_llm: bool = False):
        ...

    def plan(self, user_question: str) -> dict:
        ...
```

Output schema:

```json
{
  "population": "...",
  "exposure": "...",
  "intervention": "...",
  "outcome": "...",
  "question_type": "...",
  "pubmed_query": "..."
}
```

Fallback behavior without LLM:

- Return empty/null PICO fields.
- Use the original user question as `pubmed_query`.

LLM prompt requirements:

- Extract population, exposure, intervention, outcome, and question type.
- Generate PubMed search query.
- Return strict JSON only.
- Do not answer the medical question.

Acceptance criteria:

- Without LLM, planner still works.
- With LLM, planner returns valid JSON or falls back safely.
- The planner never generates diagnosis/treatment advice.

---

## Step 8.2 — Evidence Extraction Agent

File:

```text
src/agents/evidence_extractor.py
```

Implement:

```python
class EvidenceExtractionAgent:
    def __init__(self, use_llm: bool = False):
        ...

    def extract(self, article: dict) -> dict:
        ...
```

Output schema:

```json
{
  "pmid": "...",
  "population": "...",
  "exposure_or_intervention": "...",
  "outcome": "...",
  "main_finding": "...",
  "limitations": "..."
}
```

Fallback behavior:

- Return empty fields plus title/PMID.
- Optionally use simple keyword heuristics.

Acceptance criteria:

- Works without LLM.
- Does not hallucinate study results not present in abstract.
- Clearly returns `"not_available"` when information cannot be extracted.

---

## Step 8.3 — Recommendation Explanation Agent

File:

```text
src/agents/recommendation_explainer.py
```

Implement:

```python
class RecommendationExplainer:
    def __init__(self, use_llm: bool = False):
        ...

    def explain(self, seed_article: dict | None, recommended_article: dict) -> str:
        ...
```

Fallback explanation:

```text
This article is recommended because it is similar to the query or selected article based on semantic similarity, shared MeSH terms, or graph embedding proximity.
```

If shared MeSH terms are available, include them.

Acceptance criteria:

- Every recommendation can be explained.
- Explanation mentions method: semantic, graph, hybrid, or MeSH overlap.
- Explanation does not make clinical claims.

---

## Step 8.4 — Answer Generation Agent

File:

```text
src/agents/answer_generator.py
```

Implement:

```python
class AnswerGenerator:
    def __init__(self, use_llm: bool = False):
        ...

    def generate(self, user_question: str, recommendations: list[dict], evidence_items: list[dict]) -> str:
        ...
```

Output format:

```md
## Short Answer

## Evidence Summary

## Recommended Papers

## Limitations

## Safety Note
```

Fallback behavior:

- Generate a template-based answer listing recommended papers.
- Include PMID and title.
- Include safety note.

Acceptance criteria:

- Works without LLM.
- If LLM is used, only summarize based on provided abstracts/evidence.
- Always includes safety note.
- Does not provide diagnosis or treatment instructions.

---

## Step 8.5 — Safety Checker Agent

File:

```text
src/agents/safety_checker.py
```

Implement:

```python
class SafetyChecker:
    def validate(self, text: str) -> str:
        ...
```

Behavior:

- Ensure safety note exists.
- Optionally replace risky phrases:
  - `"you should take"` → `"the literature discusses"`
  - `"this treatment is recommended"` → `"some studies evaluate this intervention"`
- Add disclaimer if missing.

Required disclaimer:

```text
This output is for educational and research purposes only and should not be interpreted as medical advice, diagnosis, or treatment recommendation.
```

Acceptance criteria:

- Every generated answer ends with a safety note.
- The system avoids direct patient-level advice.

---

# Phase 9 — Evaluation

## Step 9.1 — Implement metrics

File:

```text
src/evaluation/metrics.py
```

Implement:

```python
def precision_at_k(relevances: list[int], k: int = 5, threshold: int = 1) -> float:
    ...

def dcg_at_k(relevances: list[int], k: int = 5) -> float:
    ...

def ndcg_at_k(relevances: list[int], k: int = 5) -> float:
    ...

def reciprocal_rank(relevances: list[int], threshold: int = 1) -> float:
    ...

def mean_reciprocal_rank(all_relevances: list[list[int]], threshold: int = 1) -> float:
    ...
```

Acceptance criteria:

- Metrics work for empty inputs.
- Metrics handle fewer than `k` results.
- Unit-testable pure functions.

---

## Step 9.2 — Create annotation file

File:

```text
data/evaluation_annotations.csv
```

Columns:

```csv
query_id,seed_pmid,recommended_pmid,method,rank,relevance
```

Relevance scale:

```text
0 = not relevant
1 = somewhat relevant
2 = relevant
3 = highly relevant
```

---

## Step 9.3 — Evaluation script

Optional file:

```text
scripts/evaluate_recommenders.py
```

Behavior:

- Load annotations.
- Group by query/method.
- Compute:
  - Precision@5
  - nDCG@5
  - MRR
- Print markdown table.

Expected output:

```md
| Method | Precision@5 | nDCG@5 | MRR |
|---|---:|---:|---:|
| MeSH Overlap | ... | ... | ... |
| Semantic | ... | ... | ... |
| Graph node2vec | ... | ... | ... |
| Hybrid | ... | ... | ... |
```

---

# Phase 10 — Streamlit UI

## Step 10.1 — Implement `app.py`

The UI should support two modes.

### Mode A — Search by research question

Input:

```text
What is the evidence linking gut microbiome and Parkinson's disease?
```

Flow:

1. Query Planner Agent generates `pubmed_query`.
2. Retrieve from cache first.
3. If not enough cached results and NCBI config exists, call PubMed.
4. Fit/load semantic recommender.
5. Show Top 5 semantic recommendations.
6. Generate summary/explanations.
7. Show safety note.

### Mode B — Recommend by selected article

Input:

- Dropdown of article titles from cached dataset.

Flow:

1. User selects a seed article.
2. Show semantic recommendations.
3. Show graph recommendations if graph model exists.
4. Show hybrid recommendations if both exist.
5. Show shared MeSH terms.
6. Show explanation.

UI sections:

```text
Project description
Question input
Generated PubMed query
Recommended papers table
Evidence summary
Graph-based recommendations
Evaluation note
Safety disclaimer
```

Acceptance criteria:

- `streamlit run app.py` starts the app.
- App works with cached dataset and no API keys.
- App does not crash if graph artifacts are missing.
- App explains which recommender method was used.

---

# Phase 11 — Scripts

Create these scripts if possible:

```text
scripts/build_pubmed_dataset.py
scripts/build_semantic_embeddings.py
scripts/build_graph.py
scripts/train_node2vec.py
scripts/evaluate_recommenders.py
```

## Script responsibilities

### `build_pubmed_dataset.py`

- Retrieve articles from sample queries.
- Save `data/pubmed_articles.json`.

### `build_semantic_embeddings.py`

- Load articles.
- Fit semantic recommender.
- Save embeddings.

### `build_graph.py`

- Load articles.
- Build Article–MeSH graph.
- Save graph artifact.

### `train_node2vec.py`

- Load graph.
- Train node2vec.
- Save embeddings.

### `evaluate_recommenders.py`

- Load annotations.
- Compute metrics.

Acceptance criteria:

- Scripts can be run independently.
- Scripts print clear success/error messages.

---

# Phase 12 — README

Create a complete `README.md` with these sections:

```md
# NutriEvidence Agent

## Overview
## What the System Does
## What the System Does Not Do
## Architecture
## Dataset
## Recommender Methods
## Knowledge Graph Extension
## LLM Agentic Layer
## Evaluation
## Installation
## How to Run
## Example Queries
## Example Output
## Limitations
## Future Work
```

Include this safety statement:

```md
This system is designed for educational and research purposes only. It does not provide medical diagnosis, treatment recommendations, or clinical decision-making.
```

Include example CV bullet:

```md
Built an agentic RAG-based biomedical literature recommender using PubMed retrieval, semantic ranking, MeSH-based knowledge graph construction, node2vec graph embeddings, and LLM-generated citation-grounded explanations.
```

---

# 13. Implementation Order for Codex

Use this exact order:

1. Create repository structure.
2. Add `requirements.txt`, `.env.example`, and config loader.
3. Implement PubMed client and cache.
4. Create sample queries.
5. Implement preprocessing.
6. Implement semantic recommender.
7. Implement basic Streamlit UI using semantic recommender only.
8. Implement MeSH overlap baseline.
9. Implement graph builder.
10. Implement node2vec trainer.
11. Implement graph recommender.
12. Implement hybrid recommender.
13. Implement evaluation metrics.
14. Add optional LLM agents with non-LLM fallbacks.
15. Improve Streamlit UI to show semantic, graph, and hybrid recommendations.
16. Add scripts for dataset building, embeddings, graph, node2vec, and evaluation.
17. Write README.
18. Test end-to-end.

---

# 14. End-to-End Acceptance Criteria

The project is complete when:

- A user can run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

- The app loads cached articles from:

```text
data/pubmed_articles.json
```

- The user can enter a biomedical/nutrition research question.
- The app returns Top 5 recommended articles.
- The user can select an article and receive similar article recommendations.
- Semantic recommender works.
- MeSH overlap baseline works.
- Graph recommender works if graph artifacts are built.
- Hybrid recommender works if both semantic and graph models exist.
- Every recommendation includes:
  - title
  - PMID
  - year
  - journal
  - score
  - method
  - explanation
- The generated answer includes a safety disclaimer.
- Evaluation metrics are implemented.
- README explains the project clearly.

---

# 15. Minimal Demo Dataset Fallback

If PubMed retrieval is unavailable, create a small `data/pubmed_articles.json` manually with at least 10–20 sample records using the same schema.

The app must support this mode.

---

# 16. Important Safety Requirements

Do not implement features that:

- diagnose a user;
- recommend a treatment for a specific patient;
- tell the user to take or stop medication;
- interpret personal medical symptoms;
- replace medical professionals.

Allowed behavior:

- retrieve articles;
- summarize abstracts;
- recommend papers;
- explain why papers are relevant;
- describe limitations;
- include citations/PMIDs;
- provide educational research support.

---

# 17. Suggested Example Queries for Testing

Use these queries:

```text
nutrition risk in children with cerebral palsy
body composition assessment in cerebral palsy children
gut microbiome and Parkinson disease
gut microbiome and Alzheimer disease
Mediterranean diet during pregnancy and child neurodevelopment
maternal diet and child cognition
urbanization and dietary patterns
```

---

# 18. Suggested First Streamlit Layout

Use this structure in `app.py`:

```python
import streamlit as st

st.set_page_config(page_title="NutriEvidence Agent", layout="wide")

st.title("NutriEvidence Agent")
st.caption("Agentic RAG Recommender for Biomedical Literature")

st.warning(
    "Educational and research purposes only. "
    "Not medical advice, diagnosis, or treatment recommendation."
)

mode = st.sidebar.radio(
    "Choose mode",
    ["Search by question", "Recommend by article"]
)

# Load cached articles
# Load/build semantic recommender
# Optionally load graph recommender
# Render selected mode
```

---

# 19. Suggested Result Table Columns

For recommendation tables use:

```text
Rank
Title
PMID
Year
Journal
Score
Method
Shared MeSH Terms
Why recommended
```

---

# 20. Future Work

Add these as future improvements, not MVP requirements:

- Elasticsearch hybrid search
- FAISS or ChromaDB vector store
- MeSH hierarchy relations
- FoodOn ontology integration
- RDF triples and SPARQL queries
- LangGraph workflow
- GraphSAGE or GCN
- User feedback loop
- BibTeX export
- Markdown/CSV export
- Full-text open-access retrieval
- Risk-of-bias assessment

---

# 21. Definition of Done

The final project should demonstrate:

```text
Recommender Systems:
- semantic recommender
- graph-based recommender
- hybrid ranking
- evaluation metrics

Knowledge Base:
- MeSH-based Article–MeSH graph
- node2vec graph embeddings
- graph-based recommendations

LLM / Agentic AI:
- query planning
- evidence extraction
- recommendation explanations
- safety checking

Biomedical NLP:
- PubMed retrieval
- title + abstract processing
- MeSH metadata usage

Demo:
- Streamlit UI
- cached dataset fallback
- clear README
```

The first working version should prioritize correctness and simplicity over advanced infrastructure.
