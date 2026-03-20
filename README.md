# COMP64702 – RAG Coursework: Mediterranean Culinary Assistant

## Overview

This repository implements a full **Retrieval-Augmented Generation (RAG)** pipeline for a Mediterranean cuisine question-answering system. It covers all three coursework phases and all five RAG components.

**Cuisine:** Mediterranean  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (as required)

---

## Project Structure

```
rag_project/
├── requirements.txt
├── data/
│   ├── corpus/
│   │   ├── mediterranean_corpus.json        # Deliverable 1
│   │   └── chunks_sentence.json
│   ├── benchmark/
│   │   └── rag_benchmark.json               # Deliverable 2
│   └── vector_store_minilm/                 # Built at runtime
├── outputs/
│   ├── evaluation_metrics.json
│   └── evaluation_detailed.json
└── notebooks/
    ├── build_corpus.py                      # Phase A – corpus builder
    ├── build_benchmark.py                   # Phase A – benchmark builder
    ├── chunking.py                          # Component 1 – Chunking
    ├── vectorisation.py                     # Component 2 – Vectorisation
    ├── retrieval_ranking.py                 # Component 3 – Retrieval & Ranking
    ├── generation.py                        # Component 4 – Prompting & Generation
    ├── evaluation.py                        # Component 5 – Evaluation metrics
    ├── inference_pipeline.py                # Deliverable 5 – Inference pipeline
    ├── evaluation_pipeline.py               # Evaluation pipeline
    ├── input_payload_sample.json            # Sample input
    └── output_payload_sample.json           # Sample output
```

> **Note on imports:** The component modules (`chunking.py`, `vectorisation.py`, etc.) must keep these exact names so that Python's import system can locate them. The pipeline scripts import them directly, e.g. `from vectorisation import VectorStore`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Phase A: Data Collection

### Step 1 – Build Background Corpus

```bash
cd rag_project/notebooks
python build_corpus.py
```

Scrapes **30 Wikipedia pages** and **10 Wikibooks pages** covering Mediterranean cuisine dishes, ingredients, techniques, and dietary traditions. All scraping is performed ethically:
- 1-second delay between requests
- Identifies as an academic coursework bot in the User-Agent header
- Uses only sources permitted by the coursework brief

Output: `data/corpus/mediterranean_corpus.json`

### Step 2 – Build RAG Benchmark Dataset

```bash
python build_benchmark.py
```

Creates **25 manually crafted Q&A pairs** spanning:
- Categories: dish, ingredient, technique, culture, nutrition
- Difficulties: easy (11), medium (11), hard (3)

Output: `data/benchmark/rag_benchmark.json`

---

## Phase B: RAG Components

All scripts below should be run from inside the `notebooks/` directory.

### Component 1 – Chunking

```bash
python chunking.py
```

Four strategies implemented and compared by running each through the full retrieval pipeline on the benchmark dataset and measuring MRR and Recall@5:

| Strategy | Description | Avg chunk size |
|----------|-------------|----------------|
| `fixed_size` | Sliding window with overlap | ~400 chars |
| `sentence` | Sentence-boundary splitting | ~350 chars |
| `paragraph` | Paragraph/section splitting | ~600 chars |
| `semantic` | Embedding-similarity grouping | variable |

To compare strategies and select the best one, run the retrieval evaluation for each and compare MRR / Recall@5 scores.

### Component 2 – Vectorisation / Embedding

```bash
python vectorisation.py
```

Three approaches implemented and compared under the same chunking and retrieval settings:

| Model | Type | Dim | Notes |
|-------|------|-----|-------|
| `tfidf` | Sparse | 50k | Fast, no GPU needed |
| `bm25` | Probabilistic | — | Lexical baseline |
| `all-MiniLM-L6-v2` | Dense | 384 | Semantic embeddings |

To compare models, build a separate vector store for each and evaluate with `evaluate_retrieval()` from `evaluation.py`.

### Component 3 – Retrieval and Ranking

```bash
python retrieval_ranking.py
```

Four strategies implemented:

| Strategy | Description |
|----------|-------------|
| `dense` | Cosine similarity on sentence embeddings |
| `bm25` | Probabilistic lexical scoring |
| `hybrid` | RRF fusion of dense + BM25 |
| `rerank` | Cross-encoder re-ranking on top candidates |

Returns **at most 5 chunks** (coursework requirement).

### Component 4 – Prompting and Generation

```bash
python generation.py   # prints prompt examples without loading the model
```

Model: `Qwen/Qwen2.5-0.5B-Instruct`

Four prompting strategies implemented:

| Strategy | Description |
|----------|-------------|
| `basic` | Instruction + context + question |
| `chain_of_thought` | Step-by-step reasoning prompt |
| `few_shot` | 2-shot examples + context + question |
| `structured` | JSON-output prompt |

### Component 5 – Evaluation

```bash
python evaluation.py   # runs a sanity check on sample predictions
```

Metrics implemented:

**Retrieval:** MRR, Recall@K, Precision@K, NDCG@K

**Generation:** Exact Match, F1 token overlap, ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore F1 (optional)

---

## Running the Inference Pipeline (Deliverable 5)

```bash
cd rag_project/notebooks
python inference_pipeline.py \
    --input  input_payload_sample.json \
    --output output_payload.json \
    --vector_store ../data/vector_store_minilm \
    --strategy few_shot \
    --retrieval hybrid \
    --top_k 5
```

**Input format:**
```json
{
  "queries": [
    {"query_id": "0", "query": "What are the main ingredients of hummus?"}
  ]
}
```

**Output format:**
```json
{
  "results": [
    {
      "query_id": "0",
      "query": "What are the main ingredients of hummus?",
      "response": "Hummus is made from chickpeas, tahini, lemon juice, garlic, and salt.",
      "retrieved_context": [
        {"doc_id": "0_2", "text": "Hummus is a Middle Eastern dip made from..."}
      ]
    }
  ]
}
```

---

## Running the Evaluation Pipeline

```bash
# Evaluate pre-generated answers:
python evaluation_pipeline.py \
    --benchmark ../data/benchmark/rag_benchmark.json \
    --predictions output_payload.json \
    --vector_store ../data/vector_store_minilm

# End-to-end (generates + evaluates, loads the Qwen model):
python evaluation_pipeline.py --end_to_end
```

---

## Ethical and Responsible Data Collection

- Only sources listed in the coursework brief were used (Wikipedia, Wikibooks)
- Polite scraping: 1-second delay, academic User-Agent header
- No personal data collected
- All data is freely licensed under Creative Commons (Wikipedia/Wikibooks)
- The benchmark Q&A pairs were manually created with reference to the source URLs

---

## Team Contributions

| Task | Owner |
|------|-------|
| Corpus & Benchmark | All members |
| Chunking | Member A |
| Vectorisation | Member B |
| Retrieval & Ranking | Member C |
| Generation & Evaluation | Member D |
| Integration & Report | All members |
