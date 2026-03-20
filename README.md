# COMP64702 – RAG Coursework: East Asian Culinary Assistant

## Overview

This repository implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for an East Asian cuisine question-answering system. The project covers all three coursework phases and all five RAG components, from corpus collection to end-to-end evaluation.

**Cuisine Domain:** East Asian (Chinese, Japanese, Korean, Taiwanese, Hong Kong)  
**Generator Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Embedding Model:** `all-MiniLM-L6-v2`  
**Vector Index:** FAISS (L2 distance)

---

## Project Structure

```
rag_project/
├── requirements.txt
├── data/
│   ├── corpus/
│   │   ├── east_asian_corpus.json          # Deliverable 1 – raw scraped corpus
│   │   └── chunked_corpus.json             # Chunked version for indexing
│   ├── benchmark/
│   │   └── rag_benchmark_dataset.json      # Deliverable 2 – Q&A benchmark
│   └── vector_store_minilm/                # Built at runtime
│       ├── vector_store.index              # FAISS index
│       └── chunk_metadata.json             # Chunk text + source metadata
├── outputs/
│   ├── input_payload.json                  # Sample input payload
│   ├── output_payload.json                 # Sample output payload
│   ├── benchmark_output.json               # Inference results on benchmark
│   ├── evaluation_metrics.json             # Aggregated metrics
│   └── evaluation_detailed.json           # Per-sample evaluation results
└── notebooks/
    ├── build_corpus.py                     # Phase A – corpus builder
    ├── construct_benchmark.py              # Phase A – benchmark helper
    ├── chunking.py                         # Component 1 – Chunking
    ├── embedding.py                        # Component 2 – Vectorisation
    ├── retriever.py                        # Component 3 – Retrieval
    ├── generator.py                        # Component 4 – Generation
    ├── eval.py                             # Component 5 – Evaluation
    └── pipeline.py                         # Inference pipeline (Deliverable 5)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies include `transformers`, `sentence-transformers`, `faiss-cpu`, `rouge-score`, `bert-score`, `torch`, `beautifulsoup4`, and `requests`.

---

## Phase A: Data Collection

### Step 1 – Build Background Corpus

```bash
cd rag_project/notebooks
python build_corpus.py
```

Scrapes Wikipedia, Wikibooks, and the Around the World in 80 Cuisines blog, targeting pages covering Chinese (northern/southern), Japanese, Korean, Taiwanese, and Hong Kong cuisines. Ethical scraping practices are followed:

- 1-second delay between all requests
- Academic User-Agent header identifying the project
- Only publicly accessible, freely licensed sources are used

Output: `data/corpus/east_asian_corpus.json`

### Step 2 – Chunk the Corpus

```bash
python chunking.py
```

Applies a sliding-window word-count chunking strategy (chunk size: 200 words, overlap: 30 words) to all documents in the corpus.

Output: `data/corpus/chunked_corpus.json`

---

## Phase B: RAG Components

All scripts below should be run from inside the `notebooks/` directory.

### Component 1 – Chunking (`chunking.py`)

Implements fixed-size word-level chunking with configurable `chunk_size` and `overlap` parameters.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 200 | Number of words per chunk |
| `overlap` | 30 | Word overlap between consecutive chunks |

### Component 2 – Vectorisation (`embedding.py`)

Encodes all chunks using `sentence-transformers/all-MiniLM-L6-v2` and builds a FAISS `IndexFlatL2` index for efficient nearest-neighbour search.

```bash
python embedding.py
```

Output: `data/vector_store_minilm/vector_store.index` and `chunk_metadata.json`

### Component 3 – Retrieval (`retriever.py`)

The `Retriever` class loads the FAISS index and embedding model at initialisation (embedding runs on CPU to preserve GPU memory for the LLM). At query time it encodes the question and returns the top-K most similar chunks by L2 distance.

```python
retriever = Retriever(
    "../data/vector_store_minilm/vector_store.index",
    "../data/vector_store_minilm/chunk_metadata.json",
)
chunks = retriever.search("What is wok hei?", top_k=5)
```

Each result contains `doc_id`, `text`, and `score`.

### Component 4 – Generation (`generator.py`)

The `Generator` class loads `Qwen/Qwen2.5-0.5B-Instruct` in half-precision (`float16`) with `device_map="auto"`. Retrieved chunks are concatenated into a context block and injected into a system prompt that instructs the model to answer only from the provided context.

```python
generator = Generator()
answer = generator.generate(query, retrieved_chunks)
```

Prompt design: the model is instructed to act as an expert East Asian culinary assistant and to respond only from the provided context, falling back to "I do not know" when the answer is absent.

### Component 5 – Evaluation (`eval.py`)

The `RAGEvaluator` class computes both retrieval and generation metrics.

**Retrieval metrics:**

| Metric | Description |
|--------|-------------|
| Hit Rate @ K | Fraction of queries where the gold chunk appears in the top-K results |
| MRR | Mean Reciprocal Rank – rewards earlier hits |

**Generation metrics:**

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | N-gram overlap with gold answers |
| BLEU | N-gram precision (smoothed) |
| BERTScore F1 | Semantic similarity using contextual embeddings |

```bash
python eval.py
```

---

## Running the Inference Pipeline (Deliverable 5)

The `pipeline.py` script wires together all five components into a single end-to-end pipeline.

### Production mode (arbitrary queries)

```bash
cd rag_project/notebooks
python pipeline.py
# Edit the __main__ block to call run_pipeline() with your input file
```

**Input format (`outputs/input_payload.json`):**
```json
{
  "queries": [
    {"query_id": "q1", "query": "What is wok hei?"},
    {"query_id": "q2", "query": "How is kimchi fermented?"}
  ]
}
```

**Output format (`outputs/output_payload.json`):**
```json
{
  "results": [
    {
      "query_id": "q1",
      "query": "What is wok hei?",
      "response": "Wok hei (breath of the wok) refers to ...",
      "retrieved_context": [
        {"doc_id": "https://..._chunk_7", "text": "...", "score": 0.42}
      ]
    }
  ]
}
```

### Benchmark evaluation mode

```bash
python pipeline.py
# Edit __main__ to call run_benchmark_inference()
```

This reads questions from the benchmark dataset, runs inference (without exposing gold answers to the model), and writes results to `outputs/benchmark_output.json`.

---

## Evaluating Results

```bash
python eval.py
```

Reads `outputs/benchmark_output.json` and `data/benchmark/rag_benchmark_dataset.json`, computes all metrics, and prints a summary report. Missed queries (those in the benchmark but absent from the output) are flagged individually.

Sample output:
```
===================================
Final Evaluation Report
-----------------------------------
Evaluated Samples : 30
Missing  Samples  : 0

[ Retrieval ]
  Hit Rate @ 10   : 86.67%
  MRR             : 0.6912

[ Generation ]
  ROUGE-1         : 0.3841
  ROUGE-2         : 0.1934
  ROUGE-L         : 0.3102
  BLEU            : 0.1587
  BERTScore F1    : 0.8741
===================================
```

---

## Sample Results

The table below shows representative query–response pairs from `outputs/benchmark_output.json`.

| Query | Response | Hit? |
|-------|----------|------|
| According to the JRO, how many Japanese restaurants were there in Thailand as of June 2012? | 1,676 in June 2012, a 2.2-fold increase from 2007 | ✅ |
| What health risk has been associated with frequent kimchi consumption? | An increased risk of gastric cancer was found in some case-control studies | ✅ |
| What saying in 1970s Hong Kong manifested the nouveau riche mentality? | "Mixing shark fin soup with rice" | ✅ |
| What is the predominant food source for the Yamis and Thao tribes in Taiwan? | Fish | ✅ |
| In which city in Northern England did salt and pepper chips originate? | Liverpool | ✅ |

---

## Ethical and Responsible Data Collection

- Only sources listed in the coursework brief were scraped (Wikipedia, Wikibooks, approved food blogs)
- Polite scraping: 1-second delay between requests, academic bot User-Agent header
- No personal data was collected at any point
- Wikipedia and Wikibooks content is freely licensed under Creative Commons (CC BY-SA)
- The benchmark Q&A pairs were manually written with reference to specific source URLs
- The LLM is instructed to answer only from retrieved context, reducing hallucination risk

---

## Team Contributions

| Task | Owner |
|------|-------|
| Corpus collection & cleaning | All members |
| Benchmark dataset creation | All members |
| Chunking (`chunking.py`) | Member A |
| Vectorisation (`embedding.py`) | Member B |
| Retrieval (`retriever.py`) | Member C |
| Generation (`generator.py`) | Member D |
| Evaluation (`eval.py`) | Member D |
| Pipeline integration (`pipeline.py`) | All members |
| Report writing | All members |