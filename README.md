# COMP64702 – RAG Coursework: East Asian Culinary Assistant

## Overview

This repository implements a complete **Retrieval-Augmented Generation (RAG)** pipeline for an East Asian cuisine question-answering system. The project covers all three coursework phases and all five RAG components, from corpus collection through to end-to-end evaluation.

**Cuisine Domain:** East Asian (Chinese, Japanese, Korean, Taiwanese, Hong Kong)
**Generator Model:** `Qwen/Qwen2.5-0.5B-Instruct` (local) or any model via [OpenRouter API](https://openrouter.ai/models)
**Embedding Model:** `BAAI/bge-small-en-v1.5`
**Vector Index:** FAISS (`IndexFlatIP` with cosine similarity)
**Retrieval Strategy:** Hybrid dense + BM25 with RRF fusion and cross-encoder reranking

---

## Project Structure

```
rag_project/
├── .env                                        # API key configuration (git-ignored)
├── .gitignore
├── requirements.txt
├── data/
│   ├── corpus/
│   │   ├── east_asian_corpus.json              # Deliverable 1 – raw scraped corpus
│   │   ├── fixed_size/chunked_corpus.json      # Chunked with fixed-size strategy
│   │   ├── sentence/chunked_corpus.json        # Chunked with sentence strategy
│   │   └── paragraph/chunked_corpus.json       # Chunked with paragraph strategy
│   ├── benchmark/
│   │   └── rag_benchmark_dataset.json          # Deliverable 2 – Q&A benchmark
│   └── vector_store/
│       ├── fixed_size/                         # FAISS index for fixed-size chunks
│       │   ├── vector_store.index
│       │   └── chunk_metadata.json
│       ├── sentence/                           # FAISS index for sentence chunks
│       │   ├── vector_store.index
│       │   └── chunk_metadata.json
│       └── paragraph/                          # FAISS index for paragraph chunks
│           ├── vector_store.index
│           └── chunk_metadata.json
├── outputs/
│   ├── input_payload.json                      # Sample input payload
│   ├── output_payload.json                     # Sample output payload
│   ├── benchmark_output.json                   # Benchmark results (sentence strategy)
│   ├── benchmark_output_fixed_size.json        # Benchmark results (fixed-size strategy)
│   ├── benchmark_output_paragraph.json         # Benchmark results (paragraph strategy)
│   ├── evaluation_metrics.json                 # Aggregated metrics
│   └── evaluation_detailed.json                # Per-sample evaluation results
└── notebooks/
    ├── build_corpus.py                         # Phase A – corpus builder
    ├── construct_benchmark.py                  # Phase A – benchmark helper
    ├── chunking.py                             # Component 1 – Chunking
    ├── embedding.py                            # Component 2 – Vectorisation
    ├── retriever.py                            # Component 3 – Retrieval
    ├── generator.py                            # Component 4 – Generation (local + OpenRouter)
    ├── eval.py                                 # Component 5 – Evaluation
    └── pipeline.py                             # Inference pipeline (Deliverable 5)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies include `transformers`, `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `rouge-score`, `bert-score`, `torch`, `beautifulsoup4`, `requests`, and `python-dotenv`.

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

Sources are split across three categories, each with its own discovery strategy. Wikipedia uses anchor-based section extraction from the List of Asian Cuisines index page, supplemented by a set of hardcoded extra pages for important articles that may be missed automatically. Wikibooks uses a two-layer crawl: first discovering cuisine-category pages from the Cookbook:Cuisines index, then following links within each category to individual recipe pages. The blog uses a set of hardcoded category URLs as the primary source, with a keyword-based fallback scan of the homepage sidebar. Content extraction is handled by separate functions for MediaWiki pages (`clean_mediawiki`) and WordPress blog posts (`clean_blog`), each targeting the relevant HTML containers and stripping noise such as navboxes, infoboxes, and reference lists.

Output: `data/corpus/east_asian_corpus.json`

### Step 2 – Sample Chunks for Benchmark Construction

```bash
python construct_benchmark.py
```

Randomly samples 30 chunks from a chunked corpus file and writes them to a text file for manual review. This is a helper used during benchmark creation rather than a fully automated step.

Output: `data/benchmark/chunks_for_prompt.txt`

---

## Phase B: RAG Components

All scripts below should be run from inside the `notebooks/` directory.

### Component 1 – Chunking (`chunking.py`)

Implements three text chunking strategies. All strategies preserve `source` and `chunk_id` metadata alongside the chunk text.

**Strategy 1 – Fixed Size** (`chunk_fixed_size`): Splits text into overlapping windows of a fixed word count. A configurable overlap between consecutive windows preserves cross-boundary context.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 200 | Number of words per chunk |
| `overlap` | 30 | Word overlap between consecutive chunks |

**Strategy 2 – Sentence** (`chunk_sentence`): Accumulates sentences until a character limit is reached, then starts a new chunk. A configurable number of trailing sentences are carried over into the next chunk as overlap.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 800 | Maximum characters per chunk |
| `overlap_sentences` | 1 | Number of sentences to carry over |

**Strategy 3 – Paragraph** (`chunk_paragraph`): Splits on double newlines. Short paragraphs below a minimum character threshold are merged with subsequent paragraphs. Oversized paragraphs that exceed the character limit fall back to the sentence strategy for further subdivision.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 1200 | Maximum characters per chunk |
| `min_chars` | 100 | Minimum paragraph size before merging |

```bash
python chunking.py                       # default: sentence strategy
python chunking.py --strategy fixed_size
python chunking.py --strategy paragraph
python chunking.py --strategy all        # builds all three strategies
```

Each strategy outputs its chunks to a subdirectory under `data/corpus/`.

### Component 2 – Vectorisation (`embedding.py`)

Encodes all chunks using `BAAI/bge-small-en-v1.5` with `normalize_embeddings=True`, then builds a FAISS `IndexFlatIP` index. Because embeddings are L2-normalised, inner product is equivalent to cosine similarity, so higher scores indicate closer matches.

```bash
python embedding.py                       # default: sentence strategy
python embedding.py --strategy paragraph
python embedding.py --strategy all        # builds indexes for all three strategies
```

Output per strategy: `data/vector_store/{strategy}/vector_store.index` and `chunk_metadata.json`.

### Component 3 – Retrieval (`retriever.py`)

The `Retriever` class implements a three-stage pipeline:

**Stage 1 – Dual recall.** The query is sent to two independent retrieval systems in parallel. BGE dense retrieval encodes the query with a task-specific prefix (`"Represent this sentence for searching relevant passages: ..."`) and performs approximate nearest-neighbour search against the FAISS index. BM25 sparse retrieval tokenises the query and scores all chunks using term frequency. Both systems return the top `recall_k=20` candidates.

**Stage 2 – RRF fusion.** Reciprocal Rank Fusion combines the two ranked lists using the formula `score = 1/(k + rank_dense) + 1/(k + rank_bm25)`, with `k=60`. This merges lexical and semantic signals without requiring score normalisation, and the top 10 candidates are passed forward.

**Stage 3 – Cross-encoder reranking.** A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder scores each `(query, chunk_text)` pair jointly, producing a more precise relevance estimate than the bi-encoder embedding. The final `top_k` (default 5) highest-scoring chunks are returned.

Both the embedding model and cross-encoder run on CPU, reserving GPU memory for the LLM.

```python
retriever = Retriever(
    "../data/vector_store/sentence/vector_store.index",
    "../data/vector_store/sentence/chunk_metadata.json",
)
chunks = retriever.search("What is wok hei?", top_k=5)
# Each result: {"doc_id": ..., "text": ..., "rrf_score": ..., "ce_score": ...}
```

### Component 4 – Generation (`generator.py`)

The generator module supports two backends, selectable via the `create_generator()` factory function:

**Backend 1 – Local (`LocalGenerator`):** Loads `Qwen/Qwen2.5-0.5B-Instruct` in half-precision (`float16`) with `device_map="auto"` and `low_cpu_mem_usage=True` to keep peak memory low during loading. Generation uses `max_new_tokens=64`, `temperature=0.0`, and greedy decoding (`do_sample=False`).

**Backend 2 – OpenRouter (`OpenRouterGenerator`):** Calls the [OpenRouter API](https://openrouter.ai/models) to use any supported remote model (e.g. `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`, `anthropic/claude-3.5-sonnet`). The API key is read from the `.env` file or the `OPENROUTER_API_KEY` environment variable.

Both backends share the same prompt construction and post-processing logic. Retrieved chunks are concatenated with `---` separators and injected into a system prompt that instructs the model to answer only from the provided context and to respond exclusively in English. A post-processing step strips any non-ASCII characters from the output to enforce the English-only constraint.

```python
# Local model (default)
from generator import create_generator
gen = create_generator(backend="local")
answer = gen.generate(query, retrieved_chunks)

# OpenRouter API
gen = create_generator(backend="openrouter", model_id="openai/gpt-4o-mini")
answer = gen.generate(query, retrieved_chunks)
```

### Component 5 – Evaluation (`eval.py`)

The `RAGEvaluator` class computes both retrieval and generation metrics. All generation metrics are computed on normalised text (lowercased, punctuation removed) to reduce sensitivity to minor formatting differences.

**Retrieval metrics:**

| Metric | Description |
|--------|-------------|
| Hit Rate @ K | Fraction of queries where a relevant chunk appears in the top-K results |
| MRR | Mean Reciprocal Rank – rewards earlier hits with higher scores |

Hit detection uses a dual strategy: it first checks for an exact `chunk_id` match, and falls back to checking whether the normalised gold answer text is contained within a retrieved chunk's normalised text. This makes the metric robust across different chunking strategies where the same content may have different chunk IDs.

**Generation metrics:**

| Metric | Description |
|--------|-------------|
| ROUGE-1 / 2 / L | Unigram, bigram, and longest-common-subsequence overlap with gold answers |
| BLEU | N-gram precision with smoothing (Method 1) |
| BERTScore F1 | Semantic similarity using contextual embeddings (`lang="en"`) |

```bash
python eval.py
```

The script reads from `outputs/benchmark_output_fixed_size.json` and `data/benchmark/rag_benchmark_dataset.json` by default. Edit the `__main__` block to point at a different output file. Missing queries (present in the benchmark but absent from the output) are flagged individually.

---

## Running the Inference Pipeline (Deliverable 5)

`pipeline.py` wires all five components into a single end-to-end pipeline. It supports two modes and two generator backends.

### Benchmark evaluation mode

```bash
cd rag_project/notebooks

# Using local model (default)
python pipeline.py --strategy sentence
python pipeline.py --strategy fixed_size
python pipeline.py --strategy paragraph

# Using OpenRouter API
python pipeline.py --strategy sentence --backend openrouter --api_model openai/gpt-4o-mini
```

Reads the benchmark dataset, runs inference without exposing gold answers to the model, and writes results to `outputs/benchmark_output_{strategy}.json`.

### Production mode (arbitrary queries)

```bash
# Local model
python pipeline.py --strategy sentence --mode production \
    --input ../outputs/input_payload.json \
    --output ../outputs/output_payload.json

# OpenRouter API
python pipeline.py --strategy sentence --mode production \
    --backend openrouter --api_model google/gemini-2.0-flash-001 \
    --input ../outputs/input_payload.json \
    --output ../outputs/output_payload.json
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

**Output format:**
```json
{
  "results": [
    {
      "query_id": "q1",
      "query": "What is wok hei?",
      "response": "Wok hei refers to the caramelised flavour...",
      "retrieved_context": [
        {
          "doc_id": "https://..._chunk_7",
          "text": "...",
          "rrf_score": 0.0315,
          "ce_score": 4.21
        }
      ]
    }
  ]
}
```

### Manually specifying index paths

```bash
python pipeline.py \
    --index_path  ../data/vector_store/sentence/vector_store.index \
    --metadata_path ../data/vector_store/sentence/chunk_metadata.json
```

When `--index_path` and `--metadata_path` are both provided, the `--strategy` argument is ignored.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--strategy` | `sentence` | Chunking strategy index to use (`fixed_size`, `sentence`, `paragraph`) |
| `--mode` | `benchmark` | `benchmark` or `production` |
| `--top_k` | `5` | Number of chunks passed to the generator (course requirement: at most 5) |
| `--backend` | `local` | Generator backend: `local` (HuggingFace) or `openrouter` (API) |
| `--api_model` | `openai/gpt-4o-mini` | OpenRouter model ID (see [model list](https://openrouter.ai/models)) |
| `--api_key` | from `.env` | OpenRouter API key (overrides `.env` and env var) |
| `--benchmark` | `../data/benchmark/rag_benchmark_dataset.json` | Path to benchmark file |
| `--input` | `../outputs/input_payload.json` | Path to production input file |
| `--output` | auto-derived | Path to write results |

---

## API Key Configuration

To use the OpenRouter backend, you need an API key from [openrouter.ai](https://openrouter.ai/).

**Option 1 – `.env` file (recommended):** Edit the `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

The `.env` file is git-ignored and will be loaded automatically when the generator module is imported.

**Option 2 – Environment variable:**

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

**Option 3 – CLI flag:**

```bash
python pipeline.py --backend openrouter --api_key "sk-or-v1-your-key-here" --api_model openai/gpt-4o-mini
```

Priority: CLI flag > `.env` file / environment variable.

---

## Evaluating Results

```bash
python eval.py
```

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

Representative query–response pairs from the benchmark output files.

| Query | Response | Hit? |
|-------|----------|------|
| According to the JRO, how many Japanese restaurants were there in Thailand as of June 2012? | 1,676 in June 2012, a 2.2-fold increase from 2007 | ✅ |
| What health risk has been associated with frequent kimchi consumption? | An increased risk of gastric cancer was found in some case-control studies | ✅ |
| What saying in 1970s Hong Kong manifested the nouveau riche mentality? | "Mixing shark fin soup with rice" | ✅ |
| What is the predominant food source for the Yamis and Thao tribes in Taiwan? | Fish | ✅ |
| What do the Chinese words 'la' and 'mian' literally mean in Lamian? | 'La' means to pull or stretch; 'mian' means noodle | ✅ |
| Which ruling class in the Goryeo period forbade beef consumption? | The Buddhist ruling class | ✅ |

---

## Ethical and Responsible Data Collection

- Only sources listed in the coursework brief were scraped (Wikipedia, Wikibooks, approved food blogs)
- Polite scraping: 1-second delay between requests, academic bot User-Agent header
- No personal data was collected at any point
- Wikipedia and Wikibooks content is freely licensed under Creative Commons (CC BY-SA)
- The benchmark Q&A pairs were manually written with reference to specific source URLs
- The LLM is instructed to answer only from retrieved context, reducing hallucination risk