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
│   │   └── semantic/chunked_corpus.json        # Chunked with semantic strategy
│   ├── benchmark/
│   │   └── rag_benchmark_dataset.json          # Deliverable 2 – Q&A benchmark
│   └── vector_store/
│       └── semantic/                           # FAISS index for semantic chunks
│           ├── vector_store.index
│           └── chunk_metadata.json
├── outputs/
│   ├── input_payload.json                      # Sample input payload
│   ├── output_payload.json                     # Sample output payload
│   ├── semantic/benchmark_output_*.json        # Benchmark results
│   ├── evaluation_metrics.json                 # Aggregated metrics
│   └── evaluation_detailed.json                # Per-sample evaluation results
└── notebooks/
    ├── build_corpus.py                         # Phase A – corpus builder
    ├── construct_benchmark.py                  # Phase A – benchmark helper
    ├── chunking.py                             # Component 1 – Semantic Chunking
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

### Component 1 – Semantic Chunking (`chunking.py`)

Implements a semantic chunking strategy that splits documents at points where the topic shifts, rather than at fixed character or word boundaries. This produces chunks where each chunk is a semantically coherent unit, improving retrieval quality. The `source` and `chunk_id` metadata are preserved alongside the chunk text.

**Algorithm:**

1. Split each document into sentences using regex-based sentence boundary detection.
2. Encode every sentence using `BAAI/bge-small-en-v1.5` to obtain a dense embedding.
3. Compute the cosine similarity between each pair of adjacent sentence embeddings.
4. Convert similarities to distances (`1 - similarity`) and apply a percentile-based dynamic threshold to identify semantic breakpoints — positions where the topic changes most sharply.
5. Split the sentence sequence at these breakpoints to form raw chunks.
6. Post-process: merge chunks shorter than `min_chars` with their neighbours; subdivide chunks longer than `max_chars` at sentence boundaries.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `breakpoint_percentile` | 85 | Semantic distance percentile threshold. Higher = fewer, larger chunks. Lower = more, smaller chunks. |
| `max_chars` | 1500 | Maximum characters per chunk; oversized chunks are subdivided at sentence boundaries |
| `min_chars` | 100 | Minimum characters per chunk; undersized chunks are merged with neighbours |

```bash
python chunking.py                                  # default parameters
python chunking.py --breakpoint_percentile 80        # more aggressive splitting
python chunking.py --max_chars 1200 --min_chars 150  # tighter length constraints
```

Output: `data/corpus/semantic/chunked_corpus.json`

### Component 2 – Vectorisation (`embedding.py`)

Encodes all semantic chunks using `BAAI/bge-small-en-v1.5` with `normalize_embeddings=True`, then builds a FAISS `IndexFlatIP` index. Because embeddings are L2-normalised, inner product is equivalent to cosine similarity, so higher scores indicate closer matches.

```bash
python embedding.py
```

Output: `data/vector_store/semantic/vector_store.index` and `chunk_metadata.json`.

### Component 3 – Retrieval (`retriever.py`)

The `Retriever` class implements a three-stage pipeline:

**Stage 1 – Dual recall.** The query is sent to two independent retrieval systems in parallel. BGE dense retrieval encodes the query with a task-specific prefix (`"Represent this sentence for searching relevant passages: ..."`) and performs approximate nearest-neighbour search against the FAISS index. BM25 sparse retrieval tokenises the query and scores all chunks using term frequency. Both systems return the top `recall_k=20` candidates.

**Stage 2 – RRF fusion.** Reciprocal Rank Fusion combines the two ranked lists using the formula `score = 1/(k + rank_dense) + 1/(k + rank_bm25)`, with `k=60`. This merges lexical and semantic signals without requiring score normalisation, and the top 10 candidates are passed forward.

**Stage 3 – Cross-encoder reranking.** A `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder scores each `(query, chunk_text)` pair jointly, producing a more precise relevance estimate than the bi-encoder embedding. The final `top_k` (default 5) highest-scoring chunks are returned.

Both the embedding model and cross-encoder run on CPU, reserving GPU memory for the LLM.

```python
retriever = Retriever(
    "../data/vector_store/semantic/vector_store.index",
    "../data/vector_store/semantic/chunk_metadata.json",
)
chunks = retriever.search("What is wok hei?", top_k=5)
# Each result: {"doc_id": ..., "text": ..., "rrf_score": ..., "ce_score": ...}
```

### Component 4 – Generation (`generator.py`)

The generator module supports two backends, selectable via the `create_generator()` factory function:

**Backend 1 – Local (`LocalGenerator`):** Loads `Qwen/Qwen2.5-0.5B-Instruct` in half-precision (`float16`) with `device_map="auto"` and `low_cpu_mem_usage=True` to keep peak memory low during loading. Generation uses `max_new_tokens=64`, `temperature=0.0`, and greedy decoding (`do_sample=False`).

**Backend 2 – OpenRouter (`OpenRouterGenerator`):** Calls the [OpenRouter API](https://openrouter.ai/models) to use any supported remote model (e.g. `openai/gpt-4o-mini`, `google/gemini-2.0-flash-001`, `anthropic/claude-3.5-sonnet`). The API key is read from the `.env` file or the `OPENROUTER_API_KEY` environment variable.

Both backends share the same prompt construction and post-processing logic. Retrieved chunks are concatenated with `---` separators and injected into a system prompt that instructs the model to answer only from the provided context and to respond in the same language as the user's question.

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
python eval.py                                                # 自动扫描 outputs/semantic/ 下最新的推理输出
python eval.py --inference ../outputs/semantic/benchmark_output_gpt-4o-mini.json  # 指定文件
python eval.py --output ../outputs/evaluation_metrics.json    # 保存指标为 JSON
```

By default, the script automatically discovers the most recent `benchmark_output_*.json` file under `outputs/semantic/`. A specific file can be passed via the `--inference` flag. Missing queries (present in the benchmark but absent from the output) are flagged individually.

| Argument | Default | Description |
|----------|---------|-------------|
| `--inference` | auto-discover | Path to inference output JSON |
| `--benchmark` | `../data/benchmark/rag_benchmark_dataset.json` | Path to benchmark file |
| `--output` | none | If set, saves aggregated metrics to this JSON path |

---

## Running the Pipeline

### One-click script (`inference.sh`)

`inference.sh` is the recommended way to run the pipeline. It executes inference followed by automatic evaluation in a single command.

```bash
./inference.sh                            # 推理 + 自动评估
./inference.sh --no_eval                  # 只推理，不评估
./inference.sh --eval_only                # 只评估（跳过推理，使用最新输出）
./inference.sh --eval_output metrics.json # 评估结果保存为 JSON
```

#### `inference.sh` CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--top_k` | `5` | Number of chunks passed to the generator |
| `--backend` | `openrouter` | Generator backend: `local` or `openrouter` |
| `--api_model` | from `.env` | OpenRouter model ID |
| `--api_key` | from `.env` | OpenRouter API key |
| `--mode` | `benchmark` | `benchmark` or `production` |
| `--benchmark` | `../data/benchmark/rag_benchmark_dataset.json` | Path to benchmark file |
| `--input` | `../outputs/input_payload.json` | Path to production input file |
| `--output` | auto-derived | Path to write inference results |
| `--no_eval` | — | Skip evaluation after inference |
| `--eval_only` | — | Skip inference, run evaluation on latest output |
| `--eval_output` | — | Save evaluation metrics to this JSON path |

---

### Running `pipeline.py` directly

`pipeline.py` wires all five components into a single end-to-end pipeline. It supports two modes and two generator backends.

### Benchmark evaluation mode

```bash
cd rag_project/notebooks

# Using local model (default)
python pipeline.py

# Using OpenRouter API
python pipeline.py --backend openrouter --api_model openai/gpt-4o-mini
```

Reads the benchmark dataset, runs inference without exposing gold answers to the model, and writes results to `outputs/semantic/benchmark_output_{model}.json`.

### Production mode (arbitrary queries)

```bash
# Local model
python pipeline.py --mode production \
    --input ../outputs/input_payload.json \
    --output ../outputs/output_payload.json

# OpenRouter API
python pipeline.py --mode production \
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
    --index_path  ../data/vector_store/semantic/vector_store.index \
    --metadata_path ../data/vector_store/semantic/chunk_metadata.json
```

When `--index_path` and `--metadata_path` are both provided, the default index path is overridden.

### CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
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

Evaluation runs automatically when using `inference.sh` in benchmark mode. To run manually:

```bash
cd rag_project/notebooks

# Auto-discover latest output
python eval.py

# Specify a file and save metrics
python eval.py --inference ../outputs/semantic/benchmark_output_gpt-4o-mini.json \
               --output ../outputs/evaluation_metrics.json

# Or use the shell script
./inference.sh --eval_only
./inference.sh --eval_only --eval_output ../outputs/evaluation_metrics.json
```

Sample output:
```
===================================
Final Evaluation Report
-----------------------------------
Inference File    : benchmark_output_gpt-4o-mini.json
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