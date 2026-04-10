"""
pipeline.py — RAG 推理主流程

用法：
  # 评测模式（跑 benchmark）
  python pipeline.py --strategy sentence

  # 生产模式（跑自定义输入）
  python pipeline.py --strategy paragraph --mode production \
      --input ../outputs/input_payload.json \
      --output ../outputs/output_payload.json

  # 手动指定索引路径（不依赖 strategy 目录结构）
  python pipeline.py \
      --index_path  ../data/vector_store/sentence/vector_store.index \
      --metadata_path ../data/vector_store/sentence/chunk_metadata.json
"""

import gc
import json
import argparse
from retriever import Retriever
from generator import create_generator

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class RAGInferencePipeline:
    def __init__(
        self,
        index_path,
        metadata_path,
        embed_model_name=EMBED_MODEL,
        llm_model_id=None,
        top_k=5,
        backend="local",
        api_key=None,
    ):
        self.retriever = Retriever(index_path, metadata_path, model_name=embed_model_name)

        gc.collect()
        self.generator = create_generator(
            backend=backend,
            model_id=llm_model_id,
            api_key=api_key,
        )
        self.top_k = top_k

        print("[Pipeline] All modules loaded.")

    def retrieve(self, query, top_k=None):
        return self.retriever.search(query, top_k=top_k or self.top_k)

    def retrieve_full(self, query, top_k=None):
        retrieved_context = self.retriever.search(query, top_k=top_k or self.top_k)
        context_texts = [c["text"] for c in retrieved_context]
        return retrieved_context, context_texts

    def _run(self, queries):
        results = []
        total = len(queries)

        for i, item in enumerate(queries):
            query    = item["query"]
            query_id = item["query_id"]

            print(f"[{i+1}/{total}] Retrieving — Query ID: {query_id}")
            retrieved_context = self.retriever.search(query, top_k=self.top_k)

            print(f"[{i+1}/{total}] Generating...")
            response = self.generator.generate(query, retrieved_context)

            results.append({
                "query_id":          query_id,
                "query":             query,
                "response":          response,
                "retrieved_context": retrieved_context,
            })
            print(f"[{i+1}/{total}] Done.\n")

        return results

    def run_benchmark_inference(self, benchmark_path, output_json_path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        print(f"\nStart benchmark inference on {len(benchmark_data)} samples...")
        results = self._run(benchmark_data)
        self._save(results, output_json_path)
        return {"results": results}

    def run_pipeline(self, input_json_path, output_json_path):
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        queries_list = input_data.get("queries", [])
        print(f"\nStart processing {len(queries_list)} queries...")
        results = self._run(queries_list)
        self._save(results, output_json_path)
        return {"results": results}

    def _save(self, results, output_json_path):
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_json_path}.")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Inference Pipeline")

    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=["fixed_size", "sentence", "paragraph"],
        help="使用哪种 chunking 策略对应的向量索引",
    )
    parser.add_argument("--index_path",    default=None, help="手动指定 FAISS 索引路径")
    parser.add_argument("--metadata_path", default=None, help="手动指定 chunk metadata 路径")
    parser.add_argument(
        "--vector_store_base",
        default="../data/vector_store",
        help="向量索引的父目录（各策略在其子目录下）",
    )
    parser.add_argument(
        "--mode",
        default="benchmark",
        choices=["benchmark", "production"],
        help="benchmark: 跑评测集；production: 跑自定义输入",
    )
    parser.add_argument(
        "--benchmark",
        default="../data/benchmark/rag_benchmark_dataset.json",
    )
    parser.add_argument(
        "--input",
        default="../outputs/input_payload.json",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="最终返回给生成模型的 chunk 数量（课程要求 at most 5）",
    )
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openrouter"],
        help="生成后端: local (本地 HF 模型) 或 openrouter (API)",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenRouter API Key (也可设置 OPENROUTER_API_KEY 环境变量)",
    )
    parser.add_argument(
        "--api_model",
        default=None,
        help="OpenRouter 模型 ID, 例如 openai/gpt-4o-mini, google/gemini-2.0-flash-001 等",
    )

    args = parser.parse_args()

    if args.index_path and args.metadata_path:
        index_path    = args.index_path
        metadata_path = args.metadata_path
    else:
        index_path    = f"{args.vector_store_base}/{args.strategy}/vector_store.index"
        metadata_path = f"{args.vector_store_base}/{args.strategy}/chunk_metadata.json"

    if args.output:
        output_path = args.output
    elif args.mode == "benchmark":
        output_path = f"../outputs/benchmark_output_{args.strategy}.json"
    else:
        output_path = f"../outputs/output_payload_{args.strategy}.json"

    print(f"策略:     {args.strategy}")
    print(f"嵌入模型: {EMBED_MODEL}")
    print(f"生成后端: {args.backend}")
    if args.api_model:
        print(f"API 模型: {args.api_model}")
    print(f"索引:     {index_path}")
    print(f"Top-K:    {args.top_k}")
    print(f"输出:     {output_path}\n")

    pipeline = RAGInferencePipeline(
        index_path=index_path,
        metadata_path=metadata_path,
        top_k=args.top_k,
        backend=args.backend,
        llm_model_id=args.api_model,
        api_key=args.api_key,
    )

    if args.mode == "benchmark":
        pipeline.run_benchmark_inference(
            benchmark_path=args.benchmark,
            output_json_path=output_path,
        )
    else:
        pipeline.run_pipeline(
            input_json_path=args.input,
            output_json_path=output_path,
        )