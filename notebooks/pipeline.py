import gc
import json
from retriever import Retriever
from generator import Generator


class RAGInferencePipeline:
    def __init__(
        self,
        index_path,
        metadata_path,
        embed_model_name="all-MiniLM-L6-v2",
        llm_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        top_k=10,
    ):
        # 检索模块
        self.retriever = Retriever(index_path, metadata_path, model_name=embed_model_name)

        # 生成模块（LLM 加载前先释放内存）
        gc.collect()
        self.generator = Generator(model_id=llm_model_id)
        self.top_k = top_k

        print("[Pipeline] All modules loaded.")

    def retrieve(self, query, top_k=None):
        """ 供外部直接调用 """
        return self.retriever.search(query, top_k=top_k or self.top_k)

    def retrieve_full(self, query, top_k=None):
        """ eval.py 调用的接口，返回 (retrieved_context, context_texts) """
        retrieved_context = self.retriever.search(query, top_k=top_k or self.top_k)
        context_texts = [c["text"] for c in retrieved_context]
        return retrieved_context, context_texts

    def _run(self, queries):
        """ 内部公共推理循环，queries 为 [{"query_id", "query"}, ...] """
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
        """
        评测专用：从 benchmark 文件读取问题推理，输出供 eval.py 使用。
        gold_answer / source_chunk_id 不会被模型看到。
        """
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        print(f"\nStart benchmark inference on {len(benchmark_data)} samples...")
        results = self._run(benchmark_data)
        self._save(results, output_json_path)
        return {"results": results}

    def run_pipeline(self, input_json_path, output_json_path):
        """
        生产模式：读取 input_payload.json
        格式为 {"queries": [{"query_id": ..., "query": ...}, ...]}
        """
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


if __name__ == "__main__":
    pipeline = RAGInferencePipeline(
        "../data/vector_store_word_fixed/vector_store.index",
        "../data/vector_store_word_fixed//chunk_metadata.json",
    )

    # 评测模式
    pipeline.run_benchmark_inference(
        benchmark_path="../data/benchmark/rag_benchmark_dataset.json",
        output_json_path="../outputs/benchmark_output.json",
    )

    # 生产模式（取消注释即可）
    # pipeline.run_pipeline(
    #     "../outputs/input_payload.json",
    #     "../outputs/output_payload.json",
    # )