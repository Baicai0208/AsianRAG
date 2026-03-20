import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path, metadata_path, model_name="all-MiniLM-L6-v2"):
        # Embedding 模型跑 CPU，显存完整留给 LLM
        print("[Retriever] Loading embedding model (CPU)...")
        self.model = SentenceTransformer(model_name)

        print("[Retriever] Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print("[Retriever] Ready.")

    def search(self, query, top_k=5):
        """
        返回 list of dict，每条包含：
          doc_id  — chunk 的唯一 ID（供 eval 命中率计算）
          text    — chunk 文本（供 generator 生成）
          score   — L2 距离（越小越相关）
        """
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                chunk_meta = self.metadata[idx]
                results.append({
                    "doc_id": chunk_meta.get("chunk_id", str(idx)),
                    "text":   chunk_meta.get("text", ""),
                    "score":  float(dist),
                })

        return results


if __name__ == "__main__":
    retriever = Retriever(
        "../data/vector_store_minilm/vector_store.index",
        "../data/vector_store_minilm/chunk_metadata.json",
    )

    test_query = "What are the main characteristics of Sichuan cuisine?"
    chunks = retriever.search(test_query, top_k=5)

    for i, chunk in enumerate(chunks):
        print(f"Rank {i+1} (Score: {chunk['score']:.4f}):\n{chunk['text']}\n")