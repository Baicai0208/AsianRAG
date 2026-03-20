import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(self, index_path, metadata_path, model_name="BAAI/bge-small-en-v1.5"):
        # BGE 模型跑 CPU，显存完整留给 LLM
        print("[Retriever] Loading embedding model (CPU)...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        print("[Retriever] Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # BM25 索引：对每个 chunk 的文本分词后建立
        print("[Retriever] Building BM25 index...")
        tokenized = [m["text"].lower().split() for m in self.metadata]
        self.bm25 = BM25Okapi(tokenized)

        print("[Retriever] Ready.")

    def _dense_search(self, query, top_k):
        """BGE dense 检索，返回 {faiss_idx: rank} 字典"""
        # BGE 模型推荐在 query 前加指令前缀
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, indices = self.index.search(query_vector, top_k)
        return {int(idx): rank for rank, idx in enumerate(indices[0]) if idx != -1}

    def _bm25_search(self, query, top_k):
        """BM25 稀疏检索，返回 {faiss_idx: rank} 字典"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): rank for rank, idx in enumerate(top_indices)}

    def _rrf(self, dense_ranks, bm25_ranks, k=60):
        """
        Reciprocal Rank Fusion
        score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        两路都没出现的候选不参与融合
        """
        all_ids = set(dense_ranks) | set(bm25_ranks)
        fused = {}
        for idx in all_ids:
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (k + dense_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[idx])
            fused[idx] = score
        # 分数越高越相关，降序排列
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def search(self, query, top_k=10, recall_k=20):
        """
        双路召回 + RRF 融合
        recall_k : 每路召回的候选数量（融合前）
        top_k    : 融合后返回的最终结果数量
        """
        dense_ranks = self._dense_search(query, top_k=recall_k)
        bm25_ranks  = self._bm25_search(query, top_k=recall_k)
        fused       = self._rrf(dense_ranks, bm25_ranks)

        results = []
        for idx, rrf_score in fused[:top_k]:
            chunk_meta = self.metadata[idx]
            results.append({
                "doc_id":    chunk_meta.get("chunk_id", str(idx)),
                "text":      chunk_meta.get("text", ""),
                "rrf_score": rrf_score,
            })

        return results


if __name__ == "__main__":
    retriever = Retriever(
        "../data/vector_store_word_fixed/vector_store.index",
        "../data/vector_store_word_fixed/chunk_metadata.json",
    )

    test_query = "What are the main characteristics of Sichuan cuisine?"
    chunks = retriever.search(test_query, top_k=10)

    for i, chunk in enumerate(chunks):
        print(f"Rank {i+1} (RRF: {chunk['rrf_score']:.4f}):\n{chunk['text'][:100]}...\n")