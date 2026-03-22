import json
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class Retriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="BAAI/bge-small-en-v1.5",
        reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # BGE dense 模型跑 CPU，显存完整留给 LLM
        print("[Retriever] Loading embedding model (CPU)...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        # Cross-encoder 精排模型
        print("[Retriever] Loading cross-encoder reranker (CPU)...")
        self.reranker = CrossEncoder(reranker_name, max_length=512)

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
        """BGE dense 检索，返回 {faiss_idx: rank} 字典。"""
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        query_vector = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        _, indices = self.index.search(query_vector, top_k)
        return {int(idx): rank for rank, idx in enumerate(indices[0]) if idx != -1}

    def _bm25_search(self, query, top_k):
        """BM25 稀疏检索，返回 {faiss_idx: rank} 字典。"""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {int(idx): rank for rank, idx in enumerate(top_indices)}

    def _rrf(self, dense_ranks, bm25_ranks, k=60):
        """
        Reciprocal Rank Fusion：
          score = 1/(k + rank_dense) + 1/(k + rank_bm25)
        k=60 对两路权重相对均衡，降低单路排名偶然性的影响。
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
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query, candidates, top_k):
        """
        Cross-encoder 精排：
          对每个 (query, chunk_text) pair 打分，取分数最高的 top_k 个。
          cross-encoder 精度最高但速度慢，仅对 RRF 后的少量候选精排，
          是工业界标准的两阶段检索做法。
        """
        pairs = [(query, self.metadata[idx]["text"]) for idx, _ in candidates]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def search(self, query, top_k=5, recall_k=20):
        """
        三阶段检索流程：
          1. 双路召回：BGE dense + BM25，各取 recall_k=20 个候选
          2. RRF 融合：合并两路结果，取 top-10
          3. Cross-encoder 精排：对 top-10 候选精排，最终返回 top_k=5

        最终返回至多 top_k 个 chunks（满足课程要求：at most 5 chunks）。
        """
        # 第一阶段：双路召回
        dense_ranks = self._dense_search(query, top_k=recall_k)
        bm25_ranks  = self._bm25_search(query, top_k=recall_k)

        # 第二阶段：RRF 融合，取 top-10 作为精排候选集
        fused      = self._rrf(dense_ranks, bm25_ranks)
        candidates = fused[:10]

        # 第三阶段：cross-encoder 精排，最终取 top_k
        reranked = self._rerank(query, candidates, top_k)

        results = []
        for (idx, rrf_score), ce_score in reranked:
            chunk_meta = self.metadata[idx]
            results.append({
                "doc_id":    chunk_meta.get("chunk_id", str(idx)),
                "text":      chunk_meta.get("text", ""),
                "rrf_score": float(rrf_score),   # numpy float → Python float
                "ce_score":  float(ce_score),    # numpy float32 → Python float
            })

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Retriever")
    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=["fixed_size", "sentence", "paragraph", "semantic"],
        help="使用哪种 chunking 策略对应的向量索引",
    )
    parser.add_argument(
        "--vector_store_base",
        default="../data/vector_store",
        help="向量索引的父目录",
    )
    parser.add_argument(
        "--query",
        default="What are the main characteristics of Sichuan cuisine?",
        help="测试查询",
    )
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    index_path    = f"{args.vector_store_base}/{args.strategy}/vector_store.index"
    metadata_path = f"{args.vector_store_base}/{args.strategy}/chunk_metadata.json"

    retriever = Retriever(index_path, metadata_path)
    chunks = retriever.search(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}\n")
    for i, chunk in enumerate(chunks):
        print(f"Rank {i+1} (RRF: {chunk['rrf_score']:.4f} | CE: {chunk['ce_score']:.4f}):")
        print(f"{chunk['text'][:120]}...\n")