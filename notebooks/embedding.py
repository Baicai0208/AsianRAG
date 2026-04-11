"""
embedding.py — 向量化 & 构建 FAISS 索引

嵌入模型：BAAI/bge-small-en-v1.5
  - 专为信息检索优化，MTEB 检索榜比 all-MiniLM-L6-v2 高约 3-5 个点
  - 模型体积约 33MB，不影响实验效率
  - BGE passage 编码时不加前缀（query 前缀在 retriever.py 的检索阶段处理）
  - normalize_embeddings=True + IndexFlatIP = 余弦相似度检索

用法：
  python embedding.py                              # 构建 semantic 策略的索引
  python embedding.py --corpus_base ../data/corpus  # 指定语料目录
"""

import json
import argparse
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_index(corpus_base: str, vector_store_base: str):
    """对 semantic chunking 的 chunked_corpus.json 构建 FAISS 索引。"""

    corpus_path = os.path.join(corpus_base, "semantic", "chunked_corpus.json")
    output_dir  = os.path.join(vector_store_base, "semantic")

    if not os.path.exists(corpus_path):
        print(f"⚠️  找不到 {corpus_path}，请先运行 chunking.py")
        return

    print(f"读取语料库: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    print(f"共 {len(texts)} 个 chunks，开始编码（模型: {EMBED_MODEL}）...")

    # BGE passage 编码：不加前缀，normalize_embeddings=True
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,  # BGE 推荐归一化，配合 IndexFlatIP 实现余弦相似度
    )

    # IndexFlatIP（内积）+ 归一化向量 = 余弦相似度，分数越高越相关
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    index_path    = os.path.join(output_dir, "vector_store.index")
    metadata_path = os.path.join(output_dir, "chunk_metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"✅ 索引已保存 → {output_dir}")
    print(f"   index:    {index_path}")
    print(f"   metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector store for semantic chunks")
    parser.add_argument(
        "--corpus_base",
        default="../data/corpus",
        help="chunked_corpus.json 所在的父目录（semantic 子目录下）",
    )
    parser.add_argument(
        "--vector_store_base",
        default="../data/vector_store",
        help="FAISS 索引的输出父目录（semantic 子目录下）",
    )
    args = parser.parse_args()

    build_index(args.corpus_base, args.vector_store_base)
    print("\n✅ 全部完成。")