"""
embedding.py — 向量化 & 构建 FAISS 索引

用法：
  python embedding.py                              # 默认用 sentence 策略
  python embedding.py --strategy paragraph         # 指定策略
  python embedding.py --strategy all               # 依次构建全部四种策略的索引
"""

import json
import argparse
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

STRATEGIES = ["fixed_size", "sentence", "paragraph", "semantic"]


def build_index(strategy: str, corpus_base: str, vector_store_base: str):
    """对指定策略的 chunked_corpus.json 构建 FAISS 索引。"""

    corpus_path = os.path.join(corpus_base, strategy, "chunked_corpus.json")
    output_dir  = os.path.join(vector_store_base, strategy)

    if not os.path.exists(corpus_path):
        print(f"[{strategy}] ⚠️  找不到 {corpus_path}，跳过。请先运行 chunking.py --strategy {strategy}")
        return

    print(f"\n[{strategy}] 读取语料库: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    print(f"[{strategy}] 共 {len(texts)} 个 chunks，开始编码...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # 构建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    index_path    = os.path.join(output_dir, "vector_store.index")
    metadata_path = os.path.join(output_dir, "chunk_metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"[{strategy}] ✅ 索引已保存 → {output_dir}")
    print(f"           index:    {index_path}")
    print(f"           metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS vector store for each chunking strategy")
    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=STRATEGIES + ["all"],
        help="指定要向量化的策略，'all' 依次处理全部四种",
    )
    parser.add_argument(
        "--corpus_base",
        default="../data/corpus",
        help="chunked_corpus.json 所在的父目录（各策略在其子目录下）",
    )
    parser.add_argument(
        "--vector_store_base",
        default="../data/vector_store",
        help="FAISS 索引的输出父目录（各策略在其子目录下）",
    )
    args = parser.parse_args()

    targets = STRATEGIES if args.strategy == "all" else [args.strategy]
    for s in targets:
        build_index(s, args.corpus_base, args.vector_store_base)

    print("\n✅ 全部完成。")