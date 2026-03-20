import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def build_index(chunked_corpus_path, output_index_path, output_metadata_path,
                model_name="BAAI/bge-small-en-v1.5"):
    with open(chunked_corpus_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    print(f"[Embedding] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # BGE 建索引时不需要加指令前缀，只有 query 端需要
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    # 使用内积索引（配合 normalize_embeddings=True 等价于余弦相似度）
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, output_index_path)

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"Vectorised {len(texts)} chunks  →  {output_index_path}")


if __name__ == "__main__":
    # 只对最优策略 word_fixed 重建索引
    print("[Embedding] Building BGE index for word_fixed...")
    build_index(
        chunked_corpus_path  = "../data/corpus/chunked_word_fixed.json",
        output_index_path    = "../data/vector_store_word_fixed/vector_store.index",
        output_metadata_path = "../data/vector_store_word_fixed/chunk_metadata.json",
    )
    print("\nDone.")