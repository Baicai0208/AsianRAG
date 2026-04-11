"""
chunking.py — Semantic Chunking 策略实现

核心思想：
  用 embedding 模型计算相邻句子的语义相似度，在相似度"骤降"的位置切分。
  切分点不靠固定字符数，而靠**意义的转折**，使每个 chunk 内部语义高度一致。

工作流程：
  1. 将文档按句子分割
  2. 对每个句子做 embedding（使用 BAAI/bge-small-en-v1.5）
  3. 计算相邻句子 embedding 的余弦相似度
  4. 相似度低于动态阈值（百分位数）的位置作为语义断点
  5. 在断点处切分，形成语义连贯的 chunk
  6. 过短 chunk 与邻居合并，过长 chunk 做二次切分

运行方式：
  python chunking.py                                          # 使用默认参数
  python chunking.py --breakpoint_percentile 80               # 调整切分灵敏度
  python chunking.py --max_chars 1200 --min_chars 150         # 调整长度约束
"""

import json
import re
import argparse
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# 模块级缓存，避免多文档重复加载
_embed_model = None

def _get_embed_model():
    """懒加载 embedding 模型，全局只加载一次。"""
    global _embed_model
    if _embed_model is None:
        print(f"[Semantic Chunking] Loading embedding model: {EMBED_MODEL} ...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
        print("[Semantic Chunking] Model loaded.")
    return _embed_model


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """用正则按句子边界切分，保留标点。"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。"""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


# ──────────────────────────────────────────────
# Semantic Chunking 策略
# ──────────────────────────────────────────────

def chunk_semantic(
    text: str,
    breakpoint_percentile: int = 85,
    max_chars: int = 1500,
    min_chars: int = 100,
) -> list[str]:
    """
    基于语义相似度的智能分块。

    算法：
      1. 将文档拆分为句子
      2. 对每个句子做 embedding
      3. 计算相邻句子之间的余弦相似度
      4. 以百分位阈值确定语义断点（相似度骤降处）
      5. 在断点处切分
      6. 过短 chunk 与邻居合并，过长 chunk 退回句子级细分

    参数：
      breakpoint_percentile: 相似度"距离"高于此百分位的位置作为切分点。
                             值越高 → 切分越少、chunk 越大。
                             值越低 → 切分越多、chunk 越小。
      max_chars: 单个 chunk 的字符上限：超过则做二次细分
      min_chars: 单个 chunk 的字符下限：过短则与相邻 chunk 合并
    """
    sentences = split_sentences(text)

    # 不足 2 句 → 无法计算相邻相似度，直接返回整段
    if len(sentences) < 2:
        return [text.strip()] if text.strip() else []

    # ── Step 1：句子 embedding ──
    model = _get_embed_model()
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)

    # ── Step 2：计算相邻句子的"语义距离" ──
    # 距离 = 1 - 余弦相似度，距离越大说明语义转折越强
    distances = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        distances.append(1.0 - sim)

    # ── Step 3：动态阈值确定断点 ──
    threshold = float(np.percentile(distances, breakpoint_percentile))

    breakpoints = []
    for i, dist in enumerate(distances):
        if dist >= threshold:
            breakpoints.append(i + 1)  # 断点在第 i 句之后

    # ── Step 4：按断点切分句子 ──
    raw_chunks = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp])
        if chunk_text.strip():
            raw_chunks.append(chunk_text.strip())
        start = bp
    # 最后一个 chunk
    if start < len(sentences):
        chunk_text = " ".join(sentences[start:])
        if chunk_text.strip():
            raw_chunks.append(chunk_text.strip())

    # ── Step 5：后处理 — 合并过短、拆分过长 ──
    chunks = _postprocess_chunks(raw_chunks, max_chars=max_chars, min_chars=min_chars)
    return chunks


def _postprocess_chunks(
    raw_chunks: list[str],
    max_chars: int = 1500,
    min_chars: int = 100,
) -> list[str]:
    """
    后处理：
      1. 合并过短 chunk（< min_chars）到前一个 chunk
      2. 拆分过长 chunk（> max_chars）为基于句子边界的子块
    """
    # ── 合并过短 chunk ──
    merged = []
    buffer = ""
    for chunk in raw_chunks:
        if buffer:
            candidate = buffer + " " + chunk
        else:
            candidate = chunk

        if len(candidate) < min_chars:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""

    # 处理残余 buffer
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    # ── 拆分过长 chunk ──
    final = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            # 过长 chunk：按句子边界做子切分
            sub_sentences = split_sentences(chunk)
            current = []
            current_len = 0
            for sent in sub_sentences:
                if current_len + len(sent) > max_chars and current:
                    final.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += len(sent)
            if current:
                final.append(" ".join(current))

    return final


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run_chunking(
    corpus_path: str,
    output_dir: str,
    breakpoint_percentile: int = 85,
    max_chars: int = 1500,
    min_chars: int = 100,
):
    """对整个语料库执行 Semantic Chunking，输出 chunked_corpus.json。"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    chunked_data = []
    total = len(corpus)

    for doc_idx, doc in enumerate(corpus):
        text = doc.get("text", "").strip()
        if not text:
            continue

        if (doc_idx + 1) % 50 == 0 or doc_idx == 0:
            print(f"[Semantic Chunking] Processing doc {doc_idx + 1}/{total} ...")

        chunks = chunk_semantic(
            text,
            breakpoint_percentile=breakpoint_percentile,
            max_chars=max_chars,
            min_chars=min_chars,
        )

        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "source":   doc["source"],
                "chunk_id": f"{doc['source']}_chunk_{idx}",
                "text":     chunk,
            })

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "chunked_corpus.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Semantic Chunking] 生成 {len(chunked_data)} 个 chunks → {out_path}")
    _print_stats(chunked_data)
    return chunked_data


def _print_stats(chunked_data: list):
    """打印 chunk 长度统计，方便调参。"""
    lengths = [len(c["text"]) for c in chunked_data]
    if not lengths:
        return
    print(f"  字符数统计 | min={min(lengths)}  "
          f"avg={int(sum(lengths)/len(lengths))}  "
          f"max={max(lengths)}  "
          f"total_chunks={len(lengths)}")


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Chunking — 基于语义相似度的智能文档分块"
    )
    parser.add_argument(
        "--corpus",
        default="../data/corpus/east_asian_corpus.json",
        help="输入语料库路径",
    )
    parser.add_argument(
        "--output_dir",
        default="../data/corpus/semantic",
        help="输出目录",
    )
    parser.add_argument(
        "--breakpoint_percentile",
        type=int,
        default=85,
        help="语义距离百分位阈值，越高切分越少 (默认: 85)",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1500,
        help="单 chunk 字符上限 (默认: 1500)",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=100,
        help="单 chunk 字符下限，过短则合并 (默认: 100)",
    )
    args = parser.parse_args()

    run_chunking(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        breakpoint_percentile=args.breakpoint_percentile,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    print("\n✅ Semantic Chunking 完成。")