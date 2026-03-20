"""
chunking.py — 四种 Chunking 策略实现

策略：
  1. fixed_size   固定词数滑动窗口（基线）
  2. sentence     按句子边界分割，积累到目标大小
  3. paragraph    按 \n\n 段落边界分割，过长段落再细分
  4. semantic     按嵌入相似度动态合并相邻句子

运行方式：
  python chunking.py                      # 默认用 sentence 策略
  python chunking.py --strategy all       # 生成全部四种，输出到各自子目录
  python chunking.py --strategy paragraph # 只生成指定策略
"""

import json
import re
import argparse
import os
import numpy as np

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """用正则按句子边界切分，保留标点。"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_paragraphs(text: str) -> list[str]:
    """按双换行符切分段落。"""
    paragraphs = re.split(r'\n\n+', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


# ──────────────────────────────────────────────
# 策略 1：Fixed Size（固定词数滑动窗口）
# ──────────────────────────────────────────────

def chunk_fixed_size(text: str, chunk_size: int = 200, overlap: int = 30) -> list[str]:
    """
    按词数做滑动窗口分块。
    overlap 保证相邻块之间有词级别的上下文重叠。
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


# ──────────────────────────────────────────────
# 策略 2：Sentence-based（句子边界分块）
# ──────────────────────────────────────────────

def chunk_sentence(text: str, max_chars: int = 800, overlap_sentences: int = 1) -> list[str]:
    """
    按句子边界积累分块，不超过 max_chars 字符。
    overlap_sentences：相邻块共享的句子数，保留上下文。
    """
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # 当前块加上新句子超过限制，且当前块非空 → 先保存
        if current_len + sent_len > max_chars and current:
            chunks.append(" ".join(current))
            # 保留末尾几句作为下一块的重叠
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_len = sum(len(s) for s in current)

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ──────────────────────────────────────────────
# 策略 3：Paragraph-based（段落边界分块）
# ──────────────────────────────────────────────

def chunk_paragraph(text: str, max_chars: int = 1200, min_chars: int = 100) -> list[str]:
    """
    按 \\n\\n 段落边界分块。
    - 短段落（< min_chars）与下一段合并，避免碎片
    - 过长段落（> max_chars）退回 sentence 策略细分
    - 相邻段落累积不超过 max_chars 后换新块
    """
    paragraphs = split_paragraphs(text)
    chunks = []
    buffer = []
    buffer_len = 0

    for para in paragraphs:
        para_len = len(para)

        # 超长段落：用 sentence 策略细分后直接加入结果
        if para_len > max_chars:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer, buffer_len = [], 0
            sub_chunks = chunk_sentence(para, max_chars=max_chars)
            chunks.extend(sub_chunks)
            continue

        # 短段落：先暂存，与后续段落合并
        if para_len < min_chars:
            buffer.append(para)
            buffer_len += para_len
            continue

        # 普通段落：放不下时先保存 buffer，再开新块
        if buffer_len + para_len > max_chars and buffer:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_len = [], 0

        buffer.append(para)
        buffer_len += para_len

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


# ──────────────────────────────────────────────
# 策略 4：Semantic（嵌入相似度动态分块）
# ──────────────────────────────────────────────

def chunk_semantic(
    text: str,
    model=None,
    similarity_threshold: float = 0.45,
    max_chars: int = 1000,
    min_sentences: int = 2,
) -> list[str]:
    """
    按句子嵌入相似度动态合并。
    当相邻句子的余弦相似度低于阈值时，视为语义边界，开启新块。

    参数：
        model               SentenceTransformer 实例（外部传入，避免重复加载）
        similarity_threshold 相似度低于此值时切块（越高切得越细）
        max_chars           单块字符上限（防止语义相近但过长）
        min_sentences       每块最少包含的句子数（避免过碎）
    """
    sentences = split_sentences(text)
    if len(sentences) <= min_sentences:
        return [text.strip()] if text.strip() else []

    # 懒加载：仅在需要时导入
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("[Warning] sentence_transformers 未安装，semantic 策略退回 sentence 策略")
            return chunk_sentence(text)

    embeddings = model.encode(sentences, show_progress_bar=False)

    # 计算相邻句子的余弦相似度
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    chunks = []
    current = [sentences[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        new_len = current_len + len(sentences[i])

        # 切块条件：相似度低 或 超出长度限制（且已满足最少句子数）
        if (sim < similarity_threshold or new_len > max_chars) and len(current) >= min_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_len = len(sentences[i])
        else:
            current.append(sentences[i])
            current_len += len(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

STRATEGY_MAP = {
    "fixed_size": chunk_fixed_size,
    "sentence":   chunk_sentence,
    "paragraph":  chunk_paragraph,
    "semantic":   chunk_semantic,   # 需要 sentence_transformers
}


def run_chunking(strategy: str, corpus_path: str, output_dir: str):
    """对整个语料库执行指定 chunking 策略，输出 chunked_corpus.json。"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # semantic 策略预加载模型，避免每篇文章重复加载
    model = None
    if strategy == "semantic":
        try:
            from sentence_transformers import SentenceTransformer
            print("[Semantic] 加载嵌入模型...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("[Warning] sentence_transformers 未安装，退回 sentence 策略")
            strategy = "sentence"

    chunk_fn = STRATEGY_MAP[strategy]
    chunked_data = []

    for doc in corpus:
        text = doc.get("text", "").strip()
        if not text:
            continue

        if strategy == "semantic":
            chunks = chunk_fn(text, model=model)
        else:
            chunks = chunk_fn(text)

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

    print(f"[{strategy}] 生成 {len(chunked_data)} 个 chunks → {out_path}")
    _print_stats(chunked_data, strategy)
    return chunked_data


def _print_stats(chunked_data: list, strategy: str):
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
    parser = argparse.ArgumentParser(description="Chunking strategies for RAG corpus")
    parser.add_argument(
        "--strategy",
        default="sentence",
        choices=["fixed_size", "sentence", "paragraph", "semantic", "all"],
        help="选择 chunking 策略，'all' 会生成全部四种",
    )
    parser.add_argument(
        "--corpus",
        default="../data/corpus/east_asian_corpus.json",
        help="输入语料库路径",
    )
    parser.add_argument(
        "--output_dir",
        default="../data/corpus",
        help="输出目录（'all' 模式下会在此目录下创建子目录）",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        for s in ["fixed_size", "sentence", "paragraph", "semantic"]:
            out = os.path.join(args.output_dir, s)
            run_chunking(s, args.corpus, out)
        print("\n✅ 全部策略完成。各策略输出在对应子目录，")
        print("   修改 embedding.py 中的路径以切换策略进行对比。")
    else:
        run_chunking(args.strategy, args.corpus, args.output_dir)