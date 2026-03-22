"""
eval.py — RAG 评估脚本

指标：
  检索端
    - Hit Rate @ K : 正确 chunk 是否出现在 Top-K 结果中
    - MRR          : Mean Reciprocal Rank，命中排名越靠前分越高

  生成端
    - ROUGE-1/2/L  : n-gram 字面重合度（归一化后计算，避免标点/大小写误判）
    - BLEU         : n-gram 精确率
    - BERTScore    : 语义相似度
"""

import json
import re
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


def normalize(text: str) -> str:
    """小写 + 去标点 + 合并空格，用于宽松匹配和 ROUGE 计算。"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class RAGEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.bleu_smoother = SmoothingFunction().method1

    def _bleu(self, gold, generated):
        ref = gold.lower().split()
        hyp = generated.lower().split()
        return sentence_bleu([ref], hyp, smoothing_function=self.bleu_smoother)

    def _is_hit(self, retrieved_chunks, target_chunk_id, gold_answer):
        """
        双重命中判断：
          1. chunk_id 精确匹配（兼容原始策略）
          2. gold_answer 归一化文本包含匹配（跨策略通用）
        """
        gold_norm = normalize(gold_answer)
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if chunk["doc_id"] == target_chunk_id:
                return 1, rank
            if gold_norm and gold_norm in normalize(chunk["text"]):
                return 1, rank
        return 0, 0

    def _mrr(self, retrieved_chunks, target_chunk_id, gold_answer):
        is_hit, rank = self._is_hit(retrieved_chunks, target_chunk_id, gold_answer)
        return 1.0 / rank if is_hit else 0.0

    def evaluate(self, inference_output_path, benchmark_path):
        with open(inference_output_path, "r", encoding="utf-8") as f:
            inference_data = json.load(f)
        inference_map = {item["query_id"]: item for item in inference_data["results"]}

        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

        hits          = 0
        mrr_scores    = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores   = []
        results       = []
        missing       = 0

        generated_list = []
        gold_list      = []

        print(f"Starting evaluation on {len(benchmark_data)} samples...")

        for item in tqdm(benchmark_data):
            query_id        = item["query_id"]
            gold_answer     = item["gold_answer"]
            target_chunk_id = item["source_chunk_id"]

            inferred = inference_map.get(query_id)
            if inferred is None:
                print(f"[Warning] query_id '{query_id}' not found, skipping.")
                missing += 1
                continue

            retrieved_chunks = inferred["retrieved_context"]
            generated_answer = inferred["response"]

            # ---- 检索指标 ----
            is_hit, _ = self._is_hit(retrieved_chunks, target_chunk_id, gold_answer)
            hits += is_hit
            mrr_scores.append(self._mrr(retrieved_chunks, target_chunk_id, gold_answer))

            # ---- 生成指标（归一化后计算，减少标点/大小写误判）----
            rouge = self.rouge_scorer.score(normalize(gold_answer), normalize(generated_answer))
            rouge1_scores.append(rouge["rouge1"].fmeasure)
            rouge2_scores.append(rouge["rouge2"].fmeasure)
            rougeL_scores.append(rouge["rougeL"].fmeasure)
            bleu_scores.append(self._bleu(gold_answer, generated_answer))

            generated_list.append(generated_answer)
            gold_list.append(gold_answer)

            results.append({
                "query_id":  query_id,
                "hit":       is_hit,
                "mrr":       mrr_scores[-1],
                "rouge1":    rouge["rouge1"].fmeasure,
                "rouge2":    rouge["rouge2"].fmeasure,
                "rougeL":    rouge["rougeL"].fmeasure,
                "bleu":      bleu_scores[-1],
                "generated": generated_answer,
                "gold":      gold_answer,
            })

        # ---- BERTScore（批量计算）----
        print("\nCalculating BERTScore (this may take a moment)...")
        P, R, F1 = bert_score(
            generated_list, gold_list,
            lang="en",
            verbose=False,
        )
        bert_f1_scores = F1.tolist()
        for i, r in enumerate(results):
            r["bertscore"] = bert_f1_scores[i]

        evaluated = len(benchmark_data) - missing
        return {
            "avg_hit_rate":      hits / evaluated if evaluated > 0 else 0.0,
            "avg_mrr":           float(np.mean(mrr_scores)),
            "avg_rouge1":        float(np.mean(rouge1_scores)),
            "avg_rouge2":        float(np.mean(rouge2_scores)),
            "avg_rougeL":        float(np.mean(rougeL_scores)),
            "avg_bleu":          float(np.mean(bleu_scores)),
            "avg_bertscore":     float(np.mean(bert_f1_scores)),
            "evaluated_samples": evaluated,
            "missing_samples":   missing,
            "detailed_results":  results,
        }


if __name__ == "__main__":
    evaluator = RAGEvaluator()

    report = evaluator.evaluate(
        inference_output_path="../outputs/benchmark_output_fixed_size.json",
        benchmark_path="../data/benchmark/rag_benchmark_dataset.json",
    )

    print("\n" + "=" * 35)
    print("Final Evaluation Report")
    print("-" * 35)
    print(f"Evaluated Samples : {report['evaluated_samples']}")
    print(f"Missing  Samples  : {report['missing_samples']}")
    print()
    print("[ Retrieval ]")
    print(f"  Hit Rate @ 10   : {report['avg_hit_rate']:.4%}")
    print(f"  MRR             : {report['avg_mrr']:.4f}")
    print()
    print("[ Generation ]")
    print(f"  ROUGE-1         : {report['avg_rouge1']:.4f}")
    print(f"  ROUGE-2         : {report['avg_rouge2']:.4f}")
    print(f"  ROUGE-L         : {report['avg_rougeL']:.4f}")
    print(f"  BLEU            : {report['avg_bleu']:.4f}")
    print(f"  BERTScore F1    : {report['avg_bertscore']:.4f}")
    print("=" * 35)

    missed = [r for r in report["detailed_results"] if r["hit"] == 0]
    if missed:
        print(f"\nMissed {len(missed)} samples:")
        print("-" * 35)
        for r in missed:
            print(f"Query ID   : {r['query_id']}")
            print(f"ROUGE-L    : {r['rougeL']:.4f}  BERTScore: {r['bertscore']:.4f}")
            print(f"Generated  : {r['generated']}")
            print(f"Gold       : {r['gold']}")
            print("-" * 35)