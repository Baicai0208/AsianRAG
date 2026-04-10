"""
generator.py — RAG 生成模块

支持两种后端:
  1. local   — 本地 HuggingFace 模型 (默认, 原有行为)
  2. openrouter — 通过 OpenRouter API 调用远程模型

示例:
  # 本地模型
  gen = create_generator(backend="local")

  # OpenRouter (从环境变量 OPENROUTER_API_KEY 读取, 或直接传入)
  gen = create_generator(
      backend="openrouter",
      api_key="sk-or-...",
      model_id="openai/gpt-4o-mini",
  )
"""

import gc
import os
import re
import json
from pathlib import Path

import torch
import requests
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# 自动加载项目根目录下的 .env 文件
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# ────────────────────────────────────────────
# 公共 prompt 构建 & 后处理
# ────────────────────────────────────────────

def _build_prompt_and_messages(query, retrieved_chunks):
    """返回 (system_msg, user_msg) 供两种后端复用."""

    # CE score 过滤：丢弃负分 chunk，减少噪声干扰
    filtered = [c for c in retrieved_chunks if c.get("ce_score", 0) >= 0]

    # 兜底：若存活不足 2 个，按 rrf_score 降序取 top-2
    if len(filtered) < 2:
        filtered = sorted(
            retrieved_chunks,
            key=lambda x: x.get("rrf_score", 0),
            reverse=True,
        )[:2]

    # OOD confidence guard
    max_ce = max((c.get("ce_score", -99) for c in filtered), default=-99)
    if max_ce < 1.0:
        confidence_note = (
            "IMPORTANT: Retrieval confidence is low. "
            "If the context does not contain a clear answer, "
            "you MUST respond with exactly 'I do not know' and nothing else.\n\n"
        )
    else:
        confidence_note = ""

    context = "\n---\n".join([c["text"] for c in filtered])

    user_msg = (
        "You are an East Asian culinary expert.\n"
        "Please provide a complete, natural, and helpful answer to the question below.\n\n"
        "RULES for answering:\n"
        "1. Base your answer STRICTLY and ONLY on the provided context to prevent hallucinations.\n"
        "2. Do NOT include any outside knowledge, assumptions, or extra information not found in the context.\n"
        "3. If the provided context does not contain the answer, you MUST say exactly 'I do not know' and nothing else.\n"
        "4. Respond directly to the user. Do not include your internal reasoning or say 'Based on the context...' — just give the answer naturally.\n"
        "5. Respond in the same language as the user's question (e.g. answer in Chinese if the question is in Chinese).\n\n"
        f"{confidence_note}"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    system_msg = (
        "You are a knowledgeable East Asian culinary expert. "
        "Provide complete, natural answers based exclusively on the provided context. "
        "Never hallucinate or add outside information. "
        "Respond in the language of the user's query."
    )
    return system_msg, user_msg


def _postprocess(text: str) -> str:
    """后处理：简单清除首尾空格."""
    return text.strip()


# ────────────────────────────────────────────
# Backend 1: 本地 HuggingFace 模型
# ────────────────────────────────────────────

class LocalGenerator:
    """使用本地 HuggingFace 模型进行生成."""

    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("[Generator] Loading LLM (this may take a moment)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("[Generator] LLM loaded.")

    def generate(self, query, retrieved_chunks):
        system_msg, user_msg = _build_prompt_and_messages(query, retrieved_chunks)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = _postprocess(result)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result


# ────────────────────────────────────────────
# Backend 2: OpenRouter API
# ────────────────────────────────────────────

class OpenRouterGenerator:
    """通过 OpenRouter API 调用远程模型进行生成.

    API Key 读取优先级:
      1. 构造函数参数 api_key
      2. 环境变量 OPENROUTER_API_KEY
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model_id="openai/gpt-4o-mini",
        api_key=None,
        max_tokens=256,
        temperature=0.0,
    ):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY env var or pass api_key= to constructor."
            )

        print(f"[Generator] Using OpenRouter — model: {self.model_id}")

    def generate(self, query, retrieved_chunks):
        system_msg, user_msg = _build_prompt_and_messages(query, retrieved_chunks)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/asian-rag-coursework",  # OpenRouter 推荐
            "X-Title": "East Asian Culinary RAG",
        }

        resp = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)

        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()

        # 错误处理
        if "error" in data:
            raise RuntimeError(f"OpenRouter returned error: {data['error']}")

        result = data["choices"][0]["message"]["content"]
        result = _postprocess(result)
        return result


# ────────────────────────────────────────────
# 工厂函数
# ────────────────────────────────────────────

def create_generator(backend="local", model_id=None, api_key=None, **kwargs):
    """创建 Generator 实例.

    Args:
        backend: "local" 或 "openrouter"
        model_id: 模型 ID。
            - local 默认 "Qwen/Qwen2.5-0.5B-Instruct"
            - openrouter 默认 "openai/gpt-4o-mini"
        api_key: OpenRouter API Key (也可通过环境变量设置)
        **kwargs: 传递给对应 Generator 类的额外参数
    """
    if backend == "local":
        mid = model_id or "Qwen/Qwen2.5-0.5B-Instruct"
        return LocalGenerator(model_id=mid, **kwargs)
    elif backend == "openrouter":
        mid = model_id or "openai/gpt-4o-mini"
        return OpenRouterGenerator(model_id=mid, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'local' or 'openrouter'.")


# 向后兼容: 保留 Generator 类名指向 LocalGenerator
Generator = LocalGenerator


# ────────────────────────────────────────────
# 测试
# ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generator quick test")
    parser.add_argument("--backend", default="local", choices=["local", "openrouter"])
    parser.add_argument("--model", default=None, help="模型 ID")
    parser.add_argument("--api_key", default=None, help="OpenRouter API Key")
    args = parser.parse_args()

    gen = create_generator(
        backend=args.backend,
        model_id=args.model,
        api_key=args.api_key,
    )

    dummy_query = "What is Dim Sum?"
    dummy_chunks = [
        {
            "text": "Dim sum is a large range of small Chinese dishes "
                    "that are traditionally enjoyed in restaurants for brunch."
        }
    ]

    answer = gen.generate(dummy_query, dummy_chunks)
    print(f"Answer: {answer}")