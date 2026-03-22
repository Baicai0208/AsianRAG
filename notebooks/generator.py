import gc
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载前先释放内存，减少峰值
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print("[Generator] Loading LLM (this may take a moment)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # 半精度，节省显存
            device_map="auto",
            low_cpu_mem_usage=True,     # 逐层加载，避免 CPU 内存峰值翻倍
        )
        print("[Generator] LLM loaded.")

    def generate(self, query, retrieved_chunks):
        """
        retrieved_chunks: list of {"text": ..., ...}
        """
        context = "\n---\n".join([chunk["text"] for chunk in retrieved_chunks])

        prompt = (
            "You are an expert culinary assistant specialized in East Asian cuisine.\n"
            "Answer the question based ONLY on the provided context. "
            "If the answer is not in the context, say 'I do not know'.\n"
            "Give a direct, concise answer. "
            "Always respond in English only. Do not use any Chinese, Korean, Japanese, or other non-English characters.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"  # 末尾加 Answer: 引导模型直接输出答案
        )

        messages = [
            {"role": "system", "content": "You are a helpful and precise culinary assistant."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 后处理：强制清除非 ASCII 字符（中文、韩文、日文等）
        result = re.sub(r'[^\x00-\x7F]+', '', result).strip()

        # 每次生成后释放显存碎片，防止批量推理时显存持续累积
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result


if __name__ == "__main__":
    generator = Generator()

    dummy_query = "What is Dim Sum?"
    dummy_chunks = [{"text": "Dim sum is a large range of small Chinese dishes that are traditionally enjoyed in restaurants for brunch."}]

    answer = generator.generate(dummy_query, dummy_chunks)
    print(f"Answer: {answer}")