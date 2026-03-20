import json
import random

def sample_chunks_for_llm(input_file='../data/corpus/chunked_corpus.json', output_file='../data/benchmark/chunks_for_prompt.txt', sample_size=30):
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    sampled = random.sample(chunks, min(sample_size, len(chunks)))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(sampled):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"ID: {chunk['chunk_id']}\n")
            f.write(f"Text: {chunk['text']}\n\n")

if __name__ == "__main__":
    sample_chunks_for_llm()