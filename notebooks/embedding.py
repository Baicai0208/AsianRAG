from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss

def main():
    with open('../data/corpus/chunked_corpus.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    texts = [chunk['text'] for chunk in chunks]
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, '../data/vector_store_word_fixed/vector_store.index')
    
    with open('../data/vector_store_word_fixed/chunk_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False)
        
    print(f"Vectorised {len(texts)} chunks and saved FAISS index.")

if __name__ == "__main__":
    main()