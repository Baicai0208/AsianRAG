import json

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def main():
    with open('../data/corpus/east_asian_corpus.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    chunked_data = []
    for doc in corpus:
        text = doc.get('text', '')
        if not text:
            continue
            
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "source": doc['source'],
                "chunk_id": f"{doc['source']}_chunk_{idx}",
                "text": chunk
            })

    with open('../data/corpus/chunked_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(chunked_data)} chunks.")

if __name__ == "__main__":
    main()