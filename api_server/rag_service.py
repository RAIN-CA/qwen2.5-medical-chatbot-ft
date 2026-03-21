from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_store import get_normalized_text

_CHUNK_CACHE = {}


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def build_chunks_for_file(filename: str, chunk_size: int = 800, overlap: int = 120) -> List[Dict]:
    cache_key = (filename, chunk_size, overlap)
    if cache_key in _CHUNK_CACHE:
        return _CHUNK_CACHE[cache_key]

    text = get_normalized_text(filename)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    records = [{"filename": filename, "chunk_id": i, "text": c} for i, c in enumerate(chunks)]

    _CHUNK_CACHE[cache_key] = records
    return records


def clear_chunk_cache(filename: str = None):
    if filename is None:
        _CHUNK_CACHE.clear()
        return

    keys_to_delete = []
    for k in _CHUNK_CACHE.keys():
        if k[0] == filename:
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del _CHUNK_CACHE[k]


def retrieve_context(
    query: str,
    selected_files: List[str],
    top_k: int = 4,
    chunk_size: int = 800,
    overlap: int = 120,
) -> Tuple[str, List[Dict]]:
    all_chunks = []
    for fname in selected_files:
        all_chunks.extend(build_chunks_for_file(fname, chunk_size=chunk_size, overlap=overlap))

    if not all_chunks:
        return "", []

    corpus = [c["text"] for c in all_chunks] + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    doc_vectors = tfidf[:-1]
    query_vector = tfidf[-1]

    sims = cosine_similarity(query_vector, doc_vectors).flatten()

    scored = sorted(
        [
            {
                "filename": all_chunks[i]["filename"],
                "chunk_id": all_chunks[i]["chunk_id"],
                "text": all_chunks[i]["text"],
                "score": float(sims[i]),
            }
            for i in range(len(all_chunks))
        ],
        key=lambda x: x["score"],
        reverse=True,
    )

    top_hits = [x for x in scored[:top_k] if x["score"] > 0]

    if not top_hits:
        return "", []

    context_parts = []
    for idx, hit in enumerate(top_hits, start=1):
        context_parts.append(
            f"[Document {idx} | file={hit['filename']} | chunk={hit['chunk_id']} | score={hit['score']:.4f}]\n{hit['text']}"
        )

    context = "\n\n".join(context_parts)
    return context, top_hits
