from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_store import get_normalized_text

_CHUNK_CACHE = {}

def reset_rag_runtime_cache():
    _CHUNK_CACHE.clear()


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


def _emit(progress_callback, event_type: str, step_id: str, text: str):
    if progress_callback:
        progress_callback(event_type, step_id, text)


def retrieve_context(
    query: str,
    selected_files: List[str],
    top_k: int = 4,
    chunk_size: int = 800,
    overlap: int = 120,
    progress_callback=None,
) -> Tuple[str, List[Dict]]:
    all_chunks = []

    # Step 1: load
    _emit(progress_callback, "status", "rag_load", f"Loading {len(selected_files)} selected document(s)")
    for idx, fname in enumerate(selected_files, start=1):
        _emit(progress_callback, "status_update", "rag_load", f"Loading file {idx}/{len(selected_files)}: {fname}")
        _ = get_normalized_text(fname)
    _emit(progress_callback, "status_done", "rag_load", f"Loaded {len(selected_files)} document(s)")

    # Step 2: chunk
    _emit(progress_callback, "status", "rag_chunk", "Chunking document text")
    for idx, fname in enumerate(selected_files, start=1):
        _emit(progress_callback, "status_update", "rag_chunk", f"Chunking file {idx}/{len(selected_files)}: {fname}")
        file_chunks = build_chunks_for_file(fname, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        _emit(progress_callback, "status_done", "rag_chunk", "No chunks produced from selected files")
        return "", []

    _emit(progress_callback, "status_done", "rag_chunk", f"Prepared {len(all_chunks)} text chunk(s)")

    # Step 3: vectorize
    _emit(progress_callback, "status", "rag_vectorize", "Building TF-IDF vectors")
    corpus = [c["text"] for c in all_chunks] + [query]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    doc_vectors = tfidf[:-1]
    query_vector = tfidf[-1]
    _emit(progress_callback, "status_done", "rag_vectorize", f"Built TF-IDF vectors for {len(all_chunks)} chunk(s)")

    # Step 4: rank
    _emit(progress_callback, "status", "rag_rank", "Computing similarity scores")
    sims = cosine_similarity(query_vector, doc_vectors).flatten()
    _emit(progress_callback, "status_update", "rag_rank", "Selecting top relevant chunks")

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
        _emit(progress_callback, "status_done", "rag_rank", "No strongly relevant chunks found")
        return "", []

    _emit(progress_callback, "status_done", "rag_rank", f"Selected top {len(top_hits)} relevant chunk(s)")

    context_parts = []
    for idx, hit in enumerate(top_hits, start=1):
        context_parts.append(
            f"[Document {idx} | file={hit['filename']} | chunk={hit['chunk_id']} | score={hit['score']:.4f}]\n{hit['text']}"
        )

    context = "\n\n".join(context_parts)
    return context, top_hits
