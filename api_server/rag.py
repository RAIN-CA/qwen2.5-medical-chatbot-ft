from pathlib import Path
from typing import List, Dict, Tuple
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_TEXT_CACHE = {}


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def list_uploaded_files() -> List[str]:
    files = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
            files.append(p.name)
    return files


def get_file_metadata_list() -> List[Dict]:
    items = []
    for name in list_uploaded_files():
        path = UPLOAD_DIR / name
        items.append({
            "name": name,
            "size_kb": round(path.stat().st_size / 1024, 1),
            "type": path.suffix.lower().replace(".", ""),
        })
    return items


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return normalize_text("\n".join(texts))

    if suffix == ".docx":
        doc = Document(str(path))
        return normalize_text("\n".join([p.text for p in doc.paragraphs]))

    return ""


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


def load_chunks_for_file(filename: str) -> List[Dict]:
    path = UPLOAD_DIR / filename
    if not path.exists():
        return []

    stat = path.stat()
    cache_key = (str(path), stat.st_mtime, stat.st_size)

    if cache_key in _TEXT_CACHE:
        return _TEXT_CACHE[cache_key]

    text = extract_text_from_file(path)
    chunks = chunk_text(text)

    records = [{"filename": filename, "chunk_id": i, "text": c} for i, c in enumerate(chunks)]
    _TEXT_CACHE.clear()
    _TEXT_CACHE[cache_key] = records
    return records


def retrieve_context(query: str, selected_files: List[str], top_k: int = 4) -> Tuple[str, List[Dict]]:
    all_chunks = []
    for fname in selected_files:
        all_chunks.extend(load_chunks_for_file(fname))

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
