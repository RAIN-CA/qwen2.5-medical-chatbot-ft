from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from docx import Document

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
_TEXT_CACHE = {}

def reset_document_runtime_cache():
    _TEXT_CACHE.clear()


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def list_uploaded_files() -> List[str]:
    files = []
    for p in sorted(UPLOAD_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS:
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


def get_normalized_text(filename: str) -> str:
    path = UPLOAD_DIR / filename
    if not path.exists():
        return ""

    stat = path.stat()
    cache_key = (str(path), stat.st_mtime, stat.st_size, "text")

    if cache_key in _TEXT_CACHE:
        return _TEXT_CACHE[cache_key]

    text = extract_text_from_file(path)
    _TEXT_CACHE[cache_key] = text
    return text


def clear_file_cache(filename: str):
    keys_to_delete = []
    for k in _TEXT_CACHE.keys():
        if str(UPLOAD_DIR / filename) in str(k):
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del _TEXT_CACHE[k]


def delete_file(filename: str) -> bool:
    path = UPLOAD_DIR / filename
    if not path.exists():
        return False
    path.unlink()
    clear_file_cache(filename)
    return True
