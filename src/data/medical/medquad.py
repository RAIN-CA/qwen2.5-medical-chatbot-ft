from pathlib import Path
from typing import Optional

from src.data.io import read_jsonl
from src.data.schemas import make_record

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide clear, concise, educational medical answers based on the given question. "
    "Do not claim to provide diagnosis or treatment. "
    "When appropriate, mention uncertainty and encourage consulting healthcare professionals."
)


def convert_medquad(raw_dir: Path, split: str = "train", max_samples: Optional[int] = None):
    path = raw_dir / "medquad" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    for row in rows:
        question = str(row.get("Question", "")).strip()
        answer = str(row.get("Answer", "")).strip()
        qtype = str(row.get("qtype", "")).strip()

        if not question or not answer:
            continue

        out.append(
            make_record(
                dataset_name="medquad",
                domain="medical",
                task_type="medical_qa",
                system_prompt=SYSTEM_PROMPT,
                user_text=question,
                assistant_text=answer,
                metadata={"qtype": qtype, "source_split": split},
            )
        )
    return out
