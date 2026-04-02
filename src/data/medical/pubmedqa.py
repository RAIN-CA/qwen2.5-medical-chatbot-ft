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


def convert_pubmedqa(raw_dir: Path, split: str = "train", max_samples: Optional[int] = None):
    path = raw_dir / "pubmedqa" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    for row in rows:
        question = str(row.get("QUESTION", "")).strip()
        contexts = row.get("CONTEXTS", [])
        final_decision = str(row.get("final_decision", "")).strip()
        long_answer = str(row.get("LONG_ANSWER", "")).strip()
        year = row.get("YEAR", None)
        meshes = row.get("MESHES", [])

        if not question:
            continue

        context_text = ""
        if isinstance(contexts, list) and contexts:
            context_text = "\n".join([str(c).strip() for c in contexts if str(c).strip()])

        user_text = question
        if context_text:
            user_text = f"Question: {question}\n\nContext:\n{context_text}"

        answer_parts = []
        if final_decision:
            answer_parts.append(f"Answer: {final_decision}")
        if long_answer:
            answer_parts.append(f"Explanation: {long_answer}")

        assistant_text = "\n\n".join(answer_parts).strip()
        if not assistant_text:
            continue

        out.append(
            make_record(
                dataset_name="pubmedqa",
                domain="medical",
                task_type="biomedical_research_qa",
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
                assistant_text=assistant_text,
                metadata={
                    "year": year,
                    "meshes": meshes,
                    "source_split": split,
                },
            )
        )
    return out
