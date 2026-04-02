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


def convert_medmcqa(raw_dir: Path, split: str = "train", max_samples: Optional[int] = None):
    path = raw_dir / "medmcqa" / split / f"{split}.jsonl"
    rows = read_jsonl(path)
    out = []

    if max_samples is not None:
        rows = rows[:max_samples]

    option_map = {1: "A", 2: "B", 3: "C", 4: "D"}

    for row in rows:
        question = str(row.get("question", "")).strip()
        opa = str(row.get("opa", "")).strip()
        opb = str(row.get("opb", "")).strip()
        opc = str(row.get("opc", "")).strip()
        opd = str(row.get("opd", "")).strip()
        exp = str(row.get("exp", "")).strip()
        cop = row.get("cop", None)
        subject_name = str(row.get("subject_name", "")).strip()
        topic_name = str(row.get("topic_name", "")).strip()

        if not question:
            continue

        answer_letter = option_map.get(cop, "")
        answer_text = ""
        if answer_letter == "A":
            answer_text = opa
        elif answer_letter == "B":
            answer_text = opb
        elif answer_letter == "C":
            answer_text = opc
        elif answer_letter == "D":
            answer_text = opd

        if not answer_letter or not answer_text:
            continue

        user_text = (
            f"Answer the following medical multiple-choice question.\n\n"
            f"Question: {question}\n"
            f"A. {opa}\n"
            f"B. {opb}\n"
            f"C. {opc}\n"
            f"D. {opd}"
        )

        assistant_text = f"The correct answer is {answer_letter}. {answer_text}"
        if exp:
            assistant_text += f"\n\nExplanation: {exp}"

        out.append(
            make_record(
                dataset_name="medmcqa",
                domain="medical",
                task_type="medical_mcqa",
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
                assistant_text=assistant_text,
                metadata={
                    "subject_name": subject_name,
                    "topic_name": topic_name,
                    "source_split": split,
                },
            )
        )
    return out
