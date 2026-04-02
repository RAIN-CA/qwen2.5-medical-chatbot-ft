from pathlib import Path
from typing import Optional

from datasets import load_dataset
from src.data.schemas import make_record

SYSTEM_PROMPT = (
    "You are a helpful academic chatbot. "
    "Provide clear, concise, and accurate answers."
)


def convert_squad(split: str = "train", max_samples: Optional[int] = None):
    dataset = load_dataset("squad", split=split)

    out = []

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    for item in dataset:
        question = item.get("question", "").strip()
        answers = item.get("answers", {}).get("text", [])

        if not question or not answers:
            continue

        answer = answers[0].strip()

        out.append(
            make_record(
                dataset_name="squad",
                domain="general",
                task_type="general_qa",
                system_prompt=SYSTEM_PROMPT,
                user_text=question,
                assistant_text=answer,
                metadata={"source": "squad"},
            )
        )

    return out
