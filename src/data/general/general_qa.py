from datasets import load_from_disk
from src.data.schemas import make_record


SYSTEM_PROMPT = (
    "You are a general knowledge assistant. "
    "Provide accurate and concise answers."
)


def convert_squad(path):
    ds = load_from_disk(path)
    out = []

    for split in ds:
        for item in ds[split]:
            if len(item["answers"]["text"]) == 0:
                continue

            out.append(
                make_record(
                    dataset_name="squad",
                    domain="general",
                    task_type="qa",
                    system_prompt=SYSTEM_PROMPT,
                    user_text=item["question"],
                    assistant_text=item["answers"]["text"][0],
                )
            )
    return out
