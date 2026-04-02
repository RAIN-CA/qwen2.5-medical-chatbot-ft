from datasets import load_from_disk
from src.data.schemas import make_record


SYSTEM_PROMPT = (
    "You are a financial knowledge assistant. "
    "Provide clear and concise financial explanations."
)


def convert_fiqa(path):
    ds = load_from_disk(path)
    out = []

    for split in ds:
        for item in ds[split]:
            q = item["question"]
            a = item["ground_truths"][0]

            out.append(
                make_record(
                    dataset_name="fiqa",
                    domain="finance",
                    task_type="qa",
                    system_prompt=SYSTEM_PROMPT,
                    user_text=q,
                    assistant_text=a,
                )
            )
    return out


def convert_financial_qa_10k(path):
    ds = load_from_disk(path)["train"]
    out = []

    for item in ds:
        out.append(
            make_record(
                dataset_name="financial_qa_10k",
                domain="finance",
                task_type="qa",
                system_prompt=SYSTEM_PROMPT,
                user_text=item["question"],
                assistant_text=item["answer"],
            )
        )
    return out
