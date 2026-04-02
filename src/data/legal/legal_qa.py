from datasets import load_from_disk
from src.data.schemas import make_record


SYSTEM_PROMPT = (
    "You are a legal assistant. "
    "Provide clear and accurate legal explanations."
)


def _extract_answer_from_legalqaeval(item):
    answers = item.get("answers")

    if answers is None:
        return None

    if isinstance(answers, dict):
        texts = answers.get("text", [])
        if isinstance(texts, list) and len(texts) > 0:
            value = str(texts[0]).strip()
            return value if value else None
        if isinstance(texts, str) and texts.strip():
            return texts.strip()

    if isinstance(answers, list):
        if len(answers) == 0:
            return None

        first = answers[0]

        if isinstance(first, str):
            first = first.strip()
            return first if first else None

        if isinstance(first, dict):
            if "text" in first:
                value = first["text"]
                if isinstance(value, list) and len(value) > 0:
                    text = str(value[0]).strip()
                    return text if text else None
                if isinstance(value, str) and value.strip():
                    return value.strip()

            for v in first.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if isinstance(v, list) and len(v) > 0:
                    text = str(v[0]).strip()
                    return text if text else None

    if isinstance(answers, str) and answers.strip():
        return answers.strip()

    return None


def convert_legalqaeval(path):
    ds = load_from_disk(path)
    out = []

    for split in ds:
        for item in ds[split]:
            question = str(item.get("question", "")).strip()
            answer = _extract_answer_from_legalqaeval(item)

            if not question or not answer:
                continue

            out.append(
                make_record(
                    dataset_name="legalqaeval",
                    domain="legal",
                    task_type="qa",
                    system_prompt=SYSTEM_PROMPT,
                    user_text=question,
                    assistant_text=answer,
                )
            )
    return out


def convert_australian_legal(path):
    ds = load_from_disk(path)["train"]
    out = []

    for item in ds:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question or not answer:
            continue

        out.append(
            make_record(
                dataset_name="australian_legal",
                domain="legal",
                task_type="qa",
                system_prompt=SYSTEM_PROMPT,
                user_text=question,
                assistant_text=answer,
            )
        )
    return out
