def build_text_from_messages(messages):
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")

    return "\n".join(parts)


def format_example(example, tokenizer, max_length: int):
    text = build_text_from_messages(example["messages"])

    if tokenizer.eos_token:
        text += tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
