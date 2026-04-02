def extract_user_and_assistant(messages):
    user_text = ""
    assistant_text = ""

    for msg in messages:
        if msg["role"] == "user":
            user_text = msg["content"].strip()
        elif msg["role"] == "assistant":
            assistant_text = msg["content"].strip()

    return user_text, assistant_text


def format_example_seq2seq(example, tokenizer, max_source_length: int, max_target_length: int):
    source_text, target_text = extract_user_and_assistant(example["messages"])
    source_text = "question: " + source_text

    model_inputs = tokenizer(
        source_text,
        truncation=True,
        max_length=max_source_length,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=target_text,
        truncation=True,
        max_length=max_target_length,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
