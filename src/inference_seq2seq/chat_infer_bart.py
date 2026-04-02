import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.eval()

    inputs = tokenizer(
        args.query,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n===== USER =====")
    print(args.query)
    print("\n===== ASSISTANT =====")
    print(response.strip())


if __name__ == "__main__":
    main()
