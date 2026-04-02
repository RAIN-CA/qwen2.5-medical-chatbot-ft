import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--model_name_for_report", type=str, default="bart_model")
    args = parser.parse_args()

    samples = read_jsonl(args.test_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.eval()

    results = []
    for idx, sample in enumerate(samples, start=1):
        question = sample["question"]
        reference = sample["reference"]
        domain = sample["domain"]

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        results.append({
            "id": idx,
            "model": args.model_name_for_report,
            "domain": domain,
            "dataset": sample.get("dataset", ""),
            "question": question,
            "reference": reference,
            "prediction": prediction,
        })

        if idx % 20 == 0:
            print(f"Processed {idx}/{len(samples)}")

    write_jsonl(results, Path(args.output_file))
    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
