import argparse
import json
from pathlib import Path

from src.inference.engine import (
    GenerationConfig,
    load_model_and_tokenizer,
    generate_answer,
    unload_model,
)


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
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--model_name_for_report", type=str, default="model")
    args = parser.parse_args()

    samples = read_jsonl(args.test_file)

    tokenizer, model = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=False,
    )

    results = []
    for idx, sample in enumerate(samples, start=1):
        domain = sample["domain"]
        question = sample["question"]
        reference = sample["reference"]

        prediction = generate_answer(
            tokenizer=tokenizer,
            model=model,
            query=question,
            domain=domain,
            generation_config=gen_cfg,
        )

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
    unload_model(tokenizer, model)
    print(f"Saved predictions to: {args.output_file}")


if __name__ == "__main__":
    main()
