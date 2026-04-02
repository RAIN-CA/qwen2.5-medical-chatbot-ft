import argparse

from src.inference.engine import (
    GenerationConfig,
    load_model_and_tokenizer,
    generate_answer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--domain", type=str, default="medical")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    print(f"Loading tokenizer/model from: {args.base_model}")
    tokenizer, model = load_model_and_tokenizer(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )

    response = generate_answer(
        tokenizer=tokenizer,
        model=model,
        query=args.query,
        domain=args.domain,
        system_prompt=args.system_prompt,
        generation_config=gen_cfg,
    )

    print("\n===== USER =====")
    print(args.query)
    print("\n===== ASSISTANT =====")
    print(response)


if __name__ == "__main__":
    main()
