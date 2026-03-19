import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def build_messages(system_prompt, user_query):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    system_prompt = (
        "You are a medical knowledge chatbot for academic coursework. "
        "Provide clear, concise, educational medical answers. "
        "Do not provide diagnosis or treatment decisions."
    )

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    messages = build_messages(system_prompt, args.query)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n===== USER =====")
    print(args.query)
    print("\n===== ASSISTANT =====")
    print(response.strip())


if __name__ == "__main__":
    main()
