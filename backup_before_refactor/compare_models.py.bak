import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


QUESTIONS = [
    "What are the common symptoms of diabetes?",
    "What is hypertension?",
    "What is the difference between CT and MRI?",
    "What are common risk factors for heart disease?",
    "What is anemia?",
]

SYSTEM_PROMPT = (
    "You are a medical knowledge chatbot for academic coursework. "
    "Provide clear, concise, educational medical answers. "
    "Do not provide diagnosis or treatment decisions."
)

OUTPUT_PATH = Path("outputs/model_comparison_results.json")


def load_model_and_tokenizer(base_model, adapter_path=None, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def main():
    models = [
        {
            "name": "base_0.5b",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_path": None,
            "load_in_4bit": False,
        },
        {
            "name": "ft_0.5b",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter_path": "outputs/qwen25_medchat_smoke",
            "load_in_4bit": False,
        },
        {
            "name": "base_3b",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "adapter_path": None,
            "load_in_4bit": True,
        },
        {
            "name": "ft_3b",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "adapter_path": "outputs/qwen25_medchat_3b_qlora_smoke",
            "load_in_4bit": True,
        },
    ]

    results = []

    for model_cfg in models:
        print(f"\n===== Loading {model_cfg['name']} =====")
        tokenizer, model = load_model_and_tokenizer(
            base_model=model_cfg["base_model"],
            adapter_path=model_cfg["adapter_path"],
            load_in_4bit=model_cfg["load_in_4bit"],
        )

        model_result = {
            "model_name": model_cfg["name"],
            "base_model": model_cfg["base_model"],
            "adapter_path": model_cfg["adapter_path"],
            "answers": [],
        }

        for q in QUESTIONS:
            print(f"Generating for: {q}")
            ans = generate_answer(tokenizer, model, q)
            model_result["answers"].append({
                "question": q,
                "answer": ans,
            })

        results.append(model_result)

        del model
        del tokenizer
        torch.cuda.empty_cache()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
