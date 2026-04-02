from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.inference.prompts import get_system_prompt, build_messages


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


def build_quantization_config(load_in_4bit: bool = False):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_model_and_tokenizer(
    base_model: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quantization_config(load_in_4bit=load_in_4bit)

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


def generate_answer(
    tokenizer,
    model,
    query: str,
    domain: Optional[str] = "medical",
    system_prompt: Optional[str] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> str:
    if generation_config is None:
        generation_config = GenerationConfig()

    final_system_prompt = get_system_prompt(domain=domain, override_prompt=system_prompt)
    messages = build_messages(final_system_prompt, query)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            do_sample=generation_config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def unload_model(tokenizer, model):
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
