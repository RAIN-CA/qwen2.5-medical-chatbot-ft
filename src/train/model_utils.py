import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def parse_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_quantization_config(
    load_in_4bit: bool,
    bnb_4bit_quant_type: str,
    bnb_4bit_use_double_quant: bool,
    bnb_4bit_compute_dtype,
):
    if not load_in_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )


def load_base_model(
    model_name: str,
    fp16: bool,
    quantization_config=None,
):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )


def prepare_model_for_training(
    model,
    load_in_4bit: bool,
    gradient_checkpointing: bool,
):
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False
    return model


def attach_lora_adapter(
    model,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules,
):
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model
