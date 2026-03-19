import os
import yaml
import random
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def format_example(example, tokenizer, max_length):
    text = build_text_from_messages(example["messages"])
    if tokenizer.eos_token:
        text += tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def parse_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg["model_name"]
    train_file = cfg["data"]["train_file"]
    val_file = cfg["data"]["val_file"]
    max_length = cfg["data"]["max_length"]

    output_dir = cfg["training"]["output_dir"]
    num_train_epochs = cfg["training"]["num_train_epochs"]
    per_device_train_batch_size = cfg["training"]["per_device_train_batch_size"]
    per_device_eval_batch_size = cfg["training"]["per_device_eval_batch_size"]
    gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"]
    learning_rate = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    logging_steps = cfg["training"]["logging_steps"]
    eval_steps = cfg["training"]["eval_steps"]
    save_steps = cfg["training"]["save_steps"]
    warmup_ratio = float(cfg["training"]["warmup_ratio"])
    lr_scheduler_type = cfg["training"]["lr_scheduler_type"]
    max_grad_norm = float(cfg["training"]["max_grad_norm"])
    bf16 = bool(cfg["training"]["bf16"])
    fp16 = bool(cfg["training"]["fp16"])
    gradient_checkpointing = bool(cfg["training"]["gradient_checkpointing"])

    quant_cfg = cfg.get("quantization", {})
    load_in_4bit = bool(quant_cfg.get("load_in_4bit", False))
    bnb_4bit_quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
    bnb_4bit_use_double_quant = bool(quant_cfg.get("bnb_4bit_use_double_quant", True))
    bnb_4bit_compute_dtype = parse_dtype(quant_cfg.get("bnb_4bit_compute_dtype", "float16"))

    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_dropout = float(cfg["lora"]["dropout"])
    target_modules = cfg["lora"]["target_modules"]

    seed = cfg["misc"]["seed"]
    report_to = cfg["misc"]["report_to"]

    set_seed(seed)
    random.seed(seed)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": val_file,
        }
    )
    print(dataset)

    print(f"Loading model: {model_name}")
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: format_example(x, tokenizer, max_length),
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        report_to=report_to,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=2,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
