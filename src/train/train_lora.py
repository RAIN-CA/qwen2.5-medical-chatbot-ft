import os
import random
import argparse

from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from src.train.config_utils import load_config                  
from src.train.formatting import format_example
from src.train.model_utils import (
    parse_dtype,
    load_tokenizer,
    build_quantization_config,
    load_base_model,
    prepare_model_for_training,
    attach_lora_adapter,
)


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
    bnb_4bit_compute_dtype = parse_dtype(
        quant_cfg.get("bnb_4bit_compute_dtype", "float16")
    )

    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_dropout = float(cfg["lora"]["dropout"])
    target_modules = cfg["lora"]["target_modules"]

    seed = cfg["misc"]["seed"]
    report_to = cfg["misc"]["report_to"]

    set_seed(seed)
    random.seed(seed)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name)

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
    quantization_config = build_quantization_config(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

    model = load_base_model(
        model_name=model_name,
        fp16=fp16,
        quantization_config=quantization_config,
    )

    model = prepare_model_for_training(
        model=model,
        load_in_4bit=load_in_4bit,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = attach_lora_adapter(
        model=model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model.print_trainable_parameters()

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: format_example(x, tokenizer, max_length),
        remove_columns=dataset["train"].column_names,
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
        data_collator=default_data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
