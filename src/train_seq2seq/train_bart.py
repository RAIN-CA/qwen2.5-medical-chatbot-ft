import os
import random
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

from src.train.config_utils import load_config
from src.train_seq2seq.formatting_bart import format_example_seq2seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_name = cfg["model_name"]
    train_file = cfg["data"]["train_file"]
    val_file = cfg["data"]["val_file"]
    max_source_length = cfg["data"].get("max_source_length", 512)
    max_target_length = cfg["data"].get("max_target_length", 128)

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

    seed = cfg["misc"]["seed"]
    report_to = cfg["misc"]["report_to"]

    set_seed(seed)
    random.seed(seed)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: format_example_seq2seq(x, tokenizer, max_source_length, max_target_length),
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Tokenizing seq2seq dataset",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
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

    print("Saving final model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
