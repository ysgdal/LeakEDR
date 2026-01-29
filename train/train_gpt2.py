#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset



class PerplexityLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.history = {
            "steps": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "elapsed_time_sec": [],
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return

        step = state.global_step
        eval_loss = metrics["eval_loss"]
        ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        elapsed = time.time() - self.start_time

        self.history["steps"].append(step)
        self.history["eval_loss"].append(eval_loss)
        self.history["eval_perplexity"].append(ppl)
        self.history["elapsed_time_sec"].append(elapsed)


def main():
    parser = argparse.ArgumentParser(description="Train GPT2 / GPT2-XL LM")
    
    # Data & model
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Pretrained model path or name")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer path or name")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation data file")

    # Training parameters
    parser.add_argument("--output_dir", type=str, default="./out_gpt2", help="Output directory")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,      # Do not truncate here
            add_special_tokens=True,
        )

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_len, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)


    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"] if args.do_train else None,
        eval_dataset=lm_datasets["validation"] if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    ppl_callback = PerplexityLoggerCallback()
    trainer.add_callback(ppl_callback)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    if args.do_train:
        trainer.train()
        trainer.save_model()

    with open(Path(args.output_dir) / "training_history.json", "w") as f:
        json.dump(ppl_callback.history, f, indent=2)

    resource = {}
    if torch.cuda.is_available():
        resource["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

    with open(Path(args.output_dir) / "resource_usage.json", "w") as f:
        json.dump(resource, f, indent=2)

    print("Training finished.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
