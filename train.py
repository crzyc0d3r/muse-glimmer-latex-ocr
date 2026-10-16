"""Run baseline evaluation, real vision LoRA training, and held-out evaluation."""

import argparse
import json
from pathlib import Path

from metrics import score
from ocr_data import conversation, split_records
from runtime import MODEL, load_model, transcribe
from training_config import make_sft_config


def run(args):
    # load_model imports Unsloth before the training stack.
    model, processor = load_model(args.model, args.max_length)
    from unsloth import FastModel
    from unsloth.trainer import UnslothVisionDataCollator
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    raw = list(load_dataset(args.dataset, split=f"train[:{args.samples}]"))
    train_rows, eval_rows = split_records(raw, args.eval_samples, args.seed)
    references = [row["text"] for row in eval_rows]
    baseline = [transcribe(model, processor, row["image"], args.max_new_tokens)
                for row in eval_rows]

    model = FastModel.get_peft_model(
        model, finetune_vision_layers=args.train_vision,
        finetune_language_layers=True, finetune_attention_modules=True,
        finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0,
        bias="none", random_state=args.seed, use_rslora=False, loftq_config=None,
    )
    # Restore checkpointing, cache policy, tokenizer padding, and generation flags
    # after the baseline phase; model.train() alone does not restore these.
    FastModel.for_training(model)
    collator = UnslothVisionDataCollator(
        model, processor, max_seq_length=args.max_length,
        train_on_responses_only=True,
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant to=user<|message|>",
    )
    converted = [conversation(row) for row in train_rows]
    labels = collator(converted[:min(2, len(converted))])["labels"]
    if any(int((row != -100).sum()) == 0 for row in labels):
        raise RuntimeError("All assistant labels were masked; inspect the chat template")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model, train_dataset=converted, processing_class=processor.tokenizer,
        data_collator=collator,
        args=make_sft_config(SFTConfig,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            warmup_steps=min(5, args.steps), max_steps=args.steps,
            learning_rate=2e-4, logging_steps=1, optim="adamw_8bit",
            weight_decay=0.001, lr_scheduler_type="linear", seed=args.seed,
            output_dir=str(output / "checkpoints"), report_to="none",
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=args.max_length,
        ),
    )
    FastModel.for_training(model)
    result = trainer.train()
    adapter = output / "adapter"
    model.save_pretrained(str(adapter))
    processor.save_pretrained(str(adapter))
    after = [transcribe(model, processor, row["image"], args.max_new_tokens)
             for row in eval_rows]
    report = {
        "model": args.model, "dataset": args.dataset,
        "train_examples": len(train_rows), "eval_examples": len(eval_rows),
        "baseline": score(baseline, references), "fine_tuned": score(after, references),
        "training_metrics": result.metrics,
        "predictions": [{"reference": r, "baseline": b, "fine_tuned": a}
                        for r, b, a in zip(references, baseline, after)],
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("baseline", "fine_tuned")}, indent=2))
    print(f"Adapter: {adapter}; held-out report: {output / 'evaluation.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dataset", default="unsloth/LaTeX_OCR")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    if args.samples <= args.eval_samples or min(args.steps, args.eval_samples) < 1:
        parser.error("Need positive steps, positive eval samples, and more samples than eval samples")
    run(args)


if __name__ == "__main__":
    main()
