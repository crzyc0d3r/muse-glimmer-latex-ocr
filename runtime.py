"""Local Muse Glimmer loading and direct-answer image transcription."""

from ocr_data import INSTRUCTION

MODEL = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"


def direct_prompt(processor):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": INSTRUCTION},
    ]}]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, reasoning_strength="low",
    )
    if not text.rstrip().endswith("<|start|>assistant"):
        raise ValueError("Unexpected model chat template; inspect it before appending ATEM markers")
    return text.rstrip() + " to=user<|message|>"


def load_model(model_name=MODEL, max_length=1024):
    # Unsloth must be imported before Transformers and TRL to install its patches.
    from unsloth import FastModel
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Full model execution requires a supported NVIDIA CUDA GPU")
    return FastModel.from_pretrained(
        model_name=model_name, max_seq_length=max_length, load_in_4bit=True,
    )


def transcribe(model, processor, image, max_new_tokens=256):
    from unsloth import FastModel
    import torch

    inputs = processor(
        text=[direct_prompt(processor)], images=[[image.convert("RGB")]],
        add_special_tokens=False, return_tensors="pt",
    ).to(model.device)
    FastModel.for_inference(model)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        )
    continuation = generated[0][inputs["input_ids"].shape[-1]:]
    text = processor.tokenizer.decode(continuation, skip_special_tokens=True)
    # Strip generation control markers, preserving all LaTeX whitespace for exact match.
    return text.split("<|eot|>", 1)[0]
