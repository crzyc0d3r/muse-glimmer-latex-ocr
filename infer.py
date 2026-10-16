"""Transcribe a local image using a base checkpoint or saved LoRA adapter."""

import argparse
from PIL import Image
from runtime import MODEL, load_model, transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    model, processor = load_model(args.model)
    print(transcribe(model, processor, image, args.max_new_tokens))


if __name__ == "__main__":
    main()
