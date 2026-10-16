"""Offline data and evaluation smoke run; it does not run or train a language model."""

import json
from PIL import Image, ImageDraw
from ocr_data import conversation, split_records
from metrics import score


def sample(formula):
    image = Image.new("RGB", (300, 45), "white")
    ImageDraw.Draw(image).text((8, 12), formula, fill="black")
    return {"image": image, "text": formula}


def main():
    rows = [sample(text) for text in ("x + y = z", "a + b = c", "1 + 2 = 3", "x = 2")]
    train, held_out = split_records(rows + [rows[0]], eval_count=1)
    prepared = conversation(train[0])
    # Deliberately fixed candidate strings test scoring; these are not model outputs.
    report = {
        "mode": "offline data/metric check; no model inference or training",
        "train_examples": len(train), "held_out_examples": len(held_out),
        "conversation_roles": [item["role"] for item in prepared["messages"]],
        "fixture_metrics": score(["x+y=z", "a + b = c"], ["x + y = z", "a + b = c"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
