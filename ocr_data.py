"""Create vision conversations and separate training and evaluation images."""

import hashlib
import random

INSTRUCTION = "Write the LaTeX representation for this image. Return only the LaTeX."


def image_key(image):
    """Hash decoded pixels so identical images cannot cross the split."""
    rgb = image.convert("RGB")
    header = f"{rgb.width}:{rgb.height}:".encode()
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def validate_record(record):
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError("Every record needs a nonempty LaTeX text label")
    image = record.get("image")
    if not hasattr(image, "convert"):
        raise ValueError("Every record needs a decoded Pillow image")
    if image.width < 1 or image.height < 1:
        raise ValueError("Images must have positive dimensions")


def split_records(records, eval_count=8, seed=3407):
    """Deduplicate images, reject conflicting labels, then hold out fixed examples."""
    unique = {}
    for record in records:
        validate_record(record)
        key = image_key(record["image"])
        if key in unique and unique[key]["text"] != record["text"]:
            raise ValueError("An identical image has conflicting LaTeX labels")
        unique.setdefault(key, record)
    rows = list(unique.values())
    if eval_count < 1 or len(rows) <= eval_count:
        raise ValueError("Need at least one unique training image and one held-out image")
    random.Random(seed).shuffle(rows)
    return rows[eval_count:], rows[:eval_count]


def conversation(record):
    validate_record(record)
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": record["image"].convert("RGB")},
                {"type": "text", "text": INSTRUCTION},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": record["text"]},
            ]},
        ]
    }
