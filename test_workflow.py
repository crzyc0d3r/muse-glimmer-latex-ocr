"""Regression checks for data leakage, scoring, and model template compatibility."""

import unittest
from dataclasses import dataclass
from PIL import Image
from metrics import edit_distance, score
from ocr_data import conversation, image_key, split_records
from runtime import direct_prompt
from training_config import make_sft_config


def row(color, text):
    return {"image": Image.new("RGB", (3, 2), color), "text": text}


class WorkflowTests(unittest.TestCase):
    def test_old_trl_config_rejects_unknown_loss_fields_but_builder_works(self):
        @dataclass
        class LegacySFTConfig:
            max_steps: int = 30
        config = make_sft_config(LegacySFTConfig, max_steps=2)
        self.assertEqual(config.max_steps, 2)
        self.assertFalse(hasattr(config, "loss_type"))

    def test_new_trl_loss_default_is_explicitly_changed_to_nll(self):
        @dataclass
        class NewSFTConfig:
            max_steps: int = 30
            loss_type: str = "chunked_nll"
        self.assertEqual(make_sft_config(NewSFTConfig).loss_type, "nll")

    def test_repeated_images_are_not_leaked(self):
        a, b, c = row("white", "a"), row("black", "b"), row("red", "c")
        train, evaluation = split_records([a, b, a, c], 1)
        self.assertEqual(len(train) + len(evaluation), 3)
        self.assertTrue({image_key(x["image"]) for x in train}.isdisjoint(
            image_key(x["image"]) for x in evaluation))

    def test_ambiguous_labels_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            split_records([row("white", "x"), row("white", "y"), row("red", "z")], 1)

    def test_exact_match_is_distinct_from_whitespace_metric(self):
        report = score(["x+y", "z"], ["x + y", "x"])
        self.assertEqual(report["exact_match"], 0)
        self.assertEqual(report["whitespace_ignored_match"], 0.5)
        self.assertAlmostEqual(report["character_error_rate"], 3 / 6)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_no_silent_missing_predictions(self):
        with self.assertRaises(ValueError):
            score(["x"], ["x", "y"])

    def test_conversation_label_is_assistant_only(self):
        messages = conversation(row("white", r"\frac{1}{2}"))["messages"]
        self.assertEqual([x["role"] for x in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], [{"type": "text", "text": r"\frac{1}{2}"}])

    def test_changed_chat_template_is_rejected(self):
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "<|im_start|>assistant"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            direct_prompt(Processor())

    def test_direct_answer_uses_verified_atem_boundary(self):
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "<|start|>assistant"
        self.assertEqual(direct_prompt(Processor()), "<|start|>assistant to=user<|message|>")


if __name__ == "__main__":
    unittest.main()
