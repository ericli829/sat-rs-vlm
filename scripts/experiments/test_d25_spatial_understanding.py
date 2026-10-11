"""Tests for the experimental prompt plumbing and conservative spatial mapper."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import run_d25_spatial_understanding_qwen as experiment
import analyze_d25_spatial_results as offline


class SpatialExperimentTests(unittest.TestCase):
    def test_offline_prose_recovery_is_conservative(self):
        offline.self_test()
        self.assertEqual(offline.prose_mapping("A building in the upper-right corner.")["mapped_region"], "top_right")

    def test_offline_recovery_rejects_negation_and_whole_rows(self):
        for description in ("There is no change in the upper-left.",
                            "Buildings at the bottom of the scene.",
                            "Buildings on the right side.",
                            "Buildings in the upper-left and lower-right."):
            with self.subTest(description=description):
                self.assertIsNone(offline.prose_mapping(description)["mapped_region"])

    def test_nine_directions_and_ambiguous_phrases(self):
        self.assertTrue(experiment.validate_mapper()["passed"])

    def test_location_field_supports_markdown_and_inline_format(self):
        result = experiment.map_location_text("Change: A new building. **Location:** lower-left.")
        self.assertEqual(result["mapped_region"], "bottom_left")

    def test_missing_and_conflicting_locations_are_unresolved(self):
        for description in ("A new building in the upper-left.",
                            "Location: upper-left or lower-right", "Location: 7",
                            "Location: left side", "Location: upper half center"):
            with self.subTest(description=description):
                self.assertIsNone(experiment.map_location_text(description)["mapped_region"])

    def test_composite_phrase_does_not_also_match_center(self):
        for phrase, expected in (("top center", "top"), ("middle left", "left"),
                                 ("center-right", "right"), ("bottom middle", "bottom")):
            with self.subTest(phrase=phrase):
                self.assertEqual(experiment.map_location_text("Location: " + phrase)["mapped_region"], expected)

    def test_actual_messages_are_replaced_without_mutating_source(self):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": "before"}, {"type": "image", "image": "after"},
            {"type": "text", "text": "old prompt"}]},
            {"role": "assistant", "content": ""}]
        original = copy.deepcopy(messages)
        changed = experiment.with_prompt_and_images(messages, "new prompt", ["grid_before", "grid_after"])
        self.assertEqual(messages, original)
        self.assertEqual(changed[0]["content"], [
            {"type": "image", "image": "grid_before"}, {"type": "image", "image": "grid_after"},
            {"type": "text", "text": "new prompt"}])

    def test_A_matches_previous_valid_control_prompt(self):
        question = "Which region of the image pair contains the primary change?"
        expected = "\n".join([
            question, "The two images are ordered BEFORE then AFTER.",
            "Choose the single cell containing the primary change.",
            "Answer with exactly one label from: " + ", ".join(experiment.OUTPUT_LABELS) + ".", "Answer:"])
        self.assertEqual(experiment.prompt_for(question, "A_original"), expected)

    def test_grid_preserves_dimensions_and_limits_overlay(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.png", root / "grid.png"
            Image.new("RGB", (256, 256), (100, 100, 100)).save(source)
            metadata = experiment.make_grid_visual(source, output)
            with Image.open(output) as image:
                self.assertEqual(image.size, (256, 256))
            self.assertGreater(metadata["overlay_touched_pixel_fraction"], 0)
            self.assertLess(metadata["overlay_touched_pixel_fraction"], 0.05)

    def test_unclear_is_not_counted_as_correct_and_unresolved_is_preserved(self):
        review = {"one": "top", "two": "unclear"}
        score = experiment._score({"one": None, "two": "bottom_left"}, review)
        self.assertEqual(score["clear"], {"hits": 0, "count": 1, "accuracy": 0.0})
        self.assertEqual(score["strict_all25"]["count"], 2)
        self.assertEqual(score["prediction_distribution_clear"], {"unresolved": 1})


if __name__ == "__main__":
    unittest.main()
