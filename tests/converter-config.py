#!/usr/bin/env python3
"""Dependency-light tests for the AudioVAE converter config contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "convert-voxcpm2-audiovae.py"
SPEC = importlib.util.spec_from_file_location("convert_voxcpm2_audiovae", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class LoadConfigTests(unittest.TestCase):
    def load(self, payload: object) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return CONVERTER.load_config(path)

    def test_defaults_and_unrelated_keys_are_allowed(self) -> None:
        payload = {
            "audio_vae_config": {
                **CONVERTER.DEFAULT_CONFIG,
                "training_only": {"dropout": 0.5},
            },
            "unrelated_top_level": True,
        }
        self.assertEqual(self.load(payload), CONVERTER.DEFAULT_CONFIG)

    def test_every_recognized_runtime_mismatch_is_rejected(self) -> None:
        def mismatched(value: object) -> object:
            if isinstance(value, bool):
                return not value
            if isinstance(value, int):
                return value + 1
            if isinstance(value, str):
                return value + "-unsupported"
            if isinstance(value, list):
                changed = list(value)
                changed[0] += 1
                return changed
            raise AssertionError(f"unhandled default type: {type(value)}")

        for key, expected in CONVERTER.DEFAULT_CONFIG.items():
            with self.subTest(key=key):
                payload = {"audio_vae_config": {key: mismatched(expected)}}
                with self.assertRaisesRegex(ValueError, key):
                    self.load(payload)

    def test_equal_python_values_with_wrong_json_types_are_rejected(self) -> None:
        cases = (
            ("encoder_dim", True),
            ("depthwise", 1),
            ("encoder_rates", [2.0, 5.0, 8.0, 8.0]),
        )
        for key, value in cases:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self.load({"audio_vae_config": {key: value}})

    def test_recognized_array_length_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoder_rates"):
            self.load({"encoder_rates": [2, 5, 8]})

    def test_non_object_top_level_fails_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an object"):
            self.load([])


if __name__ == "__main__":
    unittest.main()
