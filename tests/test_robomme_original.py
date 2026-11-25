"""Regression checks for selecting the original-data experiment on the training host."""
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
_check_original_data = runpy.run_path(str(ROOT / "scripts/train_cfg.py"))["_check_original_data"]


class OriginalRoboMMETest(unittest.TestCase):
    def test_original_training_target(self):
        cfg = runpy.run_path(str(ROOT / "configs/train/robomme_1x_musa_8gpu.py"))
        self.assertEqual(cfg["preset"], "pi05_robomme_1x_dreamdojo_fixedlag5_h50")
        self.assertEqual(cfg["cuda_visible_devices"], list(range(8)))
        self.assertEqual(cfg["train"]["batch_size"], 128)
        self.assertEqual(cfg["train"]["num_train_steps"], 40_000)
        self.assertEqual(cfg["lr_schedule"]["decay_steps"], 40_000)
        self.assertEqual(cfg["data"]["memory_random_drop_execution_tail_min"], 5)
        self.assertEqual(cfg["data"]["memory_random_drop_execution_tail_max"], 5)
        self.assertFalse(cfg["model"]["discrete_state_input"])
        self.assertEqual(cfg["model"]["action_horizon"], 50)
        self.assertEqual(cfg["checkpoint"]["save_steps"][-1], 40_000)

    def test_data_scale_and_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            info = root / "info.json"
            info.write_text(json.dumps(dict(total_episodes=16_000, total_frames=7_000_000)))
            with self.assertRaisesRegex(ValueError, "1,600"):
                _check_original_data(info, {}, {}, root / "norm_stats.json", bounded=False)
            info.write_text(json.dumps(dict(total_episodes=1600, total_frames=768897)))
            episodes = root / "episodes.jsonl"
            episodes.write_text("{}\n")
            identity = dict(dataset_info_sha256=hashlib.sha256(info.read_bytes()).hexdigest(),
                            dataset_episodes_sha256=hashlib.sha256(episodes.read_bytes()).hexdigest())
            memory = dict(identity, cached_valid_transitions=767297,
                          checkpoint_sha256="d77bf1b307b6e6d0a2800a2636afee8223a7bf19f15a8583eebd3f8979f1c44f")
            (root / "manifest.json").write_text(json.dumps(dict(
                identity, total_frames=768897, execution_frames=476857, action_horizon=50
            )))
            _check_original_data(info, memory, identity, root / "norm_stats.json", bounded=False)
            with self.assertRaisesRegex(ValueError, "caches must match"):
                _check_original_data(info, memory, dict(identity, dataset_episodes_sha256="bad"),
                                     root / "norm_stats.json", bounded=False)
