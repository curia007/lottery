import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from idaho.pick3.upload_to_hf_hub import (
    create_model_card,
    ensure_hf_assets,
    main,
    normalize_repo_id,
    upload_checkpoint_to_hub,
)


class TestUploadToHFHub(unittest.TestCase):
    def test_normalize_repo_id(self):
        self.assertEqual(normalize_repo_id("hf://username/my-model"), "username/my-model")
        self.assertEqual(normalize_repo_id("huggingface://org/model-name"), "org/model-name")
        self.assertEqual(normalize_repo_id("username/model-name/"), "username/model-name")
        self.assertEqual(normalize_repo_id("standalone-model"), "standalone-model")

    def test_create_model_card(self):
        metrics = {
            "val_loss": 0.4521,
            "val_token_accuracy": 0.875,
            "val_exact_match_rate": 0.045,
        }
        config_dict = {
            "vocab_size": 17,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 3,
            "max_position_embeddings": 128,
        }
        card = create_model_card("test-user/pick3-model", metrics=metrics, config_dict=config_dict)
        self.assertIn("test-user/pick3-model", card)
        self.assertIn("0.4521", card)
        self.assertIn("87.50%", card)
        self.assertIn("4.50%", card)
        self.assertIn("pipeline_tag: text-generation", card)
        self.assertIn("Vocabulary Size", card)

    def test_ensure_hf_assets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            # Create a dummy weights file
            (model_dir / "model.safetensors").write_bytes(b"dummy weights")

            ensure_hf_assets(model_dir, repo_id="user/repo", metrics={"val_loss": 0.5})

            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue((model_dir / "generation_config.json").exists())
            self.assertTrue((model_dir / "tokenizer_config.json").exists())
            self.assertTrue((model_dir / "tokenizer.json").exists())
            self.assertTrue((model_dir / "vocab.json").exists())
            self.assertTrue((model_dir / "special_tokens_map.json").exists())
            self.assertTrue((model_dir / "README.md").exists())

            # Verify contents
            with (model_dir / "vocab.json").open("r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.assertEqual(vocab.get("<PAD>"), 10)
            self.assertEqual(vocab.get("<PRED>"), 16)

            with (model_dir / "tokenizer_config.json").open("r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            self.assertEqual(tok_cfg.get("tokenizer_class"), "PreTrainedTokenizerFast")
            self.assertIn("chat_template", tok_cfg)
            self.assertIn("<BOS>", tok_cfg["chat_template"])

            with (model_dir / "generation_config.json").open("r", encoding="utf-8") as f:
                gen_cfg = json.load(f)
            self.assertEqual(gen_cfg.get("bos_token_id"), 11)

    @patch("idaho.pick3.upload_to_hf_hub.HAS_HF_HUB", True)
    @patch("idaho.pick3.upload_to_hf_hub.HfApi")
    def test_upload_checkpoint_to_hub(self, mock_hf_api_cls):
        mock_api_instance = MagicMock()
        mock_hf_api_cls.return_value = mock_api_instance

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "model.safetensors").write_bytes(b"dummy weights")

            url = upload_checkpoint_to_hub(
                model_directory=model_dir,
                repo_id="testuser/idaho-pick3-hub",
                token="test_hf_token",
                private=True,
                commit_message="Test commit",
            )

            mock_hf_api_cls.assert_called_once_with(token="test_hf_token")
            mock_api_instance.create_repo.assert_called_once_with(
                repo_id="testuser/idaho-pick3-hub",
                token="test_hf_token",
                private=True,
                exist_ok=True,
            )
            mock_api_instance.upload_folder.assert_called_once_with(
                folder_path=str(model_dir.resolve()),
                repo_id="testuser/idaho-pick3-hub",
                commit_message="Test commit",
                token="test_hf_token",
            )
            self.assertEqual(url, "https://huggingface.co/testuser/idaho-pick3-hub")

    @patch("idaho.pick3.upload_to_hf_hub.upload_checkpoint_to_hub")
    def test_main_cli(self, mock_upload):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "model.safetensors").write_bytes(b"dummy weights")

            main([
                "--model-dir", str(model_dir),
                "--repo-id", "username/pick3-test",
                "--token", "sample_token",
                "--private",
                "--commit-msg", "Custom message",
                "--metrics", json.dumps({"val_loss": 0.35}),
            ])

            mock_upload.assert_called_once_with(
                model_directory=model_dir,
                repo_id="username/pick3-test",
                token="sample_token",
                private=True,
                commit_message="Custom message",
                metrics={"val_loss": 0.35},
            )

    @patch("idaho.pick3.upload_to_hf_hub.upload_checkpoint_to_hub")
    def test_main_cli_default_repo_id(self, mock_upload):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "model.safetensors").write_bytes(b"dummy weights")

            main([
                "--model-dir", str(model_dir),
            ])

            mock_upload.assert_called_once_with(
                model_directory=model_dir,
                repo_id="curia007/pick3_llm_model",
                token=None,
                private=False,
                commit_message="Upload Idaho Pick 3 Transformer checkpoint and artifacts",
                metrics=None,
            )


if __name__ == "__main__":
    unittest.main()
