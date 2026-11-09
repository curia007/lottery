import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess
import sys

from idaho.pick3.pick3_hf_model import (
    Pick3ForCausalLM,
    Pick3HFConfig,
    Pick3HFHub,
)
from idaho.pick3.pick3_llm_ticket_model import (
    Pick3LLM,
    Pick3LLMConfig,
    Pick3Tokenizer,
    RemoteLottoModelLoader,
)

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import torch
except ImportError:
    torch = None


class TestPick3HFModel(unittest.TestCase):
    def setUp(self):
        self.config = Pick3HFConfig(
            vocab_size=17,
            max_position_embeddings=64,
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )

    def test_config_serialization(self):
        data = self.config.to_dict()
        self.assertEqual(data["model_type"], "gpt2")
        self.assertEqual(data["vocab_size"], 17)
        self.assertEqual(data["hidden_size"], 32)
        self.assertEqual(data["num_attention_heads"], 2)
        self.assertEqual(data["num_hidden_layers"], 2)
        # Check standard HF gpt2 fields
        self.assertEqual(data["n_positions"], 64)
        self.assertEqual(data["n_ctx"], 64)
        self.assertEqual(data["n_embd"], 32)
        self.assertEqual(data["n_head"], 2)
        self.assertEqual(data["n_layer"], 2)
        self.assertEqual(data["n_inner"], 128)
        self.assertEqual(data["activation_function"], "gelu")
        self.assertEqual(data["layer_norm_epsilon"], 1e-5)
        self.assertEqual(data["initializer_range"], 0.02)
        self.assertTrue(data["scale_attn_weights"])
        self.assertTrue(data["use_cache"])
        # Check MLX compatibility aliases
        self.assertEqual(data["max_seq_len"], 64)
        self.assertEqual(data["mlp_ratio"], 4)

        # Roundtrip from dict
        loaded = Pick3HFConfig.from_dict(data)
        self.assertEqual(loaded.model_type, "gpt2")
        self.assertEqual(loaded.vocab_size, 17)
        self.assertEqual(loaded.hidden_size, 32)
        self.assertEqual(loaded.num_attention_heads, 2)
        self.assertEqual(loaded.layer_norm_epsilon, 1e-5)

        # JSON file roundtrip
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "config.json"
            self.config.to_json_file(cfg_path)
            self.assertTrue(cfg_path.exists())
            from_file = Pick3HFConfig.from_json_file(cfg_path)
            self.assertEqual(from_file.model_type, "gpt2")
            self.assertEqual(from_file.hidden_size, 32)

    def test_hf_hub_utilities(self):
        self.assertTrue(Pick3HFHub.is_hf_repo_id("lottery-ai/pick3-transformer"))
        self.assertTrue(Pick3HFHub.is_hf_repo_id("hf://username/pick3-transformer"))
        self.assertFalse(Pick3HFHub.is_hf_repo_id(__file__))

        self.assertEqual(
            Pick3HFHub.normalize_repo_id("hf://lottery-ai/pick3-transformer"),
            "lottery-ai/pick3-transformer",
        )
        self.assertEqual(
            Pick3HFHub.normalize_repo_id("huggingface://lottery-ai/pick3-transformer"),
            "lottery-ai/pick3-transformer",
        )

        # Test model card generation
        card = Pick3HFHub.generate_model_card(
            self.config,
            metrics={"val_loss": 0.42, "val_token_accuracy": 0.88, "val_exact_match_rate": 0.05},
            repo_id="lottery-ai/pick3-transformer",
        )
        self.assertIn("lottery-ai/pick3-transformer", card)
        self.assertIn("0.4200", card)
        self.assertIn("88.00%", card)
        self.assertIn("pipeline_tag: text-generation", card)

        # Test generation config
        gen_cfg = Pick3HFHub.generate_generation_config(self.config)
        self.assertEqual(gen_cfg["bos_token_id"], 11)
        self.assertEqual(gen_cfg["eos_token_id"], 12)
        self.assertEqual(gen_cfg["pad_token_id"], 10)

    def test_export_hf_assets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "hf_assets"
            Pick3HFHub.export_hf_assets(out_dir, self.config, repo_id="test/repo")

            self.assertTrue((out_dir / "config.json").exists())
            self.assertTrue((out_dir / "generation_config.json").exists())
            self.assertTrue((out_dir / "tokenizer_config.json").exists())
            self.assertTrue((out_dir / "tokenizer.json").exists())
            self.assertTrue((out_dir / "vocab.json").exists())
            self.assertTrue((out_dir / "special_tokens_map.json").exists())
            self.assertTrue((out_dir / "README.md").exists())

            # Validate tokenizer_config.json
            with (out_dir / "tokenizer_config.json").open("r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
            self.assertEqual(tok_cfg["tokenizer_class"], "PreTrainedTokenizerFast")
            self.assertIn("chat_template", tok_cfg)
            self.assertIn("<BOS>", tok_cfg["chat_template"])

            # Validate tokenizer.json fast tokenizer format
            with (out_dir / "tokenizer.json").open("r", encoding="utf-8") as f:
                tok_json = json.load(f)
            self.assertEqual(tok_json["version"], "1.0")
            self.assertEqual(tok_json["model"]["type"], "WordLevel")
            self.assertEqual(tok_json["model"]["vocab"]["0"], 0)
            self.assertEqual(tok_json["model"]["vocab"]["<PAD>"], 10)
            self.assertEqual(tok_json["model"]["vocab"]["[UNK]"], 10)
            self.assertEqual(tok_json["model"]["vocab"]["<UNK>"], 10)
            self.assertEqual(tok_json["model"]["unk_token"], "[UNK]")

            # Validate tokenizer JSON format
            with (out_dir / "vocab.json").open("r", encoding="utf-8") as f:
                vocab = json.load(f)
            self.assertEqual(vocab["0"], 0)
            self.assertEqual(vocab["<PAD>"], 10)
            self.assertEqual(vocab["<PRED>"], 16)
            self.assertEqual(vocab["[UNK]"], 10)
            self.assertEqual(vocab["<UNK>"], 10)

    def test_tokenizer_hf_save_and_load(self):
        tokenizer = Pick3Tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "tok_test"
            tokenizer.save_pretrained(out_dir)
            self.assertTrue((out_dir / "tokenizer_config.json").exists())
            self.assertTrue((out_dir / "tokenizer.json").exists())
            self.assertTrue((out_dir / "vocab.json").exists())

            loaded_tok = Pick3Tokenizer.from_pretrained(out_dir)
            self.assertEqual(loaded_tok.vocab_size, 17)
            self.assertEqual(loaded_tok.pad_id, 10)

    def test_chat_template_rendering(self):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "chat_tok_test"
            Pick3HFHub.export_hf_assets(out_dir, self.config, repo_id="test/chat_repo")

            tokenizer = AutoTokenizer.from_pretrained(out_dir)
            self.assertIsNotNone(tokenizer.chat_template)

            messages = [
                {"role": "user", "content": "<DAY> 1 2 3 <SEP> <NIGHT>"}
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self.assertEqual(rendered, "<BOS><DAY> 1 2 3 <SEP> <NIGHT> <PRED> ")

            token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            self.assertEqual(token_ids, [11, 13, 1, 2, 3, 15, 14, 16])

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_mlx_model_save_and_from_pretrained_with_hf_assets(self):
        config = Pick3LLMConfig(
            vocab_size=17,
            max_seq_len=64,
            n_embd=32,
            n_head=2,
            n_layer=2,
        )
        model = Pick3LLM(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "saved_mlx_model"
            model.save_pretrained(
                model_dir,
                metrics={"val_loss": 0.55, "val_token_accuracy": 0.82},
                repo_id="test-org/pick3-mlx",
            )

            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue((model_dir / "model.safetensors").exists())
            self.assertTrue((model_dir / "generation_config.json").exists())
            self.assertTrue((model_dir / "tokenizer_config.json").exists())
            self.assertTrue((model_dir / "README.md").exists())

            # Load back
            loaded_model = Pick3LLM.from_pretrained(model_dir)
            self.assertEqual(loaded_model.config.n_embd, 32)
            self.assertEqual(loaded_model.config.n_layer, 2)

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_push_to_hub_mocked(self):
        config = Pick3LLMConfig(vocab_size=17, max_seq_len=32, n_embd=16, n_head=2, n_layer=1)
        model = Pick3LLM(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "hf_push_test"
            with patch("idaho.pick3.pick3_hf_model.Pick3HFHub.push_to_hub") as mock_push:
                mock_push.return_value = "https://huggingface.co/lottery-ai/pick3-test"
                url = model.push_to_hub(
                    repo_id="lottery-ai/pick3-test",
                    save_directory=model_dir,
                    token="test_token",
                )
                self.assertEqual(url, "https://huggingface.co/lottery-ai/pick3-test")
                mock_push.assert_called_once()

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_pytorch_causal_lm(self):
        pt_model = Pick3ForCausalLM(self.config)
        pt_model.eval()

        input_ids = torch.tensor([[11, 13, 1, 2, 3, 15, 16, 13]], dtype=torch.long)
        with torch.no_grad():
            outputs = pt_model(input_ids=input_ids)
            logits = outputs["logits"]

        self.assertEqual(logits.shape, (1, 8, 17))

        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_save_dir = Path(tmp_dir) / "pt_model"
            pt_model.save_pretrained(pt_save_dir)
            self.assertTrue((pt_save_dir / "pytorch_model.bin").exists())
            self.assertTrue((pt_save_dir / "config.json").exists())

            loaded_pt = Pick3ForCausalLM.from_pretrained(pt_save_dir)
            self.assertEqual(loaded_pt.config.hidden_size, 32)

    def test_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "cli_model"
            model_dir.mkdir()
            self.config.to_json_file(model_dir / "config.json")

            cmd = [
                sys.executable,
                "pick3_hf_model.py",
                "--help",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI help error: {res.stderr}")
            self.assertIn("Hugging Face Model & Hub Management Tool", res.stdout)

            # Test resolving existing local model dir
            cmd_local = [
                sys.executable,
                "pick3_hf_model.py",
                "--hf-repo", str(model_dir),
            ]
            res_local = subprocess.run(cmd_local, capture_output=True, text=True)
            self.assertEqual(res_local.returncode, 0, f"CLI local error: {res_local.stderr}")
            self.assertIn(str(model_dir), res_local.stdout)

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_omlx_and_mlx_lm_gpt2_weight_compatibility(self):
        """Verifies exported model.safetensors matches exact mlx-lm / oMLX GPT-2 parameter names."""
        from dataclasses import dataclass
        import mlx.nn as mlx_nn

        @dataclass
        class SimOMLXArgs:
            model_type: str = "gpt2"
            vocab_size: int = 17
            n_positions: int = 128
            n_embd: int = 64
            n_layer: int = 3
            n_head: int = 4
            layer_norm_epsilon: float = 1e-5

        class SimOMLXAttention(mlx_nn.Module):
            def __init__(self, args):
                super().__init__()
                self.c_attn = mlx_nn.Linear(args.n_embd, 3 * args.n_embd)
                self.c_proj = mlx_nn.Linear(args.n_embd, args.n_embd)

        class SimOMLXMLP(mlx_nn.Module):
            def __init__(self, args):
                super().__init__()
                self.c_fc = mlx_nn.Linear(args.n_embd, 4 * args.n_embd)
                self.c_proj = mlx_nn.Linear(4 * args.n_embd, args.n_embd)

        class SimOMLXBlock(mlx_nn.Module):
            def __init__(self, args):
                super().__init__()
                self.attn = SimOMLXAttention(args)
                self.mlp = SimOMLXMLP(args)
                self.ln_1 = mlx_nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
                self.ln_2 = mlx_nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

        class SimOMLXGPT2Model(mlx_nn.Module):
            def __init__(self, args):
                super().__init__()
                self.wte = mlx_nn.Embedding(args.vocab_size, args.n_embd)
                self.wpe = mlx_nn.Embedding(args.n_positions, args.n_embd)
                self.h = [SimOMLXBlock(args) for _ in range(args.n_layer)]
                self.ln_f = mlx_nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)

        class SimOMLXModel(mlx_nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args
                self.model = SimOMLXGPT2Model(args)

            def sanitize(self, weights):
                new_weights = {}
                for i in range(self.args.n_layer):
                    if f"h.{i}.attn.bias" in weights:
                        del weights[f"h.{i}.attn.bias"]
                    if f"h.{i}.attn.c_attn.weight" in weights:
                        weights[f"h.{i}.attn.c_attn.weight"] = weights[f"h.{i}.attn.c_attn.weight"].transpose(1, 0)
                    if f"h.{i}.attn.c_proj.weight" in weights:
                        weights[f"h.{i}.attn.c_proj.weight"] = weights[f"h.{i}.attn.c_proj.weight"].transpose(1, 0)
                    if f"h.{i}.mlp.c_fc.weight" in weights:
                        weights[f"h.{i}.mlp.c_fc.weight"] = weights[f"h.{i}.mlp.c_fc.weight"].transpose(1, 0)
                    if f"h.{i}.mlp.c_proj.weight" in weights:
                        weights[f"h.{i}.mlp.c_proj.weight"] = weights[f"h.{i}.mlp.c_proj.weight"].transpose(1, 0)
                for weight in weights:
                    if not weight.startswith("model."):
                        new_weights[f"model.{weight}"] = weights[weight]
                    else:
                        new_weights[weight] = weights[weight]
                return new_weights

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "omlx_compat_test"
            cfg = Pick3LLMConfig()
            model = Pick3LLM(cfg)
            model.save_pretrained(model_dir)

            raw_saved = dict(mx.load(str(model_dir / "model.safetensors")))
            omlx_loader = SimOMLXModel(SimOMLXArgs())
            sanitized = omlx_loader.sanitize(raw_saved)
            # Must load cleanly without raising ValueError: Received X parameters not in model
            omlx_loader.load_weights(list(sanitized.items()))


if __name__ == "__main__":
    unittest.main()
