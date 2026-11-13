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
            vocab_size=75,
            max_position_embeddings=64,
            hidden_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
        )

    def test_config_serialization(self):
        data = self.config.to_dict()
        self.assertEqual(data["model_type"], "gpt2")
        self.assertEqual(data["vocab_size"], 75)
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

        # Test id2label and label2id serialization
        self.assertIn("id2label", data)
        self.assertIn("label2id", data)
        self.assertEqual(data["id2label"]["0"], "0")
        self.assertEqual(data["id2label"]["1"], "1")
        self.assertEqual(data["id2label"]["10"], "<PAD>")
        self.assertEqual(data["id2label"]["16"], "<PRED>")
        self.assertEqual(data["label2id"]["0"], 0)
        self.assertEqual(data["label2id"]["<PRED>"], 16)

        # Roundtrip from dict
        loaded = Pick3HFConfig.from_dict(data)
        self.assertEqual(loaded.model_type, "gpt2")
        self.assertEqual(loaded.vocab_size, 75)
        self.assertEqual(loaded.hidden_size, 32)
        self.assertEqual(loaded.num_attention_heads, 2)
        self.assertEqual(loaded.layer_norm_epsilon, 1e-5)
        self.assertEqual(loaded.id2label[0], "0")
        self.assertEqual(loaded.id2label[10], "<PAD>")
        self.assertEqual(loaded.label2id["<PAD>"], 10)
        self.assertEqual(loaded.map_id_to_label(1), "1")
        self.assertEqual(loaded.map_id_to_label("13"), "<DAY>")
        self.assertEqual(loaded.map_ids_to_labels([1, 2, 3]), ["1", "2", "3"])

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
            self.assertEqual(tok_json["added_tokens"][3]["special"], False)
            self.assertEqual(tok_json["added_tokens"][4]["special"], False)
            self.assertEqual(tok_json["added_tokens"][5]["special"], False)
            self.assertEqual(tok_json["added_tokens"][6]["special"], False)
            self.assertEqual(tok_json["decoder"]["type"], "Sequence")
            self.assertTrue(tok_json["decoder"]["decoders"])

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
            self.assertEqual(loaded_tok.vocab_size, 75)
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

            draw_text = "<DAY> 6 5 2 <SEP> <NIGHT> 4 1 3 <EOS>"
            draw_ids = tokenizer.encode(draw_text)
            self.assertEqual(draw_ids, [13, 6, 5, 2, 15, 14, 4, 1, 3, 12])
            # Decode produces labeled text; exact spacing depends on fast tokenizer decoder
            decoded = tokenizer.decode(draw_ids)
            self.assertIn("6 5 2", decoded)
            self.assertIn("4 1 3", decoded)

            # Test messages with content=None (chat streaming initialization and tool chunks)
            none_messages = [
                {"role": "system", "content": None},
                {"role": "user", "content": "<DAY> 1 2 3 <SEP> <NIGHT>"},
                {"role": "assistant", "content": None},
            ]
            rendered_none = tokenizer.apply_chat_template(none_messages, tokenize=False, add_generation_prompt=True)
            self.assertEqual(rendered_none, "<BOS><DAY> 1 2 3 <SEP> <NIGHT> <PRED> ")

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_mlx_model_save_and_from_pretrained_with_hf_assets(self):
        config = Pick3LLMConfig(
            vocab_size=75,
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
        config = Pick3LLMConfig(vocab_size=75, max_seq_len=32, n_embd=16, n_head=2, n_layer=1)
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

        self.assertEqual(logits.shape, (1, 8, 75))

        with tempfile.TemporaryDirectory() as tmp_dir:
            pt_save_dir = Path(tmp_dir) / "pt_model"
            pt_model.save_pretrained(pt_save_dir)
            self.assertTrue((pt_save_dir / "pytorch_model.bin").exists())
            self.assertTrue((pt_save_dir / "config.json").exists())

            loaded_pt = Pick3ForCausalLM.from_pretrained(pt_save_dir)
            self.assertEqual(loaded_pt.config.hidden_size, 32)

            # Test PyTorch localized response text generation
            prompt = [11, 13, 1, 2, 3, 15, 16, 14]
            resp_text = loaded_pt.generate_response_text(prompt_tokens=prompt, draw_type="Night")
            self.assertTrue(resp_text.startswith("The next winning Pick 3 numbers are"))
            localized_dict = loaded_pt.generate_localized_prediction(prompt_tokens=prompt, draw_type="Night")
            self.assertIn("primary", localized_dict)
            self.assertIn("alternatives", localized_dict)
            self.assertIn("box_permutations", localized_dict)

    def test_tokenizer_and_hub_decode_localized_response(self):
        tokenizer = Pick3Tokenizer()
        # Decode standard tokens
        raw_text = tokenizer.decode([1, 2, 3])
        self.assertEqual(raw_text, "1 2 3")

        # Decode with skip_special_tokens
        stripped = tokenizer.decode([11, 13, 1, 2, 3, 15, 12], skip_special_tokens=True)
        self.assertEqual(stripped, "1 2 3")

        # Decode with localize=True
        localized = tokenizer.decode([1, 2, 3], localize=True)
        self.assertEqual(localized, "The next winning Pick 3 numbers are 1 2 3.")

        # Decode response dictionary
        resp = tokenizer.decode_response([4, 5, 6], draw_type="Day")
        self.assertEqual(resp["primary"], "The next winning Pick 3 numbers are 4 5 6.")
        self.assertIn("4-5-6", resp["box_permutations"])

        # Decode through Pick3HFHub helper
        hub_resp = Pick3HFHub.decode_response_text([7, 8, 9], draw_type="Night")
        self.assertEqual(hub_resp, "The next winning Pick 3 numbers are 7 8 9.")

        # Test mapping integer IDs back to label names using model.config.id2label
        labels = tokenizer.decode_ids_to_labels([11, 13, 1, 2, 3, 15, 16, 14])
        self.assertEqual(labels, ["<BOS>", "<DAY>", "1", "2", "3", "<SEP>", "<PRED>", "<NIGHT>"])

        # Test skipping special tokens
        digits_only = tokenizer.decode_ids_to_labels([11, 13, 1, 2, 3, 15, 16, 14], skip_special_tokens=True)
        self.assertEqual(digits_only, ["1", "2", "3"])

        # Test hub decode_ids_to_labels
        hub_labels = Pick3HFHub.decode_ids_to_labels([1, 2, 3])
        self.assertEqual(hub_labels, ["1", "2", "3"])

        # Test hub decode with continuous stream
        raw_stream = "6 5 2 4 1 3 0 3 0 4 1 3 6 5 2 0 5 6 1 1 8 6 6 6 0 3 2 4 6 2 1"
        hub_text = Pick3HFHub.decode_response_text(raw_stream, draw_type="Night")
        self.assertEqual(hub_text, "The next winning Pick 3 numbers are 6 5 2.")
        hub_dict = Pick3HFHub.decode_response(raw_stream, draw_type="Night")
        self.assertEqual(hub_dict["primary"], "The next winning Pick 3 numbers are 6 5 2.")
        self.assertIn("4 1 3", hub_dict["alternatives"])

        # Test creating oMLX client via hub
        omlx_client = Pick3HFHub.create_omlx_client(base_url="http://localhost:8000/v1")
        self.assertEqual(omlx_client.base_url, "http://localhost:8000/v1")
        self.assertEqual(omlx_client.model_name, "pick3_llm_model")
        parsed_stream = omlx_client.parse_response_text(raw_stream, draw_type="Night")
        self.assertEqual(parsed_stream, "The next winning Pick 3 numbers are 6 5 2.")

    @unittest.skipIf(mx is None, "MLX is not installed")
    def test_mlx_generate_response_text(self):
        config = Pick3LLMConfig(vocab_size=75, max_seq_len=32, n_embd=16, n_head=2, n_layer=1)
        model = Pick3LLM(config)
        prompt = [11, 13, 1, 2, 3, 15, 16, 14]
        resp_text = model.generate_response_text(prompt_tokens=prompt, draw_type="Night")
        self.assertTrue(resp_text.startswith("The next winning Pick 3 numbers are"))

        loc_dict = model.generate_localized_prediction(prompt_tokens=prompt, draw_type="Night")
        self.assertIn("primary", loc_dict)
        self.assertIn("alternatives", loc_dict)

        # Test model.decode_ids_to_labels using model.config.id2label
        mapped_labels = model.decode_ids_to_labels([1, 2, 3])
        self.assertEqual(mapped_labels, ["1", "2", "3"])
        self.assertEqual(model.id2label[1], "1")
        self.assertEqual(model.label2id["1"], 1)

    def test_cli_execution(self):
        repo_root = Path(__file__).resolve().parent.parent
        script_path = repo_root / "pick3_hf_model.py"
        if not script_path.exists():
            script_path = repo_root / "idaho" / "pick3" / "pick3_hf_model.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / "cli_model"
            model_dir.mkdir()
            self.config.to_json_file(model_dir / "config.json")

            cmd = [
                sys.executable,
                str(script_path),
                "--help",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
            self.assertEqual(res.returncode, 0, f"CLI help error: {res.stderr}")
            self.assertIn("Hugging Face Model & Hub Management Tool", res.stdout)

            # Test resolving existing local model dir
            cmd_local = [
                sys.executable,
                str(script_path),
                "--hf-repo", str(model_dir),
            ]
            res_local = subprocess.run(cmd_local, capture_output=True, text=True, cwd=str(repo_root))
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
            vocab_size: int = 75
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
