import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from idaho.pick3.pick3_llm_ticket_model import (
    Pick3LLM,
    Pick3LLMConfig,
    Pick3Tokenizer,
    RemoteLottoModelLoader,
    build_training_sequences,
    calculate_ticket_probabilities,
    compute_frequency_and_overdue_scores,
    format_pick3_localized_response,
    generate_pick3_winning_tickets,
    load_pick3_history,
    matches_ticket_type,
    normalize_draw,
    train_pick3_llm,
)


class TestPick3LLM(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Pick3Tokenizer()
        self.sample_df = pd.DataFrame({
            "Date": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02", "2026-08-03"],
            "Draw": ["Day", "Night", "Day", "Night", "Day"],
            "Num1": [1, 5, 2, 8, 9],
            "Num2": [2, 6, 3, 4, 1],
            "Num3": [3, 7, 4, 0, 5],
        })

    def test_tokenizer(self):
        self.assertEqual(self.tokenizer.vocab_size, 75)
        draw_tokens = self.tokenizer.encode_draw("Day", 1, 2, 3)
        self.assertEqual(draw_tokens, [self.tokenizer.day_id, 1, 2, 3])
        night_tokens = self.tokenizer.encode_draw("Night", 9, 0, 5)
        self.assertEqual(night_tokens, [self.tokenizer.night_id, 9, 0, 5])

    def test_normalize_draw(self):
        self.assertEqual(normalize_draw("day"), "Day")
        self.assertEqual(normalize_draw("NIGHT"), "Night")
        self.assertEqual(normalize_draw("combo"), "Combo")
        self.assertEqual(normalize_draw("combined"), "Combo")
        self.assertEqual(normalize_draw("both"), "Both")
        with self.assertRaises(ValueError):
            normalize_draw("InvalidDraw")

    def test_ticket_types(self):
        self.assertTrue(matches_ticket_type(1, 2, 3, "6-way"))
        self.assertFalse(matches_ticket_type(1, 1, 3, "6-way"))
        self.assertTrue(matches_ticket_type(1, 1, 3, "3-way"))
        self.assertFalse(matches_ticket_type(1, 2, 3, "3-way"))
        self.assertTrue(matches_ticket_type(1, 1, 1, "exact"))

    def test_remote_model_loader_and_local_save(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_model"
            model = RemoteLottoModelLoader.fetch_or_create_model(
                remote_model_uri="remote://lottery-ai/base-pick3-test",
                local_model_dir=tmp_path,
                config=Pick3LLMConfig(n_embd=32, n_head=2, n_layer=2, max_seq_len=64),
            )
            self.assertIsInstance(model, Pick3LLM)
            self.assertTrue((tmp_path / "config.json").exists())
            self.assertTrue((tmp_path / "model.safetensors").exists())

            # Load back
            loaded_model = Pick3LLM.from_pretrained(tmp_path)
            self.assertEqual(loaded_model.config.n_embd, 32)
            self.assertEqual(loaded_model.config.n_layer, 2)

    def test_training_and_generation(self):
        # Create small dataset for rapid test
        rows = []
        for i in range(25):
            rows.append({
                "Date": f"2026-01-{i+1:02d}",
                "Draw": "Day" if i % 2 == 0 else "Night",
                "Num1": (i * 2) % 10,
                "Num2": (i * 3) % 10,
                "Num3": (i * 5) % 10,
            })
        df = pd.DataFrame(rows)

        config = Pick3LLMConfig(vocab_size=75, max_seq_len=64, n_embd=32, n_head=2, n_layer=2)
        model = Pick3LLM(config)

        inputs, targets = build_training_sequences(
            df=df,
            tokenizer=self.tokenizer,
            history_window=4,
            max_seq_len=64,
        )
        self.assertGreater(len(inputs), 0)
        self.assertEqual(inputs.shape, targets.shape)

        trained_model = train_pick3_llm(
            model=model,
            inputs=inputs,
            targets=targets,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            verbose=False,
        )

        # Generate tickets
        tickets = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type="Night",
            num_tickets=5,
            ticket_type="exact",
            history_window=4,
        )
        self.assertEqual(len(tickets), 5)
        for t in tickets:
            self.assertEqual(len(t.ticket), 3)
            self.assertGreater(t.score, 0.0)
            self.assertEqual(t.draw, "Night")

        # Test any / 6-way / 3-way
        tickets_6way = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type="Day",
            num_tickets=3,
            ticket_type="6-way",
            history_window=4,
        )
        self.assertLessEqual(len(tickets_6way), 3)
        for t in tickets_6way:
            self.assertEqual(len(set(t.ticket)), 3)

        # Test combined Day and Night draw predictions (combo)
        tickets_combo = generate_pick3_winning_tickets(
            model=trained_model,
            df=df,
            draw_type="combo",
            num_tickets=4,
            ticket_type="exact",
            history_window=4,
        )
        self.assertEqual(len(tickets_combo), 4)
        for t in tickets_combo:
            self.assertEqual(len(t.ticket), 3)
            self.assertEqual(t.draw, "Combo")

    def test_localized_responses_formatting(self):
        # Case 1: Standard 3 distinct digits
        res = format_pick3_localized_response(
            predicted_digits=[1, 2, 3],
            alternative_tickets=["456", "789", "012"],
            draw_type="Night",
        )
        self.assertEqual(res["primary"], "The next winning Pick 3 numbers are 1 2 3.")
        self.assertEqual(res["primary_dash"], "The next winning Pick 3 numbers are 1-2-3.")
        self.assertIn("4 5 6", res["alternatives"])
        self.assertIn("6-way Box", res["box_permutations"])
        self.assertIn("1-2-3", res["box_permutations"])
        self.assertIn("Sequence sum is 6", res["profile"])
        self.assertIn("The next winning Pick 3 numbers are 1 2 3.", res["narrative"])

        # Case 2: Double digits (3-way box)
        res_double = format_pick3_localized_response(
            predicted_digits=[7, 7, 2],
            draw_type="Day",
        )
        self.assertEqual(res_double["primary"], "The next winning Pick 3 numbers are 7 7 2.")
        self.assertIn("3-way Box", res_double["box_permutations"])
        self.assertIn("7-7-2", res_double["box_permutations"])

        # Case 3: Triple digits (Straight only)
        res_triple = format_pick3_localized_response(
            predicted_digits=[5, 5, 5],
            draw_type="Night",
        )
        self.assertEqual(res_triple["primary"], "The next winning Pick 3 numbers are 5 5 5.")
        self.assertIn("Straight Only", res_triple["box_permutations"])

        # Case 4: With probability matrix
        prob_mat = np.zeros((3, 10))
        prob_mat[0, 1] = 0.45
        prob_mat[1, 2] = 0.52
        prob_mat[2, 3] = 0.61
        res_probs = format_pick3_localized_response(
            predicted_digits=[1, 2, 3],
            prob_matrix=prob_mat,
            draw_type="Night",
        )
        self.assertIn("45.0%", res_probs["positional_summary"])
        self.assertIn("52.0%", res_probs["positional_summary"])
        self.assertIn("61.0%", res_probs["positional_summary"])

    def test_decode_raw_digit_stream_into_localized_text(self):
        raw_stream = "6 5 2 4 1 3 0 3 0 4 1 3 6 5 2 0 5 6 1 1 8 6 6 6 0 3 2 4 6 2 1"
        raw_token_list = [6, 5, 2, 4, 1, 3, 0, 3, 0, 4, 1, 3, 6, 5, 2, 0, 5, 6, 1, 1, 8, 6, 6, 6, 0, 3, 2, 4, 6, 2, 1]

        # Test extraction of all triplets
        triplets = self.tokenizer.decode_all_draw_digits(raw_stream)
        self.assertEqual(triplets[0], (6, 5, 2))
        self.assertEqual(triplets[1], (4, 1, 3))
        self.assertEqual(triplets[2], (0, 3, 0))
        self.assertEqual(len(triplets), 10)

        # Test localized text decoding from string
        text_from_str = self.tokenizer.decode_response_text(raw_stream, draw_type="Night")
        self.assertEqual(text_from_str, "The next winning Pick 3 numbers are 6 5 2.")

        # Test rich response dictionary from list
        resp_dict = self.tokenizer.decode_response(raw_token_list, draw_type="Night")
        self.assertEqual(resp_dict["primary"], "The next winning Pick 3 numbers are 6 5 2.")
        self.assertIn("4 1 3", resp_dict["alternatives"])
        self.assertIn("0 3 0", resp_dict["alternatives"])
        self.assertIn("6-5-2", resp_dict["box_permutations"])

        # Test tokenizer decode with localize=True
        loc_decoded = self.tokenizer.decode(raw_token_list, localize=True, draw_type="Night")
        self.assertEqual(loc_decoded, "The next winning Pick 3 numbers are 6 5 2.")

    def test_omlx_pick3_client_verification(self):
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(base_url="http://localhost:8000/v1")

        # 1. Verify prompt generators
        chat_prompt = client.create_chat_prompt(draw_type="Day")
        self.assertEqual(len(chat_prompt), 1)
        self.assertIn("Day", chat_prompt[0]["content"])

        comp_prompt = client.create_completion_prompt(draw_type="Night")
        self.assertEqual(comp_prompt, "<BOS> <PRED> <NIGHT>")

        # 2. Verify raw stream decoding via client
        raw_stream = "6 5 2 4 1 3 0 3 0 4 1 3 6 5 2 0 5 6 1 1 8 6 6 6 0 3 2 4 6 2 1"
        res_text = client.parse_response_text(raw_stream, draw_type="Night")
        self.assertEqual(res_text, "The next winning Pick 3 numbers are 6 5 2.")

        res_dict = client.parse_response(raw_stream, draw_type="Night")
        self.assertEqual(res_dict["primary"], "The next winning Pick 3 numbers are 6 5 2.")
        self.assertIn("4 1 3", res_dict["alternatives"])
        self.assertIn("0 3 0", res_dict["alternatives"])
        self.assertIn("6-5-2", res_dict["box_permutations"])

        # 3. Verify OpenAI-compatible chat response payload
        openai_chat_payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "6 5 2 4 1 3 0 3 0",
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        parsed_chat = client.parse_response(openai_chat_payload, draw_type="Night")
        self.assertEqual(parsed_chat["primary"], "The next winning Pick 3 numbers are 6 5 2.")
        self.assertIn("4 1 3", parsed_chat["alternatives"])

        # 4. Verify OpenAI-compatible completion response payload
        openai_comp_payload = {
            "id": "cmpl-test",
            "object": "text_completion",
            "choices": [
                {
                    "text": "7 8 9",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }
        parsed_comp = client.parse_response(openai_comp_payload, draw_type="Day")
        self.assertEqual(parsed_comp["primary"], "The next winning Pick 3 numbers are 7 8 9.")
        self.assertEqual(parsed_comp["draw"], "Day")

        # 5. Verify streaming chunk extraction and None delta resilience
        chunk_init = {"choices": [{"delta": {"role": "assistant", "content": None}}]}
        self.assertEqual(client.extract_delta_content(chunk_init), "")

        chunk_data = {"choices": [{"delta": {"content": "6 "}}]}
        self.assertEqual(client.extract_delta_content(chunk_data), "6 ")

        chunk_finish = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.assertEqual(client.extract_delta_content(chunk_finish), "")

        chunk_empty = {}
        self.assertEqual(client.extract_delta_content(chunk_empty), "")

        # 6. Verify client parse_response with None or empty payloads
        none_payload = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        res_none = client.parse_response(none_payload, draw_type="Night")
        self.assertIn("primary", res_none)

    def test_cli_execution(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_csv = Path(tmp_dir) / "preds.csv"
            model_dir = Path(tmp_dir) / "test_model_cli"

            cmd = [
                sys.executable,
                "idaho/pick3/pick3_llm_ticket_model.py",
                "--csv", "idaho/pick3/data/idaho_pick3_history.csv",
                "--local-model-dir", str(model_dir),
                "--draw", "both",
                "--tickets", "3",
                "--epochs", "1",
                "--batch-size", "64",
                "--output", str(output_csv),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}")
            self.assertTrue(output_csv.exists())
            df_out = pd.read_csv(output_csv)
            self.assertEqual(len(df_out), 6)  # 3 Day + 3 Night

            # Test combo draw CLI
            output_combo_csv = Path(tmp_dir) / "preds_combo.csv"
            cmd_combo = [
                sys.executable,
                "idaho/pick3/pick3_llm_ticket_model.py",
                "--csv", "idaho/pick3/data/idaho_pick3_history.csv",
                "--local-model-dir", str(model_dir),
                "--draw", "combo",
                "--tickets", "5",
                "--epochs", "1",
                "--batch-size", "64",
                "--output", str(output_combo_csv),
            ]
            res_combo = subprocess.run(cmd_combo, capture_output=True, text=True)
            self.assertEqual(res_combo.returncode, 0, f"Combo CLI failed: {res_combo.stderr}")
            self.assertTrue(output_combo_csv.exists())
            df_combo_out = pd.read_csv(output_combo_csv)
            self.assertEqual(len(df_combo_out), 5)
            self.assertTrue((df_combo_out["draw"] == "Combo").all())


    def test_tokenizer_count_tokens(self):
        """Tokenizer count_tokens estimates token count for plain text."""
        tok = Pick3Tokenizer()
        # Empty string
        self.assertEqual(tok.count_tokens(""), 0)
        # Simple digits
        count = tok.count_tokens("6 5 2")
        self.assertGreater(count, 0)
        # Longer text
        count = tok.count_tokens("Day: 1 2 3 Night: 4 5 6")
        self.assertGreater(count, 3)

    def test_tokenizer_truncate_to_fit(self):
        """Truncate_to_fit drops oldest draws until prompt fits budget."""
        tok = Pick3Tokenizer()
        # Create 50 draw records (more than will fit in 64 tokens)
        records = [
            {"draw": "Day" if i % 2 == 0 else "Night", "num1": i % 10, "num2": (i + 1) % 10, "num3": (i + 2) % 10}
            for i in range(50)
        ]
        target = {"draw": "Night"}

        # With max_seq_len=64, should truncate significantly
        truncated = tok.truncate_to_fit(records, target, max_seq_len=64)
        self.assertLess(len(truncated), 50)
        # Verify the truncated set fits
        prompt = tok.encode_history_prompt(truncated, target["draw"])
        total = len(prompt) + 4  # +4 for target digits + EOS
        self.assertLessEqual(total, 64)

        # With max_seq_len=128, should fit more
        truncated_128 = tok.truncate_to_fit(records, target, max_seq_len=128)
        self.assertGreaterEqual(len(truncated_128), len(truncated))

        # With very large budget, all records should fit
        all_fit = tok.truncate_to_fit(records, target, max_seq_len=512)
        self.assertEqual(len(all_fit), 50)

        # Empty records
        self.assertEqual(tok.truncate_to_fit([], target, max_seq_len=64), [])

    def test_tokenizer_truncate_history_summary(self):
        """Truncate_history_summary removes oldest lines until text fits budget."""
        tok = Pick3Tokenizer()
        # Create a long history summary (many draw lines)
        lines = [f"Day: {i} {(i+1)%10} {(i+2)%10}" for i in range(30)]
        summary = "\n".join(lines)

        # Truncate to fit within a small budget
        truncated = tok.truncate_history_summary(summary, max_tokens=30)
        truncated_lines = truncated.strip().split("\n") if truncated else []
        self.assertLess(len(truncated_lines), 30)

        # With large budget, should keep most lines
        truncated_large = tok.truncate_history_summary(summary, max_tokens=200)
        self.assertGreaterEqual(len(truncated_large.strip().split("\n")), len(truncated_lines))

        # Empty summary
        self.assertEqual(tok.truncate_history_summary("", max_tokens=64), "")

    def test_client_truncates_oversized_history(self):
        """OMLXPick3Client auto-truncates history records to fit context limit."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(base_url="http://localhost:8000/v1", max_context_tokens=64)

        # Create 50 draw records (too many for 64-token budget)
        records = [
            {"draw": "Day" if i % 2 == 0 else "Night", "num1": i % 10, "num2": (i + 1) % 10, "num3": (i + 2) % 10}
            for i in range(50)
        ]

        # With history_records, should truncate and build prompt
        messages = client.create_chat_prompt(
            draw_type="Night",
            history_records=records,
        )
        self.assertEqual(len(messages), 1)
        # The content should include some history (not all 50)
        content = messages[0]["content"]
        self.assertIn("Predict the next Pick 3 Night draw numbers.", content)
        # Should have some history lines
        history_part = content.split("\nPredict")[0]
        history_lines = [l for l in history_part.strip().split("\n") if l.strip()]
        self.assertLess(len(history_lines), 50)
        self.assertGreater(len(history_lines), 0)

    def test_client_truncates_text_history_summary(self):
        """OMLXPick3Client auto-truncates plain-text history summaries."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(base_url="http://localhost:8000/v1", max_context_tokens=40)

        # Create a long text summary
        lines = [f"Day: {i} {(i+1)%10} {(i+2)%10}" for i in range(30)]
        summary = "\n".join(lines)

        messages = client.create_chat_prompt(
            draw_type="Day",
            history_summary=summary,
        )
        content = messages[0]["content"]
        # Should be truncated
        self.assertIn("Predict the next Pick 3 Day draw numbers.", content)

    def test_client_max_context_tokens_override(self):
        """Client respects per-call max_context_tokens override."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(base_url="http://localhost:8000/v1", max_context_tokens=128)

        records = [
            {"draw": "Night", "num1": i % 10, "num2": (i + 1) % 10, "num3": (i + 2) % 10}
            for i in range(30)
        ]

        # Default budget (128)
        msg_default = client.create_chat_prompt(draw_type="Night", history_records=records)
        # Override with smaller budget (32)
        msg_small = client.create_chat_prompt(draw_type="Night", history_records=records, max_context_tokens=32)

        # Smaller budget should have fewer history lines
        default_lines = len(msg_default[0]["content"].split("\nPredict")[0].strip().split("\n"))
        small_lines = len(msg_small[0]["content"].split("\nPredict")[0].strip().split("\n"))
        self.assertLessEqual(small_lines, default_lines)

    def test_client_init_max_context_tokens(self):
        """Client stores max_context_tokens from constructor."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(max_context_tokens=256)
        self.assertEqual(client.max_context_tokens, 256)

        default_client = OMLXPick3Client()
        self.assertEqual(default_client.max_context_tokens, 128)

    def test_sliding_window_preserves_system_messages(self):
        """Sliding window keeps all system messages regardless of max_history."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client()
        messages = [
            {"role": "system", "content": "You are a lottery predictor."},
            {"role": "user", "content": "What are the numbers?"},
            {"role": "assistant", "content": "The numbers are 1 2 3."},
            {"role": "user", "content": "What about tonight?"},
            {"role": "assistant", "content": "Tonight: 4 5 6."},
        ]
        # Keep only last 2 non-system messages
        result = client.sliding_window_messages(messages, max_history=2)
        # System message should be preserved
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "You are a lottery predictor.")
        # Only 2 non-system messages should remain
        non_system = [m for m in result if m["role"] != "system"]
        self.assertEqual(len(non_system), 2)
        # Should be the last 2 messages
        self.assertEqual(non_system[0]["content"], "What about tonight?")
        self.assertEqual(non_system[1]["content"], "Tonight: 4 5 6.")

    def test_sliding_window_no_truncation_when_fitting(self):
        """Sliding window returns all messages when within limit."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client()
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = client.sliding_window_messages(messages, max_history=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["role"], "system")

    def test_sliding_window_none_max_history(self):
        """Sliding window with None max_history returns messages unchanged."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(max_history=None)
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Reply 2"},
        ]
        result = client.sliding_window_messages(messages)
        self.assertEqual(len(result), 5)

    def test_sliding_window_uses_constructor_max_history(self):
        """Sliding window uses max_history from constructor when not overridden."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(max_history=2)
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Reply 2"},
        ]
        # No per-call override — should use constructor's max_history=2
        result = client.sliding_window_messages(messages)
        non_system = [m for m in result if m["role"] != "system"]
        self.assertEqual(len(non_system), 2)
        self.assertEqual(result[0]["role"], "system")

    def test_sliding_window_per_call_override(self):
        """Per-call max_history overrides constructor default."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(max_history=4)
        messages = [
            {"role": "system", "content": "Instructions."},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Reply 3"},
        ]
        # Override to 2
        result = client.sliding_window_messages(messages, max_history=2)
        non_system = [m for m in result if m["role"] != "system"]
        self.assertEqual(len(non_system), 2)

    def test_create_chat_prompt_with_messages_sliding_window(self):
        """create_chat_prompt applies sliding window when messages are provided."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client()
        messages = [
            {"role": "system", "content": "You are a lottery predictor."},
            {"role": "user", "content": "Predict Day draw."},
            {"role": "assistant", "content": "Day: 1 2 3."},
            {"role": "user", "content": "Predict Night draw."},
            {"role": "assistant", "content": "Night: 4 5 6."},
        ]
        result = client.create_chat_prompt(messages=messages, max_history=2)
        # System preserved, only last 2 non-system kept
        self.assertEqual(result[0]["role"], "system")
        non_system = [m for m in result if m["role"] != "system"]
        self.assertEqual(len(non_system), 2)

    def test_sliding_window_multiple_system_messages(self):
        """Sliding window preserves multiple system messages."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client()
        messages = [
            {"role": "system", "content": "System instruction 1."},
            {"role": "user", "content": "Msg 1"},
            {"role": "system", "content": "System instruction 2."},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Reply 2"},
        ]
        result = client.sliding_window_messages(messages, max_history=2)
        system_msgs = [m for m in result if m["role"] == "system"]
        non_system = [m for m in result if m["role"] != "system"]
        # Both system messages preserved
        self.assertEqual(len(system_msgs), 2)
        # Only last 2 non-system messages
        self.assertEqual(len(non_system), 2)


    def test_client_frequency_and_presence_penalty_defaults(self):
        """Client initializes frequency and presence penalties to 0.5 by default."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client()
        self.assertEqual(client.frequency_penalty, 0.5)
        self.assertEqual(client.presence_penalty, 0.5)
        self.assertEqual(client.stop_sequences, ["<EOS>", "[EOS]"])

    def test_client_custom_penalty_values(self):
        """Client accepts custom penalty and stop sequence values."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(
            frequency_penalty=1.0,
            presence_penalty=0.5,
            stop_sequences=["<EOS>", "STOP"]
        )
        self.assertEqual(client.frequency_penalty, 1.0)
        self.assertEqual(client.presence_penalty, 0.5)
        self.assertEqual(client.stop_sequences, ["<EOS>", "STOP"])

    def test_client_reset_session(self):
        """reset_session resets generation parameters to defaults."""
        from idaho.pick3.pick3_llm_ticket_model import OMLXPick3Client

        client = OMLXPick3Client(frequency_penalty=0.0, presence_penalty=0.0)
        self.assertEqual(client.frequency_penalty, 0.0)
        self.assertEqual(client.presence_penalty, 0.0)

        client.reset_session()
        self.assertEqual(client.frequency_penalty, 0.5)
        self.assertEqual(client.presence_penalty, 0.5)
        self.assertEqual(client.stop_sequences, ["<EOS>", "[EOS]"])

    def test_model_generate_with_frequency_penalty(self):
        """Pick3LLM generation applies frequency penalty to reduce repetition."""
        import mlx.core as mx
        from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, Pick3Tokenizer

        config = Pick3LLMConfig(vocab_size=75, max_seq_len=128, n_embd=32, n_head=2, n_layer=2)
        model = Pick3LLM(config)
        mx.eval(model.parameters())

        tokenizer = Pick3Tokenizer()
        history = [
            {"draw": "Day", "num1": 1, "num2": 2, "num3": 3},
            {"draw": "Night", "num1": 4, "num2": 5, "num3": 6},
        ]
        prompt = tokenizer.encode_history_prompt(history, "Night")

        # Generate with frequency penalty
        response = model.generate_response_text(
            prompt_tokens=prompt,
            draw_type="Night",
            temperature=0.8,
            frequency_penalty=1.0,
            presence_penalty=0.5,
        )
        self.assertIn("Pick 3", response)

    def test_model_generate_with_stop_tokens(self):
        """Pick3LLM generation stops at configured stop token IDs."""
        import mlx.core as mx
        from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, Pick3Tokenizer

        config = Pick3LLMConfig(vocab_size=75, max_seq_len=128, n_embd=32, n_head=2, n_layer=2)
        model = Pick3LLM(config)
        mx.eval(model.parameters())

        tokenizer = Pick3Tokenizer()
        history = [
            {"draw": "Day", "num1": 1, "num2": 2, "num3": 3},
        ]
        prompt = tokenizer.encode_history_prompt(history, "Night")

        # Without stop tokens, should generate 3 digits
        response_no_stop = model.generate_response_text(
            prompt_tokens=prompt,
            draw_type="Night",
            temperature=0.8,
        )
        self.assertIn("Pick 3", response_no_stop)

        # With stop tokens, generation may stop early — verify it runs without error
        response_with_stop = model.generate_response_text(
            prompt_tokens=prompt,
            draw_type="Night",
            temperature=0.8,
            stop=[0],
        )
        # Should return a valid response string
        self.assertIsInstance(response_with_stop, str)
        self.assertGreater(len(response_with_stop), 0)

    def test_model_generate_localized_with_penalties(self):
        """Pick3LLM generate_localized_prediction supports penalty parameters."""
        import mlx.core as mx
        from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3LLMConfig, Pick3Tokenizer

        config = Pick3LLMConfig(vocab_size=75, max_seq_len=128, n_embd=32, n_head=2, n_layer=2)
        model = Pick3LLM(config)
        mx.eval(model.parameters())

        tokenizer = Pick3Tokenizer()
        history = [
            {"draw": "Day", "num1": 1, "num2": 2, "num3": 3},
        ]
        prompt = tokenizer.encode_history_prompt(history, "Night")

        result = model.generate_localized_prediction(
            prompt_tokens=prompt,
            draw_type="Night",
            temperature=0.8,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        self.assertIn("primary", result)
        self.assertIn("Pick 3", result["primary"])


if __name__ == "__main__":
    unittest.main()
