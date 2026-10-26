import os
import re
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SEQUENCE_LENGTH = 10
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_MODEL_DIRS = (
    REPO_ROOT / "idaho" / "pick3" / "models" / "pick3_llm_model",
    REPO_ROOT / "models" / "pick3_llm_model",
)
OS_EXPORT_DIR = SCRIPT_DIR / "generated"
_UNLOADED = object()
_prediction_generator = _UNLOADED


class DrawingHistory(list):
    def __init__(self, drawings=(), draw_types=None):
        super().__init__(drawings)
        self.draw_types = list(draw_types) if draw_types is not None else [None] * len(self)


def _is_local_model_dir(path):
    path = Path(path).expanduser()
    return (path / "config.json").is_file() and any(
        (path / filename).is_file() for filename in ("model.safetensors", "model.npz")
    )


def resolve_model_source(model_override=None, model_dirs=None):
    """Choose an explicit local/Hub override, then bundled local checkpoints."""
    if model_override is None:
        model_override = os.environ.get("PICK3_HF_MODEL")

    if model_override and _is_local_model_dir(model_override):
        return "local", Path(model_override).expanduser()

    for model_dir in (DEFAULT_MODEL_DIRS if model_dirs is None else model_dirs):
        if _is_local_model_dir(model_dir):
            return "local", Path(model_dir)

    if model_override:
        return "hub", model_override
    return "none", None


def _sample_digit(logits, temperature, top_k, top_p):
    logits = np.asarray(logits, dtype=np.float64).copy()
    temperature = max(float(temperature), 1e-8)
    logits /= temperature

    if top_k and int(top_k) > 0 and int(top_k) < len(logits):
        logits[np.argsort(logits)[:-int(top_k)]] = -np.inf

    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    top_p = min(max(float(top_p), np.finfo(float).eps), 1.0)
    if top_p < 1.0:
        sorted_indices = np.argsort(probabilities)[::-1]
        cumulative = np.cumsum(probabilities[sorted_indices])
        remove = cumulative - probabilities[sorted_indices] >= top_p
        probabilities[sorted_indices[remove]] = 0
        probabilities /= probabilities.sum()

    return int(np.random.choice(len(probabilities), p=probabilities))


class LocalPick3Generator:
    def __init__(self, model, tokenizer, mlx):
        self.model = model
        self.tokenizer = tokenizer
        self.mlx = mlx

    def __call__(self, history, temperature, top_k, top_p):
        tokens = [self.tokenizer.bos_id]
        draw_types = getattr(history, "draw_types", [])
        for index, drawing in enumerate(history[-SEQUENCE_LENGTH:]):
            draw_type = draw_types[index] if index < len(draw_types) else None
            draw_marker = self.tokenizer.night_id if draw_type == "Night" else self.tokenizer.day_id
            tokens.extend([draw_marker, *map(int, drawing), self.tokenizer.sep_id])
        tokens.extend([self.tokenizer.pred_id, self.tokenizer.day_id])

        logits_rows = []
        predicted_numbers = []
        for _ in range(3):
            input_ids = self.mlx.array([tokens], dtype=self.mlx.int32)
            next_logits = np.asarray(self.model(input_ids)[0, -1, :10])
            logits_rows.append(next_logits)
            digit = _sample_digit(next_logits, temperature, top_k, top_p)
            predicted_numbers.append(digit)
            tokens.append(digit)

        return torch.tensor(np.asarray(logits_rows), dtype=torch.float32), predicted_numbers


def _load_prediction_generator():
    source_type, source = resolve_model_source()
    if source_type == "local":
        from idaho.pick3.pick3_llm_ticket_model import Pick3LLM, Pick3Tokenizer
        import mlx.core as mx

        model = Pick3LLM.from_pretrained(source)
        model.eval()
        tokenizer = Pick3Tokenizer()
        print(f"Loaded local Pick 3 model from: {source}")
        return LocalPick3Generator(model, tokenizer, mx)

    if source_type == "none":
        local_paths = ", ".join(str(path) for path in DEFAULT_MODEL_DIRS)
        raise FileNotFoundError(
            "No local Pick 3 checkpoint was found at: "
            f"{local_paths}. Set PICK3_HF_MODEL to opt in to a Hugging Face model."
        )

    try:
        from transformers import pipeline

        print(f"Loading explicitly configured Hugging Face model: {source}...")
        generator_pipeline = pipeline(
            task="idaho-pick3-generation",
            model=source,
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        underlying_model = generator_pipeline.model
        underlying_model.eval()

        def predict(history, temperature, top_k, top_p):
            input_tensor = torch.tensor([history], dtype=torch.float32)
            with torch.no_grad():
                outputs = underlying_model(sequence_history=input_tensor)
                logits = outputs["logits"].squeeze(0)
            result = generator_pipeline(history, temperature=temperature, top_k=top_k, top_p=top_p)
            predicted_numbers = [int(num) for num in re.findall(r"\d", result["generated_text"])]
            return logits, predicted_numbers

        return predict
    except Exception as error:
        local_paths = ", ".join(str(path) for path in DEFAULT_MODEL_DIRS)
        print(
            f"Hugging Face model {source!r} could not be loaded ({error}). "
            f"No bundled local checkpoint was found at: {local_paths}. "
            "Using mock simulation for demonstration only."
        )
        return None


def get_prediction_generator():
    global _prediction_generator
    if _prediction_generator is _UNLOADED:
        _prediction_generator = _load_prediction_generator()
    return _prediction_generator


# ---------------------------------------------------------
# 2. DATA PARSING ENGINE
# ---------------------------------------------------------
def parse_drawing_data(file_path_or_text):
    parsed_drawings = DrawingHistory()
    file_path = None
    if isinstance(file_path_or_text, (str, os.PathLike)) and "\n" not in str(file_path_or_text):
        candidate_path = Path(file_path_or_text)
        try:
            if candidate_path.is_file():
                file_path = candidate_path
        except OSError:
            pass

    if file_path is not None and file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, header=None, dtype=str, keep_default_na=False)
        header = [str(value).strip().lower() for value in df.iloc[0].tolist()] if not df.empty else []
        required_columns = {"date", "draw", "num1", "num2", "num3"}
        if required_columns.issubset(header):
            df.columns = header
            df = df.iloc[1:].copy()
            df["_date_order"] = pd.to_datetime(df["date"], errors="coerce")
            df["_draw_order"] = df["draw"].str.lower().map({"day": 0, "night": 1})
            df = df.sort_values(["_date_order", "_draw_order"], kind="stable", na_position="last")
            for _, row in df.iterrows():
                digits = [str(row[column]).strip() for column in ("num1", "num2", "num3")]
                if all(re.fullmatch(r"\d", value) for value in digits):
                    parsed_drawings.append([int(value) for value in digits])
                    parsed_drawings.draw_types.append(
                        "Night" if str(row["draw"]).strip().lower() == "night" else "Day"
                    )
            return parsed_drawings

        for _, row in df.iterrows():
            fields = [str(value).strip() for value in row.tolist()]
            if len(fields) >= 3 and all(re.fullmatch(r"\d", value) for value in fields[-3:]):
                parsed_drawings.append([int(value) for value in fields[-3:]])
                parsed_drawings.draw_types.append(None)
            elif len(fields) == 1:
                text_history = _parse_text_lines(fields[0].splitlines())
                parsed_drawings.extend(text_history)
                parsed_drawings.draw_types.extend(text_history.draw_types)
        return parsed_drawings
    elif file_path is not None and file_path.suffix.lower() == ".txt":
        lines = file_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = str(file_path_or_text).strip().splitlines()

    return _parse_text_lines(lines)


def _parse_text_lines(lines):
    parsed_drawings = DrawingHistory()
    for line in lines:
        if not line.strip():
            continue
        draw_text = line.rsplit(":", 1)[-1]
        digits = [int(num) for num in re.findall(r"(?<!\d)\d(?!\d)", draw_text)]
        if len(digits) == 3:
            parsed_drawings.append(digits)
            draw_type = "Night" if re.search(r"\bnight\b", line, re.IGNORECASE) else None
            if draw_type is None and re.search(r"\bday\b", line, re.IGNORECASE):
                draw_type = "Day"
            parsed_drawings.draw_types.append(draw_type)
    return parsed_drawings


# ---------------------------------------------------------
# 3. ANOMALY TRACKING & AUDIT ENGINE
# ---------------------------------------------------------
def analyze_sequence_anomalies(history_window, prediction):
    """
    Performs audit tracking across historical inputs and projected targets:
    - Flags consecutive duplicate drawings
    - Tracks position-wise character entropy drops
    - Measures statistical skewness variances
    """
    alerts = []
    history_arr = np.array(history_window)  # Shape: [10, 3]

    # 1. Audit Check: Concurrent Identity Repetition
    for idx in range(len(history_window) - 1):
        if history_window[idx] == history_window[idx + 1]:
            alerts.append(
                f"⚠️ FLAG: Consecutive matching sequences detected at historical step {idx + 1} ({history_window[idx]}).")

    # 2. Audit Check: Positional Entropy Drops (Lack of uniform dispersion)
    for ball_pos in range(3):
        unique_elements, counts = np.unique(history_arr[:, ball_pos], return_counts=True)
        # If a single digit occupies more than 40% of the active context window, flag variance skew
        for element, count in zip(unique_elements, counts):
            if count >= 4:
                alerts.append(
                    f"⚠️ SKEW DETECTED: Digit [{element}] occupies {count * 10}% of the historical track for Ball Position {ball_pos + 1}.")

    # 3. Audit Check: Prediction Convergence/Symmetry
    if len(set(prediction)) == 1:
        alerts.append(
            f"🚨 ANOMALY: Prediction engine output contains identical values across all positions ({prediction[0]}-{prediction[1]}-{prediction[2]}).")

    if not alerts:
        alerts.append(
            "✅ SYSTEM NORMAL: No anomalous structural skew or repetitive patterns detected in the active context window.")

    return "\n".join(alerts)


# ---------------------------------------------------------
# 4. BUSINESS INTROSPECTION & EXCEL EXPORT ENGINE
# ---------------------------------------------------------
def compile_and_export_report(history, temperature, top_k, top_p, prediction_generator=None, export_dir=None):
    if len(history) < SEQUENCE_LENGTH:
        error_msg = f"❌ Error: Found {len(history)} valid records. Exactly {SEQUENCE_LENGTH} are required."
        return error_msg, pd.DataFrame(), None

    history_draw_types = getattr(history, "draw_types", None)
    target_draw_types = history_draw_types[-SEQUENCE_LENGTH:] if history_draw_types is not None else None
    target_window = DrawingHistory(history[-SEQUENCE_LENGTH:], target_draw_types)

    # 1. Evaluate model metrics, loading a local checkpoint only when first needed.
    if prediction_generator is None:
        prediction_generator = get_prediction_generator()
    if prediction_generator is not None:
        logits, predicted_numbers = prediction_generator(target_window, temperature, top_k, top_p)
    else:
        logits = torch.randn(3, 10) * 1.5
        predicted_numbers = [np.random.randint(0, 10) for _ in range(3)]

    logits = torch.as_tensor(logits, dtype=torch.float32)
    predicted_numbers = [int(num) for num in predicted_numbers]
    if logits.shape != (3, 10) or len(predicted_numbers) != 3 or any(not 0 <= num <= 9 for num in predicted_numbers):
        raise ValueError("Pick 3 predictor must return a 3x10 logits matrix and three digits from 0 to 9")

    # Calculate Probability Matrix Percentages
    prob_matrix = F.softmax(logits, dim=-1).cpu().numpy() * 100

    # 2. Compute Structural Anomaly Summary
    anomaly_log = analyze_sequence_anomalies(target_window, predicted_numbers)

    # 3. Assemble Output Table Frames
    prob_records = []
    for digit in range(10):
        prob_records.append({
            "Digit (Token)": f"[{digit}]",
            "Ball 1 Prob (%)": f"{prob_matrix[0][digit]:.2f}%",
            "Ball 2 Prob (%)": f"{prob_matrix[1][digit]:.2f}%",
            "Ball 3 Prob (%)": f"{prob_matrix[2][digit]:.2f}%"
        })
    df_probs = pd.DataFrame(prob_records)

    # 4. Compile Executive Natural Text Presentation Summary
    prediction_str = "-".join(str(d) for d in predicted_numbers)
    report_text = (
        f"========================================================================\n"
        f"              IDAHO PICK-3 SEQUENCE INTELLIGENCE EXECUTIVE REPORT       \n"
        f"========================================================================\n"
        f"▶ TARGET PREDICTED DRAWING:  [{prediction_str}]\n"
        f"------------------------------------------------------------------------\n"
        f"▶ INFERENCE TUNING PARAMETERS:\n"
        f"  - Sampling Temperature : {temperature}   - Top-K: {top_k if top_k else 'Off'}   - Top-P: {top_p}\n"
        f"========================================================================\n"
        f"📈 AUDIT & ANOMALY TRACKING LOG:\n"
        f"{anomaly_log}\n"
        f"========================================================================\n"
    )

    # 5. EXECUTE EXCEL SHEET SPREADSHEET WRITER (openpyxl Integration)
    export_path = Path(export_dir) if export_dir is not None else OS_EXPORT_DIR
    export_path.mkdir(parents=True, exist_ok=True)
    export_filename = export_path / "Pick3_Sequence_Intelligence_Report.xlsx"

    with pd.ExcelWriter(export_filename, engine='openpyxl') as writer:
        # Sheet 1: Forecast Summary Data
        summary_data = {
            "Metric Parameter Description": [
                "Predicted Drawing Sequence", "Sequence History Length",
                "Tuning Temperature", "Top-K Filter Bound", "Top-P Distribution Density"
            ],
            "Value Log Assignment": [
                prediction_str, len(target_window),
                temperature, str(top_k), top_p
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Executive Summary", index=False)

        # Sheet 2: Probability Analysis Table
        # Clean string identifiers to float values for raw excel formatting preservation
        df_export_probs = pd.DataFrame({
            "Digit (Token)": [f"[{d}]" for d in range(10)],
            "Ball 1 Probability (%)": prob_matrix[0] / 100,
            "Ball 2 Probability (%)": prob_matrix[1] / 100,
            "Ball 3 Probability (%)": prob_matrix[2] / 100
        })
        df_export_probs.to_excel(writer, sheet_name="Probability Distributions", index=False)

        # Sheet 3: Ingested History Logs
        df_history_log = pd.DataFrame(target_window, columns=["Ball 1 Source", "Ball 2 Source", "Ball 3 Source"])
        df_history_log.insert(0, "Historical Chronological Step", [f"Step T-{10 - i}" for i in range(10)])
        df_history_log.to_excel(writer, sheet_name="Ingested Sequence Logs", index=False)

    return report_text, df_probs, str(export_filename)


# ---------------------------------------------------------
# 5. CORE SYSTEM CONTROLLER ROUTER
# ---------------------------------------------------------
def process_interface(text_input, file_input, temperature, top_k, top_p):
    source_input = file_input if isinstance(file_input, (str, os.PathLike)) else (
        file_input.name if file_input is not None else text_input
    )
    if not source_input or str(source_input).strip() == "":
        return "❌ Ingestion Error: Please provide text rows or upload a file.", pd.DataFrame(), None

    history = parse_drawing_data(source_input)
    text_report, dataframe_output, generated_filepath = compile_and_export_report(history, temperature, top_k, top_p)
    return text_report, dataframe_output, generated_filepath


# ---------------------------------------------------------
# 6. GRADIO LAYOUT ASSEMBLY
# ---------------------------------------------------------
default_history_string = (
    "Drawing 1: 5-2-1\nDrawing 2: 0-9-4\nDrawing 3: 3-3-8\n"
    "Drawing 4: 7-1-2\nDrawing 5: 4-6-0\nDrawing 6: 8-8-9\n"
    "Drawing 7: 1-5-3\nDrawing 8: 2-0-7\nDrawing 9: 6-4-4\n"
    "Drawing 10: 9-1-5"
)

with gr.Blocks(title="Enterprise Sequence Analyzer") as demo:
    gr.Markdown("# 🎰 Idaho Pick-3 Predictive Intelligence & Audit Hub")
    gr.Markdown("Enter ten or more Pick 3 draws, oldest first, or upload a CSV/TXT file.")
    with gr.Row():
        text_input = gr.Textbox(label="Draw history", value=default_history_string, lines=10)
        file_input = gr.File(label="History file", file_types=[".csv", ".txt"], type="filepath")
    with gr.Row():
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
        top_k = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="Top-K (0 disables)")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-P")
    run_button = gr.Button("Generate report", variant="primary")
    report_output = gr.Textbox(label="Analysis report", lines=12)
    probability_output = gr.Dataframe(label="Digit probabilities")
    export_output = gr.File(label="Download Excel report")
    run_button.click(
        fn=process_interface,
        inputs=[text_input, file_input, temperature, top_k, top_p],
        outputs=[report_output, probability_output, export_output],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), share=True)