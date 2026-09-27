"""Streamlit live demo for the steel defect classifier.

Runs the classifier and LLM report in-process (no separate API server
needed), so this same file works both for `streamlit run app.py` locally
and as a Hugging Face Spaces app.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_report import generate_report  # noqa: E402
from model import CLASS_NAMES, build_transform, load_model  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "defect_classifier.pt"

DEFECT_LABELS = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

st.set_page_config(page_title="Steel Defect Detection", page_icon="\U0001F50D", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_transform():
    return build_transform()


def classify(image: Image.Image) -> dict:
    model = get_model()
    transform = get_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    class_probabilities = {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)}
    best_idx = int(probabilities.argmax())
    return {
        "defect_type": CLASS_NAMES[best_idx],
        "confidence": float(probabilities[best_idx]),
        "class_probabilities": class_probabilities,
    }


st.title("Steel Defect Detection")
st.caption(
    "Upload a steel surface image to classify its defect and get an "
    "LLM-generated inspection report. Trained on the NEU-CLS dataset "
    "(99.3% held-out test accuracy) — see the Limitations note below."
)

with st.sidebar:
    st.header("About")
    st.write(
        "MobileNetV2 transfer learning classifier + Ollama "
        "(`qwen2.5:7b-instruct`) inspection report, grounded in real "
        "metallurgical causes per defect type."
    )
    st.write("[Full write-up on GitHub](https://github.com/Tiasha-17/steel-defect-detection)")
    st.divider()
    st.subheader("Try a sample")
    sample_files = sorted((PROJECT_ROOT / "tests" / "fixtures").glob("*.jpg"))
    sample_choice = st.selectbox(
        "Pick a sample image",
        ["-- none --"] + [f.stem for f in sample_files],
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

image_bytes = None
image_name = None
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image_name = uploaded_file.name
elif sample_choice != "-- none --":
    sample_path = PROJECT_ROOT / "tests" / "fixtures" / f"{sample_choice}.jpg"
    image_bytes = sample_path.read_bytes()
    image_name = sample_path.name

if image_bytes is not None:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image.")
        st.stop()

    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.image(image, caption=image_name, use_container_width=True)

    with st.spinner("Classifying..."):
        result = classify(image)
    with st.spinner("Generating inspection report..."):
        report = generate_report(result["defect_type"], result["confidence"])

    with col2:
        label = DEFECT_LABELS[result["defect_type"]]
        st.metric("Predicted defect", label, f"{result['confidence']:.1%} confidence")

        probs_df = pd.DataFrame(
            {
                "Defect type": [DEFECT_LABELS[c] for c in CLASS_NAMES],
                "Probability": [result["class_probabilities"][c] for c in CLASS_NAMES],
            }
        ).set_index("Defect type")
        st.bar_chart(probs_df, horizontal=True)

    st.subheader("Inspection report")
    if report.report_source == "template_fallback":
        st.info(
            "Ollama wasn't reachable from this app, so this report is the "
            "grounding fact sheet directly rather than LLM-composed prose. "
            "See the README for why `/predict` is designed to degrade this "
            "way instead of failing.",
            icon="ℹ️",
        )
    else:
        st.success("LLM-composed report (Ollama, qwen2.5:7b-instruct)", icon="✅")

    st.markdown(f"**Likely cause:** {report.likely_cause}")
    st.markdown(f"**Recommended action:** {report.recommended_action}")

    with st.expander("Raw JSON"):
        st.json(
            {
                "classification": result,
                "inspection_report": report.model_dump(),
            }
        )
else:
    st.info("Upload an image above, or pick a sample from the sidebar, to get started.")

st.divider()
st.caption(
    "NEU-CLS is a small, fairly clean academic dataset collected in controlled "
    "conditions. Real factory imagery (lighting, camera angle, sensor noise) "
    "would need re-validation before this model is production-ready — see the "
    "full write-up for details."
)
