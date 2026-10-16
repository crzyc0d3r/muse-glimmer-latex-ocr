"""Upload and transcribe formula images with a locally saved adapter."""

import os
from pathlib import Path
import streamlit as st
from PIL import Image
from runtime import load_model, transcribe


@st.cache_resource
def local_model(path):
    return load_model(path)


st.title("Formula image to LaTeX")
st.caption("Local model inference. Start training first to create the adapter.")
adapter = Path(os.environ.get("OCR_ADAPTER_DIR", "artifacts/adapter"))
if not adapter.is_dir():
    st.info(f"Adapter directory not found: {adapter}. Run train.py first.")
    st.stop()
upload = st.file_uploader("Formula image", type=["png", "jpg", "jpeg", "webp"])
if upload:
    image = Image.open(upload).convert("RGB")
    st.image(image)
    if st.button("Transcribe"):
        with st.spinner("Running the local model"):
            model, processor = local_model(str(adapter))
            result = transcribe(model, processor, image)
        st.code(result, language="latex")
        st.latex(result)
