from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from potato_leaf_inference import load_model, predict_pil_image


st.set_page_config(page_title="Potato Leaf Disease Demo", layout="wide")


@st.cache_resource
def get_runtime(checkpoint_path: str | None, class_info_path: str | None, device: str | None):
    return load_model(checkpoint_path=checkpoint_path, class_info_path=class_info_path, device=device)


def main() -> None:
    st.title("Potato Leaf Disease Identification")
    st.write("Upload a leaf image to run the final EfficientNetB0 (fine-tune) model.")

    default_checkpoint = Path("artifacts/phase_2_benchmarking/models/efficientnet_b0_finetune_best.pt")
    default_class_info = Path("submission_ready/final_package/deployment/class_info.json")

    with st.sidebar:
        st.header("Runtime")
        checkpoint_path = st.text_input("Checkpoint path", value=str(default_checkpoint))
        class_info_path = st.text_input("class_info.json path", value=str(default_class_info))
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        top_k = st.slider("Top-k predictions", min_value=1, max_value=5, value=3)

    runtime_device = None if device == "auto" else device
    model, active_device, info = get_runtime(checkpoint_path, class_info_path, runtime_device)
    uploaded_file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded_file is None:
        st.info("Upload an image to start the demo.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    prediction = predict_pil_image(
        image,
        model=model,
        device=active_device,
        class_names=info["class_names"],
        disease_info=info["disease_info"],
        top_k=top_k,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", prediction["predicted_class"])
        st.metric("Confidence", f"{prediction['confidence']:.4f}")
        st.caption(f"Running on `{active_device}`")

    st.subheader("Top predictions")
    for row in prediction["top_k"]:
        with st.container(border=True):
            st.write(f"Rank {row['rank']}: {row['class_name']} ({row['probability']:.4f})")
            st.write(row["disease_note"])


if __name__ == "__main__":
    main()
