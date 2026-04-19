from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from potato_leaf_inference import (
    load_model, load_gate_model, predict_pil_image,
)
from explainability import generate_explanation


st.set_page_config(page_title="Potato Leaf Disease Demo", layout="wide")


@st.cache_resource
def get_runtime(checkpoint_path: str | None, class_info_path: str | None, device: str | None):
    model, active_device, info = load_model(
        checkpoint_path=checkpoint_path,
        class_info_path=class_info_path,
        device=device,
    )
    # Load ImageNet gate model for object labelling
    gate_model, gate_categories = load_gate_model(active_device)
    return model, active_device, info, gate_model, gate_categories


def main() -> None:
    st.title("Potato Leaf Disease Identification")
    st.write("Upload a leaf image to run the final Hybrid CNN-Transformer model.")

    default_checkpoint = Path("artifacts/phase_2_benchmarking/models/hybrid_cnn_transformer_best.pt")
    default_class_info = Path("AdvancePractice/deployment/class_info.json")

    with st.sidebar:
        st.header("Runtime")
        checkpoint_path = st.text_input("Checkpoint path", value=str(default_checkpoint))
        class_info_path = st.text_input("class_info.json path", value=str(default_class_info))
        device = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        top_k = st.slider("Top-k predictions", min_value=1, max_value=5, value=3)

        st.header("Explainability")
        xai_enabled = st.checkbox("Show Grad-CAM / Score-CAM", value=False)
        xai_method = st.selectbox("XAI method", options=["gradcam", "scorecam"], index=0)
        xai_alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)

    runtime_device = None if device == "auto" else device
    model, active_device, info, gate_model, gate_categories = get_runtime(
        checkpoint_path, class_info_path, runtime_device,
    )

    centroids_data = info.get("_centroids")
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
        centroids_data=centroids_data,
        gate_model=gate_model,
        gate_categories=gate_categories,
    )

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
        brightness = prediction.get("brightness_check")
        if brightness and brightness["level"] == "WARNING_LOW_LIGHT":
            st.warning(f"Low-light warning: {brightness['message']}")
        elif brightness and brightness["level"] == "WARNING_OVEREXPOSED":
            st.warning(f"Overexposure warning: {brightness['message']}")
    with col2:
        is_rejected = prediction.get("rejected", False)

        if is_rejected:
            stage = prediction.get("rejection_stage", "")
            detected = prediction.get("detected_object", "unknown")

            if stage == "leaf_gate":
                st.error(f"**This is not a potato leaf.**")
                st.write(f"Detected object: **{detected}**")
                st.caption(prediction.get("rejection_reason", ""))
                leaf_gate = prediction.get("leaf_gate", {})
                if leaf_gate:
                    st.caption(
                        f"Similarity: {leaf_gate['max_similarity']:.3f} "
                        f"(closest class: {leaf_gate['closest_class']})"
                    )
                imagenet = prediction.get("imagenet_labels", [])
                if imagenet:
                    with st.expander("ImageNet detection details"):
                        for lbl in imagenet:
                            tag = " 🌿" if lbl["is_plant_related"] else ""
                            st.write(f"- {lbl['label']} ({lbl['confidence']:.3f}){tag}")

            elif stage == "confidence_gate":
                st.warning("**Uncertain prediction — possibly not a potato leaf.**")
                st.write(f"Detected object: **{detected}**")
                best = prediction.get("best_guess", "")
                conf = prediction.get("confidence", 0)
                st.caption(f"Best guess: {best} ({conf:.3f}) — too low to trust")
                st.caption(prediction.get("rejection_reason", ""))

        else:
            st.subheader("Prediction")
            st.metric("Predicted class", prediction["predicted_class"])
            st.metric("Confidence", f"{prediction['confidence']:.4f}")
            st.caption(f"Running on `{active_device}`")
            leaf_gate = prediction.get("leaf_gate", {})
            if leaf_gate:
                st.caption(
                    f"Leaf similarity: {leaf_gate['max_similarity']:.3f} "
                    f"(closest: {leaf_gate['closest_class']})"
                )

    if not is_rejected and prediction.get("top_k"):
        st.subheader("Top predictions")
        for row in prediction["top_k"]:
            with st.container(border=True):
                st.write(f"Rank {row['rank']}: {row['class_name']} ({row['probability']:.4f})")
                st.write(row["disease_note"])

    # ── Explainability (Grad-CAM / Score-CAM) ─────────────────────────────
    if not is_rejected and xai_enabled:
        st.subheader(f"Explainability — {xai_method.upper()}")
        with st.spinner(f"Generating {xai_method} heatmap…"):
            explanation = generate_explanation(
                img=image,
                model=model,
                device=active_device,
                class_names=info["class_names"],
                method=xai_method,
                alpha=xai_alpha,
            )
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(explanation["overlay"], caption=f"{xai_method.upper()} overlay", use_container_width=True)
        with cam_col2:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            im = ax.imshow(explanation["heatmap"], cmap="jet")
            ax.set_title(f"Activation map for: {explanation['target_class']}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
            st.pyplot(fig)
        st.caption(
            f"Target class: **{explanation['target_class']}** | "
            f"Predicted: **{explanation['predicted_class']}** "
            f"({explanation['confidence']:.4f})"
        )


if __name__ == "__main__":
    main()
