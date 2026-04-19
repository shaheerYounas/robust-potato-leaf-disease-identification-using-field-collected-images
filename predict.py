from __future__ import annotations

import argparse
import json
from pathlib import Path

from potato_leaf_inference import load_model, load_gate_model, predict_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-image inference for the final potato leaf disease model.")
    parser.add_argument("image_path", help="Path to the image file to classify.")
    parser.add_argument("--checkpoint", help="Optional checkpoint path override.")
    parser.add_argument("--class-info", help="Optional class_info.json path override.")
    parser.add_argument("--device", help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to return.")
    parser.add_argument("--json-out", help="Optional path to save the prediction JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, device, info = load_model(
        checkpoint_path=args.checkpoint,
        class_info_path=args.class_info,
        device=args.device,
    )
    gate_model, gate_categories = load_gate_model(device)
    centroids_data = info.get("_centroids")
    if gate_model is None:
        print("Warning: ImageNet gate labels are unavailable; continuing with the packaged disease model only.")

    prediction = predict_image(
        args.image_path,
        model=model,
        device=device,
        class_names=info["class_names"],
        disease_info=info["disease_info"],
        top_k=args.top_k,
        centroids_data=centroids_data,
        gate_model=gate_model,
        gate_categories=gate_categories,
    )

    print(f"Model: {info.get('final_model', 'Hybrid CNN-Transformer')}")
    print(f"Device: {device}")
    print(f"Image: {prediction['image_path']}")

    brightness = prediction.get("brightness_check")
    if brightness and brightness["level"] != "OK":
        print(f"\n** {brightness['level']}: {brightness['message']} **\n")

    is_rejected = prediction.get("rejected", False)

    if is_rejected:
        stage = prediction.get("rejection_stage", "")
        detected = prediction.get("detected_object", "unknown")

        if stage == "leaf_gate":
            print(f"\n** REJECTED: This is not a potato leaf. **")
            print(f"   Detected object: {detected}")
            leaf_gate = prediction.get("leaf_gate", {})
            print(f"   Leaf similarity: {leaf_gate.get('max_similarity', 0):.3f} "
                  f"(threshold: {leaf_gate.get('threshold', 0):.3f})")
        elif stage == "confidence_gate":
            print(f"\n** UNCERTAIN: Possibly not a potato leaf. **")
            print(f"   Detected object: {detected}")
            print(f"   Best guess: {prediction.get('best_guess', '?')} "
                  f"({prediction.get('confidence', 0):.3f})")

        imagenet = prediction.get("imagenet_labels", [])
        if imagenet:
            print(f"   ImageNet top-3:")
            for lbl in imagenet:
                tag = " [plant]" if lbl["is_plant_related"] else ""
                print(f"     - {lbl['label']} ({lbl['confidence']:.3f}){tag}")
    else:
        print(f"Predicted class: {prediction['predicted_class']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        leaf_gate = prediction.get("leaf_gate", {})
        if leaf_gate:
            print(f"Leaf similarity: {leaf_gate.get('max_similarity', 0):.3f}")

    if prediction.get("top_k"):
        print("\nTop predictions:")
        for row in prediction["top_k"]:
            print(f"{row['rank']}. {row['class_name']}  {row['probability']:.4f}")
            print(f"   {row['disease_note']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(prediction, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved prediction JSON to {out_path}")


if __name__ == "__main__":
    main()
