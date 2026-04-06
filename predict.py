from __future__ import annotations

import argparse
import json
from pathlib import Path

from potato_leaf_inference import load_model, predict_image


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
    prediction = predict_image(
        args.image_path,
        model=model,
        device=device,
        class_names=info["class_names"],
        disease_info=info["disease_info"],
        top_k=args.top_k,
    )

    print(f"Model: {info.get('final_model', 'EfficientNetB0 (fine-tune)')}")
    print(f"Device: {device}")
    print(f"Image: {prediction['image_path']}")

    brightness = prediction.get("brightness_check")
    if brightness and brightness["level"] != "OK":
        print(f"\n** {brightness['level']}: {brightness['message']} **\n")

    print(f"Predicted class: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("")
    print("Top predictions:")
    for row in prediction["top_k"]:
        print(f"{row['rank']}. {row['class_name']}  {row['probability']:.4f}")
        print(f"   {row['disease_note']}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
        print("")
        print(f"Saved prediction JSON to {out_path}")


if __name__ == "__main__":
    main()
