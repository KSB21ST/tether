from __future__ import annotations

import argparse
import socket

import torch
from PIL import Image
from multiprocessing.managers import BaseManager
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


MODEL_ID = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Grounding-DINO ({MODEL_ID}) on {device}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print("Grounding-DINO loaded.")


class GroundingDino:
    """
    Wrapper around HuggingFace Grounding-DINO for zero-shot object detection.

    detect() returns bounding boxes in full-image pixel coordinates (xyxy format).
    """

    def detect(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[tuple[float, float, float, float, float, str]]:
        """
        Run Grounding-DINO on a single image.

        Args:
            image_path: Path to the image file.
            text_prompt: Free-form text describing the objects to detect.
                         Multiple objects can be separated by " . " (DINO convention).
            box_threshold: Minimum objectness score for a detected box.
            text_threshold: Minimum text-matching score for a detected box.

        Returns:
            List of (x1, y1, x2, y2, score, label) tuples in full-image pixel coords.
            Empty list if no detections pass the thresholds.
        """
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        # Grounding-DINO expects the caption to end with a period
        caption = text_prompt.strip()
        if not caption.endswith("."):
            caption = caption + "."
        print(caption)
        print("*"*100)
        inputs = processor(images=image, text=caption, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )[0]

        boxes = results["boxes"].cpu().numpy()    # (N, 4) in xyxy pixel coords
        scores = results["scores"].cpu().numpy()  # (N,)
        labels = results["labels"]                # list of strings

        return [
            (float(x1), float(y1), float(x2), float(y2), float(s), str(lbl))
            for (x1, y1, x2, y2), s, lbl in zip(boxes, scores, labels)
        ]


class GroundingDinoManager(BaseManager):
    pass


GroundingDinoManager.register("GroundingDino", GroundingDino)


def serve_grounding_dino(port: int = 50033):
    manager = GroundingDinoManager(address=("", port), authkey=b"groundingdino")
    server = manager.get_server()
    server_ip = get_local_ip()
    print(f"Serving GroundingDino on {server_ip}:{port}...")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50033, help="Port to serve GroundingDino on")
    args = parser.parse_args()
    serve_grounding_dino(args.port)
