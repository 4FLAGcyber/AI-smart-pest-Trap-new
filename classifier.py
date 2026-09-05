"""
AI classification stage.

Loads a TFLite image classifier (by default: MobileNetV2 pretrained on
ImageNet as a stand-in) and maps its prediction onto the trap's three
categories via pest_categories.py.

The wrapper adapts automatically to the model it is given:
  * input layout  NHWC or NCHW
  * input dtype   uint8 (quantized models) or float32
  * output        scores or raw logits

so you can drop in your own Teachable Machine / fine-tuned model without
changing this code.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image

import config
from pest_categories import label_to_category

# ImageNet normalization, used by PyTorch-converted models (NCHW float input)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Prediction:
    label: str           # top classifier label, e.g. "ladybug"
    confidence: float    # 0.0 - 1.0
    category: str        # Harmful / Beneficial / Harmless
    top: List[Tuple[str, float]] = field(default_factory=list)  # top-k (label, score)
    subject_top: List[Tuple[int, str, float]] = field(default_factory=list)


class InsectClassifier:
    def __init__(self, model_path=None, labels_path=None):
        import os
        import warnings

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        Interpreter = None
        try:
            # Preferred on-device runtime
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            pass
        if Interpreter is None:
            try:
                # Newer LiteRT package (tf.lite.Interpreter is deprecated in TF 2.20)
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from tensorflow.lite.python.interpreter import Interpreter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interpreter = Interpreter(model_path=model_path or config.MODEL_PATH)
            self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        shape = self.input_details["shape"]  # [1, H, W, 3] or [1, 3, H, W]
        if shape[1] == 3:
            self.input_nchw = True
            self.input_h, self.input_w = int(shape[2]), int(shape[3])
        else:
            self.input_nchw = False
            self.input_h, self.input_w = int(shape[1]), int(shape[2])

        self.input_is_quantized = self.input_details["dtype"] == np.uint8

        with open(labels_path or config.LABELS_PATH, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame_rgb).resize((self.input_w, self.input_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0

        if self.input_is_quantized:
            # Quantized models consume raw 0-255 bytes
            out = (arr * 255.0).astype(np.uint8)
        elif self.input_nchw:
            # PyTorch-style export: ImageNet-normalized, channels-first
            out = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
            out = out.transpose(2, 0, 1).astype(np.float32)
        else:
            # Classic TFLite float model: [-1, 1]
            out = (arr * 2.0 - 1.0).astype(np.float32)

        return np.expand_dims(out, axis=0)

    def classify(self, frame_rgb: np.ndarray) -> Prediction:
        input_tensor = self._preprocess(frame_rgb)
        if input_tensor.dtype != self.input_details["dtype"]:
            input_tensor = input_tensor.astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details["index"])[0]

        if output.dtype == np.uint8:
            # Quantized output: uint8 scores -> normalize to 0-1
            scale, zero = self.output_details.get("quantization", (1.0 / 255.0, 0))
            scale = scale or 1.0 / 255.0
            scores = (output.astype(np.float32) - zero) * scale
        else:
            scores = output.astype(np.float32)
            # Logits need softmax; probability outputs can be used as-is
            if scores.max() > 1.5 or scores.min() < -0.1:
                exp = np.exp(scores - scores.max())
                scores = exp / exp.sum()

        candidate_count = min(
            max(config.TOP_K, config.SUBJECT_TOP_K), len(scores), len(self.labels)
        )
        candidate_idx = np.argsort(scores)[::-1][:candidate_count]
        subject_top = [
            (int(i), self.labels[i] if i < len(self.labels) else f"class_{i}", float(scores[i]))
            for i in candidate_idx
        ]
        top = [(label, score) for _, label, score in subject_top[:config.TOP_K]]

        label, confidence = top[0]
        return Prediction(
            label=label,
            confidence=confidence,
            category=label_to_category(label),
            top=top,
            subject_top=subject_top,
        )
