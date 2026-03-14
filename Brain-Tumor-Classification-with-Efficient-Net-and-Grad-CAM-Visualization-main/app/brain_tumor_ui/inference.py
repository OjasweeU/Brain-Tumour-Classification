from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imutils
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

CLASS_NAMES = (
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
)
IMAGE_SIZE = (240, 240)
DEFAULT_MODEL_CANDIDATES = (
    Path("models/model.keras"),
    Path("models/model.h5"),
)


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    original_image: np.ndarray
    cropped_image: np.ndarray
    overlay_image: np.ndarray


def load_trained_model(model_path: str | Path) -> keras.Model:
    path = Path(model_path)
    if path.is_dir():
        raise FileNotFoundError(f"Model path '{path}' points to a directory, not a Keras model file.")
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at '{path}'. Export or copy the notebook's best checkpoint to this path."
        )
    return keras.models.load_model(path)


def resolve_model_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_MODEL_CANDIDATES[0]


def read_image(upload: bytes | str | Path) -> np.ndarray:
    if isinstance(upload, (str, Path)):
        image = Image.open(upload)
    else:
        image = Image.open(BytesIO(upload))
    rgb = image.convert("RGB")
    return np.array(rgb)


def crop_brain_region(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        return image_rgb.copy()

    contour = max(contours, key=cv2.contourArea)
    ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
    ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
    ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
    ext_bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    cropped_bgr = image_bgr[ext_top[1] : ext_bottom[1], ext_left[0] : ext_right[0]]
    if cropped_bgr.size == 0:
        return image_rgb.copy()

    return cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)


def prepare_image(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped = crop_brain_region(image_rgb)
    resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(resized.astype(np.float32), axis=0)
    return cropped, batch


def _resolve_last_conv_layer(model: keras.Model) -> keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer was found in the loaded model. Grad-CAM cannot be generated.")


def generate_gradcam_overlay(
    model: keras.Model,
    image_batch: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    last_conv_layer = _resolve_last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_batch)
        target_index = tf.argmax(predictions[0])
        loss = predictions[:, target_index]

    gradients = tape.gradient(loss, conv_output)
    pooled_gradients = tf.reduce_mean(gradients[0], axis=(0, 1))
    conv_output = conv_output[0]

    activation_map = tf.reduce_sum(conv_output * pooled_gradients, axis=-1).numpy()
    activation_map = np.maximum(activation_map, 0)
    max_value = float(activation_map.max())
    if max_value > 0:
        activation_map = activation_map / max_value

    activation_map = cv2.resize(activation_map, IMAGE_SIZE)
    heatmap = np.uint8(255 * activation_map)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    base_image = image_batch[0].astype(np.float32)
    base_image = np.clip(base_image, 0, 255)
    overlay = cv2.addWeighted(base_image, alpha, heatmap.astype(np.float32), 1 - alpha, 0)
    return overlay.astype(np.uint8)


def predict_image(model: keras.Model, image_rgb: np.ndarray) -> PredictionResult:
    cropped_image, image_batch = prepare_image(image_rgb)
    probabilities_array = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities_array))
    probabilities = {
        label: float(probabilities_array[index]) for index, label in enumerate(CLASS_NAMES)
    }

    overlay = generate_gradcam_overlay(model, image_batch)
    return PredictionResult(
        predicted_label=CLASS_NAMES[predicted_index],
        confidence=probabilities[CLASS_NAMES[predicted_index]],
        probabilities=probabilities,
        original_image=image_rgb,
        cropped_image=cropped_image,
        overlay_image=overlay,
    )


def format_probabilities(probabilities: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"Tumor Type": label.replace("_", " ").title(), "Probability": score}
        for label, score in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    ]
