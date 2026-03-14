from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imutils
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

CLASS_NAMES = (
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
)
IMAGE_SIZE = (240, 240)


def crop_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        return image

    contour = max(contours, key=cv2.contourArea)
    ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
    ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
    ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
    ext_bottom = tuple(contour[contour[:, :, 1].argmax()][0])
    cropped = image[ext_top[1] : ext_bottom[1], ext_left[0] : ext_right[0]]
    return cropped if cropped.size else image


def preprocess_split(source_root: Path, destination_root: Path) -> None:
    for class_name in CLASS_NAMES:
        source_dir = source_root / class_name
        target_dir = destination_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(sorted(source_dir.glob("*"))):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            cropped = crop_image(image)
            resized = cv2.resize(cropped, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target_dir / f"{index}.jpg"), resized)


def build_model() -> Model:
    base = EfficientNetB1(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    head = base.output
    head = GlobalAveragePooling2D()(head)
    head = Dropout(0.5)(head)
    outputs = Dense(4, activation="softmax")(head)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_export(
    dataset_root: Path,
    workspace_root: Path,
    epochs: int,
    batch_size: int,
) -> Path:
    train_source = dataset_root / "Training"
    test_source = dataset_root / "Testing"

    prepared_root = workspace_root / "artifacts" / "prepared_data"
    prepared_train = prepared_root / "train"
    prepared_test = prepared_root / "test"

    preprocess_split(train_source, prepared_train)
    preprocess_split(test_source, prepared_test)

    datagen = ImageDataGenerator(
        rotation_range=10,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_data = datagen.flow_from_directory(
        prepared_train,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    valid_data = datagen.flow_from_directory(
        prepared_train,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    test_data = ImageDataGenerator().flow_from_directory(
        prepared_test,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    model = build_model()
    keras_output_path = workspace_root / "models" / "model.keras"
    h5_output_path = workspace_root / "models" / "model.h5"
    keras_output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            str(keras_output_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="auto",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="auto",
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.3,
            patience=2,
            min_delta=0.001,
            mode="auto",
            verbose=1,
        ),
    ]

    model.fit(
        train_data,
        epochs=epochs,
        validation_data=valid_data,
        verbose=2,
        callbacks=callbacks,
    )

    print("Test metrics:", model.evaluate(test_data, verbose=2))
    model.save(keras_output_path)
    try:
        model.save(h5_output_path)
    except Exception as exc:
        print(f"Legacy H5 export skipped: {exc}")
    print(f"Saved trained model to '{keras_output_path}'.")
    return keras_output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the original EfficientNetB1 brain tumor classifier and export models/model.keras."
    )
    parser.add_argument(
        "--dataset-root",
        default=r"C:\Users\hp\Downloads\btdata\Brain-MRI",
        help="Root directory containing Training and Testing folders.",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root where artifacts and models/model.h5 will be written.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    args = parser.parse_args()

    tf.random.set_seed(42)
    np.random.seed(42)

    train_and_export(
        dataset_root=Path(args.dataset_root),
        workspace_root=Path(args.workspace_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
