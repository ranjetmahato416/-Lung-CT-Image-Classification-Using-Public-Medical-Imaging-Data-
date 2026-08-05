from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMAGE_SIZE

from PIL import (
    Image,
    UnidentifiedImageError
)


def load_image_for_model(image_path):
    """
    Load and preprocess an image exactly like the dissertation pipeline.

    Important:
    - image resized to 128x128
    - converted to RGB
    - float32
    - NOT divided by 255
    - DenseNet preprocessing is handled inside the saved model
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    image = tf.io.read_file(
        str(image_path)
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear"
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image


def prepare_image_batch(image_path):
    """
    Convert a single image to model-ready batch format.

    Output:
        shape = (1, 128, 128, 3)
    """

    image = load_image_for_model(
        image_path
    )

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    return image_batch


def get_display_image(image_path):
    """
    Load an uploaded image for display only.
    This does not affect model inference.
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    return np.array(
        image
    )


def validate_image_file(
    image_path
):
    """
    Verify that the uploaded file is a real readable image.

    Returns:
        True

    Raises:
        ValueError if the file is invalid.
    """

    try:

        with Image.open(
            image_path
        ) as image:

            image.verify()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        raise ValueError(
            "The uploaded file is not "
            "a valid readable image."
        ) from error

    return True