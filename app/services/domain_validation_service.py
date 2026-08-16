# ============================================================
# Domain Validation Service
# ============================================================

import json

import numpy as np
import tensorflow as tf
from PIL import Image

from config import (
    DOMAIN_MODEL_PATH,
    DOMAIN_THRESHOLD_PATH
)


# ============================================================
# Constants
# ============================================================

IMAGE_SIZE = (
    128,
    128
)


# ============================================================
# Cached Resources
# ============================================================

_domain_model = None
_domain_threshold = None


# ============================================================
# Load Domain Validator
# ============================================================

def load_domain_model():
    """
    Load and cache the final supervised MobileNetV2
    input-domain validator.
    """

    global _domain_model

    if _domain_model is None:

        if not DOMAIN_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Domain validation model not found: "
                f"{DOMAIN_MODEL_PATH}"
            )

        _domain_model = (
            tf.keras.models.load_model(
                DOMAIN_MODEL_PATH,
                compile=False
            )
        )

    return _domain_model


# ============================================================
# Load Validation Threshold
# ============================================================

def load_domain_threshold():
    """
    Load the validation-selected operating threshold.
    """

    global _domain_threshold

    if _domain_threshold is None:

        if not DOMAIN_THRESHOLD_PATH.exists():

            raise FileNotFoundError(
                "Domain threshold file not found: "
                f"{DOMAIN_THRESHOLD_PATH}"
            )

        with open(
            DOMAIN_THRESHOLD_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            config = json.load(
                file
            )

        _domain_threshold = float(
            config[
                "Threshold"
            ]
        )

    return _domain_threshold


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_domain_image(
    image_path
):
    """
    Reproduce the preprocessing used in Notebook 16B.

    Steps:
    - load RGB image
    - resize to 128x128
    - convert to grayscale
    - convert back to three channels
    - preserve 0-255 float32 range

    MobileNetV2 preprocessing is already embedded
    inside the saved model.
    """

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = (
        tf.image.rgb_to_grayscale(
            image_tensor
        )
    )

    image_tensor = (
        tf.image.grayscale_to_rgb(
            image_tensor
        )
    )

    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    return image_tensor


# ============================================================
# Validate Input Domain
# ============================================================

def validate_input_domain(
    image_path
):
    """
    Determine whether the uploaded image is compatible
    with the supported lung CT nodule input domain.

    Returns:
        {
            probability,
            threshold,
            accepted,
            status
        }
    """

    model = load_domain_model()

    threshold = (
        load_domain_threshold()
    )

    image = preprocess_domain_image(
        image_path
    )

    probability = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    accepted = (
        probability
        >= threshold
    )

    return {

        "probability":
            probability,

        "threshold":
            threshold,

        "accepted":
            bool(
                accepted
            ),

        "status":
            (
                "Supported"
                if accepted
                else "Unsupported"
            )
    }