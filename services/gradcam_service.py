from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from config import (
    IMAGE_SIZE,
    GRADCAM_FOLDER
)

from services.model_service import (
    load_model
)

from services.preprocessing import (
    load_image_for_model
)


LAST_CONV_LAYER_NAME = (
    "conv5_block16_2_conv"
)


def get_densenet_backbone(model):
    """
    Locate the nested DenseNet121 backbone.
    """

    candidates = [
        layer
        for layer in model.layers
        if (
            isinstance(
                layer,
                tf.keras.Model
            )
            and
            "densenet" in layer.name.lower()
        )
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify "
            "DenseNet121 backbone."
        )

    return candidates[0]


def build_feature_extractor(
    backbone
):
    """
    Build a model that returns:
    - final selected convolutional feature map
    - normal DenseNet backbone output
    """

    last_conv_layer = (
        backbone.get_layer(
            LAST_CONV_LAYER_NAME
        )
    )

    return tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ],
        name="DenseNet121_GradCAM_FeatureExtractor"
    )


def make_gradcam_heatmap(
    image_batch,
    target_class
):
    """
    Generate class-targeted Grad-CAM heatmap.

    target_class:
        0 = Benign
        1 = Malignant
    """

    model = load_model()

    backbone = get_densenet_backbone(
        model
    )

    feature_extractor = (
        build_feature_extractor(
            backbone
        )
    )

    backbone_index = (
        model.layers.index(
            backbone
        )
    )

    image_batch = tf.cast(
        image_batch,
        tf.float32
    )

    with tf.GradientTape() as tape:

        # ----------------------------------------
        # Outer model layers before DenseNet
        # ----------------------------------------

        x = image_batch

        for layer in model.layers[
            1:backbone_index
        ]:

            try:
                x = layer(
                    x,
                    training=False
                )

            except TypeError:
                x = layer(x)

        # ----------------------------------------
        # DenseNet feature extraction
        # ----------------------------------------

        conv_outputs, backbone_output = (
            feature_extractor(
                x,
                training=False
            )
        )

        # ----------------------------------------
        # Outer classification head
        # ----------------------------------------

        y = backbone_output

        for layer in model.layers[
            backbone_index + 1:
        ]:

            try:
                y = layer(
                    y,
                    training=False
                )

            except TypeError:
                y = layer(y)

        malignant_probability = (
            y[:, 0]
        )

        if target_class == 1:

            target_score = (
                malignant_probability
            )

        elif target_class == 0:

            target_score = (
                1.0
                - malignant_probability
            )

        else:

            raise ValueError(
                "target_class must be "
                "0 or 1."
            )

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are None."
        )

    pooled_gradients = (
        tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )
    )

    conv_outputs = (
        conv_outputs[0]
    )

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


def create_gradcam_overlay(
    image_path,
    heatmap,
    alpha=0.40
):
    """
    Overlay the Grad-CAM heatmap on the original image.
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    image = image.resize(
        IMAGE_SIZE
    )

    image_np = np.array(
        image
    ).astype(
        np.uint8
    )

    heatmap_resized = (
        tf.image.resize(
            heatmap[..., np.newaxis],
            IMAGE_SIZE
        )
        .numpy()
        .squeeze()
    )

    heatmap_uint8 = np.uint8(
        255
        * heatmap_resized
    )

    colormap = plt.get_cmap(
        "jet"
    )

    heatmap_rgb = (
        colormap(
            heatmap_uint8
        )[
            ...,
            :3
        ]
        * 255
    ).astype(
        np.uint8
    )

    overlay = (
        image_np
        * (1 - alpha)
        +
        heatmap_rgb
        * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return overlay


def generate_gradcam(
    image_path,
    predicted_label,
    output_filename
):
    """
    Generate and save Grad-CAM overlay.

    Returns the saved filename.
    """

    image = load_image_for_model(
        image_path
    )

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    heatmap = make_gradcam_heatmap(
        image_batch=image_batch,
        target_class=int(
            predicted_label
        )
    )

    overlay = create_gradcam_overlay(
        image_path=image_path,
        heatmap=heatmap
    )

    GRADCAM_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        GRADCAM_FOLDER
        / output_filename
    )

    Image.fromarray(
        overlay
    ).save(
        output_path
    )

    return output_filename