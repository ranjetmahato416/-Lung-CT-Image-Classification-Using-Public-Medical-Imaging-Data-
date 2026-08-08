import tensorflow as tf

from config import (
    MODEL_PATH,
    F1_THRESHOLD
)

from services.preprocessing import (
    prepare_image_batch
)


@tf.keras.utils.register_keras_serializable(
    package="LungNoduleClassification"
)
class DenseNetPreprocessing(
    tf.keras.layers.Layer
):
    """
    Custom preprocessing layer used in the saved DenseNet121 model.
    """

    def call(
        self,
        inputs
    ):

        return (
            tf.keras.applications
            .densenet
            .preprocess_input(
                inputs
            )
        )

    def get_config(
        self
    ):

        return super().get_config()


_model = None


def load_model():

    global _model

    if _model is None:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        print("=" * 60)
        print("LOADING DENSENET121")
        print("=" * 60)

        print(
            "TensorFlow version:",
            tf.__version__
        )

        print(
            "Model path:",
            MODEL_PATH
        )

        print(
            "Model size:",
            MODEL_PATH.stat().st_size,
            "bytes"
        )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "DenseNetPreprocessing":
                    DenseNetPreprocessing,

                "LungNoduleClassification>DenseNetPreprocessing":
                    DenseNetPreprocessing
            },
            compile=False
        )

        print(
            "✓ DenseNet121 loaded successfully."
        )

        print(
            "Input shape:",
            _model.input_shape
        )

        print(
            "Output shape:",
            _model.output_shape
        )

        print("=" * 60)

    return _model

def predict_image(
    image_path
):
    """
    Run inference on a single image.

    Returns:
        probability
        predicted label
        predicted class
        threshold
    """

    model = load_model()

    image_batch = prepare_image_batch(
        image_path
    )

    prediction = model.predict(
        image_batch,
        verbose=0
    )

    probability = float(
        prediction[
            0,
            0
        ]
    )

    predicted_label = int(
        probability
        >= F1_THRESHOLD
    )

    predicted_class = (
        "Malignant"
        if predicted_label == 1
        else "Benign"
    )

    return {
        "probability":
            probability,

        "predicted_label":
            predicted_label,

        "predicted_class":
            predicted_class,

        "threshold":
            float(
                F1_THRESHOLD
            )
    }