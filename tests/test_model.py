import tensorflow as tf

from services.model_service import (
    DenseNetPreprocessing
)

from config import MODEL_PATH


print("TensorFlow version:")
print(tf.__version__)

print("\nModel path:")
print(MODEL_PATH)

print("\nModel exists:")
print(MODEL_PATH.exists())

print("\nModel size:")
print(MODEL_PATH.stat().st_size)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "DenseNetPreprocessing":
            DenseNetPreprocessing,

        "LungNoduleClassification>DenseNetPreprocessing":
            DenseNetPreprocessing
    },
    compile=False
)


print("\n✓ Model loaded successfully.")

print("\nModel name:")
print(model.name)

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)