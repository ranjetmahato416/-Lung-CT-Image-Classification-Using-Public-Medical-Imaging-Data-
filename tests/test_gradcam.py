from pathlib import Path

from services.model_service import (
    predict_image
)

from services.gradcam_service import (
    generate_gradcam
)


IMAGE_PATH = Path(
    r"C:\Users\Dell\Desktop\Colab_Notebook\Dataset\Processed\CNN_Dataset\Malignant\LIDC-IDRI-0385_Nodule_002.png"
)


result = predict_image(
    IMAGE_PATH
)


print(
    "Prediction:"
)

print(
    result
)


output_filename = (
    "test_gradcam.png"
)


saved_filename = generate_gradcam(
    image_path=IMAGE_PATH,
    predicted_label=result[
        "predicted_label"
    ],
    output_filename=output_filename
)


print(
    "\nGrad-CAM saved:"
)

print(
    saved_filename
)