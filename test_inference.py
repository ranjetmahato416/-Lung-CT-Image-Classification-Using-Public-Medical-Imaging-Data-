from pathlib import Path

from services.model_service import (
    predict_image
)


IMAGE_PATH = Path(
    r"C:\Users\Dell\Desktop\Colab_Notebook\Dataset\Processed\CNN_Dataset\Malignant\LIDC-IDRI-0385_Nodule_002.png"
)


result = predict_image(
    IMAGE_PATH
)


print(
    "Prediction result:"
)

print(
    result
)