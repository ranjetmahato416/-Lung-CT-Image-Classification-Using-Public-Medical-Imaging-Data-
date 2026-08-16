from pathlib import Path

from services.model_service import (
    predict_image
)


IMAGE_PATH = Path(
    r"C:\Users\Dell\Desktop\-Lung-CT-Image-Classification-Using-Public-Medical-Imaging-Data-\app\tests\fixtures\LIDC-IDRI-0044_Nodule_003_Malignant.png"
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