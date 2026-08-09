# Explainable Lung Nodule Malignancy Classification from CT Images

MSc Computer Science Dissertation Project

## Overview

This project investigates deep-learning-based binary classification
of lung nodules from CT images using the LIDC-IDRI dataset.

Three transfer-learning architectures were evaluated:

- EfficientNetB0
- DenseNet121
- ResNet50

The final architecture was selected using validation PR-AUC to avoid
using the test set for model selection.

Fine-tuned DenseNet121 was selected as the final model.

The project also includes a Flask web application with:

- CT image upload
- DenseNet121 inference
- validation-selected decision threshold
- Grad-CAM visual explainability
- Gemini-generated textual explanation
- automated application tests

## Final Model Performance

| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.861586 |
| Validation PR-AUC | 0.587677 |
| Test ROC-AUC | 0.855333 |
| Test PR-AUC | 0.588136 |
| Validation F1 Threshold | 0.669041 |
| High-Recall Threshold | 0.445648 |

## Architecture Comparison

| Architecture | Validation PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|
| DenseNet121 | 0.587677 | 0.855333 | 0.588136 |
| EfficientNetB0 | 0.570277 | 0.830252 | 0.633736 |
| ResNet50 | 0.497598 | 0.822883 | 0.562283 |

DenseNet121 was selected using validation PR-AUC.
Test metrics were used only for final evaluation.

## Explainability

Grad-CAM is applied to:

`conv5_block16_2_conv`

to generate class-targeted visual explanations.

Gemini is used only to explain the DenseNet121 prediction and
Grad-CAM metadata in plain language. Gemini does not perform the
classification.

## Repository Structure

```text
notebooks/
app/
results/
docs/
scripts/
```


## Running the Web Application Locally

### Prerequisites

- Python 3.x
- Git
- Internet access for Gemini API explanations
- A Gemini Developer API key

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```


### 2. Open the Flask application

`cd app`

### 3. Open the Flask application
- Windows
`python -m venv .venv
.venv\Scripts\activate`

- macOS/Linux
`python3 -m venv .venv
source .venv/bin/activate`

### 4. Install dependencies
`Install dependencies`

### 5. Download the trained DenseNet121 model
- Download model from the GitHub Releases page.
`best_model.keras`
-Place it at:
`app/models/best_model.keras`
Resulting structure must be 
app/
├── app.py
├── config.py
├── models/
│   └── best_model.keras
└── ...

### 6. Configure environment variables
`GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
FLASK_SECRET_KEY=replace_with_a_private_random_value`

### 7. Verify the model
`python -c "from config import MODEL_PATH; print('Model available:', MODEL_PATH.exists())"`

### 8. Run automated test 
`pytest -v`

### 9. Start flask 
`python app.py`

Then open:
`http://127.0.0.1:5000/`


### 10. Health check

Open `http://127.0.0.1:5000/health`

The endpoint should return a successful JSON response.


```markdown
## Application Architecture

The final research prototype follows this inference pipeline:

CT nodule image
        │
        ▼
Flask upload validation
        │
        ▼
Image preprocessing
128 × 128 × 3, float32
        │
        ▼
Fine-Tuned DenseNet121
        │
        ▼
Malignancy output score
        │
        ▼
Validation-selected threshold
0.669041
        │
        ├───────────────┐
        ▼               ▼
Classification       Grad-CAM
Benign/Malignant     visual explanation
        │               │
        └───────┬───────┘
                ▼
          Gemini API
     textual explanation
                │
                ▼
          Flask result page

```

## Web Application

### Upload Interface

![Application homepage](docs/screenshots/Homepage.png)

### DenseNet121 Prediction with Grad-CAM

![Prediction and Grad-CAM](docs/screenshots/Grad-CAM_Visualization.png)
![Prediction and Grad-CAM](docs/screenshots/benign_result.png)

### Automated Test Suite

![Pytest results](docs/screenshots/pytest_results.png)

### AI Assistant Explanation

![AI Explanation] (docs/screenshhots/Gemini-AI Explanation.png)


## Experimental Results

### CNN Architecture Comparison

![CNN model comparison](results/figures/cnn_architecture_test_comparison.png)

### Final DenseNet121 ROC Curve

![DenseNet ROC curve](results/figures/densenet121_test_roc_curve.png)

### Final DenseNet121 Precision-Recall Curve

![DenseNet PR curve](results/figures/densenet121_test_pr_curve.png)

### Final Confusion Matrix

![DenseNet confusion matrix](results/figures/densenet121_final_confusion_matrix.png)

### Grad-CAM Explainability

![DenseNet Grad-CAM](results/figures/densenet121_gradcam_typical_cases_panel.png)