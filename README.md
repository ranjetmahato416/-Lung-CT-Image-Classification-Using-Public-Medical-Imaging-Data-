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