# DenseNet121 Model Setup

The trained model is intentionally not stored directly in the Git
repository because it is a large binary artifact.

## Final Selected Model

The Flask application uses:

`best_model.keras`

Architecture:

`Fine-Tuned DenseNet121`

The architecture was selected using validation PR-AUC rather than test
performance.

### Performance

| Metric | Value |
|---|---:|
| Validation ROC-AUC | 0.861586 |
| Validation PR-AUC | 0.587677 |
| Test ROC-AUC | 0.855333 |
| Test PR-AUC | 0.588136 |
| Validation F1 Threshold | 0.669041 |
| High-Recall Threshold | 0.445648 |

## Download

Download `best_model.keras` from the repository's GitHub Releases page.

## Required Location

After downloading the file, place it at:

```text
app/
└── models/
    └── best_model.keras