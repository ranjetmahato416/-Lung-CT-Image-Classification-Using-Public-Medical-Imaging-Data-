# Model File

The Flask application requires the final validation-selected
fine-tuned DenseNet121 checkpoint:

`best_model.keras`

Place the file in:

`app/models/best_model.keras`

The model file is not stored in this repository due to its size.

Final model:
- Architecture: DenseNet121
- Selection metric: Validation PR-AUC
- Validation PR-AUC: 0.587677
- Test ROC-AUC: 0.855333
- Test PR-AUC: 0.588136
- Validation-selected F1 threshold: 0.669041