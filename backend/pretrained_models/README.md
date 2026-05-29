# Pre-Trained Model Directory

Each subdirectory here represents one model available for inference through
the `/api/pretrained/predict` endpoint.

## Required directory layout

```
pretrained_models/
└── <model_name>/
    ├── metadata.json       ← required
    ├── preprocess.py       ← optional (but recommended for custom feature engineering)
    └── model.keras         ← OR model.h5 (Keras/LSTM) OR model.pkl (sklearn)
        scaler.pkl          ← optional — inverse-transforms outputs
```

### `preprocess.py` (Custom Inference Pipeline)

If your model requires specific feature engineering (e.g., lagged features, one-hot encoding, extracting date components) before inference, you should place a `preprocess.py` inside the model directory. 

The `/api/pretrained/evaluate` endpoint will automatically detect this file and call:
```python
X_test, y_test, test_dates = preprocess_module.preprocess_evaluate(df, target_col, date_col, test_size)
```
This isolates the model logic so the main API server doesn't need to be updated when you deploy new models!

## `metadata.json` schema

```json
{
  "description": "Human-readable description shown in the UI",
  "model_type": "LSTM",           // LSTM | MLR | ARIMA | etc.
  "input_schema": {
    "type": "array",
    "description": "Flat list of the last N scaled closing prices",
    "example": [0.43, 0.51, 0.48]
  },
  "output_schema": {
    "type": "array",
    "description": "Predicted value(s) in original scale"
  }
}
```

## Example: adding an LSTM stock-price model

```
pretrained_models/
└── lstm_saham_ihsg/
    ├── metadata.json
    ├── model.keras
    └── scaler.pkl
```

The JSON you send from the UI should match the shape the model expects.
For an LSTM with window_size=60, pass a list of 60 numbers:

```json
[0.31, 0.33, 0.30, ..., 0.45]
```
