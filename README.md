# Binance BTCUSDT 5m XGBoost Model

## 1. Google Colab training
```python
!pip install -r requirements.txt
!python train_binance_xgb.py
```

Artifacts generated:
- `xgb_binance_btcusdt_5m.joblib`
- `feature_importance.png`
- `confusion_matrix.csv`
- `test_predictions.csv`
- `walk_forward_results.csv`
- `backtest_trades.csv`
- `training_summary.json`

## 2. Run FastAPI locally
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 3. Sample predict request
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"open": 60000, "high": 60100, "low": 59900, "close": 60050, "volume": 120.5},
      {"open": 60050, "high": 60200, "low": 60000, "close": 60180, "volume": 98.1}
    ]
  }'
```

Real request must contain at least 100 candles.

## 4. Railway deploy
- Put `Dockerfile` at the repo root.
- Commit all files including the trained `.joblib` file.
- Deploy the repo on Railway.
- Railway automatically detects the root `Dockerfile`.
