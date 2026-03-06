import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="BTCUSDT XGBoost Signal API", version="1.1.0")

BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_MODEL_PATHS = [
    Path(os.getenv("MODEL_PATH", "")).resolve() if os.getenv("MODEL_PATH") else None,
    BASE_DIR / "model" / "xgb_binance_btcusdt_5m.joblib",
    BASE_DIR / "xgb_binance_btcusdt_5m.joblib",
]

model_bundle: Optional[Dict[str, Any]] = None
model_load_error: Optional[str] = None

CLASS_TO_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}


# =========================
# Request Schemas
# =========================
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: Optional[str] = None


class PredictRequest(BaseModel):
    candles: List[Candle] = Field(..., min_length=100)


class BatchPredictRequest(BaseModel):
    candles: List[Candle] = Field(..., min_length=100)
    window: int = Field(default=100, ge=30)
    step: int = Field(default=10, ge=1)


# =========================
# Indicator Functions
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_val = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_val.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di


# =========================
# Feature Engineering
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["ret_1"] = data["close"].pct_change()
    for i in range(1, 6):
        data[f"recent_ret_lag_{i}"] = data["close"].pct_change(i)

    data["rsi_14"] = rsi(data["close"], 14)

    macd_line, macd_signal, macd_hist = macd(data["close"], 12, 26, 9)
    data["macd_line"] = macd_line
    data["macd_signal"] = macd_signal
    data["macd_hist"] = macd_hist

    bb_mid, bb_upper, bb_lower = bollinger_bands(data["close"], 20, 2)
    data["bb_mid"] = bb_mid
    data["bb_upper"] = bb_upper
    data["bb_lower"] = bb_lower
    data["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    data["bb_pos"] = (data["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    data["atr_14"] = atr(data["high"], data["low"], data["close"], 14)
    data["atr_pct"] = data["atr_14"] / data["close"].replace(0, np.nan)

    adx_val, plus_di, minus_di = adx(data["high"], data["low"], data["close"], 14)
    data["adx_14"] = adx_val
    data["plus_di"] = plus_di
    data["minus_di"] = minus_di

    for span in [9, 21, 50]:
        data[f"ema_{span}"] = ema(data["close"], span)
        data[f"ema_{span}_slope_1"] = data[f"ema_{span}"].pct_change()
        data[f"ema_{span}_slope_3"] = data[f"ema_{span}"].pct_change(3)

    data["ema9_gt_ema21"] = (data["ema_9"] > data["ema_21"]).astype(int)
    data["ema21_gt_ema50"] = (data["ema_21"] > data["ema_50"]).astype(int)
    data["ema9_gt_ema50"] = (data["ema_9"] > data["ema_50"]).astype(int)

    data["close_vs_ema9"] = (data["close"] / data["ema_9"]) - 1
    data["close_vs_ema21"] = (data["close"] / data["ema_21"]) - 1
    data["close_vs_ema50"] = (data["close"] / data["ema_50"]) - 1

    data["vol_mean_20"] = data["volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["volume"] / data["vol_mean_20"].replace(0, np.nan)
    data["volume_change_1"] = data["volume"].pct_change()

    data["candle_body_pct"] = (data["close"] - data["open"]) / data["open"].replace(0, np.nan)
    data["upper_wick_pct"] = (
        data[["high", "open", "close"]].max(axis=1) - data[["open", "close"]].max(axis=1)
    ) / data["close"].replace(0, np.nan)
    data["lower_wick_pct"] = (
        data[["open", "close"]].min(axis=1) - data["low"]
    ) / data["close"].replace(0, np.nan)

    data["utc_hour"] = data["open_time"].dt.hour.astype(str)
    data["utc_dayofweek"] = data["open_time"].dt.dayofweek.astype(str)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


# =========================
# Model Utilities
# =========================
def find_existing_model_path() -> Optional[Path]:
    for p in CANDIDATE_MODEL_PATHS:
        if p is not None and p.exists():
            return p
    return None


def load_model_if_available():
    global model_bundle, model_load_error

    model_path = find_existing_model_path()
    if model_path is None:
        model_bundle = None
        model_load_error = (
            "Model file not found. Expected one of:\n"
            + "\n".join([str(p) for p in CANDIDATE_MODEL_PATHS if p is not None])
        )
        return

    try:
        loaded = joblib.load(model_path)
        if not isinstance(loaded, dict):
            raise ValueError("Loaded joblib is not a dict bundle.")
        if "pipeline" not in loaded:
            raise ValueError("Loaded bundle does not contain 'pipeline'.")
        if "features" not in loaded:
            raise ValueError("Loaded bundle does not contain 'features'.")
        model_bundle = loaded
        model_load_error = None
    except Exception as e:
        model_bundle = None
        model_load_error = f"Failed to load model from {model_path}: {repr(e)}"


@app.on_event("startup")
def startup_event():
    load_model_if_available()


def require_model():
    if model_bundle is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "MODEL_NOT_LOADED",
                "message": model_load_error,
            },
        )
    return model_bundle


def _prepare_dataframe_from_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(candles).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open_time" in df.columns and df["open_time"].notna().any():
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    else:
        df["open_time"] = pd.date_range(
            end=pd.Timestamp.utcnow(),
            periods=len(df),
            freq="5min",
            tz="UTC",
        )

    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def _predict_from_window_df(window_df: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, Any]:
    pipe = bundle["pipeline"]
    feature_names = bundle["features"]
    class_to_label = bundle.get("class_to_label", CLASS_TO_LABEL)

    feat_df = build_features(window_df)
    feat_df = feat_df.ffill().bfill()

    X_last = feat_df.iloc[[-1]][feature_names].copy()
    probs = pipe.predict_proba(X_last)[0]

    pred_idx = int(np.argmax(probs))
    signal = class_to_label.get(pred_idx, "HOLD")

    probability_map = {
        class_to_label.get(0, "SELL"): float(probs[0]),
        class_to_label.get(1, "HOLD"): float(probs[1]),
        class_to_label.get(2, "BUY"): float(probs[2]),
    }

    return {
        "signal": signal,
        "confidence": float(np.max(probs)),
        "probabilities": probability_map,
    }


# =========================
# Routes
# =========================
@app.get("/")
def root():
    model_path = find_existing_model_path()
    return {
        "service": "BTCUSDT XGBoost Signal API",
        "status": "ok" if model_bundle is not None else "model_missing",
        "model_loaded": model_bundle is not None,
        "model_path": str(model_path) if model_path else None,
        "error": model_load_error,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model_bundle is not None else "degraded",
        "model_loaded": model_bundle is not None,
        "error": model_load_error,
    }


@app.post("/reload-model")
def reload_model():
    load_model_if_available()
    model_path = find_existing_model_path()
    return {
        "model_loaded": model_bundle is not None,
        "model_path": str(model_path) if model_path else None,
        "error": model_load_error,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    bundle = require_model()

    candles = [c.model_dump() for c in req.candles]
    if len(candles) < 100:
        raise HTTPException(status_code=400, detail="At least 100 candles are required.")

    try:
        df = _prepare_dataframe_from_candles(candles)
        result = _predict_from_window_df(df, bundle)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {repr(e)}")

    return result


@app.post("/batch-predict")
def batch_predict(req: BatchPredictRequest):
    bundle = require_model()

    candles = [c.model_dump() for c in req.candles]
    total_len = len(candles)
    window = req.window
    step = req.step

    if total_len < window:
        raise HTTPException(
            status_code=400,
            detail=f"At least {window} candles are required for batch prediction.",
        )

    try:
        df = _prepare_dataframe_from_candles(candles)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Input parsing failed: {repr(e)}")

    predictions = []

    try:
        for end_idx in range(window, total_len + 1, step):
            start_idx = end_idx - window
            window_df = df.iloc[start_idx:end_idx].copy()

            pred = _predict_from_window_df(window_df, bundle)
            predictions.append(
                {
                    "index": end_idx,
                    "signal": pred["signal"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {repr(e)}")

    return {"predictions": predictions}
