from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "xgb_binance_btcusdt_5m.joblib"

# Reuse exact same feature engineering logic from training
# -------------------------------------------------------
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
    data["upper_wick_pct"] = (data[["high", "open", "close"]].max(axis=1) - data[["open", "close"]].max(axis=1)) / data["close"].replace(0, np.nan)
    data["lower_wick_pct"] = (data[["open", "close"]].min(axis=1) - data["low"]) / data["close"].replace(0, np.nan)

    if "open_time" in data.columns:
        data["open_time"] = pd.to_datetime(data["open_time"], utc=True, errors="coerce")
        data["utc_hour"] = data["open_time"].dt.hour.astype("Int64").astype(str)
        data["utc_dayofweek"] = data["open_time"].dt.dayofweek.astype("Int64").astype(str)
    else:
        # If open_time is not given, assume sequential current UTC timestamps spaced by 5 minutes.
        synthetic_index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(data), freq="5min", tz="UTC")
        data["utc_hour"] = synthetic_index.hour.astype(str)
        data["utc_dayofweek"] = synthetic_index.dayofweek.astype(str)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time: str | None = None


class PredictRequest(BaseModel):
    candles: List[Candle] = Field(..., min_length=100, description="Most recent 100+ OHLCV candles")


app = FastAPI(title="Binance BTCUSDT XGBoost Signal API")
model_bundle = joblib.load(MODEL_PATH)
pipeline = model_bundle["pipeline"]
features = model_bundle["features"]
class_to_label = {int(k): v for k, v in model_bundle["class_to_label"].items()}


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.candles) < 100:
        raise HTTPException(status_code=400, detail="At least 100 candles are required.")

    df = pd.DataFrame([c.model_dump() for c in req.candles])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feat_df = build_features(df)
    feat_df = feat_df.dropna().reset_index(drop=True)
    if feat_df.empty:
        raise HTTPException(status_code=400, detail="Not enough valid rows after feature engineering.")

    latest = feat_df.iloc[[-1]][features]
    probs = pipeline.predict_proba(latest)[0]
    pred_class = int(np.argmax(probs))
    signal = class_to_label[pred_class]

    response = {
        "signal": signal,
        "confidence": float(np.max(probs)),
        "probabilities": {
            "SELL": float(probs[0]),
            "HOLD": float(probs[1]),
            "BUY": float(probs[2]),
        },
    }
    return response
