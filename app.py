import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================
# FastAPI App
# =========================
app = FastAPI(title="BTCUSDT XGBoost Signal API", version="1.0.0")


# =========================
# Paths / Globals
# =========================
BASE_DIR = Path(__file__).resolve().parent

# 우선순위:
# 1) ENV MODEL_PATH
# 2) /app/model/xgb_binance_btcusdt_5m.joblib
# 3) /app/xgb_binance_btcusdt_5m.joblib
CANDIDATE_MODEL_PATHS = [
    Path(os.getenv("MODEL_PATH", "")).resolve() if os.getenv("MODEL_PATH") else None,
    BASE_DIR / "model" / "xgb_binance_btcusdt_5m.joblib",
    BASE_DIR / "xgb_binance_btcusdt_5m.joblib",
]

model_bundle: Optional[Dict[str, Any]] = None
model_load_error: Optional[str] = None


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


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


# =========================
# Indicator Functions
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_series

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_series = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_series.fillna(0)


# =========================
# Feature Engineering
# =========================
def build_features_from_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(candles).copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required field: {col}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open_time" in df.columns and df["open_time"].notna().any():
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    else:
        # open_time이 없으면 더미 UTC 시간 생성
        df["open_time"] = pd.date_range(
            end=utc_now(),
            periods=len(df),
            freq="5min",
            tz="UTC",
        )

    # 기본 정렬
    df = df.sort_values("open_time").reset_index(drop=True)

    # Technical indicators
    df["rsi_14"] = rsi(df["close"], 14)

    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"], 12, 26, 9)

    df["bb_mid"], df["bb_upper"], df["bb_lower"] = bollinger_bands(df["close"], 20, 2.0)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_pos"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    df["atr_14"] = atr(df, 14)
    df["adx_14"] = adx(df, 14)

    # EMA
    df["ema_9"] = ema(df["close"], 9)
    df["ema_21"] = ema(df["close"], 21)
    df["ema_50"] = ema(df["close"], 50)

    # EMA cross states
    df["ema9_gt_ema21"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema21_gt_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)
    df["ema9_gt_ema50"] = (df["ema_9"] > df["ema_50"]).astype(int)

    df["ema9_21_spread"] = (df["ema_9"] - df["ema_21"]) / df["close"].replace(0, np.nan)
    df["ema21_50_spread"] = (df["ema_21"] - df["ema_50"]) / df["close"].replace(0, np.nan)
    df["ema9_50_spread"] = (df["ema_9"] - df["ema_50"]) / df["close"].replace(0, np.nan)

    # EMA slopes
    df["ema_9_slope"] = df["ema_9"].pct_change()
    df["ema_21_slope"] = df["ema_21"].pct_change()
    df["ema_50_slope"] = df["ema_50"].pct_change()

    # Volume features
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio_20"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
    df["vol_change"] = df["volume"].pct_change()

    # Recent returns pattern (최근 5봉)
    df["ret_1"] = df["close"].pct_change(1)
    for i in range(1, 6):
        df[f"recent_ret_lag_{i}"] = df["ret_1"].shift(i)

    # Price range features
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_change"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # Time features (UTC)
    df["hour_utc"] = df["open_time"].dt.hour.astype(int)
    df["dayofweek_utc"] = df["open_time"].dt.dayofweek.astype(int)

    # 훈련 시 사용한 시간 원핫을 수동으로 고정 생성
    for h in range(24):
        df[f"hour_{h}"] = (df["hour_utc"] == h).astype(int)

    for d in range(7):
        df[f"dow_{d}"] = (df["dayofweek_utc"] == d).astype(int)

    # 최종 피처
    feature_columns = [
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "bb_pos",
        "atr_14",
        "adx_14",
        "ema_9",
        "ema_21",
        "ema_50",
        "ema9_gt_ema21",
        "ema21_gt_ema50",
        "ema9_gt_ema50",
        "ema9_21_spread",
        "ema21_50_spread",
        "ema9_50_spread",
        "ema_9_slope",
        "ema_21_slope",
        "ema_50_slope",
        "vol_ratio_20",
        "vol_change",
        "hl_range",
        "oc_change",
        "recent_ret_lag_1",
        "recent_ret_lag_2",
        "recent_ret_lag_3",
        "recent_ret_lag_4",
        "recent_ret_lag_5",
    ] + [f"hour_{h}" for h in range(24)] + [f"dow_{d}" for d in range(7)]

    features = df[feature_columns].copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill()

    return features


def build_pipeline_features_from_candles(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build features compatible with the training pipeline bundle.
    This mirrors the feature names used in train_binance_xgb.py.
    """
    df = pd.DataFrame(candles).copy()

    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required field: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "open_time" in df.columns and df["open_time"].notna().any():
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    else:
        df["open_time"] = pd.date_range(
            end=utc_now(),
            periods=len(df),
            freq="5min",
            tz="UTC",
        )

    df = df.sort_values("open_time").reset_index(drop=True)

    # Core returns
    df["ret_1"] = df["close"].pct_change()
    for i in range(1, 6):
        df[f"recent_ret_lag_{i}"] = df["close"].pct_change(i)

    # Indicators
    df["rsi_14"] = rsi(df["close"], 14)
    df["macd_line"], df["macd_signal"], df["macd_hist"] = macd(df["close"], 12, 26, 9)

    bb_mid, bb_upper, bb_lower = bollinger_bands(df["close"], 20, 2)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df["atr_14"] = atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)

    adx_val = adx(df, 14)
    df["adx_14"] = adx_val

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr_series = true_range(df).ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    df["plus_di"] = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_series)
    df["minus_di"] = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_series)

    # EMA and derived features
    for span in [9, 21, 50]:
        df[f"ema_{span}"] = ema(df["close"], span)
        df[f"ema_{span}_slope_1"] = df[f"ema_{span}"].pct_change()
        df[f"ema_{span}_slope_3"] = df[f"ema_{span}"].pct_change(3)

    df["ema9_gt_ema21"] = (df["ema_9"] > df["ema_21"]).astype(int)
    df["ema21_gt_ema50"] = (df["ema_21"] > df["ema_50"]).astype(int)
    df["ema9_gt_ema50"] = (df["ema_9"] > df["ema_50"]).astype(int)

    df["close_vs_ema9"] = (df["close"] / df["ema_9"]) - 1
    df["close_vs_ema21"] = (df["close"] / df["ema_21"]) - 1
    df["close_vs_ema50"] = (df["close"] / df["ema_50"]) - 1

    df["vol_mean_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["volume"] / df["vol_mean_20"].replace(0, np.nan)
    df["volume_change_1"] = df["volume"].pct_change()

    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["upper_wick_pct"] = (
        df[["high", "open", "close"]].max(axis=1) - df[["open", "close"]].max(axis=1)
    ) / df["close"].replace(0, np.nan)
    df["lower_wick_pct"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / df["close"].replace(0, np.nan)

    # Categorical time features expected by training pipeline
    df["utc_hour"] = df["open_time"].dt.hour.astype(str)
    df["utc_dayofweek"] = df["open_time"].dt.dayofweek.astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()

    expected_columns = [
        "open", "high", "low", "close", "volume",
        "ret_1",
        "recent_ret_lag_1", "recent_ret_lag_2", "recent_ret_lag_3", "recent_ret_lag_4", "recent_ret_lag_5",
        "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pos",
        "atr_14", "atr_pct", "adx_14", "plus_di", "minus_di",
        "ema_9", "ema_21", "ema_50",
        "ema_9_slope_1", "ema_21_slope_1", "ema_50_slope_1",
        "ema_9_slope_3", "ema_21_slope_3", "ema_50_slope_3",
        "ema9_gt_ema21", "ema21_gt_ema50", "ema9_gt_ema50",
        "close_vs_ema9", "close_vs_ema21", "close_vs_ema50",
        "volume_ratio_20", "volume_change_1",
        "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
        "utc_hour", "utc_dayofweek",
    ]
    return df[expected_columns].copy()


def get_predictor_from_bundle(bundle: Any) -> Tuple[str, Any, Optional[List[str]]]:
    """
    Returns (bundle_type, predictor, feature_columns)
    - legacy bundle_type uses a bare model + feature_columns
    - pipeline bundle_type uses sklearn Pipeline with internal preprocessor
    - direct bundle_type uses model object itself (non-dict artifact)
    """
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle is empty.")

    if not isinstance(bundle, dict):
        if hasattr(bundle, "predict_proba"):
            return "direct", bundle, None
        raise HTTPException(status_code=500, detail="Invalid model bundle structure.")

    legacy_model = bundle.get("model")
    legacy_columns = bundle.get("feature_columns")
    if legacy_model is not None and legacy_columns is not None:
        return "legacy", legacy_model, legacy_columns

    pipeline = bundle.get("pipeline")
    if pipeline is not None:
        return "pipeline", pipeline, None

    raise HTTPException(
        status_code=500,
        detail=f"Invalid model bundle structure. keys={sorted(bundle.keys())}",
    )


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
        model_bundle = joblib.load(model_path)
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

    bundle_type, predictor, feature_columns = get_predictor_from_bundle(bundle)

    try:
        if bundle_type == "legacy":
            X_all = build_features_from_candles(candles)
            X_last = X_all.iloc[[-1]].copy()

            missing_cols = [c for c in feature_columns if c not in X_last.columns]
            for col in missing_cols:
                X_last[col] = 0.0

            X_last = X_last[feature_columns]
            X_last = X_last.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            X_all = build_pipeline_features_from_candles(candles)
            X_last = X_all.iloc[[-1]].copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature generation failed: {repr(e)}")

    try:
        probs = predictor.predict_proba(X_last)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {repr(e)}")

    # 학습 코드 기준 매핑:
    # 0 -> SELL(-1)
    # 1 -> HOLD(0)
    # 2 -> BUY(1)
    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

    pred_idx = int(np.argmax(probs))
    signal = label_map[pred_idx]

    probability_map = {
        "SELL": float(probs[0]),
        "HOLD": float(probs[1]),
        "BUY": float(probs[2]),
    }

    return {
        "signal": signal,
        "confidence": float(np.max(probs)),
        "probabilities": probability_map,
    }
