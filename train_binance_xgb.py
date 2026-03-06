import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# Configuration
# =========================
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
LOOKBACK_DAYS = 30
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
KLINE_LIMIT = 1000
MODEL_PATH = "xgb_binance_btcusdt_5m.joblib"
FEATURE_IMPORTANCE_PATH = "feature_importance.png"
CONFUSION_MATRIX_PATH = "confusion_matrix.csv"
TEST_PREDICTIONS_PATH = "test_predictions.csv"
WALK_FORWARD_RESULTS_PATH = "walk_forward_results.csv"
BACKTEST_TRADES_PATH = "backtest_trades.csv"

CLASS_TO_LABEL = {0: "SELL", 1: "HOLD", 2: "BUY"}
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}
TARGET_VALUE_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_TARGET_VALUE = {v: k for k, v in TARGET_VALUE_TO_CLASS.items()}


# =========================
# Data fetching
# =========================
def fetch_binance_futures_klines(symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
    """
    Fetch recent futures klines directly from Binance Futures API.
    Binance klines max limit is 1000 per request, so paginate by time.
    """
    end_time_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    start_time_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).timestamp() * 1000)

    all_rows: List[list] = []
    current_start = start_time_ms

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    while current_start < end_time_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time_ms,
            "limit": KLINE_LIMIT,
        }
        response = session.get(BINANCE_FUTURES_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_start = last_open_time + 1
        if next_start <= current_start:
            break
        current_start = next_start

        if len(rows) < KLINE_LIMIT:
            break

    if not all_rows:
        raise RuntimeError("No kline data returned from Binance API.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


# =========================
# Technical indicators
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
# Feature engineering
# =========================
def build_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    data = df.copy()

    # Core returns
    data["ret_1"] = data["close"].pct_change()
    for i in range(1, 6):
        data[f"recent_ret_lag_{i}"] = data["close"].pct_change(i)

    # Indicators
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

    # EMA and cross/slope states
    for span in [9, 21, 50]:
        data[f"ema_{span}"] = ema(data["close"], span)
        data[f"ema_{span}_slope_1"] = data[f"ema_{span}"].pct_change()
        data[f"ema_{span}_slope_3"] = data[f"ema_{span}"].pct_change(3)

    data["ema9_gt_ema21"] = (data["ema_9"] > data["ema_21"]).astype(int)
    data["ema21_gt_ema50"] = (data["ema_21"] > data["ema_50"]).astype(int)
    data["ema9_gt_ema50"] = (data["ema_9"] > data["ema_50"]).astype(int)

    # Distance features
    data["close_vs_ema9"] = (data["close"] / data["ema_9"]) - 1
    data["close_vs_ema21"] = (data["close"] / data["ema_21"]) - 1
    data["close_vs_ema50"] = (data["close"] / data["ema_50"]) - 1

    # Volume change vs 20-bar average
    data["vol_mean_20"] = data["volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["volume"] / data["vol_mean_20"].replace(0, np.nan)
    data["volume_change_1"] = data["volume"].pct_change()

    # Candle structure
    data["candle_body_pct"] = (data["close"] - data["open"]) / data["open"].replace(0, np.nan)
    data["upper_wick_pct"] = (data[["high", "open", "close"]].max(axis=1) - data[["open", "close"]].max(axis=1)) / data["close"].replace(0, np.nan)
    data["lower_wick_pct"] = (data[["open", "close"]].min(axis=1) - data["low"]) / data["close"].replace(0, np.nan)

    # Time features (UTC)
    data["utc_hour"] = data["open_time"].dt.hour.astype(str)
    data["utc_dayofweek"] = data["open_time"].dt.dayofweek.astype(str)

    if with_target:
        future_max = data["high"].shift(-1).rolling(window=6, min_periods=6).max().shift(-5)
        future_min = data["low"].shift(-1).rolling(window=6, min_periods=6).min().shift(-5)

        up_move = (future_max / data["close"]) - 1
        down_move = (future_min / data["close"]) - 1

        target = np.where(up_move >= 0.005, 1, np.where(down_move <= -0.005, -1, 0))
        data["target_value"] = target
        data["target_class"] = data["target_value"].map(TARGET_VALUE_TO_CLASS)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


NUMERIC_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "recent_ret_lag_1",
    "recent_ret_lag_2",
    "recent_ret_lag_3",
    "recent_ret_lag_4",
    "recent_ret_lag_5",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_pos",
    "atr_14",
    "atr_pct",
    "adx_14",
    "plus_di",
    "minus_di",
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_9_slope_1",
    "ema_21_slope_1",
    "ema_50_slope_1",
    "ema_9_slope_3",
    "ema_21_slope_3",
    "ema_50_slope_3",
    "ema9_gt_ema21",
    "ema21_gt_ema50",
    "ema9_gt_ema50",
    "close_vs_ema9",
    "close_vs_ema21",
    "close_vs_ema50",
    "volume_ratio_20",
    "volume_change_1",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]
CATEGORICAL_FEATURES = ["utc_hour", "utc_dayofweek"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================
# Split and validation
# =========================
def time_order_split(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipe


def walk_forward_splits(n_samples: int, n_folds: int = 5):
    usable = n_samples // (n_folds + 1)
    for fold in range(1, n_folds + 1):
        train_end = usable * fold
        val_end = usable * (fold + 1) if fold < n_folds else n_samples
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        if len(val_idx) == 0:
            continue
        yield fold, train_idx, val_idx


def run_walk_forward_cv(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    rows = []
    X = df[ALL_FEATURES]
    y = df["target_class"]

    for fold, train_idx, val_idx in walk_forward_splits(len(df), n_folds=n_folds):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = make_pipeline()
        pipe.fit(X_train, y_train)

        train_pred = pipe.predict(X_train)
        val_pred = pipe.predict(X_val)

        rows.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_accuracy": accuracy_score(y_train, train_pred),
                "val_accuracy": accuracy_score(y_val, val_pred),
                "val_precision_macro": precision_score(y_val, val_pred, average="macro", zero_division=0),
                "val_recall_macro": recall_score(y_val, val_pred, average="macro", zero_division=0),
                "val_f1_macro": f1_score(y_val, val_pred, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Evaluation / backtest
# =========================
def evaluate_model(pipe: Pipeline, X_train, y_train, X_test, y_test):
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, test_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2])

    print("\n" + "=" * 80)
    print("FINAL HOLD-OUT TEST METRICS")
    print("=" * 80)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Overfitting Gap (Train-Test): {train_acc - test_acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1 Score  (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_pred,
            labels=[0, 1, 2],
            target_names=[CLASS_TO_LABEL[0], CLASS_TO_LABEL[1], CLASS_TO_LABEL[2]],
            zero_division=0,
        )
    )
    print("Confusion Matrix [rows=true, cols=pred]:")
    print(pd.DataFrame(cm, index=["SELL", "HOLD", "BUY"], columns=["SELL", "HOLD", "BUY"]))

    cm_df = pd.DataFrame(cm, index=["SELL", "HOLD", "BUY"], columns=["SELL", "HOLD", "BUY"])
    cm_df.to_csv(CONFUSION_MATRIX_PATH)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm_df,
        "train_pred": train_pred,
        "test_pred": test_pred,
    }


def save_feature_importance(pipe: Pipeline, top_n: int = 30):
    preprocessor = pipe.named_steps["preprocessor"]
    model = pipe.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    plt.figure(figsize=(12, 10))
    plot_df = fi.head(top_n).sort_values("importance", ascending=True)
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=180)
    plt.close()
    return fi


def simple_backtest(test_df: pd.DataFrame, pred_classes: np.ndarray, fee_rate: float = 0.001, hold_bars: int = 6):
    """
    Very simple non-overlapping backtest.
    Entry: next bar open after prediction bar
    Exit : open of bar after hold window, if available, else close of last available bar
    Fee  : fee_rate applied on entry and exit (round-trip cost = 2 * fee_rate)
    """
    data = test_df.copy().reset_index(drop=True)
    data["pred_class"] = pred_classes
    data["pred_signal"] = data["pred_class"].map(CLASS_TO_LABEL)

    trades = []
    i = 0
    while i < len(data) - hold_bars - 1:
        signal = data.loc[i, "pred_signal"]
        if signal == "HOLD":
            i += 1
            continue

        entry_idx = i + 1
        exit_idx = min(i + hold_bars + 1, len(data) - 1)

        entry_time = data.loc[entry_idx, "open_time"]
        exit_time = data.loc[exit_idx, "open_time"]
        entry_price = data.loc[entry_idx, "open"]
        exit_price = data.loc[exit_idx, "open"]

        if signal == "BUY":
            gross_return = (exit_price / entry_price) - 1
        else:  # SELL -> short
            gross_return = (entry_price / exit_price) - 1

        net_return = gross_return - (2 * fee_rate)
        trades.append(
            {
                "signal": signal,
                "prediction_index": i,
                "entry_index": entry_idx,
                "exit_index": exit_idx,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
            }
        )
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("\nNo trades generated in simple backtest.")
        return trades_df, {}

    trades_df["equity_curve"] = (1 + trades_df["net_return"]).cumprod()
    total_return = trades_df["equity_curve"].iloc[-1] - 1
    win_rate = (trades_df["net_return"] > 0).mean()
    avg_trade = trades_df["net_return"].mean()

    running_max = trades_df["equity_curve"].cummax()
    drawdown = trades_df["equity_curve"] / running_max - 1
    max_drawdown = drawdown.min()

    stats = {
        "num_trades": int(len(trades_df)),
        "win_rate": float(win_rate),
        "avg_trade_return": float(avg_trade),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
    }

    print("\n" + "=" * 80)
    print("SIMPLE BACKTEST (includes 0.1% fee per side)")
    print("=" * 80)
    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    trades_df.to_csv(BACKTEST_TRADES_PATH, index=False)
    return trades_df, stats


# =========================
# Main train routine
# =========================
def main():
    print("Fetching Binance Futures BTCUSDT 5m candles...")
    raw_df = fetch_binance_futures_klines(SYMBOL, INTERVAL, LOOKBACK_DAYS)
    print(f"Raw candles fetched: {len(raw_df):,}")
    print(f"Time range: {raw_df['open_time'].min()} -> {raw_df['open_time'].max()}")

    feat_df = build_features(raw_df, with_target=True)
    feat_df = feat_df.dropna(subset=ALL_FEATURES + ["target_class"]).reset_index(drop=True)

    print(f"Rows after feature engineering / dropping NaNs: {len(feat_df):,}")
    print("Target distribution:")
    target_dist = feat_df["target_class"].map(CLASS_TO_LABEL).value_counts(normalize=True).sort_index()
    print(target_dist)

    train_df, test_df = time_order_split(feat_df, train_ratio=0.8)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    print("\nRunning 5-fold walk-forward validation...")
    wf_results = run_walk_forward_cv(train_df, n_folds=5)
    print(wf_results)
    wf_results.to_csv(WALK_FORWARD_RESULTS_PATH, index=False)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df["target_class"]
    X_test = test_df[ALL_FEATURES]
    y_test = test_df["target_class"]

    pipe = make_pipeline()
    pipe.fit(X_train, y_train)

    eval_results = evaluate_model(pipe, X_train, y_train, X_test, y_test)

    fi_df = save_feature_importance(pipe, top_n=30)
    print(f"\nSaved feature importance chart -> {FEATURE_IMPORTANCE_PATH}")
    print(fi_df.head(20))

    test_output = test_df[["open_time", "open", "high", "low", "close", "volume", "target_class"]].copy()
    test_output["target_label"] = test_output["target_class"].map(CLASS_TO_LABEL)
    test_output["pred_class"] = eval_results["test_pred"]
    test_output["pred_label"] = test_output["pred_class"].map(CLASS_TO_LABEL)
    test_output.to_csv(TEST_PREDICTIONS_PATH, index=False)

    trades_df, backtest_stats = simple_backtest(test_df, eval_results["test_pred"], fee_rate=0.001, hold_bars=6)

    bundle = {
        "pipeline": pipe,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_to_label": CLASS_TO_LABEL,
        "label_to_class": LABEL_TO_CLASS,
        "target_value_to_class": TARGET_VALUE_TO_CLASS,
        "class_to_target_value": CLASS_TO_TARGET_VALUE,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle -> {MODEL_PATH}")

    summary = {
        "raw_rows": int(len(raw_df)),
        "feature_rows": int(len(feat_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "walk_forward": wf_results.to_dict(orient="records"),
        "metrics": {
            "train_accuracy": float(eval_results["train_accuracy"]),
            "test_accuracy": float(eval_results["test_accuracy"]),
            "precision_macro": float(eval_results["precision_macro"]),
            "recall_macro": float(eval_results["recall_macro"]),
            "f1_macro": float(eval_results["f1_macro"]),
        },
        "backtest": backtest_stats,
        "artifacts": {
            "model": MODEL_PATH,
            "feature_importance": FEATURE_IMPORTANCE_PATH,
            "confusion_matrix": CONFUSION_MATRIX_PATH,
            "test_predictions": TEST_PREDICTIONS_PATH,
            "walk_forward_results": WALK_FORWARD_RESULTS_PATH,
            "backtest_trades": BACKTEST_TRADES_PATH,
        },
    }
    with open("training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print("Saved training summary -> training_summary.json")


if __name__ == "__main__":
    main()
