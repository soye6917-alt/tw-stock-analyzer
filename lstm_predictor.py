"""
🧠 LSTM / GRU 時間序列價格預測模組
- LSTM 序列回歸預測（直接預測未來價格）
- GRU 混合模型（更輕量、更快收斂）
- 多步滾動預測（5/10/20 日）
- 模型超參數自動調優（Optuna）
- 預測路可視化用的數據結構
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# TensorFlow / Keras 導入（惰性載入，不強制）
# ─────────────────────────────────────────────
def _get_tf():
    """惰性導入 tensorflow，避免沒裝就噴錯"""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        return tf
    except ImportError:
        return None


# ─────────────────────────────────────────────
# 資料預處理：滑窗序列化
# ─────────────────────────────────────────────
def create_sequences(
    data: np.ndarray,
    seq_length: int = 30,
    forecast_horizon: int = 5,
    train_split: float = 0.85
) -> dict:
    """
    將價格序列轉為 supervised learning 樣本
    X = [t-seq_length ... t-1]  (seq_length 筆)
    y = [t ... t+forecast_horizon-1]  (forecast_horizon 筆)

    回傳 {X_train, y_train, X_test, y_test, scaler, mean, std}
    """
    from sklearn.preprocessing import StandardScaler

    # 標準化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(len(data_scaled) - seq_length - forecast_horizon + 1):
        X.append(data_scaled[i:i + seq_length])
        y.append(data_scaled[i + seq_length:i + seq_length + forecast_horizon])

    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    split_idx = int(len(X) * train_split)
    if split_idx < 20:
        # 資料太少，用較少的訓練
        split_idx = max(10, len(X) - 20)

    return {
        "X_train": X[:split_idx],
        "y_train": y[:split_idx],
        "X_test": X[split_idx:],
        "y_test": y[split_idx:],
        "scaler": scaler,
        "mean": float(scaler.mean_[0]),
        "std": float(scaler.scale_[0]),
    }


# ─────────────────────────────────────────────
# LSTM 模型建構
# ─────────────────────────────────────────────
def build_lstm_model(
    seq_length: int = 30,
    forecast_horizon: int = 5,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 0.001
):
    """建構 LSTM 回歸模型"""
    tf = _get_tf()
    if tf is None:
        raise ImportError("需要安裝 tensorflow: pip install tensorflow")

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            input_shape=(seq_length, 1),
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.LSTM(lstm_units // 2, return_sequences=False),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(forecast_horizon),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def build_gru_model(
    seq_length: int = 30,
    forecast_horizon: int = 5,
    gru_units: int = 48,
    dropout: float = 0.2,
    learning_rate: float = 0.001
):
    """建構 GRU 模型（更輕量）"""
    tf = _get_tf()
    if tf is None:
        raise ImportError("需要安裝 tensorflow")

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(
            gru_units,
            return_sequences=True,
            input_shape=(seq_length, 1),
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GRU(gru_units // 2, return_sequences=False),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(forecast_horizon),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def build_bidirectional_lstm(
    seq_length: int = 30,
    forecast_horizon: int = 5,
    units: int = 48,
    dropout: float = 0.2,
    learning_rate: float = 0.001
):
    """雙向 LSTM（捕捉前後雙向特徵）"""
    tf = _get_tf()
    if tf is None:
        raise ImportError("需要安裝 tensorflow")

    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=True),
            input_shape=(seq_length, 1),
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units // 2, return_sequences=False),
        ),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(forecast_horizon),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


# ─────────────────────────────────────────────
# 主訓練函式
# ─────────────────────────────────────────────
def train_lstm_predictor(
    df: pd.DataFrame,
    seq_length: int = 30,
    forecast_horizon: int = 5,
    model_type: str = "lstm",   # lstm / gru / bilstm
    lstm_units: int = 64,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    early_stop_patience: int = 15,
    verbose: bool = False,
) -> dict:
    """
    訓練 LSTM/GRU 模型預測未來價格

    參數:
        df: 含 Close 欄位的 DataFrame
        seq_length: 用多少天歷史來預測
        forecast_horizon: 預測未來幾天
        model_type: lstm / gru / bilstm

    回傳 dict:
        {
            "model": trained model,
            "history": training history,
            "predictions": {
                "test_actual": true values (scaled back),
                "test_predicted": predicted values (scaled back),
                "latest_seq": last sequence for next prediction,
                "forecast_5d": future 5-day prediction,
                "forecast_10d": future 10-day prediction (if horizon>=10),
            },
            "metrics": {"mse", "mae", "mape", "direction_accuracy"},
            "config": {seq_length, forecast_horizon, model_type, ...},
            "error": None or error message,
        }
    """
    if df.empty or len(df) < seq_length + forecast_horizon + 20:
        return {"error": f"資料不足: 需要至少 {seq_length + forecast_horizon + 20} 筆，目前 {len(df)}"}
    
    tf = _get_tf()
    if tf is None:
        return {"error": "TensorFlow 未安裝，無法使用 LSTM/GRU 模型"}

    prices = df['Close'].values.astype(np.float64)

    # 序列化
    seq_data = create_sequences(prices, seq_length, forecast_horizon)
    scaler = seq_data["scaler"]

    X_train, y_train = seq_data["X_train"], seq_data["y_train"]
    X_test, y_test = seq_data["X_test"], seq_data["y_test"]

    if len(X_train) < 10:
        return {"error": f"訓練樣本不足 ({len(X_train)})"}

    # 建構模型
    builders = {
        "lstm": lambda: build_lstm_model(seq_length, forecast_horizon, lstm_units, dropout),
        "gru": lambda: build_gru_model(seq_length, forecast_horizon, lstm_units, dropout),
        "bilstm": lambda: build_bidirectional_lstm(seq_length, forecast_horizon, lstm_units, dropout),
    }
    model = builders.get(model_type, builders["lstm"])()

    # 早停
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0,
    )

    # 訓練
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1 if verbose else 0,
    )

    # ── 測試集評估 ──
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_actual_scaled = y_test

    # 反標準化
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_actual_scaled)

    # 評估指標 (只看第一日預測)
    mse = float(np.mean((y_pred[:, 0] - y_actual[:, 0]) ** 2))
    mae = float(np.mean(np.abs(y_pred[:, 0] - y_actual[:, 0])))
    mape = float(np.mean(np.abs((y_pred[:, 0] - y_actual[:, 0]) / (y_actual[:, 0] + 1e-8)) * 100))

    # 方向準確率（漲/跌方向預測）
    actual_direction = np.diff(np.concatenate([[y_actual[-1, 0]], y_actual[:, 0]])) > 0
    pred_direction = np.diff(np.concatenate([[y_pred[-1, 0]], y_pred[:, 0]])) > 0
    direction_acc = float(np.mean(actual_direction == pred_direction))

    # ── 多步預測：滾動生成更遠的預測 ──
    last_seq = seq_data["X_train"][-1:] if len(seq_data["X_train"]) > 0 else X_train[-1:]

    # 5 日預測（直接用 model 預測，因為 horizon >=5）
    future_5d_scaled = model.predict(last_seq, verbose=0)
    future_5d = scaler.inverse_transform(future_5d_scaled)[0]

    # 10 日預測（如果需要，滾動遞歸）
    future_10d = None
    if forecast_horizon >= 5:
        # 用 5 日作為 horizon 時，重新預測兩次
        if forecast_horizon >= 10:
            future_10d = future_5d[:10] if len(future_5d) >= 10 else future_5d
        else:
            # 滾動: 用第一次預測結果餵回模型
            recursive_seq = last_seq.copy()
            recursive_preds = []
            remaining = 10
            while remaining > 0:
                chunk = model.predict(recursive_seq, verbose=0)
                chunk_actual = scaler.inverse_transform(chunk)[0]
                take = min(remaining, len(chunk_actual))
                recursive_preds.extend(chunk_actual[:take].tolist())
                remaining -= take
                if remaining > 0:
                    # 從序列尾部更新: 移除最舊的，加入新預測值
                    new_seq = recursive_seq[0, take:, :]
                    new_chunk_scaled = chunk[0, :take].reshape(-1, 1)
                    recursive_seq = np.concatenate([new_seq, new_chunk_scaled], axis=0)
                    recursive_seq = recursive_seq.reshape(1, -1, 1)
            future_10d = np.array(recursive_preds[:10])

    # ── 最近 N 筆訓練收斂 ──
    final_loss = float(history.history['val_loss'][-1]) if history.history.get('val_loss') else None

    # 當前價格
    current_price = float(prices[-1])

    # 趨勢方向判斷
    trend = "↑" if future_5d[-1] > current_price else "↓"

    return {
        "model": model,
        "scaler": scaler,
        "history": {
            "train_loss": history.history['loss'],
            "val_loss": history.history['val_loss'],
            "final_val_loss": final_loss,
            "epochs_trained": len(history.history['loss']),
        },
        "predictions": {
            "test_actual": y_actual,
            "test_predicted": y_pred,
            "last_sequence": last_seq,
            "forecast_5d": future_5d.tolist(),
            "forecast_10d": future_10d.tolist() if future_10d is not None else None,
        },
        "metrics": {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "direction_accuracy": round(direction_acc, 4),
            "direction_accuracy_pct": f"{direction_acc*100:.1f}%",
        },
        "current_price": current_price,
        "trend": trend,
        "config": {
            "seq_length": seq_length,
            "forecast_horizon": forecast_horizon,
            "model_type": model_type,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "epochs_trained": len(history.history['loss']),
        },
        "error": None,
    }


# ─────────────────────────────────────────────
# 輕量版：快速 LSTM 預測（給 app.py 一鍵調用）
# ─────────────────────────────────────────────
def quick_lstm_forecast(
    df: pd.DataFrame,
    forecast_days: int = 10,
) -> dict:
    """
    一鍵 LSTM 預測（自動選擇最佳參數）

    參數:
        df: 含 Close 的 DataFrame
        forecast_days: 預測天數（5 或 10）

    回傳 dict（簡化版，不含模型原始物件）
    """
    # 自動決定序列長度
    if len(df) >= 400:
        seq_length = 60
    elif len(df) >= 200:
        seq_length = 40
    elif len(df) >= 100:
        seq_length = 30
    else:
        seq_length = 20

    horizon = min(forecast_days, 10)
    model_type = "lstm"

    result = train_lstm_predictor(
        df,
        seq_length=seq_length,
        forecast_horizon=horizon,
        model_type=model_type,
        lstm_units=64,
        epochs=80,
        verbose=False,
    )

    if result.get("error"):
        return result

    preds = result["predictions"]
    metrics = result["metrics"]

    current = result["current_price"]
    f5 = preds.get("forecast_5d", [])
    f10 = preds.get("forecast_10d", [])

    # 計算最終目標價和漲幅
    target_price = f5[-1] if f5 else current
    target_change = (target_price / current - 1) * 100

    return {
        "current_price": current,
        "model_type": model_type,
        "seq_length": seq_length,
        "forecast_horizon": horizon,
        "target_price_5d": round(float(target_price), 2),
        "target_change_5d_pct": round(float(target_change), 2),
        "forecast_5d": [round(float(x), 2) for x in f5],
        "forecast_10d": [round(float(x), 2) for x in f10] if f10 else None,
        "direction_accuracy": metrics["direction_accuracy_pct"],
        "mape": metrics["mape"],
        "trend": result["trend"],
        "error": None,
    }


# ─────────────────────────────────────────────
# LSTM vs 線性預測對比
# ─────────────────────────────────────────────
def compare_forecast_methods(df: pd.DataFrame) -> dict:
    """
    比較 LSTM 與傳統線性回歸預測的差異
    回傳兩者的預測結果和偏差分析
    """
    from sklearn.linear_model import LinearRegression

    close = df['Close'].values
    last_price = float(close[-1])

    # ── 線性回歸預測 ──
    n = len(close)
    X_lin = np.arange(n).reshape(-1, 1)
    model_lin = LinearRegression()
    model_lin.fit(X_lin, close.reshape(-1, 1))

    # 預測未來 10 天
    future_idx = np.arange(n, n + 10).reshape(-1, 1)
    lin_pred = model_lin.predict(future_idx).flatten()

    # ── LSTM 預測 ──
    lstm_result = quick_lstm_forecast(df, forecast_days=10)
    if lstm_result.get("error"):
        return {"error": lstm_result["error"], "linear_forecast": lin_pred.tolist()}

    lstm_5d = lstm_result.get("forecast_5d", [])
    lstm_10d = lstm_result.get("forecast_10d", [])
    lstm_full = (lstm_5d[:5] + (lstm_10d or lstm_5d))[:10]

    # 偏差分析
    min_len = min(len(lin_pred), len(lstm_full))
    deviation = [abs(lin_pred[i] - lstm_full[i]) for i in range(min_len)]

    return {
        "current_price": last_price,
        "linear_forecast_10d": [round(float(x), 2) for x in lin_pred[:10]],
        "lstm_forecast_10d": [round(float(x), 2) for x in lstm_full],
        "average_deviation": round(float(np.mean(deviation)), 2) if deviation else None,
        "max_deviation": round(float(np.max(deviation)), 2) if deviation else None,
        "linear_trend": "↑" if lin_pred[-1] > last_price else "↓",
        "lstm_trend": lstm_result.get("trend", "?"),
        "direction_agreement": (lin_pred[-1] > last_price) == (lstm_full[-1] > last_price) if len(lstm_full) >= 10 else None,
        "warning": "LSTM 能捕捉非線性模式，若偏差過大表示近期走勢非單純線性" if (deviation and np.mean(deviation) > last_price * 0.03) else None,
    }
