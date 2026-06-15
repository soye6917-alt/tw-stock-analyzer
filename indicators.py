"""
技術指標計算模組
- MA, EMA, RSI, MACD, Bollinger Bands, KDJ
"""

import pandas as pd
import numpy as np


def add_ma(df: pd.DataFrame, periods: list = [5, 10, 20, 60, 120]) -> pd.DataFrame:
    """移動平均線"""
    result = df.copy()
    for p in periods:
        result[f"MA{p}"] = result["Close"].rolling(window=p).mean()
    return result


def add_ema(df: pd.DataFrame, periods: list = [5, 12, 26]) -> pd.DataFrame:
    """指數移動平均"""
    result = df.copy()
    for p in periods:
        result[f"EMA{p}"] = result["Close"].ewm(span=p, adjust=False).mean()
    return result


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI 指標"""
    result = df.copy()
    delta = result["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))
    # 前 period 筆用簡化計算
    result["RSI"] = result["RSI"].fillna(50)
    return result


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD 指標"""
    result = df.copy()
    ema12 = result["Close"].ewm(span=12, adjust=False).mean()
    ema26 = result["Close"].ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_Hist"] = result["MACD"] - result["MACD_Signal"]
    return result


def add_bollinger(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.DataFrame:
    """布林通道"""
    result = df.copy()
    result["BB_Mid"] = result["Close"].rolling(window=period).mean()
    bb_std = result["Close"].rolling(window=period).std()
    result["BB_Upper"] = result["BB_Mid"] + std * bb_std
    result["BB_Lower"] = result["BB_Mid"] - std * bb_std
    result["BB_Width"] = result["BB_Upper"] - result["BB_Lower"]
    return result


def add_kdj(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """KDJ 隨機指標"""
    result = df.copy()
    low_min = result["Low"].rolling(window=period).min()
    high_max = result["High"].rolling(window=period).max()
    rsv = 100 * ((result["Close"] - low_min) / (high_max - low_min).replace(0, np.nan))
    result["RSV"] = rsv.fillna(50)
    result["K"] = 50.0
    result["D"] = 50.0
    for i in range(period, len(result)):
        result.loc[result.index[i], "K"] = (
            2/3 * result.loc[result.index[i-1], "K"] + 1/3 * result.loc[result.index[i], "RSV"]
        )
        result.loc[result.index[i], "D"] = (
            2/3 * result.loc[result.index[i-1], "D"] + 1/3 * result.loc[result.index[i], "K"]
        )
    result["J"] = 3 * result["K"] - 2 * result["D"]
    # 前 period 筆先填 50
    result.loc[:period-1, "K"] = 50
    result.loc[:period-1, "D"] = 50
    result.loc[:period-1, "J"] = 50
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次加入所有常用指標"""
    df = add_ma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_kdj(df)
    return df


def get_indicator_signals(df: pd.DataFrame) -> dict:
    """提供最新的技術訊號摘要"""
    if df.empty or len(df) < 30:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals = {}

    # RSI 訊號
    rsi = latest.get("RSI", 50)
    if rsi > 70:
        signals["RSI"] = "🟡 超買區 (>70)"
    elif rsi < 30:
        signals["RSI"] = "🟢 超賣區 (<30)"
    else:
        signals["RSI"] = f"⚪ 中性 ({rsi:.1f})"

    # MACD 交叉訊號
    if latest.get("MACD", 0) > latest.get("MACD_Signal", 0):
        if prev.get("MACD", 0) <= prev.get("MACD_Signal", 0):
            signals["MACD"] = "🟢 黃金交叉 ↑"
        else:
            signals["MACD"] = "🟢 MACD 在訊號線上"
    else:
        if prev.get("MACD", 0) >= prev.get("MACD_Signal", 0):
            signals["MACD"] = "🔴 死亡交叉 ↓"
        else:
            signals["MACD"] = "🔴 MACD 在訊號線下"

    # 布林通道
    close = latest["Close"]
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    bb_mid = latest.get("BB_Mid", 0)
    if close >= bb_upper:
        signals["布林"] = "🔴 觸頂 (壓力)"
    elif close <= bb_lower:
        signals["布林"] = "🟢 觸底 (支撐)"
    else:
        band_range = bb_upper - bb_lower
        if band_range > 0:
            pos = (close - bb_lower) / band_range * 100
            signals["布林"] = f"⚪ 通道中部 ({pos:.0f}%)"
        else:
            signals["布林"] = "⚪ -"

    # 均線訊號 (MA5 vs MA20)
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    if ma5 > ma20:
        signals["均線"] = "🟢 短線在長線上 (多頭)"
    else:
        signals["均線"] = "🔴 短線在長線下 (空頭)"

    return signals
