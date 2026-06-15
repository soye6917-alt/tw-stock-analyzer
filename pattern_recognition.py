"""
📐 技術型態辨識
W底、M頭、頭肩頂/底、箱型突破、三角收斂
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


def find_local_extrema(df, order=5):
    """
    找出區域高點與低點
    order: 兩側各看幾個 K 棒來判斷
    """
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    local_max_idx = argrelextrema(high, np.greater, order=order)[0]
    local_min_idx = argrelextrema(low, np.less, order=order)[0]

    peaks = pd.DataFrame({
        "idx": local_max_idx,
        "price": high[local_max_idx],
        "type": "peak",
    })

    troughs = pd.DataFrame({
        "idx": local_min_idx,
        "price": low[local_min_idx],
        "type": "trough",
    })

    extremas = pd.concat([peaks, troughs], ignore_index=True).sort_values("idx")
    return extremas


def detect_w_bottom(df, lookback=60):
    """
    W 底（雙重底）：
    1. 兩個相近的低點，中間有一個反彈高點
    2. 價格突破頸線（中間高點）
    3. 兩個低點價差 < 5%
    """
    ext = find_local_extrema(df.tail(lookback), order=5)
    if len(ext) < 3:
        return False, {}

    troughs = ext[ext["type"] == "trough"].tail(6)
    if len(troughs) < 2:
        return False, {}

    # 找最後兩個低點
    t1 = troughs.iloc[-2]
    t2 = troughs.iloc[-1]

    # 兩個低點價差 < 5%
    price_diff = abs(t1["price"] - t2["price"]) / max(t1["price"], t2["price"])
    if price_diff > 0.05:
        return False, {}

    # 找中間的高點（頸線）
    peaks_between = ext[(ext["idx"] > t1["idx"]) & (ext["idx"] < t2["idx"]) & (ext["type"] == "peak")]
    if peaks_between.empty:
        return False, {}

    neckline = peaks_between["price"].max()
    current_price = df["Close"].iloc[-1]

    # 確認第二個腳之後有反彈
    price_after_t2 = df["Close"].iloc[int(t2["idx"]):].max()
    if price_after_t2 <= t2["price"] * 1.03:
        return False, {}

    is_breakout = current_price > neckline * 0.99

    return True, {
        "type": "W 底（雙重底）",
        "left_bottom": round(t1["price"], 2),
        "right_bottom": round(t2["price"], 2),
        "neckline": round(neckline, 2),
        "current_price": round(current_price, 2),
        "breakout": is_breakout,
        "target": round(neckline + (neckline - min(t1["price"], t2["price"])), 2),
        "confidence": "高" if is_breakout else "中",
    }


def detect_m_top(df, lookback=60):
    """
    M 頭（雙重頂）：
    1. 兩個相近的高點，中間有一個回檔低點
    2. 價格跌破頸線（中間低點）
    """
    ext = find_local_extrema(df.tail(lookback), order=5)
    if len(ext) < 3:
        return False, {}

    peaks = ext[ext["type"] == "peak"].tail(6)
    if len(peaks) < 2:
        return False, {}

    p1 = peaks.iloc[-2]
    p2 = peaks.iloc[-1]

    price_diff = abs(p1["price"] - p2["price"]) / max(p1["price"], p2["price"])
    if price_diff > 0.05:
        return False, {}

    troughs_between = ext[(ext["idx"] > p1["idx"]) & (ext["idx"] < p2["idx"]) & (ext["type"] == "trough")]
    if troughs_between.empty:
        return False, {}

    neckline = troughs_between["price"].min()
    current_price = df["Close"].iloc[-1]

    is_breakdown = current_price < neckline * 1.01

    return True, {
        "type": "M 頭（雙重頂）",
        "left_top": round(p1["price"], 2),
        "right_top": round(p2["price"], 2),
        "neckline": round(neckline, 2),
        "current_price": round(current_price, 2),
        "breakdown": is_breakdown,
        "target": round(neckline - (max(p1["price"], p2["price"]) - neckline), 2),
        "confidence": "高" if is_breakdown else "中",
    }


def detect_head_shoulders(df, lookback=120):
    """
    頭肩頂：
    左肩 → 頭（最高） → 右肩 → 跌破頸線
    """
    ext = find_local_extrema(df.tail(lookback), order=7)
    peaks = ext[ext["type"] == "peak"].tail(10)
    if len(peaks) < 3:
        return False, {}

    # 取最後三個高點
    p3 = peaks.iloc[-3:]  # left shoulder, head, right shoulder
    vals = p3["price"].values

    # 頭要比左右肩高
    if not (vals[1] > vals[0] and vals[1] > vals[2]):
        return False, {}

    # 左右肩高度相近
    shoulder_diff = abs(vals[0] - vals[2]) / max(vals[0], vals[2])
    if shoulder_diff > 0.08:
        return False, {}

    # 頸線：左右肩之間的低點
    troughs_between = ext[(ext["idx"] > p3.iloc[0]["idx"]) & (ext["idx"] < p3.iloc[2]["idx"]) & (ext["type"] == "trough")]
    if troughs_between.empty:
        return False, {}

    neckline = troughs_between["price"].min()
    current_price = df["Close"].iloc[-1]
    is_breakdown = current_price < neckline * 1.01

    return True, {
        "type": "頭肩頂",
        "left_shoulder": round(vals[0], 2),
        "head": round(vals[1], 2),
        "right_shoulder": round(vals[2], 2),
        "neckline": round(neckline, 2),
        "current_price": round(current_price, 2),
        "breakdown": is_breakdown,
        "target": round(neckline - (vals[1] - neckline), 2),
        "confidence": "高" if is_breakdown else "中",
    }


def detect_head_shoulders_bottom(df, lookback=120):
    """頭肩底（反轉型態）"""
    ext = find_local_extrema(df.tail(lookback), order=7)
    troughs = ext[ext["type"] == "trough"].tail(10)
    if len(troughs) < 3:
        return False, {}

    t3 = troughs.iloc[-3:]
    vals = t3["price"].values

    # 頭要比左右肩低
    if not (vals[1] < vals[0] and vals[1] < vals[2]):
        return False, {}

    shoulder_diff = abs(vals[0] - vals[2]) / max(vals[0], vals[2])
    if shoulder_diff > 0.08:
        return False, {}

    peaks_between = ext[(ext["idx"] > t3.iloc[0]["idx"]) & (ext["idx"] < t3.iloc[2]["idx"]) & (ext["type"] == "peak")]
    if peaks_between.empty:
        return False, {}

    neckline = peaks_between["price"].max()
    current_price = df["Close"].iloc[-1]
    is_breakout = current_price > neckline * 0.99

    return True, {
        "type": "頭肩底",
        "left_shoulder": round(vals[0], 2),
        "head": round(vals[1], 2),
        "right_shoulder": round(vals[2], 2),
        "neckline": round(neckline, 2),
        "current_price": round(current_price, 2),
        "breakout": is_breakout,
        "target": round(neckline + (neckline - vals[1]), 2),
        "confidence": "高" if is_breakout else "中",
    }


def detect_breakout(df, lookback=40):
    """
    箱型突破：
    價格在一個區間內整理，然後突破區間上緣
    """
    chunk = df.tail(lookback)
    high = chunk["High"].max()
    low = chunk["Low"].min()
    range_pct = (high - low) / low * 100

    # 箱型需要有一定範圍（5-25%）
    if range_pct < 5 or range_pct > 25:
        return False, {}

    # 最近 80% 的 K 棒都在這個區間內
    recent = chunk.tail(int(lookback * 0.8))
    inside_count = ((recent["High"] <= high * 1.01) & (recent["Low"] >= low * 0.99)).sum()
    inside_ratio = inside_count / len(recent)
    if inside_ratio < 0.7:
        return False, {}

    current_price = df["Close"].iloc[-1]
    is_breakout = current_price > high * 1.01
    is_breakdown = current_price < low * 0.99

    if not (is_breakout or is_breakdown):
        return False, {}

    return True, {
        "type": "箱型突破" if is_breakout else "箱型跌破",
        "range_high": round(high, 2),
        "range_low": round(low, 2),
        "range_pct": round(range_pct, 1),
        "current_price": round(current_price, 2),
        "direction": "突破↑" if is_breakout else "跌破↓",
        "target": round(high + (high - low), 2) if is_breakout else round(low - (high - low), 2),
        "confidence": "高",
    }


def detect_all_patterns(df, lookback=120):
    """執行所有型態辨識，回傳找到的型態列表"""
    patterns = []

    # W 底
    found, info = detect_w_bottom(df, lookback)
    if found:
        patterns.append(info)

    # M 頭
    found, info = detect_m_top(df, lookback)
    if found:
        patterns.append(info)

    # 頭肩頂
    found, info = detect_head_shoulders(df, lookback)
    if found:
        patterns.append(info)

    # 頭肩底
    found, info = detect_head_shoulders_bottom(df, lookback)
    if found:
        patterns.append(info)

    # 箱型突破
    found, info = detect_breakout(df, lookback)
    if found:
        patterns.append(info)

    return patterns
