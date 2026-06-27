"""
🔄 馬可夫狀態轉換模型 (Markov-Switching) + Kalman Filter
- 將市場分為多頭/空頭/盤整三種隱藏狀態
- 訓練狀態轉換機率矩陣
- Kalman Filter 即時更新狀態判斷
- 比均線交叉更科學的趨勢辨識
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. 馬可夫狀態定義與轉換模型
# ─────────────────────────────────────────────
def markov_switching_model(
    prices: np.ndarray,
    lookback: int = 252,
    min_per_state: int = 20,
) -> Dict:
    """
    使用簡化版馬可夫狀態轉換模型辨識市場狀態
    不使用外部套件（從頭實作）

    狀態:
        0 = 🟢 多頭 (Bull)
        1 = ⚪ 盤整 (Neutral)
        2 = 🔴 空頭 (Bear)

    回傳: {states, transition_matrix, current_state, state_duration, ...}
    """
    if len(prices) < lookback:
        lookback = len(prices)
    data = prices[-lookback:]

    # ── 用收益率特徵分群 ──
    returns = np.diff(data) / data[:-1] * 100

    # 計算各時間點的統計特徵
    features = []
    for i in range(lookback - 1):
        chunk = data[max(0, i - 20):i + 1]
        rets = returns[max(0, i - 19):i + 1]
        n = len(rets)
        if n < 5:
            features.append([0, 0, 0, 0])
            continue
        mu = np.mean(rets)
        sigma = np.std(rets)
        slope = (data[i] - data[max(0, i - 20)]) / (data[max(0, i - 20)] + 1e-8) * 100
        
        # 離 MA20 的乖離
        ma20 = np.mean(chunk)
        ma_bias = (data[i] / ma20 - 1) * 100 if ma20 > 0 else 0
        
        features.append([mu, sigma, slope, ma_bias])

    features = np.array(features)

    # ── K-means (手寫) 分群為 3 個狀態 ──
    def _kmeans_3(X, max_iter=100):
        # 初始質心：用 33/66 百分位
        X_flat = X.flatten()
        p33 = np.percentile(X_flat, 33)
        p66 = np.percentile(X_flat, 66)
        # 用收益率和乖離率來初始
        means_sorted = np.mean(X, axis=0)
        centroids = np.vstack([
            means_sorted * 0.8,
            means_sorted,
            means_sorted * 1.2,
        ])
        for _ in range(max_iter):
            dists = np.zeros((len(X), 3))
            for k in range(3):
                dists[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.zeros_like(centroids)
            for k in range(3):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids
        return labels

    states = _kmeans_3(features)

    # 重新映射：用各狀態的平均收益排序
    state_means = {}
    for s in range(3):
        mask = states == s
        if mask.any():
            state_means[s] = np.mean(features[mask, 0])  # 平均收益
        else:
            state_means[s] = 0

    # 收益最高 = 多頭(0), 中等 = 盤整(1), 最低 = 空頭(2)
    sorted_states = sorted(state_means.keys(), key=lambda x: state_means[x], reverse=True)
    state_map = {sorted_states[0]: 0, sorted_states[1]: 1, sorted_states[2]: 2}
    states_mapped = np.array([state_map[s] for s in states])

    # ── 轉移機率矩陣 ──
    trans_matrix = np.zeros((3, 3))
    for i in range(len(states_mapped) - 1):
        trans_matrix[states_mapped[i], states_mapped[i + 1]] += 1

    for i in range(3):
        row_sum = trans_matrix[i].sum() or 1
        trans_matrix[i] = trans_matrix[i] / row_sum

    # ── 當前狀態與持續時間 ──
    current = states_mapped[-1]
    duration = 1
    for i in range(len(states_mapped) - 2, -1, -1):
        if states_mapped[i] == current:
            duration += 1
        else:
            break

    # ── 下一狀態預測 ──
    next_state_probs = trans_matrix[current]
    next_state = np.argmax(next_state_probs)

    # ── 每個狀態的統計 ──
    state_stats = {}
    for s in range(3):
        mask = states_mapped == s
        if mask.any():
            rets_in_state = returns[mask[1:] if len(returns) < len(mask) else mask]
            state_stats[s] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.mean() * 100), 1),
                "avg_return": round(float(np.mean(rets_in_state) if len(rets_in_state) > 0 else 0), 3),
            }

    state_labels = {0: "🟢 多頭", 1: "⚪ 盤整", 2: "🔴 空頭"}
    current_label = state_labels.get(current, "未知")

    # 穩定性：狀態是否快速切換
    transitions = sum(1 for i in range(1, len(states_mapped)) if states_mapped[i] != states_mapped[i-1])
    stability = max(0, 1 - transitions / max(len(states_mapped), 1)) * 100

    return {
        "states": states_mapped.tolist(),
        "current_state": int(current),
        "current_label": current_label,
        "state_duration": duration,
        "next_state": int(next_state),
        "next_label": state_labels.get(next_state, "未知"),
        "next_probs": {
            "🟢 多頭": round(float(next_state_probs[0]) * 100, 1),
            "⚪ 盤整": round(float(next_state_probs[1]) * 100, 1),
            "🔴 空頭": round(float(next_state_probs[2]) * 100, 1),
        },
        "transition_matrix": trans_matrix.tolist(),
        "state_stats": state_stats,
        "stability": round(stability, 1),
        "total_transitions": transitions,
    }


# ─────────────────────────────────────────────
# 2. Kalman Filter 即時趨勢追蹤
# ─────────────────────────────────────────────
def kalman_trend_filter(prices: np.ndarray) -> Dict:
    """
    使用 Kalman Filter 追蹤價格趨勢
    輸出: 平滑價格、趨勢方向、趨勢強度、即時轉折點
    """
    n = len(prices)
    if n < 10:
        return {"error": "資料不足"}

    # 初始化 Kalman Filter (1 維隨機漫步 + 觀測雜訊)
    # 狀態: [price, velocity]
    # 觀測: price
    x = np.array([prices[0], 0.0])  # [位置, 速度]
    P = np.eye(2) * 1000  # 初始不確定性

    # 模型參數
    dt = 1.0
    F = np.array([[1, dt], [0, 1]])  # 狀態轉移
    H = np.array([[1, 0]])           # 觀測矩陣
    Q = np.eye(2) * 0.01             # 過程雜訊
    R = np.array([[5.0]])            # 觀測雜訊（價格波動）

    smoothed = []
    velocities = []
    for z in prices:
        # 預測
        x = F @ x
        P = F @ P @ F.T + Q

        # 更新
        y = z - H @ x  # 殘差
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P

        smoothed.append(x[0])
        velocities.append(x[1])

    smoothed = np.array(smoothed)
    velocities = np.array(velocities)

    # 趨勢判定
    current_vel = velocities[-1]
    avg_vel_20 = np.mean(velocities[-20:]) if len(velocities) >= 20 else np.mean(velocities)

    # 趨勢強度 0~100
    vel_std = np.std(velocities) or 1
    strength = int(min(100, max(0, abs(current_vel) / vel_std * 25)))

    if current_vel > vel_std * 0.5:
        trend_dir = "📈 強勢上漲"
    elif current_vel > 0:
        trend_dir = "↗ 緩慢上漲"
    elif current_vel > -vel_std * 0.5:
        trend_dir = "↘ 緩慢下跌"
    else:
        trend_dir = "📉 強勢下跌"

    # 轉折偵測：速度由正轉負或由負轉正
    turns = []
    for i in range(1, len(velocities)):
        if velocities[i-1] * velocities[i] < 0:
            # 速度翻轉，檢查幅度
            if abs(velocities[i]) > vel_std * 0.3:
                direction = "🔄 轉多" if velocities[i] > 0 else "🔄 轉空"
                turns.append({
                    "index": i,
                    "price": float(prices[i]),
                    "direction": direction,
                })

    return {
        "smoothed": smoothed.tolist(),
        "velocities": velocities.tolist(),
        "current_velocity": round(float(current_vel), 4),
        "current_trend": trend_dir,
        "trend_strength": strength,
        "avg_velocity_20": round(float(avg_vel_20), 4),
        "turns": turns[-5:] if turns else [],
        "recent_turn": turns[-1] if turns else None,
        "noise_ratio": round(float(np.std(prices - smoothed) / np.std(prices)), 4),
    }


# ─────────────────────────────────────────────
# 3. Hessian 市場狀態矩陣
# ─────────────────────────────────────────────
def hessian_market_matrix(
    df: pd.DataFrame,
    markov_state: int = None,
    kalman: Dict = None,
) -> Dict:
    """
    根據多維度輸入判定市場所處的 8 種狀態之一

    維度:
    - 趨勢方向 (多/空)
    - 趨勢強度 (強/弱)
    - 波動性 (高/低)

    輸出: 8 種狀態 + 建議策略
    """
    if df.empty or len(df) < 30:
        return {"error": "資料不足"}

    close = df['Close'].values

    # ── 趨勢方向與強度 ──
    if markov_state is not None:
        trend_bull = markov_state == 0
        trend_bear = markov_state == 2
    else:
        # fallback: MA 判斷
        ma20 = np.mean(close[-20:])
        ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
        trend_bull = close[-1] > ma20 > ma60
        trend_bear = close[-1] < ma20 < ma60

    # 趨勢強度
    if len(close) >= 20:
        ret_20d = (close[-1] / close[-20] - 1) * 100
    else:
        ret_20d = 0
    ret_20d_abs = abs(ret_20d)
    trend_strong = ret_20d_abs > 5    # 20日漲跌超過5%
    trend_moderate = ret_20d_abs > 2  # 超過2%

    # ── 波動性 ──
    returns = np.diff(close) / close[:-1] * 100
    recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    hist_vol = np.std(returns) if len(returns) > 0 else 0
    vol_ratio = recent_vol / (hist_vol + 1e-8) if hist_vol > 0 else 1
    vol_high = vol_ratio > 1.3       # 近期波動高於歷史
    vol_low = vol_ratio < 0.7         # 近期波動低於歷史

    # ── 編碼狀態 ──
    # 趨勢: 多頭(1) / 空頭(0)
    # 強度: 強(1) / 弱(0) 
    # 波動: 高(1) / 低(0)
    trend_code = 1 if trend_bull else 0
    strength_code = 1 if trend_strong else 0
    vol_code = 1 if vol_high else 0

    state_matrix = {
        (1, 1, 1): {
            "name": "🔥 強多頭 + 高波動",
            "type": "主升段 (動能強)",
            "strategy": "順勢持有，設移動停利。波動高需嚴格停損",
            "risk": "追高風險，震盪可能加劇",
        },
        (1, 1, 0): {
            "name": "🟢 強多頭 + 低波動",
            "type": "穩定上漲 (整理後突破)",
            "strategy": "最理想持有區間，可加碼。波動低表示籌碼安定",
            "risk": "留意量能萎縮後的漲勢衰竭",
        },
        (1, 0, 1): {
            "name": "🧐 弱多頭 + 高波動",
            "type": "高檔震盪 (出貨嫌疑)",
            "strategy": "減碼觀望，高波動表示多空分歧",
            "risk": "高檔震盪可能是頭部型態",
        },
        (1, 0, 0): {
            "name": "🟡 弱多頭 + 低波動",
            "type": "盤堅格局 (溫和向上)",
            "strategy": "可持有但不加碼，等待方向確認",
            "risk": "可能轉盤整或轉空",
        },
        (0, 1, 1): {
            "name": "💥 強空頭 + 高波動",
            "type": "主跌段 (恐慌殺盤)",
            "strategy": "空手或放空，避免摸底，反彈即賣點",
            "risk": "殺盤力道可能持續，勿逆勢接刀",
        },
        (0, 1, 0): {
            "name": "🔴 強空頭 + 低波動",
            "type": "陰跌格局 (無量下跌)",
            "strategy": "持續觀望，低波動空頭最危險（緩跌）",
            "risk": "跌勢可能長時間延續",
        },
        (0, 0, 1): {
            "name": "🤔 弱空頭 + 高波動",
            "type": "築底過程 (多空拉鋸)",
            "strategy": "初步觀察，跌幅趨緩但波動大，等待底部成形",
            "risk": "可能只是下跌中繼",
        },
        (0, 0, 0): {
            "name": "💤 弱空頭 + 低波動",
            "type": "底部盤整 (量縮整理)",
            "strategy": "注意量能變化，量縮到極致後放量可視為買點",
            "risk": "盤整時間可能很長",
        },
    }

    current_state = state_matrix.get(
        (trend_code, strength_code, vol_code),
        {"name": "?? 未知", "type": "待判斷", "strategy": "謹慎觀望", "risk": "訊號矛盾"},
    )

    return {
        "trend": "🟢 多頭" if trend_bull else ("🔴 空頭" if trend_bear else "⚪ 中性"),
        "trend_strength": "💪 強" if trend_strong else ("💪 中" if trend_moderate else "☁️ 弱"),
        "volatility": "🌊 高波動" if vol_high else ("🍃 低波動" if vol_low else "⚪ 正常波動"),
        "20d_return": f"{ret_20d:+.2f}%",
        "state_name": current_state["name"],
        "state_type": current_state["type"],
        "strategy": current_state["strategy"],
        "risk": current_state["risk"],
        # 完整編碼
        "code": f"{'多' if trend_bull else '空'}_{'強' if trend_strong else ('中' if trend_moderate else '弱')}_{'高波動' if vol_high else ('低波動' if vol_low else '正常')}",
    }


# ─────────────────────────────────────────────
# 4. 主入口
# ─────────────────────────────────────────────
def run_market_cycle_analysis(df: pd.DataFrame) -> Dict:
    """
    市場循環分析主入口
    整合馬可夫、Kalman Filter、Hessian 矩陣

    回傳:
        markov, kalman, hessian, summary_lines
    """
    if df.empty or len(df) < 30:
        return {"error": "資料不足", "markov": None, "kalman": None, "hessian": None}

    result = {
        "markov": None,
        "kalman": None,
        "hessian": None,
        "summary_lines": [],
        "error": None,
    }
    lines = []
    prices = df['Close'].values

    # 1. 馬可夫狀態轉換
    markov = markov_switching_model(prices)
    result["markov"] = markov
    lines.append(f"**🔄 馬可夫狀態轉換分析**")
    lines.append(f"  目前狀態: {markov.get('current_label', 'N/A')} (持續 {markov.get('state_duration', 0)} 日)")
    lines.append(f"  預測下一狀態: {markov.get('next_label', 'N/A')}")
    lines.append(f"  轉移機率: 多頭 {markov.get('next_probs', {}).get('🟢 多頭', 0):.0f}% / 盤整 {markov.get('next_probs', {}).get('⚪ 盤整', 0):.0f}% / 空頭 {markov.get('next_probs', {}).get('🔴 空頭', 0):.0f}%")
    lines.append(f"  穩定性: {markov.get('stability', 0):.1f}%")
    for s, stats in markov.get("state_stats", {}).items():
        labels = {0: "多頭", 1: "盤整", 2: "空頭"}
        lines.append(f"  {labels.get(s, '?')}: {stats.get('pct', 0):.1f}% 時間 | 平均日收益 {stats.get('avg_return', 0):+.4f}%")

    # 2. Kalman Filter
    kalman = kalman_trend_filter(prices)
    result["kalman"] = kalman
    if kalman.get("error") is None:
        lines.append(f"")
        lines.append(f"**🎯 Kalman Filter 趨勢追蹤**")
        lines.append(f"  趨勢: {kalman.get('current_trend', 'N/A')}")
        lines.append(f"  強度: {kalman.get('trend_strength', 0)}/100")
        lines.append(f"  速度: {kalman.get('current_velocity', 0):+.6f}")
        if kalman.get("recent_turn"):
            turn = kalman["recent_turn"]
            lines.append(f"  近期轉折: {turn.get('direction', '?')} @ {turn.get('price', 0):.2f}")
    else:
        lines.append(f"   Kalman: {kalman['error']}")

    # 3. Hessian 市場狀態矩陣
    hessian = hessian_market_matrix(df, markov.get("current_state"), kalman)
    result["hessian"] = hessian
    if hessian.get("error") is None:
        lines.append(f"")
        lines.append(f"**🧩 Hessian 市場狀態矩陣**")
        lines.append(f"  狀態: {hessian.get('state_name', 'N/A')}")
        lines.append(f"  類型: {hessian.get('state_type', 'N/A')}")
        lines.append(f"  策略: {hessian.get('strategy', 'N/A')}")
        lines.append(f"  風險: {hessian.get('risk', 'N/A')}")
        lines.append(f"  趨勢: {hessian.get('trend', '?')} | 強度: {hessian.get('trend_strength', '?')} | 波動: {hessian.get('volatility', '?')}")

    result["summary_lines"] = lines
    return result
