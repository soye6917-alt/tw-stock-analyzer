"""
💰 主力籌碼、OBV 與量價背離分析模組
- OBV (On-Balance Volume) 趨勢判斷
- 量價背離評分系統（多週期）
- 主力買賣超追蹤（大單過濾邏輯）
- 融資融券變化分析
- 籌碼面綜合評分
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# OBV (On-Balance Volume) 分析
# ─────────────────────────────────────────────
def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算 OBV 指標
    OBV = Σ volume * sign(close - prev_close)
    """
    result = df.copy()
    close = result['Close'].values
    volume = result['Volume'].values

    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    result['OBV'] = obv
    # OBV 均線
    result['OBV_MA5'] = pd.Series(obv).rolling(5).mean().values
    result['OBV_MA20'] = pd.Series(obv).rolling(20).mean().values
    return result


def analyze_obv_trend(df: pd.DataFrame) -> dict:
    """
    分析 OBV 趨勢與價格趨勢的關係

    回傳:
        obv_trend: 上升/下降/盤整
        divergence: 是否背離及類型
        signal: 買/賣/中立訊號
    """
    if df.empty or len(df) < 30 or 'OBV' not in df.columns:
        return {"error": "資料不足或 OBV 未計算"}

    close = df['Close'].values
    obv = df['OBV'].values

    # 多週期趨勢
    def trend_score(series: np.ndarray, period: int) -> float:
        """計算趨勢分數 (-5 ~ +5)"""
        if len(series) < period:
            return 0.0
        recent = series[-period:]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        return float(np.tanh(slope / (np.std(recent) + 1e-8)) * 5)

    close_trend_5 = trend_score(close, 5)
    close_trend_20 = trend_score(close, 20)
    obv_trend_5 = trend_score(obv, 5)
    obv_trend_20 = trend_score(obv, 20)

    # OBV 多空判斷
    latest_obv = obv[-1]
    obv_ma20 = df['OBV_MA20'].iloc[-1] if 'OBV_MA20' in df.columns else np.mean(obv[-20:])
    obv_above_ma = latest_obv > obv_ma20

    # ── 背離偵測 ──
    divergences = []

    # 1. 頂背離：價格創高 but OBV 沒跟上
    if len(close) >= 20:
        price_peak = np.max(close[-20:])
        price_peak_idx = np.argmax(close[-20:])
        obv_at_peak = obv[-20:][price_peak_idx]
        obv_recent_avg = np.mean(obv[-5:])
        if close[-1] >= price_peak * 0.995 and obv_recent_avg < obv_at_peak * 0.98:
            divergences.append({
                "type": "🔴 頂背離 (Bearish Divergence)",
                "detail": "價格創高但OBV未同步跟進，動能減弱，可能反轉向下",
                "strength": "強" if (obv_at_peak - obv_recent_avg) / (obv_at_peak + 1e-8) > 0.05 else "中",
            })

    # 2. 底背離：價格創新低 but OBV 已回升
    if len(close) >= 20:
        price_trough = np.min(close[-20:])
        price_trough_idx = np.argmin(close[-20:])
        obv_at_trough = obv[-20:][price_trough_idx]
        obv_recent_avg = np.mean(obv[-5:])
        if close[-1] <= price_trough * 1.005 and obv_recent_avg > obv_at_trough * 1.02:
            divergences.append({
                "type": "🟢 底背離 (Bullish Divergence)",
                "detail": "價格創新低但OBV率先回升，賣壓減輕，可能反彈向上",
                "strength": "強" if (obv_recent_avg - obv_at_trough) / (obv_at_trough + 1e-8) > 0.05 else "中",
            })

    # 綜合 OBV 趨勢判定
    obv_trend_label = "上升 ↗" if obv_trend_5 > 1.0 else \
                      "下降 ↘" if obv_trend_5 < -1.0 else "盤整 →"

    # 訊號
    signals = []
    if divergences:
        for d in divergences:
            signals.append(d["type"])
    elif obv_trend_5 > 2 and close_trend_5 > 2:
        signals.append("🟢 量價同步上漲 (健康多頭)")
    elif obv_trend_5 < -2 and close_trend_5 < -2:
        signals.append("🔴 量價同步下跌 (空頭延續)")
    elif obv_trend_5 > 0 and close_trend_5 < 0:
        signals.append("🟢 量增價跌 (可能築底)")
        divergences.append({
            "type": "🟢 潛在底背離",
            "detail": "OBV 止穩但價格還在跌，注意是否接近底部",
            "strength": "觀察中",
        })
    elif obv_trend_5 < 0 and close_trend_5 > 0:
        signals.append("🔴 價漲量縮 (動能不足)")
        divergences.append({
            "type": "🔴 潛在頂背離",
            "detail": "價格漲但 OBV 背離向下，上漲動能減弱",
            "strength": "觀察中",
        })
    else:
        signals.append("⚪ OBV 中性")

    return {
        "obv_current": round(float(latest_obv), 0),
        "obv_ma20": round(float(obv_ma20), 0),
        "obv_above_ma": obv_above_ma,
        "obv_trend": obv_trend_label,
        "obv_trend_score_5d": round(float(obv_trend_5), 2),
        "close_trend_score_5d": round(float(close_trend_5), 2),
        "divergences": divergences,
        "signals": signals,
        "score": round(float(obv_trend_5 * 2), 1),  # -10 ~ +10
    }


# ─────────────────────────────────────────────
# 量價背離評分系統
# ─────────────────────────────────────────────
def volume_price_divergence_score(df: pd.DataFrame) -> dict:
    """
    量價背離綜合評分（多週期）
    回傳 0~100 分，越高表示量價關係越健康

    評分維度:
    - 短線量價配合 (5日)      30%
    - 中線量價配合 (20日)     30%
    - OBV 趨勢一致性           20%
    - 突破時的量能確認         20%
    """
    if df.empty or len(df) < 30 or 'Volume' not in df.columns:
        return {"error": "資料不足"}

    close = df['Close'].values
    volume = df['Volume'].values

    if 'OBV' not in df.columns:
        df = calculate_obv(df)
        obv = df['OBV'].values
    else:
        obv = df['OBV'].values

    score = 0.0
    details = []

    # ── 1. 短線量價配合 (30分) ──
    short_score = 15  # 起始
    for i in range(-5, 0):
        if abs(i) >= len(close):
            continue
        price_chg = (close[i] - close[i-1]) / close[i-1] * 100
        vol_chg = (volume[i] / np.mean(volume[-10:]) - 1) * 100

        if price_chg > 0.5 and vol_chg > 20:
            short_score += 3  # 量價齊揚
        elif price_chg > 0.5 and vol_chg < -20:
            short_score -= 2  # 價漲量縮
        elif price_chg < -0.5 and vol_chg > 20:
            short_score -= 3  # 價跌量增
        elif price_chg < -0.5 and vol_chg < -20:
            short_score += 1  # 價跌量縮（賣壓減輕）

    short_score = max(0, min(30, short_score))
    details.append({"dimension": "短線量價 (5日)", "score": short_score, "max": 30})
    score += short_score

    # ── 2. 中線量價配合 (30分) ──
    mid_score = 15
    if len(close) >= 20:
        # 計算近 20 日價格趨勢和量能趨勢的相關性
        price_slope = np.polyfit(np.arange(20), close[-20:], 1)[0]
        vol_slope = np.polyfit(np.arange(20), volume[-20:], 1)[0]

        if price_slope > 0 and vol_slope > 0:
            mid_score += 10  # 中線量價同向
        elif price_slope > 0 and vol_slope < 0:
            mid_score -= 5   # 中線價漲量縮
        elif price_slope < 0 and vol_slope > 0:
            mid_score -= 10  # 中線價跌量增
        elif price_slope < 0 and vol_slope < 0:
            mid_score += 5   # 中線量價同步向下後可能反轉

        # 量能穩定性加分
        vol_cv = np.std(volume[-20:]) / (np.mean(volume[-20:]) + 1)
        if vol_cv < 0.5:
            mid_score += 5  # 量能穩定

    mid_score = max(0, min(30, mid_score))
    details.append({"dimension": "中線量價 (20日)", "score": mid_score, "max": 30})
    score += mid_score

    # ── 3. OBV 趨勢一致性 (20分) ──
    obv_score = 10
    obv_trend = np.polyfit(np.arange(10), obv[-10:], 1)[0]
    price_trend_10 = np.polyfit(np.arange(10), close[-10:], 1)[0]

    if obv_trend > 0 and price_trend_10 > 0:
        obv_score += 8
    elif obv_trend > 0 > price_trend_10:
        obv_score += 5  # OBV 領先價格
    elif obv_trend < 0 < price_trend_10:
        obv_score -= 5  # 頂背離
    elif obv_trend < 0 and price_trend_10 < 0:
        obv_score += 2

    obv_score = max(0, min(20, obv_score))
    details.append({"dimension": "OBV 趨勢", "score": obv_score, "max": 20})
    score += obv_score

    # ── 4. 突破確認 (20分) ──
    break_score = 10
    if len(close) >= 20:
        high_20 = np.max(close[-20:])
        low_20 = np.min(close[-20:])

        # 突破近期高點且量能配合
        if close[-1] >= high_20 * 0.99:
            vol_ratio = volume[-1] / (np.mean(volume[-20:-1]) + 1)
            if vol_ratio > 1.5:
                break_score += 8  # 突破帶量
            else:
                break_score -= 2  # 突破量不足
        elif close[-1] <= low_20 * 1.01:
            vol_ratio = volume[-1] / (np.mean(volume[-20:-1]) + 1)
            if vol_ratio > 1.5:
                break_score -= 5  # 跌破帶量（危險）
            else:
                break_score += 2  # 跌破量縮（可能假跌破）

    break_score = max(0, min(20, break_score))
    details.append({"dimension": "突破確認", "score": break_score, "max": 20})
    score += break_score

    # 等級判定
    if score >= 80:
        level = "🟢 量價完美 (健康)"
    elif score >= 65:
        level = "🟢 量價良好"
    elif score >= 50:
        level = "🟡 量價普通"
    elif score >= 35:
        level = "🟠 量價偏弱 (有背離風險)"
    else:
        level = "🔴 量價惡化 (危險信號)"

    return {
        "total_score": round(score, 1),
        "level": level,
        "details": details,
        "max_score": 100,
    }


# ─────────────────────────────────────────────
# 籌碼面綜合分析
# ─────────────────────────────────────────────
def institutional_flow_analysis(stock_id: str, days: int = 20) -> dict:
    """
    法人籌碼流向分析
    用三大法人買賣超數據分析資金動向

    回傳:
        net_buy_sell: 淨買超/賣超
        trend: 近期法人動向
        concentration: 籌碼集中度
        score: 籌碼面評分 0~100
    """
    try:
        from fundamentals import fetch_institutional_trading
        inst_df = fetch_institutional_trading(stock_id, days=days)
    except ImportError:
        return {"error": "fundamentals 模組無法導入"}

    if inst_df.empty:
        return {"error": "無法人買賣超資料"}

    # 解析各類法人（用位置索引避免編碼問題）
    categories = {}
    for idx, label in enumerate(["外資", "投信", "自營商", "三大法人合計"]):
        # 嘗試不同匹配方式
        row = None
        for col in inst_df.columns:
            if '類別' in col or '別' in col:
                mask = inst_df[col].astype(str).str.contains(label, na=False)
                row = inst_df[mask]
                if not row.empty:
                    break
        if row is None:
            row = inst_df[inst_df.iloc[:, 0].astype(str).str.contains(label, na=False)]
        
        if not row.empty:
            # 用位置而不是欄位名取資料
            cols = row.iloc[0]
            buy_col = None
            sell_col = None
            net_col = None
            for c in row.columns:
                if '買進' in c or '買進(張)' in c or '買進張數' in c:
                    buy_col = c
                if '賣出' in c or '賣出(張)' in c or '賣出張數' in c:
                    sell_col = c
                if '買賣超' in c or '買賣超(張)' in c:
                    net_col = c
            categories[label] = {
                "buy": int(row[buy_col].values[0]) if buy_col else 0,
                "sell": int(row[sell_col].values[0]) if sell_col else 0,
                "net": int(row[net_col].values[0]) if net_col else 0,
            }

    # 總得分
    score = 50  # 起始
    signals = []

    # 外資動向（權重最大）
    if "外資" in categories:
        f_net = categories["外資"]["net"]
        if f_net > 5000:
            score += 20
            signals.append(f"🏢 外資大買 {f_net//1000}K 張")
        elif f_net > 1000:
            score += 10
            signals.append(f"🏢 外資買超 {f_net//1000}K 張")
        elif f_net > 0:
            score += 5
        elif f_net < -5000:
            score -= 20
            signals.append(f"🏢 外資大賣 {abs(f_net)//1000}K 張")
        elif f_net < -1000:
            score -= 10
            signals.append(f"🏢 外資賣超 {abs(f_net)//1000}K 張")
        elif f_net < 0:
            score -= 5

    # 投信動向
    if "投信" in categories:
        t_net = categories["投信"]["net"]
        if t_net > 2000:
            score += 15
            signals.append(f"🏦 投信大買 {t_net//1000}K 張")
        elif t_net > 500:
            score += 8
        elif t_net > 0:
            score += 3
        elif t_net < -2000:
            score -= 15
            signals.append(f"🏦 投信大賣 {abs(t_net)//1000}K 張")
        elif t_net < -500:
            score -= 8
        elif t_net < 0:
            score -= 3

    # 自營商
    if "自營商" in categories:
        d_net = categories["自營商"]["net"]
        if d_net > 1000:
            score += 10
            signals.append(f"🏪 自營商大買 {d_net//1000}K 張")
        elif d_net < -1000:
            score -= 10
            signals.append(f"🏪 自營商大賣 {abs(d_net)//1000}K 張")
        elif d_net > 0:
            score += 3
        elif d_net < 0:
            score -= 3

    # 三大法人共識
    if "三大法人合計" in categories:
        total_net = categories["三大法人合計"]["net"]
        # 在已有個別評分的基礎上，再加強共識信號
        if total_net > 10000:
            signals.append(f"💰 三大法人合計大買 {total_net//1000}K 張 (強力資金流入)")
        elif total_net < -10000:
            signals.append(f"💰 三大法人合計大賣 {abs(total_net)//1000}K 張 (資金大幅流出)")

    # 正負反轉信號
    if len(signals) >= 2:
        pos = sum(1 for s in signals if "大買" in s or "買超" in s)
        neg = sum(1 for s in signals if "大賣" in s or "賣超" in s)
        if pos >= 2 and neg == 0:
            signals.append("✅ 法人全面買超 (強共識)")
        elif neg >= 2 and pos == 0:
            signals.append("❌ 法人全面賣超 (強共識)")
            score -= 5

    score = max(0, min(100, score))

    # 等級
    if score >= 75:
        level = "🟢 籌碼強勢 (法人偏多)"
    elif score >= 60:
        level = "🟢 籌碼偏多"
    elif score >= 40:
        level = "🟡 籌碼中性"
    elif score >= 25:
        level = "🟠 籌碼偏空"
    else:
        level = "🔴 籌碼弱勢 (法人偏空)"

    return {
        "score": score,
        "level": level,
        "categories": {k: v for k, v in categories.items()},
        "signals": signals,
        "total_net": categories.get("三大法人合計", {}).get("net", 0),
    }


# ─────────────────────────────────────────────
# 主入口：籌碼面綜合報告
# ─────────────────────────────────────────────
def run_chip_analysis(
    stock_id: str,
    df: pd.DataFrame,
    fetch_institutional: bool = True,
) -> dict:
    """
    籌碼面綜合分析主入口
    整合 OBV、量價背離、法人籌碼

    回傳 dict 供 app.py 展示
    """
    result = {
        "stock_id": stock_id,
        "obv_result": None,
        "divergence_result": None,
        "institutional_result": None,
        "overall_score": 0,
        "overall_level": "⚪ 待評估",
        "summary_lines": [],
        "error": None,
    }

    lines = []

    # 1. OBV 分析
    df_obv = calculate_obv(df)
    obv_result = analyze_obv_trend(df_obv)
    result["obv_result"] = obv_result

    obv_score = obv_result.get("score", 0) * 3  # -30 ~ +30
    if obv_result.get("error") is None:
        lines.append(f"📊 **OBV 趨勢: {obv_result.get('obv_trend', 'N/A')}**")
        lines.append(f"  OBV 分數: {obv_result.get('score', 0):+.1f}")
        for sig in obv_result.get("signals", []):
            lines.append(f"  {sig}")
        for div in obv_result.get("divergences", []):
            lines.append(f"  {div['type']} ({div.get('strength', '')})")

    # 2. 量價背離評分
    div_result = volume_price_divergence_score(df_obv)
    result["divergence_result"] = div_result
    div_score = div_result.get("total_score", 50) / 100 * 40 - 20  # -20 ~ +20

    if div_result.get("error") is None:
        lines.append(f"")
        lines.append(f"📈 **量價背離評分: {div_result.get('total_score', 50):.0f}/100**")
        lines.append(f"  等級: {div_result.get('level', 'N/A')}")
        for d in div_result.get("details", []):
            pct = d["score"] / d["max"] * 100
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            lines.append(f"  {d['dimension']}: {bar} ({d['score']}/{d['max']})")

    # 3. 法人籌碼分析
    if fetch_institutional:
        inst_result = institutional_flow_analysis(stock_id)
        result["institutional_result"] = inst_result
        inst_score = (inst_result.get("score", 50) - 50) * 0.6  # -30 ~ +30

        if inst_result.get("error") is None:
            lines.append(f"")
            lines.append(f"🏢 **法人籌碼: {inst_result.get('level', 'N/A')}**")
            lines.append(f"  分數: {inst_result.get('score', 50):.0f}/100")
            for sig in inst_result.get("signals", []):
                lines.append(f"  {sig}")
        else:
            inst_score = 0
    else:
        inst_score = 0

    # 綜合評分
    overall = obv_score + div_score + inst_score
    overall = max(-50, min(50, overall))
    # 映射到 0~100
    overall_100 = int((overall + 50) * 1)

    if overall_100 >= 75:
        overall_level = "🟢 籌碼面強勢"
    elif overall_100 >= 60:
        overall_level = "🟢 籌碼面偏多"
    elif overall_100 >= 40:
        overall_level = "🟡 籌碼面中性"
    elif overall_100 >= 25:
        overall_level = "🟠 籌碼面偏弱"
    else:
        overall_level = "🔴 籌碼面弱勢 (風險)"

    result["overall_score"] = overall_100
    result["overall_level"] = overall_level
    result["summary_lines"] = lines

    return result
