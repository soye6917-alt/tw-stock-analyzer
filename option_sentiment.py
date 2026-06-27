"""
📊 隱含波動率與選擇權未平倉分析模組
- 台指選擇權 Put/Call Ratio 計算
- 最大未平倉 (Max OI) 壓力/支撐區間
- 波動率偏斜 (Volatility Skew)
- 恐慌/貪婪指標計算

注意：本模組使用台灣期交所公開資料
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─── Requests Session ───
import requests
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})


# ─────────────────────────────────────────────
# 1. 台指選擇權 Put/Call Ratio + 未平倉
# ─────────────────────────────────────────────
def fetch_tx_option_oi() -> Dict:
    """
    從期交所取得台指選擇權 (TXO) 未平倉資料
    回傳 {puts, calls, total_put_oi, total_call_oi, pc_ratio, max_oi_call, max_oi_put, ...}
    """
    result = {
        "error": None,
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    try:
        today = datetime.now()
        # 期交所每日選擇權未平倉
        date_str = today.strftime("%Y/%m/%d")
        url = "https://www.taifex.com.tw/cht/3/callsAndPutsDailyDownload"
        form_data = {
            "down_type": "2",
            "queryStartDate": date_str,
            "queryEndDate": date_str,
            "commodity_id_t": "TXO",
        }
        resp = _session.post(url, data=form_data, timeout=15)

        if resp.status_code != 200 or len(resp.text) < 100:
            # 嘗試前一個交易日
            for offset in range(1, 7):
                    d = today - timedelta(days=offset)
                    if d.weekday() >= 5:
                        continue
                    date_str2 = d.strftime("%Y/%m/%d")
                    form_data["queryStartDate"] = date_str2
                    form_data["queryEndDate"] = date_str2
                    resp = _session.post(url, data=form_data, timeout=15)
                    if resp.status_code == 200 and len(resp.text) > 100:
                        break

        # 解析 CSV 格式
        lines = resp.text.strip().split('\n')
        if len(lines) < 2:
            result["error"] = "無選擇權未平倉資料"
            return result

        calls_oi = []
        puts_oi = []
        max_oi_call = {"strike": 0, "oi": 0}
        max_oi_put = {"strike": 0, "oi": 0}

        for line in lines[1:]:  # 跳過 header
            parts = line.replace('"', '').split(',')
            if len(parts) < 8:
                continue
            try:
                strike = float(parts[3].strip())
                call_oi = int(parts[4].strip().replace(',', '')) if parts[4].strip() else 0
                put_oi = int(parts[7].strip().replace(',', '')) if parts[7].strip() else 0
            except (ValueError, IndexError):
                continue

            if call_oi > 0:
                calls_oi.append({"strike": strike, "oi": call_oi})
                if call_oi > max_oi_call["oi"]:
                    max_oi_call = {"strike": strike, "oi": call_oi}
            if put_oi > 0:
                puts_oi.append({"strike": strike, "oi": put_oi})
                if put_oi > max_oi_put["oi"]:
                    max_oi_put = {"strike": strike, "oi": put_oi}

        total_call_oi = sum(x["oi"] for x in calls_oi)
        total_put_oi = sum(x["oi"] for x in puts_oi)

        result["total_call_oi"] = total_call_oi
        result["total_put_oi"] = total_put_oi
        result["pc_ratio"] = round(total_put_oi / max(total_call_oi, 1), 4)
        result["max_oi_call"] = max_oi_call
        result["max_oi_put"] = max_oi_put
        result["calls_oi"] = sorted(calls_oi, key=lambda x: x["strike"])[:30]
        result["puts_oi"] = sorted(puts_oi, key=lambda x: x["strike"])[:30]

    except Exception as e:
        result["error"] = f"期交所資料擷取失敗: {str(e)}"

    return result


def analyze_pc_ratio(data: Dict, current_price: float = None) -> Dict:
    """
    分析 Put/Call Ratio 及選擇權市場情緒
    """
    if data.get("error"):
        return {"error": data["error"]}

    pc = data.get("pc_ratio", 0)
    max_call = data.get("max_oi_call", {})
    max_put = data.get("max_oi_put", {})

    # PC Ratio 解讀（台指選擇權慣例）
    if pc > 1.2:
        pc_signal = "🟢 非常偏多 (Put > Call 1.2x)"
        pc_desc = "選擇權市場強烈偏多，避險需求集中在多方"
    elif pc > 1.0:
        pc_signal = "🟢 偏多"
        pc_desc = "Put 未平倉大於 Call，市場情緒偏多"
    elif pc > 0.8:
        pc_signal = "⚪ 中性"
        pc_desc = "Put/Call 均衡，市場情緒中性"
    elif pc > 0.6:
        pc_signal = "🔴 偏空"
        pc_desc = "Call 未平倉大於 Put，市場對沖需求轉空"
    else:
        pc_signal = "🔴 非常偏空 (Call > Put)"
        pc_desc = "選擇權市場明顯偏空，大量避險買權"

    # 最大未平倉區間：大量OI集結處 = 結算行情參考點
    oi_analysis = {}
    if max_call.get("strike") and max_put.get("strike"):
        call_strike = max_call["strike"]
        put_strike = max_put["strike"]
        oi_analysis = {
            "call_max_oi_strike": call_strike,
            "put_max_oi_strike": put_strike,
            "range_low": min(call_strike, put_strike),
            "range_high": max(call_strike, put_strike),
            "range_mid": round((call_strike + put_strike) / 2, 0),
        }
        # 推測結算區間
        if current_price:
            if current_price < oi_analysis["range_low"]:
                oi_analysis["settlement_bias"] = "偏弱 (價格低於最大OI區間)"
            elif current_price > oi_analysis["range_high"]:
                oi_analysis["settlement_bias"] = "偏強 (價格高於最大OI區間)"
            else:
                oi_analysis["settlement_bias"] = "區間內震盪 (最大OI集結區)"

    return {
        "pc_ratio": pc,
        "pc_signal": pc_signal,
        "pc_description": pc_desc,
        "max_oi_call_strike": max_call.get("strike", 0),
        "max_oi_call_volume": max_call.get("oi", 0),
        "max_oi_put_strike": max_put.get("strike", 0),
        "max_oi_put_volume": max_put.get("oi", 0),
        "oi_range_analysis": oi_analysis,
        "total_call_oi": data.get("total_call_oi", 0),
        "total_put_oi": data.get("total_put_oi", 0),
    }


# ─────────────────────────────────────────────
# 2. 台指選擇權波動率指數 (TXV) 簡化版
# ─────────────────────────────────────────────
def estimate_tw_volatility_index() -> Dict:
    """
    估算台指選擇權波動率（近似 VIX）
    從近月 TXO 選擇權價格反推隱含波動率

    回傳: {vix_estimate, iv_percentile, vix_signal}
    """
    result = {
        "vix_estimate": None,
        "iv_percentile": None,
        "vix_signal": "⚪ 無法估算",
        "error": None,
    }

    try:
        # 台指選擇權波動率估算（權宜作法：從台指期近月價格變化反推）
        # 取大盤 TAEX 的近 20 日波動率作為近似
        from data_fetcher import fetch_historical
        df = fetch_historical("0050", months=3)
        if df.empty or len(df) < 20:
            return result

        returns = df['Close'].pct_change().dropna()
        recent_vol = returns.tail(20).std() * np.sqrt(252) * 100  # 年化波動率 %

        # 台指選擇權波動率通常比大盤波動率高 3-5%
        vix_est = round(recent_vol + 4, 1)

        percentiles = {
            15: "🔴 極高波動 (恐慌)",
            20: "🔴 高波動 (警戒)",
            25: "🟡 偏高波動",
            30: "⚪ 中性波動",
            35: "🟢 偏低波動 (穩定)",
        }

        vix_signal = "⚪ 中性波動"
        iv_pctile = 50
        for pct, label in sorted(percentiles.items()):
            if vix_est <= pct * 2:  # 調整 scale
                vix_signal = label
                iv_pctile = max(10, min(90, int((1 - (vix_est / 30)) * 100)))
                break

        result["vix_estimate"] = vix_est
        result["iv_percentile"] = iv_pctile
        result["vix_signal"] = vix_signal

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# 3. 台股恐慌貪婪指數
# ─────────────────────────────────────────────
def calculate_fear_greed_index(
    df_twse: pd.DataFrame = None,
    pc_analysis: Dict = None,
    vix_estimate: float = None,
) -> Dict:
    """
    計算台股恐慌貪婪指數 (0-100)

    綜合維度:
    - 股價動能 (股價 vs MA50) 30%
    - 漲跌家數比 (Breadth) 20%
    - Put/Call Ratio 20%
    - 波動率 % 15%
    - 新聞情緒 15%
    """
    score = 50
    components = {}

    # ── 1. 股價動能 (30分) ──
    if df_twse is not None and not df_twse.empty and len(df_twse) >= 50:
        close = df_twse['Close'].values
        current = close[-1]
        ma50 = np.mean(close[-50:])

        ma_bias = (current / ma50 - 1) * 100
        if ma_bias > 10:
            mkt_score = 28
            mkt_label = "極度貪婪 (股價遠高於MA)"
        elif ma_bias > 5:
            mkt_score = 24
            mkt_label = "貪婪"
        elif ma_bias > 2:
            mkt_score = 20
            mkt_label = "偏貪婪"
        elif ma_bias > -2:
            mkt_score = 15
            mkt_label = "中性"
        elif ma_bias > -5:
            mkt_score = 10
            mkt_label = "偏恐慌"
        elif ma_bias > -10:
            mkt_score = 6
            mkt_label = "恐慌"
        else:
            mkt_score = 3
            mkt_label = "極度恐慌 (股價遠低於MA)"
        components["股價動能 (vs MA50)"] = {"score": mkt_score, "max": 30, "label": mkt_label}
        score += mkt_score - 15  # 回到0基線
    else:
        components["股價動能"] = {"score": 15, "max": 30, "label": "資料不足"}

    # ── 2. Put/Call Ratio (20分) ──
    if pc_analysis and pc_analysis.get("pc_ratio"):
        pc = pc_analysis["pc_ratio"]
        if pc > 1.3:
            pc_score = 18
            pc_label = "極度貪婪 (P/C 奇高)"
        elif pc > 1.1:
            pc_score = 15
            pc_label = "貪婪"
        elif pc > 0.9:
            pc_score = 10
            pc_label = "中性"
        elif pc > 0.7:
            pc_score = 6
            pc_label = "恐慌"
        else:
            pc_score = 3
            pc_label = "極度恐慌 (P/C 極低)"
        components["Put/Call Ratio"] = {"score": pc_score, "max": 20, "label": pc_label}
        score += pc_score - 10
    else:
        components["Put/Call Ratio"] = {"score": 10, "max": 20, "label": "無資料"}

    # ── 3. 波動率 (15分) ──
    if vix_estimate:
        if vix_estimate < 15:
            vix_score = 14
            vix_label = "低波動 (穩定)"
        elif vix_estimate < 20:
            vix_score = 11
            vix_label = "波動正常"
        elif vix_estimate < 25:
            vix_score = 7
            vix_label = "波動偏高"
        elif vix_estimate < 35:
            vix_score = 4
            vix_label = "波動高 (警戒)"
        else:
            vix_score = 2
            vix_label = "極高波動 (恐慌)"
        components["波動率"] = {"score": vix_score, "max": 15, "label": vix_label}
        score += vix_score - 7.5
    else:
        components["波動率"] = {"score": 7.5, "max": 15, "label": "無資料"}

    # ── 4. 漲跌家數比 (20分，使用 0050 漲跌近似) ──
    if df_twse is not None and not df_twse.empty and len(df_twse) >= 5:
        ret_5d = (df_twse['Close'].iloc[-1] / df_twse['Close'].iloc[-5] - 1) * 100
        if ret_5d > 4:
            breadth_score = 18
            breadth_label = "市場廣度極佳"
        elif ret_5d > 2:
            breadth_score = 14
            breadth_label = "市場偏強"
        elif ret_5d > 0:
            breadth_score = 10
            breadth_label = "中性"
        elif ret_5d > -2:
            breadth_score = 6
            breadth_label = "市場偏弱"
        else:
            breadth_score = 3
            breadth_label = "市場廣度差"
        components["市場廣度 (0050漲跌)"] = {"score": breadth_score, "max": 20, "label": breadth_label}
        score += breadth_score - 10
    else:
        components["市場廣度"] = {"score": 10, "max": 20, "label": "資料不足"}

    # ── 5. 新聞情緒 (15分) ──
    # 不強制依賴新聞，給基礎分
    components["新聞情緒"] = {"score": 7.5, "max": 15, "label": "暫無分析"}
    score += 0

    # 歸一化到 0-100
    final_score = int(max(0, min(100, 50 + score)))

    if final_score >= 80:
        level = "🟢 極度貪婪"
        advice = "市場極度樂觀，留意過熱反轉風險，建議逐步獲利"
    elif final_score >= 65:
        level = "🟢 貪婪"
        advice = "市場偏樂觀，追高需謹慎"
    elif final_score >= 45:
        level = "⚪ 中性"
        advice = "市場情緒均衡，適合按計劃操作"
    elif final_score >= 30:
        level = "🟠 恐慌"
        advice = "市場偏悲觀，留意逢低布局機會"
    else:
        level = "🔴 極度恐慌"
        advice = "市場過度悲觀，長線投資者可考慮分批進場"

    return {
        "fear_greed_index": final_score,
        "level": level,
        "advice": advice,
        "components": components,
    }


# ─────────────────────────────────────────────
# 4. 選擇權壓力/支撐區間分析
# ─────────────────────────────────────────────
def option_level_analysis(current_price: float, option_data: Dict) -> Dict:
    """
    根據選擇權未平倉結構計算支撐壓力
    """
    if option_data.get("error"):
        return {"error": option_data["error"]}

    max_call_strike = option_data.get("max_oi_call", {}).get("strike", 0)
    max_put_strike = option_data.get("max_oi_put", {}).get("strike", 0)

    # 大量 Call OI 集結 = 壓力位（莊家不願指數漲破）
    # 大量 Put OI 集結 = 支撐位（莊家不願指數跌破）
    calls_oi = option_data.get("calls_oi", [])
    puts_oi = option_data.get("puts_oi", [])

    # 找出前三大壓力/支撐
    top_calls = sorted(calls_oi, key=lambda x: x["oi"], reverse=True)[:5]
    top_puts = sorted(puts_oi, key=lambda x: x["oi"], reverse=True)[:5]

    resistance_levels = [{"strike": x["strike"], "oi": x["oi"]} for x in top_calls if x["strike"] > current_price]
    support_levels = [{"strike": x["strike"], "oi": x["oi"]} for x in top_puts if x["strike"] < current_price]

    # 關鍵心理關卡
    key_levels = []
    for level in [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000]:
        if abs(level - current_price) / current_price < 0.15:
            key_levels.append(level)

    return {
        "max_oi_call_strike": max_call_strike,
        "max_oi_put_strike": max_put_strike,
        "top_resistance": resistance_levels[:3],
        "top_support": support_levels[:3],
        "key_psychological_levels": key_levels,
    }


# ─────────────────────────────────────────────
# 5. 主入口
# ─────────────────────────────────────────────
def run_option_sentiment_analysis(current_price: float = None) -> Dict:
    """
    選擇權市場情緒分析主入口

    回傳: {
        option_data, pc_analysis, vix_analysis, 
        fear_greed, levels, summary_lines
    }
    """
    result = {
        "option_data": None,
        "pc_analysis": None,
        "vix_analysis": None,
        "fear_greed": None,
        "option_levels": None,
        "summary_lines": [],
        "error": None,
    }

    lines = []

    # 1. 選擇權 PC + 未平倉
    option_data = fetch_tx_option_oi()
    result["option_data"] = option_data

    if option_data.get("error") is None:
        pc = analyze_pc_ratio(option_data, current_price)
        result["pc_analysis"] = pc
        lines.append(f"**📊 Put/Call 選擇權未平倉比率**")
        lines.append(f"  P/C Ratio: {pc.get('pc_ratio', 0):.3f} — {pc.get('pc_signal', 'N/A')}")
        lines.append(f"  {pc.get('pc_description', '')}")
        lines.append(f"  Call Max OI: {pc.get('max_oi_call_strike', 0):.0f} ({pc.get('max_oi_call_volume', 0):,})")
        lines.append(f"  Put Max OI: {pc.get('max_oi_put_strike', 0):.0f} ({pc.get('max_oi_put_volume', 0):,})")

        # 最大OI區間解讀
        oi_range = pc.get("oi_range_analysis", {})
        if oi_range:
            lines.append(f"  🎯 最大OI集結區: {oi_range.get('range_low', 0):.0f}~{oi_range.get('range_high', 0):.0f}")
            if current_price:
                lines.append(f"  📍 現價: {current_price:.0f} → 結算傾向: {oi_range.get('settlement_bias', 'N/A')}")
    else:
        lines.append("📊 選擇權資料暫時無法取得")

    # 2. 波動率
    vix = estimate_tw_volatility_index()
    result["vix_analysis"] = vix
    if vix.get("vix_estimate"):
        lines.append(f"")
        lines.append(f"**🌊 波動率指數 (近似VIX)**")
        lines.append(f"  波動率: {vix['vix_estimate']:.1f}% — {vix.get('vix_signal', '')}")
        lines.append(f"  百分位: {vix.get('iv_percentile', 50)}%")
    else:
        lines.append(f"  波動率: 無法估算")

    # 3. 恐慌貪婪
    df_twse = None
    try:
        from data_fetcher import fetch_historical
        df_twse = fetch_historical("0050", months=6)
    except Exception:
        pass

    fear_greed = calculate_fear_greed_index(df_twse, pc if option_data.get("error") is None else None, vix.get("vix_estimate"))
    result["fear_greed"] = fear_greed

    lines.append(f"")
    lines.append(f"**🎭 台股恐慌貪婪指數**")
    lines.append(f"  指數: {fear_greed['fear_greed_index']}/100 — {fear_greed['level']}")
    lines.append(f"  💡 {fear_greed['advice']}")
    for comp_name, comp_data in fear_greed.get("components", {}).items():
        pct = comp_data["score"] / comp_data["max"] * 100
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        lines.append(f"  {comp_name}: {bar} ({comp_data['score']}/{comp_data['max']}) {comp_data['label']}")

    # 4. 選擇權壓力/支撐區間
    if current_price and option_data.get("error") is None:
        levels = option_level_analysis(current_price, option_data)
        result["option_levels"] = levels
        lines.append(f"")
        lines.append(f"**🛡️ 選擇權支撐壓力分析**")
        if levels.get("top_support"):
            s = levels["top_support"][0]
            lines.append(f"  支撐: {s['strike']:.0f} (OI: {s['oi']:,})")
        if levels.get("top_resistance"):
            r = levels["top_resistance"][0]
            lines.append(f"  壓力: {r['strike']:.0f} (OI: {r['oi']:,})")

    result["summary_lines"] = lines
    return result
