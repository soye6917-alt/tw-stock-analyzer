"""
🏆 專家級每日5大推薦股票模組
多因子評分系統:技術35% + 新聞20% + 基本面15% + 籌碼15% + 動能10% + 風險5%
"""

import pandas as pd
import numpy as np
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from data_fetcher import (
    fetch_historical, fetch_stock_list, POPULAR_STOCKS,
    get_stock_name, SESSION
)
from indicators import add_all_indicators
from fundamentals import fetch_fundamentals, fetch_institutional_trading
import pattern_recognition as pr


# ─────────────────────────────────────────────
# 掃描標的清單(50 檔市場重點股)
# ─────────────────────────────────────────────
SCAN_UNIVERSE = {
    # 半導體
    "2330": "台積電", "2454": "聯發科", "2303": "聯電",
    "3034": "聯詠", "3711": "日月光投控", "3443": "創意",
    "3661": "世芯-KY", "5269": "祥碩", "6643": "M31",
    # 電子代工/系統
    "2317": "鴻海", "2382": "廣達", "2356": "英業達",
    "2357": "華碩", "2376": "技嘉", "2377": "微星",
    # 面板/光電/PCB
    "2409": "友達", "3481": "群創", "3037": "欣興", "8046": "南電",
    # 電子零組件
    "2308": "台達電", "2327": "國巨", "2456": "奇力新",
    # 網通/電信
    "4904": "遠傳", "3045": "台灣大", "2412": "中華電",
    # 金融
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
    "2886": "兆豐金", "2892": "第一金", "5880": "合庫金",
    "2884": "玉山金", "2885": "元大金",
    # 傳產龍頭
    "2002": "中鋼", "1301": "台塑", "1303": "南亞",
    "1326": "台化", "1216": "統一", "2912": "統一超",
    # 航運
    "2603": "長榮", "2609": "陽明", "2618": "長榮航",
    # 其他
    "3008": "大立光", "6515": "穎崴", "6531": "愛普",
    # ETF
    "0050": "元大台灣50", "0056": "元大高股息",
}


# ─────────────────────────────────────────────
# 新聞情緒分析
# ─────────────────────────────────────────────

def fetch_news_sentiment(stock_id: str, stock_name: str) -> tuple:
    """
    從 Yahoo 奇摩股市取得最新新聞,進行關鍵字情緒分析
    回傳 (score, headlines, summary)
    score: -10 ~ +10
    """
    score = 0.0
    headlines = []

    try:
        # Yahoo 奇摩股市新聞頁面
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = SESSION.get(url, timeout=10, headers=headers)

        if resp.status_code != 200:
            return 0, [], "無法取得新聞"

        text = resp.text

        # 提取新聞標題(Yahoo 新聞的常見結構)
        # 先抓所有 <h3> 和 <a> 裡的中文標題
        found_titles = []

        # 方法1: 找 h3 標籤內的文字
        h3_pattern = r'<h3[^>]*>(.*?)</h3>'
        h3_matches = re.findall(h3_pattern, text, re.DOTALL)
        for m in h3_matches:
            # 移除 HTML 標籤
            clean = re.sub(r'<[^>]+>', '', m).strip()
            # 只保留有中文字且長度大於5的
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean)
            if len(clean) > 10 and len(chinese_chars) >= 4:
                found_titles.append(clean)

        # 方法2: 找 title 或 article 標籤
        if len(found_titles) < 3:
            a_pattern = r'<a[^>]*title=["\'](.*?)["\']'
            a_matches = re.findall(a_pattern, text)
            for m in a_matches:
                chinese_chars = re.findall(r'[\u4e00-\u9fff]', m)
                if len(m) > 10 and len(chinese_chars) >= 4 and m not in found_titles:
                    found_titles.append(m)

        # 去重,取前10條
        headlines = list(dict.fromkeys(found_titles))[:10]

        if not headlines:
            return 0, [], "無近期新聞"

        # ── 新聞情緒分析 ──
        # 正向關鍵字
        positive_keywords = [
            "創高", "突破", "利多", "買進", "喊買", "調升", "成長", "擴產", "受惠",
            "大漲", "飆漲", "漲停", "上漲", "獲利", "創新高", "營收", "年增",
            "合作", "訂單", "大單", "轉盈", "配息", "股利", "加碼", "併購",
            "整併", "轉機", "翻身", "回溫", "復甦", "增資", "募資", "補漲",
            "反彈", "走強", "法說", "亮眼", "優於預期", "目標價", "唱多",
            "超車", "領先", "獨家", "先進", "布局AI", "AI", "HPC",
        ]
        # 負向關鍵字
        negative_keywords = [
            "大跌", "暴跌", "跌停", "重挫", "下跌", "賣壓", "利空", "調降",
            "降評", "賣出", "減碼", "衰退", "下滑", "年減", "月減", "虧損",
            "裁員", "關廠", "停工", "違約", "訴訟", "被罰", "罰款", "調查",
            "掏空", "作假", "變臉", "破底", "破線", "壓力", "爆量",
            "匯損", "庫存", "降溫", "終止", "取消", "延後", "下市",
            "地雷", "炸彈", "套牢", "解套", "換手", "出貨", "警戒",
            "降息", "升息"  # 依情境而定
        ]

        pos_count = 0
        neg_count = 0

        for title in headlines:
            title_lower = title.lower()
            for kw in positive_keywords:
                if kw in title:
                    pos_count += 1
                    break
            for kw in negative_keywords:
                if kw in title:
                    neg_count += 1
                    break

        total_signal = pos_count + neg_count
        if total_signal > 0:
            net = pos_count - neg_count
            # 標準化到 -10 ~ +10
            score = (net / total_signal) * 10

        # 加分:標題中明確提到股票本身(有人關注)
        if stock_id in text or stock_name in text:
            score += 1

        score = max(-10, min(10, score))

    except Exception as e:
        return 0, [], f"新聞分析異常: {e}"

    return score, headlines, ""


# ─────────────────────────────────────────────
# 評分核心函式
# ─────────────────────────────────────────────

@dataclass
class StockScore:
    stock_id: str
    stock_name: str
    total_score: float = 0.0
    tech_score: float = 0.0      # 技術 35
    news_score: float = 0.0      # 新聞 20
    fund_score: float = 0.0      # 基本面 15
    inst_score: float = 0.0      # 籌碼 15
    momentum_score: float = 0.0  # 動能 10
    risk_score: float = 0.0      # 風險 5
    current_price: float = 0.0
    change_pct: float = 0.0
    signals: dict = field(default_factory=dict)
    news_headlines: list = field(default_factory=list)
    rating: str = "中立"
    analysis: list = field(default_factory=list)
    error: str = None
    # === 進出場輔助 ===
    entry_zone: str = ""              # 🟢可進場 / 🟡觀察 / 🔴不宜
    entry_note: str = ""              # 進場建議說明
    ma20_price: float = 0.0           # 月線位置
    ma60_price: float = 0.0           # 季線位置
    atr: float = 0.0                  # ATR 波動指標
    stop_loss: float = 0.0            # 停損價
    target_1: float = 0.0             # 第一目標價 (1:1)
    target_2: float = 0.0             # 第二目標價 (1:2)
    risk_reward_ratio: float = 0.0    # 風報比
    support_level: float = 0.0        # 主要支撐
    resistance_level: float = 0.0     # 主要壓力
    patterns: list = field(default_factory=list)  # 技術型態辨識結果


def compute_tech_score(df: pd.DataFrame) -> tuple:
    """技術面評分 (最高35分)"""
    if df.empty or len(df) < 30:
        return 0, {}, []

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    close = latest["Close"]
    score = 0.0
    signals = {}
    notes = []

    # 1. 均線排列 (0~8分)
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)

    if ma5 > ma20 > ma60 and ma60 > 0:
        score += 8
        signals["均線"] = "🟢 多頭排列"
        notes.append("均線多頭排列 (+8)")
    elif ma5 > ma20:
        score += 4
        signals["均線"] = "🟢 短線偏多"
        notes.append("短線站上月線 (+4)")
    else:
        score -= 3
        signals["均線"] = "🔴 短線偏弱"
        notes.append("短線在月線下 (-3)")

    if ma20 > ma60 and ma60 > 0:
        score += 2
        notes.append("月線在季線上 (+2)")
    elif ma60 > ma20 and ma60 > 0:
        score -= 2
        notes.append("月線在季線下 (-2)")

    # 2. RSI (0~5分)
    rsi = latest.get("RSI", 50)
    if 40 < rsi < 60:
        score += 5
        signals["RSI"] = f"中性 ({rsi:.0f})"
        notes.append(f"RSI {rsi:.0f} 中性 (+5)")
    elif 30 <= rsi <= 40:
        score += 4
        signals["RSI"] = f"🟢 偏低 ({rsi:.0f})"
        notes.append(f"RSI {rsi:.0f} 偏低 (+4)")
    elif 60 <= rsi <= 70:
        score += 3
        signals["RSI"] = f"偏強 ({rsi:.0f})"
        notes.append(f"RSI {rsi:.0f} 偏強 (+3)")
    elif rsi > 70:
        score -= 3
        signals["RSI"] = f"🟡 超買 ({rsi:.0f})"
        notes.append(f"RSI {rsi:.0f} 超買 (-3)")
    elif rsi < 30:
        score += 2
        signals["RSI"] = f"🟢 超賣 ({rsi:.0f})"
        notes.append(f"RSI {rsi:.0f} 超賣 (+2)")

    # 3. MACD (0~5分)
    macd = latest.get("MACD", 0)
    macd_sig = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    prev_hist = prev.get("MACD_Hist", 0)

    if macd > macd_sig and macd > 0:
        score += 5
        signals["MACD"] = "🟢 多頭"
        notes.append("MACD正值+站上訊號線 (+5)")
    elif macd > macd_sig and macd < 0:
        score += 3
        signals["MACD"] = "🟡 改善中"
        notes.append("MACD負但站上訊號線 (+3)")
    elif macd < macd_sig and macd > 0:
        score -= 2
        signals["MACD"] = "🟡 減弱"
        notes.append("MACD正值但跌破訊號線 (-2)")
    else:
        score -= 4
        signals["MACD"] = "🔴 空頭"
        notes.append("MACD負+跌破訊號線 (-4)")

    if macd_hist > prev_hist and macd_hist > 0:
        score += 2
        notes.append("MACD動能增強 (+2)")

    # 4. 布林通道 (0~5分)
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if bb_upper > 0 and bb_lower > 0:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if 20 < bb_pos < 60:
            score += 5
            signals["布林"] = f"中段 ({bb_pos:.0f}%)"
            notes.append("布林通道中段 (+5)")
        elif bb_pos <= 20:
            score += 3
            signals["布林"] = f"🟢 低檔 ({bb_pos:.0f}%)"
            notes.append("布林觸底 (+3)")
        elif 60 <= bb_pos <= 80:
            score += 2
            signals["布林"] = f"偏高 ({bb_pos:.0f}%)"
            notes.append("布林偏高 (+2)")
        else:
            score -= 3
            signals["布林"] = f"🔴 觸頂 ({bb_pos:.0f}%)"
            notes.append("布林觸頂過熱 (-3)")

    # 5. KDJ (0~5分)
    k_val = latest.get("K", 50)
    d_val = latest.get("D", 50)
    if k_val > d_val and k_val < 80:
        score += 5
        signals["KDJ"] = f"🟢 金叉 ({k_val:.0f}/{d_val:.0f})"
        notes.append(f"KDJ金叉 (+5)")
    elif k_val > d_val:
        score += 2
        signals["KDJ"] = f"高檔金叉 ({k_val:.0f}/{d_val:.0f})"
        notes.append("KDJ高檔金叉 (+2)")
    else:
        score -= 2
        signals["KDJ"] = f"🔴 死叉 ({k_val:.0f}/{d_val:.0f})"
        notes.append("KDJ死叉 (-2)")

    # 6. 成交量 (0~7分)
    if "Volume" in df.columns and len(df) > 10:
        avg_vol_20 = df["Volume"].iloc[-21:-1].mean() if len(df) > 21 else df["Volume"].mean()
        cur_vol = df["Volume"].iloc[-1]
        vol_ratio = cur_vol / avg_vol_20 if avg_vol_20 > 0 else 1
        recent_return = (close / df["Close"].iloc[-6] - 1) * 100 if len(df) > 5 else 0

        if vol_ratio > 1.3 and recent_return > 0:
            score += 7
            signals["量能"] = "🟢 價漲量增"
            notes.append(f"量增{vol_ratio:.1f}倍+價漲 (+7)")
        elif vol_ratio > 1.5 and recent_return < 0:
            score -= 3
            signals["量能"] = "🔴 價跌量增"
            notes.append("價跌量增 (-3)")
        elif 0.7 <= vol_ratio <= 1.3:
            score += 3
            signals["量能"] = "⚪ 量能正常"
            notes.append("量能正常 (+3)")

    score = max(-20, min(35, score))
    return score, signals, notes


def compute_fund_score_detail(fund: dict) -> tuple:
    """基本面評分 (最高15分)"""
    score = 0.0
    notes = []

    if "error" in fund:
        return 0, ["無基本面資料"]

    pe = fund.get("pe_ratio")
    if pe:
        if pe < 12: score += 8; notes.append(f"PE {pe:.1f} 低估 (+8)")
        elif pe < 18: score += 5; notes.append(f"PE {pe:.1f} 合理偏低 (+5)")
        elif pe < 25: score += 3; notes.append(f"PE {pe:.1f} 合理 (+3)")
        elif pe < 35: score += 1; notes.append(f"PE {pe:.1f} 略高 (+1)")
        else: score -= 3; notes.append(f"PE {pe:.1f} 過高 (-3)")

    dy = fund.get("dividend_yield")
    if dy:
        if dy > 6: score += 4; notes.append(f"殖利率 {dy:.1f}% 高 (+4)")
        elif dy > 4: score += 3; notes.append(f"殖利率 {dy:.1f}% 優 (+3)")
        elif dy > 3: score += 2; notes.append(f"殖利率 {dy:.1f}% 不錯 (+2)")
        elif dy > 1: score += 1; notes.append(f"殖利率 {dy:.1f}% 尚可 (+1)")

    pb = fund.get("pb_ratio")
    if pb:
        if pb < 1: score += 3; notes.append(f"PB {pb:.2f} 低於淨值 (+3)")
        elif pb < 1.5: score += 2; notes.append(f"PB {pb:.2f} 偏低 (+2)")
        elif pb < 3: score += 1; notes.append(f"PB {pb:.2f} 合理 (+1)")
        elif pb > 5: score -= 1; notes.append(f"PB {pb:.2f} 偏高 (-1)")

    return max(-5, min(15, score)), notes


def compute_inst_score(stock_id: str) -> tuple:
    """籌碼面評分 (最高15分)"""
    score = 0.0
    notes = []

    inst_df = fetch_institutional_trading(stock_id)
    if inst_df.empty:
        return 0, ["法人資料未更新"]

    for label, col_max in [("外資", 6), ("投信", 5), ("自營商", 4)]:
        row = inst_df[inst_df["類別"].str.contains(label)]
        if not row.empty:
            net = row["買賣超"].values[0]
            net_k = net / 1000
            if net_k > 5: s = col_max; notes.append(f"{label}大買 {net_k:.0f}張 (+{col_max})")
            elif net_k > 1: s = int(col_max * 0.7); notes.append(f"{label}買超 {net_k:.0f}張 (+{s})")
            elif net > 0: s = int(col_max * 0.4); notes.append(f"{label}小買 (+{s})")
            elif net_k < -5: s = -col_max; notes.append(f"{label}大賣 {abs(net_k):.0f}張 (-{col_max})")
            elif net < 0: s = -int(col_max * 0.5); notes.append(f"{label}賣超 (-{abs(s)})")
            else: s = 0
            score += s

    return max(-10, min(15, score)), notes


def compute_momentum_score(df: pd.DataFrame) -> tuple:
    """動能評分 (最高10分)"""
    if df.empty or len(df) < 10:
        return 0, []

    score = 0.0
    notes = []
    close = df["Close"]

    ret_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0
    if ret_1w > 5: score += 5; notes.append(f"週漲 {ret_1w:.1f}% (+5)")
    elif ret_1w > 2: score += 4; notes.append(f"週漲 {ret_1w:.1f}% (+4)")
    elif ret_1w > 0: score += 2; notes.append(f"週漲 {ret_1w:.1f}% (+2)")
    elif ret_1w > -2: score -= 1; notes.append(f"週跌 {abs(ret_1w):.1f}% (-1)")
    else: score -= 3; notes.append(f"週跌 {abs(ret_1w):.1f}% 弱勢 (-3)")

    ret_1m = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) > 21 else 0
    if ret_1m > 10: score += 3; notes.append(f"月漲 {ret_1m:.1f}% (+3)")
    elif ret_1m > 5: score += 2; notes.append(f"月漲 {ret_1m:.1f}% (+2)")
    elif ret_1m > 0: score += 1; notes.append(f"月漲 {ret_1m:.1f}% (+1)")
    elif ret_1m < -10: score -= 2; notes.append(f"月跌 {abs(ret_1m):.1f}% (-2)")

    return max(-10, min(10, score)), notes


def compute_risk_score(df: pd.DataFrame) -> tuple:
    """風險評分 (最高5分)"""
    if df.empty or len(df) < 20:
        return 3, []

    score = 3.0
    notes = []
    close = df["Close"]

    returns = close.pct_change().dropna()
    if len(returns) > 20:
        recent_vol = returns.tail(20).std() * np.sqrt(252) * 100
        if recent_vol < 20: score += 2; notes.append(f"低波動 {recent_vol:.0f}% (+2)")
        elif recent_vol > 45: score -= 2; notes.append(f"高波動 {recent_vol:.0f}% (-2)")

    if len(close) > 20:
        roll_max = close.tail(20).expanding().max()
        drawdowns = (close.tail(20) - roll_max) / roll_max * 100
        max_dd = drawdowns.min()
        if max_dd > -5: score += 1; notes.append(f"回撤僅{abs(max_dd):.1f}% (+1)")
        elif max_dd < -20: score -= 2; notes.append(f"回撤{abs(max_dd):.1f}% (-2)")

    return max(0, min(5, score)), notes


# ─────────────────────────────────────────────
# 大盤多空判斷
# ─────────────────────────────────────────────

def get_market_context() -> dict:
    """分析大盤趨勢 (用0050代替加權指數)"""
    ctx = {
        "trend": "未知",
        "trend_short": "⚪ 等待",
        "ma20": 0, "ma60": 0, "ma120": 0,
        "current": 0,
        "note": "",
    }
    try:
        df = fetch_historical("0050", months=12)
        if df.empty or len(df) < 120:
            return ctx

        df = add_all_indicators(df)
        latest = df.iloc[-1]
        ctx["current"] = latest["Close"]
        ctx["ma20"] = latest.get("MA20", 0)
        ctx["ma60"] = latest.get("MA60", 0)
        ctx["ma120"] = latest.get("MA120", 0) if "MA120" in df.columns else 0

        ma20 = ctx["ma20"]
        ma60 = ctx["ma60"]
        cur = ctx["current"]

        # 多空判斷
        if ma20 > ma60 and cur > ma20:
            ctx["trend"] = "多頭"
            ctx["trend_short"] = "🟢 多頭格局"
            ctx["note"] = "月線 > 季線,指數在月線上,整體偏多"
        elif ma20 < ma60 and cur < ma20:
            ctx["trend"] = "空頭"
            ctx["trend_short"] = "🔴 空頭格局"
            ctx["note"] = "月線 < 季線,指數在月線下,謹慎為上"
        elif cur > ma20 and cur < ma60:
            ctx["trend"] = "短多中空"
            ctx["trend_short"] = "🟡 短多中空"
            ctx["note"] = "站上月線但仍在季線下,短線反彈看待"
        elif cur < ma20 and cur > ma60:
            ctx["trend"] = "短空中多"
            ctx["trend_short"] = "🟡 短空中多"
            ctx["note"] = "跌破月線但季線有撐,整理格局"
        else:
            ctx["trend"] = "盤整"
            ctx["trend_short"] = "⚪ 盤整"
            ctx["note"] = "均線糾結,方向不明"

        # 統計20日漲跌幅
        ret_20d = (cur / df["Close"].iloc[-22] - 1) * 100 if len(df) > 21 else 0
        ctx["return_20d"] = ret_20d
        if ret_20d > 5:
            ctx["note"] += f",近月強漲 {ret_20d:.1f}%"
        elif ret_20d < -5:
            ctx["note"] += f",近月回檔 {ret_20d:.1f}%"

    except Exception:
        pass
    return ctx


# ─────────────────────────────────────────────
# 進場與風險分析
# ─────────────────────────────────────────────

def compute_entry_risk(df: pd.DataFrame, current_price: float) -> dict:
    """
    計算進場建議、停損價、目標價
    回傳 dict with entry_zone, entry_note, ma20, ma60, atr, stop_loss, target_1, target_2, rr_ratio
    """
    result = {
        "entry_zone": "⚪ 資料不足",
        "entry_note": "",
        "ma20_price": 0,
        "ma60_price": 0,
        "atr": 0,
        "stop_loss": 0,
        "target_1": 0,
        "target_2": 0,
        "risk_reward_ratio": 0,
        "support_level": 0,
        "resistance_level": 0,
    }

    if df.empty or len(df) < 30:
        return result

    latest = df.iloc[-1]
    close_series = df["Close"]

    # 均線位置
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    result["ma20_price"] = ma20
    result["ma60_price"] = ma60
    result["support_level"] = ma60 if ma60 > 0 else ma20
    result["resistance_level"] = ma20 if ma20 > current_price else current_price * 1.1

    # ATR (Average True Range) - 波動率
    if all(k in df.columns for k in ["High", "Low", "Close"]) and len(df) > 14:
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-14:])
    else:
        atr = current_price * 0.02  # 預設2%

    result["atr"] = round(atr, 2)

    # 停損價:1.5倍 ATR 或跌破 MA60 (取較寬的)
    sl_by_atr = current_price - atr * 1.5
    sl_by_ma = ma60 * 0.97 if ma60 > 0 else current_price * 0.93
    stop_loss = min(sl_by_atr, sl_by_ma)
    stop_loss = max(stop_loss, current_price * 0.85)  # 最多-15%
    result["stop_loss"] = round(stop_loss, 2)

    risk_per_share = current_price - stop_loss

    # 目標價:1:1 和 1:2 風報比
    result["target_1"] = round(current_price + risk_per_share * 1.0, 2)
    result["target_2"] = round(current_price + risk_per_share * 2.0, 2)

    if risk_per_share > 0:
        result["risk_reward_ratio"] = round(risk_per_share / risk_per_share, 2)  # =1.0 at TP1
        # 更實用的風報比:潛在獲利 / 潛在虧損
        gain_to_sl = (current_price - stop_loss)
        gain_to_t1 = (result["target_1"] - current_price)
        result["risk_reward_ratio"] = round(gain_to_t1 / gain_to_sl if gain_to_sl > 0 else 0, 2)

    # 進場區間建議
    # 多頭排列 (MA5 > MA20 > MA60):拉回月線買
    ma5 = latest.get("MA5", 0)
    if ma5 > ma20 > ma60 and ma60 > 0:
        result["entry_zone"] = "🟢 可進場"
        if current_price > ma20 * 1.05:
            result["entry_note"] = f"偏離月線 {(current_price/ma20-1)*100:.1f}%,建議等拉回 {ma20:.1f} 附近進場"
        elif current_price < ma20 * 1.02:
            result["entry_note"] = f"接近月線 {ma20:.1f},多頭格局可分批布局"
        else:
            result["entry_note"] = "均線多頭排列,可順勢操作"
        result["support_level"] = ma20

    # 月線上、季線下 (短多中空)
    elif current_price > ma20 and current_price < ma60:
        result["entry_zone"] = "🟡 觀察"
        result["entry_note"] = f"短線站上月線 {ma20:.0f} 但仍在季線 {ma60:.0f} 下,屬反彈格局,需站穩季線才安全"
        result["support_level"] = ma20
        result["resistance_level"] = ma60

    # 都在線下 (空頭)
    elif current_price < ma20 and current_price < ma60:
        result["entry_zone"] = "🔴 不宜進場"
        result["entry_note"] = f"價位在月線({ma20:.0f})和季線({ma60:.0f})之下,趨勢偏弱,建議等待站上月線再考慮"
        result["resistance_level"] = ma20

    # 都在線上 (強多頭)
    elif current_price > ma20 and current_price > ma60:
        result["entry_zone"] = "🟢 可進場"
        if ma20 > ma60:
            result["entry_note"] = "均線多頭,順勢操作"
        else:
            result["entry_note"] = f"價位在均線上,但月線({ma20:.0f}) < 季線({ma60:.0f}),注意是否為反轉"

    return result


def generate_analysis(pick: StockScore) -> list:
    """生成人性化分析說明"""
    lines = []
    s = pick.total_score

    if s >= 70:
        lines.append("🏆 **最佳推薦!多因子共振,技術+新聞+基本面同步看多**")
    elif s >= 55:
        lines.append("⭐ **強勢推薦!多數指標正向,具備短中線動能**")
    elif s >= 40:
        lines.append("📈 **推薦關注!部分指標轉佳,可留意布局時機**")
    else:
        lines.append("👀 **潛力觀察中**")

    lines.append("")
    lines.append("**📊 評分細項:**")
    lines.append(f"  📊技術 {pick.tech_score:.0f}/35 | 🗞️新聞 {pick.news_score:.0f}/20 | 📋基本面 {pick.fund_score:.0f}/15")
    lines.append(f"  🏢籌碼 {pick.inst_score:.0f}/15 | 🚀動能 {pick.momentum_score:.0f}/10 | 🛡️風險 {pick.risk_score:.0f}/5")
    lines.append(f"  **總分:{s:.0f}/100**")

    if pick.signals:
        sig_parts = [f"{k} {v}" for k, v in list(pick.signals.items())[:3]]
        if sig_parts:
            lines.append("")
            lines.append("**📡 關鍵訊號:**")
            lines.append(f"  {' | '.join(sig_parts)}")

    if pick.news_headlines:
        lines.append("")
        lines.append("**🗞️ 最新新聞:**")
        for h in pick.news_headlines[:3]:
            lines.append(f"  • {h[:50]}{'...' if len(h) > 50 else ''}")

    # 技術型態辨識
    if pick.patterns:
        lines.append("")
        lines.append("**📐 技術型態:**")
        for p in pick.patterns[:2]:
            conf = p.get("confidence", "中")
            icon = "🟢" if conf == "高" else "🟡"
            target = p.get("target", 0)
            lines.append(f"  {icon} **{p['type']}**")
            if target > 0:
                lines.append(f"     🎯 目標價 {target:.1f}")

    # 進出場建議
    if pick.entry_zone:
        lines.append("")
        lines.append("**📌 進出場參考:**")
        lines.append(f"  {pick.entry_zone} - {pick.entry_note}")
        if pick.ma20_price > 0:
            lines.append(f"  月線 {pick.ma20_price:.1f} | 季線 {pick.ma60_price:.1f} | ATR {pick.atr:.1f}")
        if pick.stop_loss > 0:
            lines.append(f"  🔴 停損 {pick.stop_loss:.1f} ({((pick.stop_loss/pick.current_price)-1)*100:+.1f}%)")
        if pick.target_1 > 0 and pick.target_2 > 0:
            lines.append(f"  🎯 目標1 {pick.target_1:.1f} | 目標2 {pick.target_2:.1f}")
        if pick.risk_reward_ratio > 0:
            lines.append(f"  ⚖️ 風報比 1:{pick.risk_reward_ratio:.1f}(至目標1)")

    lines.append("")
    lines.append("─" * 30)
    lines.append(f"📌 現價:{pick.current_price:.2f} | 日漲跌:{pick.change_pct:+.2f}%")
    lines.append("⚠️ 量化篩選結果,非投資建議")

    return lines


def score_stock(stock_id: str, stock_name: str, months: int = 6, include_news: bool = True) -> StockScore:
    """對單一股票進行完整評分"""
    result = StockScore(stock_id=stock_id, stock_name=stock_name)

    try:
        df = fetch_historical(stock_id, months=months)
        if df.empty:
            result.error = "無資料"
            return result

        df = add_all_indicators(df)

        tech_score, signals, tech_notes = compute_tech_score(df)
        result.tech_score = tech_score
        result.signals = signals

        fund = fetch_fundamentals(stock_id)
        fund_score, fund_notes = compute_fund_score_detail(fund)
        result.fund_score = fund_score

        inst_score, inst_notes = compute_inst_score(stock_id)
        result.inst_score = inst_score

        mom_score, mom_notes = compute_momentum_score(df)
        result.momentum_score = mom_score

        risk_score, risk_notes = compute_risk_score(df)
        result.risk_score = risk_score

        # 新聞情緒 (最高20分)
        if include_news:
            news_score, headlines, news_msg = fetch_news_sentiment(stock_id, stock_name)
            result.news_score = news_score
            result.news_headlines = headlines
        else:
            result.news_score = 0

        # 進出場分析
        entry_data = compute_entry_risk(df, result.current_price)
        result.entry_zone = entry_data["entry_zone"]
        result.entry_note = entry_data["entry_note"]
        result.ma20_price = entry_data["ma20_price"]
        result.ma60_price = entry_data["ma60_price"]
        result.atr = entry_data["atr"]
        result.stop_loss = entry_data["stop_loss"]
        result.target_1 = entry_data["target_1"]
        result.target_2 = entry_data["target_2"]
        result.risk_reward_ratio = entry_data["risk_reward_ratio"]
        result.support_level = entry_data["support_level"]
        result.resistance_level = entry_data["resistance_level"]

        # 技術型態辨識
        try:
            result.patterns = pr.detect_all_patterns(df, lookback=120)
        except Exception:
            result.patterns = []

        # 總分
        result.total_score = (
            tech_score + result.news_score + fund_score + inst_score + mom_score + risk_score
        )

        result.current_price = df["Close"].iloc[-1]
        if len(df) > 1:
            result.change_pct = (result.current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100

        ts = result.total_score
        if ts >= 70: result.rating = "強力推薦 ⭐"
        elif ts >= 55: result.rating = "推薦買進 📈"
        elif ts >= 40: result.rating = "值得關注 👀"
        elif ts >= 25: result.rating = "中立觀望 ⚖️"
        else: result.rating = "保守觀察 ⏳"

        result.analysis = generate_analysis(result)

    except Exception as e:
        result.error = str(e)

    return result


def get_daily_picks(top_n: int = 5, months: int = 6, include_news: bool = True) -> list:
    """掃描 50 檔重點股,回傳 Top N 推薦"""
    results = []
    total = len(SCAN_UNIVERSE)

    for i, (sid, sname) in enumerate(SCAN_UNIVERSE.items()):
        result = score_stock(sid, sname, months=months, include_news=include_news)
        if result.error is None:
            results.append(result)
        time.sleep(0.12)

    results.sort(key=lambda x: x.total_score, reverse=True)
    return results[:top_n]


def get_daily_picks_with_context(top_n: int = 5, months: int = 6, include_news: bool = True) -> dict:
    """取得每日推薦(含市場背景)"""
    picks = get_daily_picks(top_n=top_n, months=months, include_news=include_news)

    avg_score = np.mean([p.total_score for p in picks]) if picks else 0
    if avg_score >= 60: market_note = "🔥 市場偏多,推薦股多因子共振明顯"
    elif avg_score >= 45: market_note = "📈 市場中性偏多"
    elif avg_score >= 30: market_note = "⚖️ 市場震盪"
    else: market_note = "⚠️ 市場觀望"

    # 大盤背景
    market_ctx = get_market_context()

    return {
        "picks": picks,
        "market_note": market_note,
        "market_ctx": market_ctx,
        "candidates_count": len(SCAN_UNIVERSE),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
