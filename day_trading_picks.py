"""
⚡ 當日當沖股推薦系統
專為當沖（當日買賣）設計的多因子掃描與推薦模組

評分因子：
- 波動率 (30%)：高波動創造當沖空間
- 成交量 (25%)：流動性是當沖命脈
- 短線動能 (20%)：明確方向才好做
- 技術位置 (15%)：RSI/KDJ 短線位置
- 風險過濾 (10%)：剔除盤整、量縮、訊號混亂

輸出入：
- 推薦原因（人性化解讀）
- 建議買點 / 賣點
- 停損價位
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from data_fetcher import fetch_historical, fetch_stock_list, POPULAR_STOCKS, get_stock_name
from indicators import add_all_indicators


# ─────────────────────────────────────────────
# 當沖掃描標的清單（高流動性 / 高波動潛力股）
# ─────────────────────────────────────────────
DAYTRADE_UNIVERSE = {
    # 半導體 (高波動)
    "2330": "台積電", "2454": "聯發科", "2303": "聯電",
    "3034": "聯詠", "3711": "日月光投控", "3443": "創意",
    "3661": "世芯-KY", "6643": "M31", "5269": "祥碩",
    # AI / 伺服器 (熱門題材、高波動)
    "2317": "鴻海", "2382": "廣達", "2356": "英業達",
    "2376": "技嘉", "2377": "微星", "2357": "華碩",
    # PCB / 載板
    "3037": "欣興", "8046": "南電", "3189": "景碩",
    # 光電
    "3008": "大立光", "3481": "群創", "2409": "友達",
    # 電子零組件
    "2308": "台達電", "2327": "國巨", "3653": "健策",
    # 網通
    "4904": "遠傳", "3045": "台灣大", "2345": "智邦",
    # 航運 (高波動)
    "2603": "長榮", "2609": "陽明", "2615": "萬海",
    "2618": "長榮航", "2610": "華航",
    # 金融 (穩定量大的可以做)
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
    "2884": "玉山金", "2885": "元大金", "2886": "兆豐金",
    # 傳產龍頭
    "2002": "中鋼", "1301": "台塑", "1303": "南亞",
    "1216": "統一", "2912": "統一超",
    # 新興強勢股
    "6515": "穎崴", "6531": "愛普", "6770": "力積電",
    "2368": "金像電", "2383": "台光電", "3231": "緯創",
    "4938": "和碩", "3017": "奇鋐", "3406": "玉晶光",
    # 其他熱門
    "2498": "宏達電", "2353": "宏碁", "2313": "華通",
}

# 當沖關鍵字（生成推薦原因時使用）
BULLISH_REASONS = {
    "high_volume_break": "爆量突破壓力區，短線攻擊訊號明確",
    "ma_support_bounce": "回測均線支撐後彈升，短線買盤進場",
    "rsi_oversold_reversal": "RSI 超賣後轉強，短線反彈機會高",
    "bullish_flag": "多頭旗型整理末端，突破在即",
    "gap_up_momentum": "跳空上漲後量縮整理，醞釀續攻",
    "institutional_buying": "法人連續買超，籌碼安定有利短多",
    "sector_rotation": "產業輪動資金進駐，短線有跟漲空間",
    "turnover_breakout": "換手量成功，短線浮額清洗完畢",
    "trend_acceleration": "短線動能加速，適合順勢當沖",
    "volume_price_divergence": "價量齊揚攻擊盤，盤中拉回即可布局",
    "support_bounce": "觸及關鍵支撐後反彈，短線安全邊際高",
    "breakout_pullback": "突破後拉回測試支撐，標準進場點",
    "tight_consolidation": "窄幅盤整末端，即將表態有爆發空間",
    "leading_strength": "族群領漲股，強者恆強適合當沖",
    "oversold_bounce": "超跌後浮現短線反彈訊號",
}


@dataclass
class DayTradePick:
    """當沖推薦股票"""
    stock_id: str
    stock_name: str
    total_score: float = 0.0
    
    # 各項評分
    volatility_score: float = 0.0    # 波動率 (30)
    volume_score: float = 0.0        # 成交量 (25)
    momentum_score: float = 0.0      # 動能 (20)
    technical_score: float = 0.0     # 技術位置 (15)
    risk_score: float = 0.0          # 風險過濾 (10)
    
    # 價格資訊
    current_price: float = 0.0
    change_pct: float = 0.0
    day_high: float = 0.0
    day_low: float = 0.0
    avg_volume: float = 0.0
    
    # 進出場建議
    buy_price: float = 0.0           # 建議買點
    sell_price: float = 0.0          # 建議賣點
    stop_loss: float = 0.0           # 停損價
    risk_reward: float = 0.0         # 風報比
    
    # 技術位置
    ma5: float = 0.0
    ma20: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    
    # 推薦原因
    reasons: list = field(default_factory=list)
    reason_summary: str = ""          # 一句話總結
    trading_note: str = ""            # 當沖操作建議
    
    # 評級
    rating: str = "觀望"
    rating_emoji: str = "⚪"
    
    # 錯誤
    error: str = None


# ─────────────────────────────────────────────
# 評分核心
# ─────────────────────────────────────────────

def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    """計算 ATR (Average True Range)"""
    if df.empty or len(df) < period + 1:
        return 0.0
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    tr = np.maximum(
        high[-period:] - low[-period:],
        np.maximum(
            np.abs(high[-period:] - np.roll(close, 1)[-period:]),
            np.abs(low[-period:] - np.roll(close, 1)[-period:])
        )
    )
    # 排除第一個 NaN
    tr = tr[~np.isnan(tr)]
    return float(np.mean(tr)) if len(tr) > 0 else 0.0


def compute_volatility_score(df: pd.DataFrame) -> tuple:
    """
    波動率評分 (最高 30 分)
    高波動 = 當沖有空間賺價差
    """
    if df.empty or len(df) < 20:
        return 0, []

    score = 0.0
    notes = []
    close = df["Close"].values
    
    # ATR-based 波動
    atr = _calc_atr(df)
    if atr <= 0:
        return 0, []
    
    atr_pct = atr / close[-1] * 100  # ATR 佔股價百分比

    # 1. ATR 波動幅度 (0~15分)
    if atr_pct > 5:
        score += 15
        notes.append(f"ATR 波動 {atr_pct:.1f}%（極高波動，當沖空間充裕）")
    elif atr_pct > 3:
        score += 13
        notes.append(f"ATR 波動 {atr_pct:.1f}%（高波動，適合當沖）")
    elif atr_pct > 2:
        score += 10
        notes.append(f"ATR 波動 {atr_pct:.1f}%（波動適中）")
    elif atr_pct > 1:
        score += 6
        notes.append(f"ATR 波動 {atr_pct:.1f}%（波動偏低，空間有限）")
    else:
        score += 2
        notes.append(f"ATR 波動 {atr_pct:.1f}%（低波動，不適合當沖）")

    # 2. 近期日內振幅 (0~10分)
    if len(df) >= 10:
        daily_ranges = (df["High"] - df["Low"]) / df["Close"] * 100
        avg_range = daily_ranges.iloc[-10:].mean()
        
        if avg_range > 5:
            score += 10
            notes.append(f"近10日平均振幅 {avg_range:.1f}%（振幅大，當沖利潤空間足）")
        elif avg_range > 3:
            score += 8
            notes.append(f"近10日平均振幅 {avg_range:.1f}%（振幅適中）")
        elif avg_range > 2:
            score += 5
            notes.append(f"近10日平均振幅 {avg_range:.1f}%（振幅尚可）")
        elif avg_range > 1:
            score += 3
            notes.append(f"近10日平均振幅 {avg_range:.1f}%（振幅偏小）")
        else:
            score += 1
            notes.append(f"近10日平均振幅 {avg_range:.1f}%（振幅太小，不適合當沖）")

    # 3. 波動率穩定性 (0~5分)：波動持續 vs 突然放大
    if len(df) >= 20:
        recent_range = daily_ranges.iloc[-5:].mean()
        older_range = daily_ranges.iloc[-20:-5].mean()
        if older_range > 0:
            ratio = recent_range / older_range
            if 0.8 < ratio < 1.5:
                score += 5
                notes.append("波動率穩定（持續有當沖空間）")
            elif ratio >= 1.5:
                score += 3
                notes.append("波動率正在放大（短線波動加劇，注意風險控管）")
            elif ratio >= 0.5:
                score += 2
                notes.append("波動率略為收縮")

    return max(0, min(30, score)), notes


def compute_volume_score(df: pd.DataFrame) -> tuple:
    """
    成交量評分 (最高 25 分)
    高成交量 = 容易進出、不會被套
    """
    if df.empty or len(df) < 20:
        return 0, []

    score = 0.0
    notes = []
    
    volume = df["Volume"].values
    close = df["Close"].values
    
    cur_vol = volume[-1]
    avg_vol_5 = np.mean(volume[-6:-1]) if len(volume) >= 6 else cur_vol
    avg_vol_20 = np.mean(volume[-21:-1]) if len(volume) >= 21 else cur_vol
    
    # 1. 絕對成交量 (0~12分) - 量大好進出
    if cur_vol > 100_000:
        score += 12
        notes.append(f"成交量 {cur_vol/1000:.0f}張（充沛流動性，進出無虞）")
    elif cur_vol > 50_000:
        score += 10
        notes.append(f"成交量 {cur_vol/1000:.0f}張（流動性佳）")
    elif cur_vol > 20_000:
        score += 8
        notes.append(f"成交量 {cur_vol/1000:.0f}張（流動性尚可）")
    elif cur_vol > 10_000:
        score += 5
        notes.append(f"成交量 {cur_vol/1000:.0f}張（流動性偏低，注意滑價）")
    elif cur_vol > 5_000:
        score += 3
        notes.append(f"成交量 {cur_vol/1000:.0f}張（成交量偏低）")
    else:
        score += 1
        notes.append(f"成交量 {cur_vol/1000:.0f}張（量太小，不適合當沖）")
    
    # 2. 相對成交量 (0~8分)
    if avg_vol_20 > 0:
        vol_ratio = cur_vol / avg_vol_20
        
        if 1.2 < vol_ratio < 2.5:
            score += 8
            notes.append(f"成交量 {vol_ratio:.1f}x 均量（溫和放量，市場關注度高）")
        elif 0.8 <= vol_ratio <= 1.2:
            score += 6
            notes.append("成交量與均量相當（穩定）")
        elif vol_ratio >= 2.5:
            score += 5
            notes.append(f"成交量 {vol_ratio:.1f}x 均量（爆量，當沖機會高但注意反轉）")
        elif vol_ratio >= 0.5:
            score += 3
            notes.append("量縮中（人氣退潮，不適合當沖）")
        else:
            score += 1
            notes.append("嚴重量縮")
    
    # 3. 量價關係 (0~5分)
    if len(close) >= 2:
        price_change = (close[-1] - close[-2]) / close[-2] * 100
        if price_change > 0 and cur_vol > avg_vol_5 * 1.1:
            score += 5
            notes.append("價漲量增（多頭攻擊訊號）")
        elif price_change > 0:
            score += 3
            notes.append("價漲量平")
        elif price_change < 0 and cur_vol < avg_vol_5 * 0.9:
            score += 3
            notes.append("價跌量縮（賣壓減輕）")
        elif price_change < 0:
            score += 1
            notes.append("價跌量增（須留意賣壓）")

    return max(0, min(25, score)), notes


def compute_momentum_score(df: pd.DataFrame) -> tuple:
    """
    短線動能評分 (最高 20 分)
    有明確方向的股票才好當沖
    """
    if df.empty or len(df) < 10:
        return 0, []

    score = 0.0
    notes = []
    close = df["Close"].values
    
    # 1. 近5日漲跌幅 (0~8分)
    if len(close) >= 6:
        ret_5d = (close[-1] / close[-6] - 1) * 100
        if 3 < ret_5d < 10:
            score += 8
            notes.append(f"近5日 {ret_5d:+.1f}%（短線強勢但未過熱）")
        elif 1 < ret_5d <= 3:
            score += 6
            notes.append(f"近5日 {ret_5d:+.1f}%（溫和上漲）")
        elif -3 < ret_5d <= 1:
            score += 4
            notes.append(f"近5日 {ret_5d:+.1f}%（盤整中）")
        elif ret_5d >= 10:
            score += 3
            notes.append(f"近5日 {ret_5d:+.1f}%（漲幅已大，追高風險高）")
        elif ret_5d <= -3:
            score += 2
            notes.append(f"近5日 {ret_5d:+.1f}%（短線偏弱）")

    # 2. 連續方向 (0~7分)
    if len(close) >= 6:
        consecutive = 0
        direction = 0
        for i in range(len(close)-1, max(len(close)-6, 0), -1):
            chg = (close[i] - close[i-1]) / close[i-1]
            if chg > 0:
                if direction == 0:
                    direction = 1
                    consecutive = 1
                elif direction == 1:
                    consecutive += 1
                else:
                    break
            elif chg < 0:
                if direction == 0:
                    direction = -1
                    consecutive = 1
                elif direction == -1:
                    consecutive += 1
                else:
                    break
            else:
                break
        
        if direction == 1 and 2 <= consecutive <= 4:
            score += 7
            notes.append(f"連{consecutive}紅（短多確立）")
        elif direction == 1 and consecutive == 1:
            score += 4
            notes.append("剛轉強")
        elif direction == -1 and consecutive >= 3:
            score += 2
            notes.append(f"連{consecutive}黑（弱勢，等待轉強訊號）")
        elif direction == -1:
            score += 3
            notes.append("短線偏弱")

    # 3. 近20日相對強度 (0~5分)
    if len(close) >= 21:
        ret_20d = (close[-1] / close[-21] - 1) * 100
        ret_10d = (close[-1] / close[-11] - 1) * 100
        # 加速上漲
        if ret_10d > 0 and ret_20d > 0 and ret_10d > ret_20d * 0.5:
            score += 5
            notes.append("短線動能加速（近10日 > 近20日半數）")
        elif ret_10d > 0:
            score += 3
            notes.append("近10日上漲中")
        elif ret_10d < 0 and ret_20d > 0:
            score += 1
            notes.append("短線回檔中，等待動能恢復")

    return max(0, min(20, score)), notes


def compute_technical_position(df: pd.DataFrame) -> tuple:
    """
    技術位置評分 (最高 15 分)
    RSI/KDJ/MACD 短線位置，不追高、不接刀
    """
    if df.empty or len(df) < 30:
        return 0, [], {}

    score = 0.0
    notes = []
    latest = df.iloc[-1]
    close = latest["Close"]
    data_out = {}
    
    # 1. RSI 位置 (0~6分)
    rsi = latest.get("RSI", 50)
    data_out["rsi"] = rsi
    if 40 < rsi < 65:
        score += 6
        notes.append(f"RSI {rsi:.0f}（安全區間，不會追高也不接刀）")
    elif 30 <= rsi <= 40:
        score += 5
        notes.append(f"RSI {rsi:.0f}（偏低，有反彈空間）")
    elif 65 <= rsi <= 75:
        score += 3
        notes.append(f"RSI {rsi:.0f}（偏高，短線須注意拉回）")
    elif rsi > 75:
        score += 1
        notes.append(f"RSI {rsi:.0f}（超買，當沖追高風險大）")
    elif rsi < 30:
        score += 4
        notes.append(f"RSI {rsi:.0f}（超賣，可搶反彈但注意趨勢）")

    # 2. 均線位置 (0~5分)
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    data_out["ma5"] = ma5
    data_out["ma20"] = ma20
    
    if ma5 > 0 and ma20 > 0:
        if close > ma5 > ma20:
            score += 5
            notes.append(f"站穩 5日線{ma5:.0f} & 月線{ma20:.0f}（短線強勢）")
        elif close > ma20 and close < ma5:
            score += 3
            notes.append(f"在月線{ma20:.0f}上但跌破5日線{ma5:.0f}（短線整理）")
        elif close < ma20 and close > ma5:
            score += 2
            notes.append(f"跌破月線{ma20:.0f}（反彈格局看待）")
        elif close < ma5 < ma20:
            score += 1
            notes.append("均線下偏弱（等站回月線再當沖）")

    # 3. MACD 短線方向 (0~4分)
    macd = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    prev_hist = df["MACD_Hist"].iloc[-2] if "MACD_Hist" in df.columns and len(df) > 1 else 0
    
    if macd > macd_signal and macd_hist > prev_hist:
        score += 4
        notes.append("MACD 多頭擴張（動能增強）")
    elif macd > macd_signal:
        score += 3
        notes.append("MACD 偏多")
    elif macd_hist > prev_hist:
        score += 2
        notes.append("MACD 柱狀圖收斂（空方減弱）")
    else:
        score += 1
        notes.append("MACD 偏空（等待翻多訊號）")

    return max(0, min(15, score)), notes, data_out


def compute_risk_filter(df: pd.DataFrame) -> tuple:
    """
    風險過濾評分 (最高 10 分)
    剔除不適合當沖的狀況：盤整、籌碼亂、波動不足
    """
    if df.empty or len(df) < 30:
        return 0, []

    score = 10.0  # 從滿分開始扣
    notes = []
    close = df["Close"].values
    latest = df.iloc[-1]
    
    # 1. 盤整過濾：布林通道寬度
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if bb_upper > 0 and bb_lower > 0:
        bb_width_pct = (bb_upper - bb_lower) / latest["Close"] * 100
        if bb_width_pct < 3:
            score -= 4
            notes.append("布林通道過窄（盤整中，方向不明）")
        elif bb_width_pct < 5:
            score -= 1
            notes.append("布林通道偏窄")
    
    # 2. 近期是否有跳空/缺口（不確定性高）
    if len(df) >= 5:
        gap_count = 0
        for i in range(max(1, len(df)-5), len(df)):
            prev_high = df["High"].iloc[i-1]
            prev_low = df["Low"].iloc[i-1]
            cur_low = df["Low"].iloc[i]
            cur_high = df["High"].iloc[i]
            if cur_low > prev_high * 1.02 or cur_high < prev_low * 0.98:
                gap_count += 1
        if gap_count >= 2:
            score -= 3
            notes.append("近期頻繁跳空（風險偏高，當沖須嚴守停損）")
    
    # 3. 價格區間（太便宜或太貴流動性有差）
    price = close[-1]
    if price < 10:
        score -= 2
        notes.append("股價 < $10（低價股波動雖大但流動性風險高）")
    elif price > 1000:
        score -= 1
        notes.append("高價股（每檔跳動金額大，注意資金管理）")

    # 4. 成交值（金額）= 成交量 x 股價
    cur_vol = df["Volume"].iloc[-1]
    turnover = cur_vol * price  # 成交值
    if turnover < 100_000_000:  # 小於 1 億
        score -= 3
        notes.append(f"成交值偏低 ({turnover/1e8:.1f}億)（流動性不足）")
    elif turnover < 500_000_000:  # 小於 5 億
        score -= 1
        notes.append(f"成交值 ({turnover/1e8:.1f}億)")

    return max(0, min(10, score)), notes


def calculate_entry_exit(df: pd.DataFrame, current_price: float, 
                         momentum_score: float, tech_score: float) -> dict:
    """
    計算當沖買點 / 賣點 / 停損
    
    邏輯：
    - 買點：支撐位附近（5日線、昨日低點、開盤參考價）+ ATR buffer
    - 賣點：壓力位附近（5日高點、布林上軌、ATR 目標）
    - 停損：ATR x 1.5 下方
    """
    result = {
        "buy_price": 0,
        "sell_price": 0,
        "stop_loss": 0,
        "risk_reward": 0,
        "atr": 0,
    }
    
    if df.empty or len(df) < 14:
        return result
    
    atr = _calc_atr(df)
    if atr <= 0:
        atr = current_price * 0.02
    result["atr"] = round(atr, 2)
    
    latest = df.iloc[-1]
    ma5 = latest.get("MA5", current_price)
    ma20 = latest.get("MA20", current_price)
    
    # ── 買點計算 ──
    # 當沖買點 = 支撐區間（5日線、昨日低、開盤價附近）
    yesterday_low = df["Low"].iloc[-2] if len(df) >= 2 else current_price
    yesterday_close = df["Close"].iloc[-2] if len(df) >= 2 else current_price
    
    # 收集支撐候選
    supports = []
    if ma5 > 0 and ma5 < current_price:
        supports.append(ma5)
    if ma20 > 0 and ma20 < current_price:
        supports.append(ma20)
    supports.append(yesterday_low)
    
    # 取最近的支撐
    valid_supports = [s for s in supports if s < current_price]
    if valid_supports:
        nearest_support = max(valid_supports)
        # 買點 = 支撐上方一點（不會剛好買在支撐）
        buy_price = nearest_support + atr * 0.3
    else:
        # 沒有下方支撐，以現價下方 1 ATR 為參考
        buy_price = current_price - atr * 0.5
    
    buy_price = max(buy_price, current_price * 0.93)  # 不低於現價 7%
    buy_price = min(buy_price, current_price * 1.00)  # 不高於現價
    result["buy_price"] = round(buy_price, 2)
    
    # ── 賣點計算 ──
    # 壓力位：布林上軌、近期高點、ATR 倍數
    bb_upper = latest.get("BB_Upper", current_price * 1.05)
    high_5 = df["High"].iloc[-5:].max() if len(df) >= 5 else current_price * 1.05
    high_10 = df["High"].iloc[-10:].max() if len(df) >= 10 else current_price * 1.05
    
    # 收集壓力候選
    resistances = []
    if bb_upper > current_price:
        resistances.append(bb_upper)
    if high_5 > current_price:
        resistances.append(high_5)
    if high_10 > current_price:
        resistances.append(high_10)
    
    # 用 ATR 倍數作為額外目標
    if resistances:
        nearest_resistance = min(resistances)
        # 取最近的壓力或 ATR 目標，兩者取合理值
        atr_target = current_price + atr * 1.5
        sell_candidates = [r for r in [nearest_resistance, atr_target] if r > current_price * 1.01]
        if sell_candidates:
            sell_price = min(sell_candidates)
        else:
            sell_price = current_price + atr
    else:
        # based on momentum: 強勢2x ATR, 中性 1.5x, 弱 1x
        multiplier = 2.0 if momentum_score >= 14 else (1.5 if momentum_score >= 8 else 1.0)
        sell_price = current_price + atr * multiplier
    
    sell_price = max(sell_price, current_price * 1.01)  # 最少 1% 目標
    sell_price = min(sell_price, current_price * 1.10)  # 最多 10%
    result["sell_price"] = round(sell_price, 2)
    
    # ── 停損價 ──
    stop_loss = current_price - atr * 1.5
    # 不能低於關鍵支撐太多
    if supports:
        support_floor = max(supports) * 0.98
        stop_loss = max(stop_loss, support_floor)
    stop_loss = max(stop_loss, current_price * 0.94)  # 當沖停損最多 -6%
    stop_loss = min(stop_loss, current_price * 0.99)  # 最少 -1%
    result["stop_loss"] = round(stop_loss, 2)
    
    # ── 風險報酬比 ──
    risk = current_price - stop_loss
    reward = sell_price - current_price
    if risk > 0 and reward > 0:
        result["risk_reward"] = round(reward / risk, 2)
    
    return result


def generate_reasons(pick: DayTradePick, df: pd.DataFrame) -> list:
    """
    生成推薦原因（使用關鍵字匹配 + 條件判斷）
    """
    reasons = []
    
    close = df["Close"].values
    volume = df["Volume"].values
    latest = df.iloc[-1]
    
    # 波動原因
    if pick.volatility_score >= 20:
        reasons.append(f"🔥 {BULLISH_REASONS['high_volume_break']}（波動率 {pick.atr/pick.current_price*100:.1f}%）")
    
    # 成交量原因
    if pick.volume_score >= 18:
        reasons.append(f"📊 {BULLISH_REASONS['volume_price_divergence']}（量 {df['Volume'].iloc[-1]/1000:.0f}張）")
    elif pick.volume_score >= 12:
        reasons.append("💧 流動性充足，當沖進出方便")
    
    # 技術面原因
    rsi = latest.get("RSI", 50)
    if 35 <= rsi <= 45 and pick.momentum_score >= 8:
        reasons.append(f"📈 {BULLISH_REASONS['support_bounce']}（RSI {rsi:.0f} 健康偏低，有上漲空間）")
    elif 45 <= rsi <= 60:
        reasons.append("✅ RSI 在安全區間，非超買亦非超賣，適合短線操作")
    
    # 均線原因
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    if close[-1] > ma5 > ma20 and ma20 > 0:
        reasons.append(f"📐 均線多頭排列(5日{ma5:.0f} > 月線{ma20:.0f})，短線趨勢明確")
    elif ma5 > 0 and close[-1] > ma5:
        reasons.append(f"📐 站穩 5 日線 {ma5:.0f} 之上，短線偏多")
    
    # MACD 原因
    macd = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    if macd > macd_signal and macd_hist > 0:
        reasons.append("🔄 MACD 多頭擴張中，順勢操作勝率高")
    
    # 動能原因
    if len(close) >= 6:
        ret_5d = (close[-1] / close[-6] - 1) * 100
        if 2 < ret_5d < 8:
            reasons.append(f"🚀 {BULLISH_REASONS['trend_acceleration']}（近5日 {ret_5d:+.1f}%）")
        elif 0 < ret_5d <= 2:
            reasons.append("📈 溫和上漲中，有醞釀突破機會")
        elif -2 < ret_5d < 0:
            reasons.append("📊 短期回檔但幅度有限，有反彈機會")
    
    # 連續方向
    if len(close) >= 4:
        up_streak = 0
        for i in range(len(close)-1, max(len(close)-5, 0), -1):
            if close[i] > close[i-1]:
                up_streak += 1
            else:
                break
        if up_streak >= 3:
            reasons.append(f"⚡ 連續 {up_streak} 日上漲，短多氣勢強")
    
    # 型態/整理
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if bb_upper > 0 and bb_lower > 0:
        bb_pos = (close[-1] - bb_lower) / (bb_upper - bb_lower) * 100
        if 30 <= bb_pos <= 60:
            reasons.append(f"📍 布林通道中段({bb_pos:.0f}%)，非極端位置，操作彈性大")
        elif bb_pos < 20:
            reasons.append(f"💡 布林下軌附近({bb_pos:.0f}%)，有反彈空間")
    
    # 量價關係
    if len(volume) >= 6:
        avg_vol_5 = np.mean(volume[-6:-1])
        vol_ratio = volume[-1] / avg_vol_5 if avg_vol_5 > 0 else 1
        if 1.2 < vol_ratio < 2.5:
            reasons.append(f"📊 量增 {vol_ratio:.1f}x 均量，市場關注度提升中")
    
    # 確保至少有2條原因
    if len(reasons) == 0:
        reasons.append("📊 技術面中性，短線有操作機會")
    if len(reasons) == 1:
        reasons.append("⚠️ 訊號較少，建議嚴守停損紀律")
    
    return reasons[:5]  # 最多5條


def generate_trading_note(pick: DayTradePick) -> str:
    """
    生成當沖操作建議
    """
    notes = []
    buy_pct = (pick.buy_price / pick.current_price - 1) * 100
    sell_pct = (pick.sell_price / pick.current_price - 1) * 100
    stop_pct = (pick.stop_loss / pick.current_price - 1) * 100
    
    notes.append(f"**🎯 操作策略**")
    notes.append(f"• 買點：{pick.buy_price:.2f}（現價{buy_pct:+.1f}%）— 拉回支撐區即可布局")
    notes.append(f"• 賣點：{pick.sell_price:.2f}（現價{sell_pct:+.1f}%）— 達標或盤中轉弱即出")
    notes.append(f"• 停損：{pick.stop_loss:.2f}（現價{stop_pct:.1f}%）— 嚴格執行！")
    
    if pick.risk_reward > 2:
        notes.append(f"• ⚖️ 風報比 1:{pick.risk_reward:.1f} — 理想")
    elif pick.risk_reward > 1:
        notes.append(f"• ⚖️ 風報比 1:{pick.risk_reward:.1f} — 可接受")
    else:
        notes.append(f"• ⚖️ 風報比 1:{pick.risk_reward:.1f} — 偏低，注意風險")
    
    notes.append("")
    notes.append("**⚠️ 當沖提醒**")
    notes.append("• 當沖有時間壓力，務必在 13:25 前平倉")
    notes.append("• 開盤前 30 分鐘波動大，新手建議 9:30 後再進場")
    notes.append("• 有賺就跑，不貪心是當沖存活關鍵")
    notes.append("• 交易成本（手續費+稅）約 0.585%，需計入損益")
    
    return "\n".join(notes)


def score_day_trade(stock_id: str, stock_name: str, months: int = 3) -> DayTradePick:
    """
    對單一股票進行當沖適合度評分
    """
    result = DayTradePick(stock_id=stock_id, stock_name=stock_name)
    
    try:
        df = fetch_historical(stock_id, months=max(months, 3))
        if df.empty or len(df) < 20:
            result.error = "資料不足"
            return result
        
        df = add_all_indicators(df)
        
        # 各項評分
        v_score, v_notes = compute_volatility_score(df)
        vol_score, vol_notes = compute_volume_score(df)
        m_score, m_notes = compute_momentum_score(df)
        t_score, t_notes, t_data = compute_technical_position(df)
        r_score, r_notes = compute_risk_filter(df)
        
        result.volatility_score = v_score
        result.volume_score = vol_score
        result.momentum_score = m_score
        result.technical_score = t_score
        result.risk_score = r_score
        
        result.total_score = v_score + vol_score + m_score + t_score + r_score
        
        # 基本資料
        result.current_price = df["Close"].iloc[-1]
        result.day_high = df["High"].iloc[-1]
        result.day_low = df["Low"].iloc[-1]
        result.avg_volume = df["Volume"].iloc[-20:].mean()
        result.rsi = t_data.get("rsi", 50)
        result.ma5 = t_data.get("ma5", 0)
        result.ma20 = t_data.get("ma20", 0)
        
        if len(df) > 1:
            result.change_pct = (result.current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
        
        # 進出場計算
        ee = calculate_entry_exit(df, result.current_price, m_score, t_score)
        result.buy_price = ee["buy_price"]
        result.sell_price = ee["sell_price"]
        result.stop_loss = ee["stop_loss"]
        result.risk_reward = ee["risk_reward"]
        result.atr = ee["atr"]
        
        # 推薦原因
        result.reasons = generate_reasons(result, df)
        result.reason_summary = result.reasons[0] if result.reasons else ""
        
        # 操作建議
        result.trading_note = generate_trading_note(result)
        
        # 評級
        ts = result.total_score
        if ts >= 75:
            result.rating = "🔥 強力推薦"
            result.rating_emoji = "🔥"
        elif ts >= 60:
            result.rating = "⭐ 推薦"
            result.rating_emoji = "⭐"
        elif ts >= 45:
            result.rating = "👍 可考慮"
            result.rating_emoji = "👍"
        elif ts >= 30:
            result.rating = "⚖️ 觀望"
            result.rating_emoji = "⚖️"
        else:
            result.rating = "❌ 不建議"
            result.rating_emoji = "❌"
        
    except Exception as e:
        result.error = str(e)
    
    return result


def get_day_trading_picks(top_n: int = 5, months: int = 3, 
                          min_score: float = 30) -> list:
    """
    掃描當沖標的，回傳 Top N 推薦（平行處理加速）
    """
    results = []
    max_workers = 15
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(score_day_trade, sid, sname, months=months): sid
            for sid, sname in DAYTRADE_UNIVERSE.items()
        }
        for future in as_completed(future_map):
            try:
                pick = future.result(timeout=30)
                if pick.error is None:
                    results.append(pick)
            except Exception:
                pass
    
    # 過濾：最低分數
    results = [r for r in results if r.total_score >= min_score]
    # 排序：總分高→低
    results.sort(key=lambda x: x.total_score, reverse=True)
    
    return results[:top_n]


def get_day_trading_summary(picks: list) -> dict:
    """
    產生當沖推薦摘要
    """
    if not picks:
        return {
            "total_candidates": len(DAYTRADE_UNIVERSE),
            "qualified_count": 0,
            "avg_score": 0,
            "market_note": "⚠️ 今日無符合條件的當沖標的",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    
    avg_score = np.mean([p.total_score for p in picks])
    
    if avg_score >= 65:
        market_note = "🔥 今日當沖環境佳，多檔標的達推薦門檻"
    elif avg_score >= 50:
        market_note = "📈 當沖環境尚可，精選標的操作"
    elif avg_score >= 35:
        market_note = "⚖️ 市場中性，建議嚴選 + 嚴守紀律"
    else:
        market_note = "⚠️ 當沖環境偏弱，建議觀望或輕倉操作"
    
    # 統計
    strong_buy = sum(1 for p in picks if p.total_score >= 60)
    buy = sum(1 for p in picks if 45 <= p.total_score < 60)
    watch = sum(1 for p in picks if p.total_score < 45)
    
    return {
        "total_candidates": len(DAYTRADE_UNIVERSE),
        "qualified_count": len(picks),
        "avg_score": round(avg_score, 1),
        "strong_buy_count": strong_buy,
        "buy_count": buy,
        "watch_count": watch,
        "market_note": market_note,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
