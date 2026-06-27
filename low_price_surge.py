"""
FIRE 低價飆股預測 — 精準掃描即將起漲的低價潛力股
專注股價 < $100 的股票，找出：爆量突破 + 技術共振 + 籌碼異動

評分機制:
  A級 (90+): 強烈買入信號
  B級 (75-89): 值得關注
  C級 (60-74): 潛在觀察
  D級 (<60): 條件不足

價格預測:
  - 短期目標 (1-5天): 基於最近阻力位 + ATR波動
  - 中期目標 (1-2週): 基於型態突破 + 波段漲幅
  - 波段潛力 (1月): 基於52週價位 + 產業動能
"""

import pandas as pd
import numpy as np
import time, sys, re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fix stdout encoding for cp950/Big5 systems
if sys.stdout.encoding and sys.stdout.encoding.upper() in ("CP950", "BIG5"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

# 低價股掃描清單（價格 < $100 的熱門標的 + 各產業小型股）
LOW_PRICE_UNIVERSE = {
    # 電子/半導體
    "2401": "凌陽", "2458": "義隆", "2329": "華泰",
    "3260": "威剛", "4968": "立積", "2363": "矽統",
    "6233": "旺玖", "6104": "創惟", "3529": "力旺",
    "3545": "敦泰", "8016": "矽創", "6214": "精誠",
    "3026": "禾伸堂", "2365": "昆盈", "2302": "麗正",
    "2387": "精元", "2392": "正崴", "2399": "映泰",
    "2402": "毅嘉", "2413": "環科", "2415": "錩新",
    "2425": "承啟", "2426": "鼎元", "2428": "興勤",
    "2436": "偉詮電", "2440": "太空梭", "2449": "京元電子",
    "2465": "麗臺", "2472": "立隆電", "2474": "可成",
    # 光電/面板
    "2409": "友達", "3481": "群創", "6116": "彩晶",
    "8105": "凌巨", "6120": "輔祥", "8215": "明基材",
    "3051": "力特", "5371": "中光電",
    # 金融/證券
    "2883": "開發金", "2888": "新光金", "2880": "華南金",
    "2834": "臺企銀", "2812": "台中銀", "2845": "遠東銀",
    "2889": "國票金", "2890": "永豐金", "5880": "合庫金",
    "2836": "高雄銀", "2838": "聯邦銀", "2849": "安泰銀",
    # 傳產/原物料
    "2002": "中鋼", "2014": "中鴻", "2017": "官田鋼",
    "2023": "燁輝", "2024": "志聯", "2025": "千興",
    "1304": "台聚", "1305": "華夏", "1312": "國喬",
    "1314": "中石化", "1321": "大洋", "1337": "再生-KY",
    "1434": "福懋", "1444": "力麗", "1455": "集盛",
    "1464": "得力", "1473": "臺南", "1474": "弘裕",
    # 營建
    "2501": "國建", "2504": "國產", "2511": "太子",
    "2515": "中工", "2520": "冠德", "2534": "宏盛",
    "2538": "基泰", "2540": "愛山林", "2542": "興富發",
    "2545": "皇翔", "2548": "華固", "2597": "潤弘",
    # 航運
    "2605": "新興", "2606": "裕民", "2612": "中航",
    "2613": "中櫃", "2617": "台航", "2630": "亞航",
    "2642": "宅配通", "5608": "四維航",
    # 電機/機械
    "1513": "中興電", "1514": "亞力", "1516": "川飛",
    "1519": "華城", "1521": "大億", "1522": "堤維西",
    "1526": "日馳", "1535": "中宇", "1536": "和大",
    "1558": "伸興", "1563": "巧新",
    # 生技
    "4105": "東洋", "4106": "雃博", "4114": "健喬",
    "4128": "中天", "4137": "麗豐-KY", "4736": "泰博",
    "4743": "合一", "4746": "台耀", "6446": "藥華藥",
    # 電線電纜
    "1603": "華電", "1604": "聲寶", "1605": "華新",
    "1608": "華榮", "1609": "大亞", "1611": "中電",
    "1612": "宏泰", "1614": "三洋電",
    # 油電/瓦斯
    "6505": "台塑化", "9918": "欣天然", "9926": "新海",
    # 百貨/貿易
    "2903": "遠百", "2905": "三商", "2910": "統領",
    "2913": "農林", "2915": "潤泰全",
    # 通信網路
    "4904": "遠傳", "4906": "正文", "4912": "聯銘",
    "4916": "事欣科", "4934": "太極",
    # 電子零組件
    "3023": "信邦", "3028": "增你強", "3031": "佰鴻",
    "3032": "偉訓", "3033": "威健", "3036": "文曄",
    "3041": "揚智", "3042": "晶技", "3043": "科風",
    "3044": "健鼎", "3090": "日電貿",
    # 其他潛力股
    "8033": "雷虎", "8112": "至上", "8114": "振樺電",
    "8163": "達方", "8210": "勤誠", "9904": "寶成",
    "9910": "豐泰", "9921": "巨大", "9924": "福興",
    "9933": "中鼎", "9940": "信義", "9945": "潤泰新",
}

# Add remaining stocks from surge predictor's POPULAR_STOCKS
def _ensure_universe_populated():
    if len(LOW_PRICE_UNIVERSE) > 220:
        return
    fetcher = sys.modules.get("data_fetcher")
    if fetcher:
        extra = getattr(fetcher, "POPULAR_STOCKS", {})
        for k, v in extra.items():
            if k not in LOW_PRICE_UNIVERSE:
                LOW_PRICE_UNIVERSE[k] = v


@dataclass
class LowPriceSurgeCandidate:
    stock_id: str
    stock_name: str
    price: float = 0.0
    score: float = 0.0
    grade: str = "D"

    # 技術面
    volume_ratio: float = 0.0
    price_change_pct: float = 0.0
    rsi_14: float = 50.0
    macd_bullish: bool = False
    ma_aligned: bool = False
    breakout_ma20: bool = False
    breakout_ma60: bool = False
    consecutive_green: int = 0
    volume_surge: bool = False
    near_52w_low: bool = False

    # 籌碼面
    inst_buy_3d: bool = False
    margin_decrease: bool = False

    # 基本面
    revenue_growth: bool = False
    pe_reasonable: bool = False

    # ===== 價格預測區 =====
    # 阻力位 (從近到遠)
    resistance_1: float = 0.0     # 第一道阻力
    resistance_2: float = 0.0     # 第二道阻力
    resistance_3: float = 0.0     # 第三道阻力

    # 支撐位
    support_1: float = 0.0        # 短線支撐 (停損參考)
    support_2: float = 0.0        # 強力支撐

    # 目標價
    target_short: float = 0.0     # 短期目標 (1-5天)
    target_medium: float = 0.0    # 中期目標 (1-2週)
    target_peak: float = 0.0      # 波段頂點 (1個月)

    # 波動率
    atr_14: float = 0.0           # 14日ATR
    avg_volume: float = 0.0       # 20日均量

    # 評級
    confidence: str = ""           # 預測信心: 高/中/低
    risk_reward: float = 0.0      # 盈虧比 (目標漲幅/止損跌幅)
    upside_pct: float = 0.0       # 短期預期漲幅%

    signals: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _get_fetcher():
    return sys.modules.get("data_fetcher")


def _find_swing_points(close: np.ndarray, lookback: int = 30) -> tuple:
    """找出最近的波段高低點"""
    recent = close[-lookback:]
    if len(recent) < 5:
        return float(np.max(recent)), float(np.min(recent))

    # 簡單方法: local max/min
    highs = []
    lows = []
    for i in range(2, len(recent) - 2):
        if recent[i] > recent[i-1] and recent[i] > recent[i-2] and recent[i] > recent[i+1] and recent[i] > recent[i+2]:
            highs.append(recent[i])
        if recent[i] < recent[i-1] and recent[i] < recent[i-2] and recent[i] < recent[i+1] and recent[i] < recent[i+2]:
            lows.append(recent[i])

    if not highs:
        highs.append(float(np.max(recent)))
    if not lows:
        lows.append(float(np.min(recent)))

    return float(max(highs)), float(min(lows))


def _calc_atr(df) -> float:
    """計算14日ATR"""
    if len(df) < 15:
        return 0.0
    high = df["High"].values if "High" in df.columns else df["Close"].values
    low = df["Low"].values if "Low" in df.columns else df["Close"].values
    close = df["Close"].values

    tr = []
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr.append(max(hl, hc, lc))

    if len(tr) < 14:
        return float(np.mean(tr))
    return float(np.mean(tr[-14:]))


def _calc_fibonacci_extensions(close: np.ndarray, lookback: int = 60) -> dict:
    """計算費氏擴展目標價"""
    recent = close[-lookback:]
    if len(recent) < 10:
        return {}

    swing_low = float(np.min(recent))
    swing_high = float(np.max(recent))
    swing_range = swing_high - swing_low

    if swing_range <= 0:
        return {}

    return {
        "1.272": swing_low + swing_range * 1.272,
        "1.382": swing_low + swing_range * 1.382,
        "1.618": swing_low + swing_range * 1.618,
        "2.000": swing_low + swing_range * 2.000,
        "low": swing_low,
        "high": swing_high,
    }


def predict_price_target(c, close, volume, df_with_idx, ma20, ma60_val):
    """精準預測目標價位 — 結合技術分析、波動率、型態學"""
    latest_close = c.price

    # ATR (波動率基準)
    atr = _calc_atr(df_with_idx)
    c.atr_14 = round(atr, 3)
    atr_pct = atr / latest_close * 100 if latest_close > 0 else 0

    # 20日均量
    avg_vol = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
    c.avg_volume = round(avg_vol, 0)

    # 波段高低點
    swing_high_20, swing_low_20 = _find_swing_points(close, 20)
    swing_high_60, swing_low_60 = _find_swing_points(close, 60)

    # 52週高低 (用完整資料)
    year_high = float(np.max(close[-250:])) if len(close) >= 250 else swing_high_60
    year_low = float(np.min(close[-250:])) if len(close) >= 250 else swing_low_60

    # 費氏擴展
    fib = _calc_fibonacci_extensions(close, 60)

    # ===== 阻力位計算 =====
    resistances = []

    # 1. 近20日波高
    if swing_high_20 > latest_close * 1.005:
        resistances.append(("近20日高點", swing_high_20))

    # 2. 60MA (如果目前股價在以下)
    if ma60_val > latest_close * 1.01:
        resistances.append(("60日均線", ma60_val))

    # 3. 費氏擴展1.272
    if fib and fib["1.272"] > latest_close * 1.01:
        resistances.append(("Fib 1.272", fib["1.272"]))

    # 4. 近60日高點
    if swing_high_60 > latest_close * 1.02 and swing_high_60 not in [r[1] for r in resistances]:
        resistances.append(("近60日高點", swing_high_60))

    # 5. 費氏擴展1.382
    if fib and fib["1.382"] > latest_close * 1.02:
        resistances.append(("Fib 1.382", fib["1.382"]))

    # 排序: 由近到遠
    resistances.sort(key=lambda x: x[1])
    # 只保留 > 現價的
    resistances = [(name, val) for name, val in resistances if val > latest_close]

    if resistances:
        c.resistance_1 = round(resistances[0][1], 2)
    if len(resistances) > 1:
        c.resistance_2 = round(resistances[1][1], 2)
    if len(resistances) > 2:
        c.resistance_3 = round(resistances[2][1], 2)

    # 設預設阻力
    if c.resistance_1 == 0:
        c.resistance_1 = round(latest_close * 1.05, 2)  # +5%
    if c.resistance_2 == 0:
        c.resistance_2 = round(latest_close * 1.10, 2)  # +10%
    if c.resistance_3 == 0:
        c.resistance_3 = round(latest_close * 1.20, 2)  # +20%

    # ===== 支撐位計算 =====
    supports = []

    # 1. 近20日波低
    if swing_low_20 < latest_close * 0.99:
        supports.append(swing_low_20)

    # 2. 20MA
    if ma20 > 0 and ma20 < latest_close * 0.995:
        supports.append(ma20)

    # 3. 近60日波低
    if swing_low_60 < latest_close * 0.98:
        supports.append(swing_low_60)

    # 4. ATR基礎 (現價 - 2*ATR)
    atr_support = latest_close - 2 * atr
    if atr_support > 0:
        supports.append(atr_support)

    supports.sort(reverse=True)  # 由近到遠

    if supports:
        c.support_1 = round(supports[0], 2)
    if len(supports) > 1:
        c.support_2 = round(supports[1], 2)

    if c.support_1 == 0:
        c.support_1 = round(latest_close * 0.95, 2)
    if c.support_2 == 0:
        c.support_2 = round(latest_close * 0.90, 2)

    # ===== 目標價計算 =====

    # 短期目標 (1-5天): 第一道阻力為目標
    short_target = c.resistance_1

    # 如果突破強勢(volume_surge + 漲幅>2%)，短期可直接看第二阻力
    if c.volume_surge and c.price_change_pct > 2:
        short_target = max(short_target, c.resistance_2)
    elif c.breakout_ma60:  # 突破60MA，動能強
        short_target = max(short_target, c.resistance_2)

    # 中期目標 (1-2週): 基於費氏擴展 + 趨勢強度
    if fib and fib["1.272"] > latest_close:
        med_target = fib["1.272"]
    elif c.resistance_2 > latest_close:
        med_target = c.resistance_2
    else:
        med_target = latest_close * 1.10

    # 如果有量，中期目標可看更遠
    if c.volume_surge and c.score > 70:
        if fib and fib["1.382"] > med_target:
            med_target = fib["1.382"]

    if c.ma_aligned and c.macd_bullish:
        # 均線多頭 + MACD黃金交叉 -> 強趨勢
        if fib and fib["1.618"] > med_target:
            med_target = fib["1.618"]

    # 波段頂點 (1個月): 看52週高或費氏2.0
    if c.breakout_ma60 and c.score > 70:
        peak_tgt = max(year_high, fib.get("2.000", year_high)) if fib else year_high
    elif c.macd_bullish:
        peak_tgt = max(year_high * 0.95, fib.get("1.618", year_high * 0.95)) if fib else year_high * 0.95
    else:
        peak_tgt = year_high

    # 預測不能低於現價
    c.target_short = round(max(short_target, latest_close * 1.01), 2)
    c.target_medium = round(max(med_target, c.target_short * 1.01), 2)
    c.target_peak = round(max(peak_tgt, c.target_medium * 1.01), 2)

    # ===== 盈虧比 =====
    upside = c.target_short - latest_close
    downside = latest_close - c.support_1
    if downside > 0:
        c.risk_reward = round(upside / downside, 2)
    else:
        c.risk_reward = 99  # 理論上無風險 (rare)

    # ===== 預期漲幅 =====
    c.upside_pct = round((c.target_short - latest_close) / latest_close * 100, 1)

    # ===== 信心評級 =====
    confidence_score = 0
    # 量能支持
    if c.volume_surge:
        confidence_score += 25
    # 技術突破
    if c.breakout_ma20:
        confidence_score += 10
    if c.breakout_ma60:
        confidence_score += 15
    if c.ma_aligned:
        confidence_score += 15
    if c.macd_bullish:
        confidence_score += 15
    if c.consecutive_green >= 3:
        confidence_score += 10
    # 籌碼
    if c.inst_buy_3d:
        confidence_score += 10

    if confidence_score >= 70:
        c.confidence = "高"
    elif confidence_score >= 40:
        c.confidence = "中"
    else:
        c.confidence = "低"

    return c


def analyze_low_price_candidate(stock_id: str, stock_name: str) -> LowPriceSurgeCandidate:
    """單一低價股完整分析 + 價格預測"""
    c = LowPriceSurgeCandidate(stock_id=stock_id, stock_name=stock_name)
    fetcher = _get_fetcher()

    if not fetcher:
        c.notes.append("無法載入 data_fetcher")
        return c

    try:
        df = fetcher.fetch_historical(stock_id, months=3)
        if df.empty or len(df) < 20:
            c.notes.append(f"資料不足 ({len(df)}筆)")
            return c

        close = df["Close"].values
        volume = df["Volume"].values
        latest_close = float(close[-1])
        c.price = round(latest_close, 2)

        # 價格門檻
        if latest_close >= 100:
            c.notes.append(f"股價{latest_close:.0f}已超過低價股門檻")
            c.score = 30
            return c

        # 短期動能分析
        chg_pct = (close[-1] - close[-2]) / close[-2] * 100
        c.price_change_pct = round(chg_pct, 2)

        # 量能分析
        avg_vol_20 = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / avg_vol_20 if avg_vol_20 > 0 else 0
        c.volume_ratio = round(vol_ratio, 2)
        vol_surge = vol_ratio > 1.5 and volume[-1] > volume[-2] * 1.3
        c.volume_surge = vol_surge

        # RSI
        from indicators import add_all_indicators
        df_with_idx = add_all_indicators(df)
        if "RSI_14" in df_with_idx.columns:
            rsi = float(df_with_idx["RSI_14"].iloc[-1])
            c.rsi_14 = round(rsi, 1)
        else:
            rsi = 50.0

        # 均線分析
        ma5 = np.mean(close[-5:]) if len(close) >= 5 else 0
        ma20 = np.mean(close[-20:]) if len(close) >= 20 else 0
        ma60_val = np.mean(close[-60:]) if len(close) >= 60 else 0

        if ma20 > 0 and latest_close > ma20:
            c.breakout_ma20 = True
        if ma60_val > 0 and latest_close > ma60_val:
            c.breakout_ma60 = True
        if ma5 > ma20 > ma60_val and ma20 > 0:
            c.ma_aligned = True

        # MACD 黃金交叉
        if "MACD" in df_with_idx.columns and "MACD_signal" in df_with_idx.columns:
            macd = df_with_idx["MACD"].values
            macd_sig = df_with_idx["MACD_signal"].values
            if len(macd) >= 2:
                if macd[-1] > macd_sig[-1] and macd[-2] <= macd_sig[-2]:
                    c.macd_bullish = True

        # 連紅K
        green_count = 0
        for i in range(min(10, len(close)-1), 0, -1):
            if close[-i] > close[-i-1]:
                green_count += 1
            else:
                break
        c.consecutive_green = green_count

        # 52週位置
        year_high = np.max(close[-250:]) if len(close) >= 250 else np.max(close)
        year_low = np.min(close[-250:]) if len(close) >= 250 else np.min(close)
        if year_high > year_low:
            price_position = (latest_close - year_low) / (year_high - year_low) * 100
            if price_position < 25:
                c.near_52w_low = True

        # 籌碼面
        try:
            from fundamentals import fetch_institutional_trading
            inst = fetch_institutional_trading(stock_id)
            if not inst.empty:
                recent = inst.tail(3)
                foreign_net = recent["foreign_net"].sum() if "foreign_net" in recent.columns else 0
                if foreign_net > 0:
                    c.inst_buy_3d = True
        except:
            pass

        # 基本面
        try:
            from fundamentals import fetch_fundamentals
            fund = fetch_fundamentals(stock_id)
            if fund:
                pe = fund.get("pe_ratio") or 0
                if 0 < pe < 15:
                    c.pe_reasonable = True
                rev = fund.get("revenue_yoy_growth") or 0
                if rev > 0:
                    c.revenue_growth = True
        except:
            pass

        # ===== 價格預測 =====
        c = predict_price_target(c, close, volume, df_with_idx, ma20, ma60_val)

        # ===== 評分系統 (滿分100) =====
        score = 0.0
        signals_list = []

        # 1. 量能異常 (最高20分)
        if vol_surge:
            vol_score = min(20, vol_ratio * 5)
            score += vol_score
            signals_list.append(f"爆量{vol_ratio:.1f}倍")

        # 2. 價格動能 (最高20分)
        if chg_pct > 0:
            score += min(20, chg_pct * 3)
        if chg_pct > 3:
            signals_list.append(f"漲{chg_pct:.1f}%")
        elif chg_pct > 1.5:
            signals_list.append("穩步上漲")

        # 3. RSI 位置 (最高15分)
        if 30 <= rsi <= 40:
            score += 10
            signals_list.append("RSI低檔即將翻多")
        elif 40 < rsi <= 55:
            score += 8
            signals_list.append("RSI中性偏多")
        elif 55 < rsi <= 70:
            score += 12
            signals_list.append("RSI強勢區")
        elif rsi > 70:
            score += 5
            signals_list.append("RSI過熱")

        # 4. 技術突破 (最高15分)
        if c.ma_aligned:
            score += 8
            signals_list.append("均線多頭排列")
        if c.macd_bullish:
            score += 7
            signals_list.append("MACD黃金交叉")

        # 5. 連續收紅 (最高10分)
        score += min(10, green_count * 2)
        if green_count >= 3:
            signals_list.append(f"連{green_count}紅")

        # 6. 籌碼面 (最高10分)
        if c.inst_buy_3d:
            score += 5
            signals_list.append("外資/投信買超")
        if c.near_52w_low:
            score += 5
            signals_list.append("低檔盤整待突破")

        # 7. 基本面 (最高10分)
        if c.revenue_growth:
            score += 5
            signals_list.append("營收成長")
        if c.pe_reasonable:
            score += 5
            signals_list.append("本益比合理")

        # 8. 目標價信心加分 (最高5分)
        if c.risk_reward >= 2.0:
            score += 3
            signals_list.append(f"盈虧比{c.risk_reward:.1f}")
        if c.confidence == "高":
            score += 2
            signals_list.append("預測信心高")

        score = min(100, score)
        c.score = round(score, 1)

        # 等級判定
        if c.score >= 90:
            c.grade = "A"
        elif c.score >= 75:
            c.grade = "B"
        elif c.score >= 60:
            c.grade = "C"
        else:
            c.grade = "D"

        c.signals = signals_list[:6]
        c.notes = c.notes[:4]

    except Exception as e:
        c.notes.append(f"分析異常: {str(e)[:60]}")

    return c


def scan_low_price_surge(top_n: int = 20, max_workers: int = 10) -> list:
    """掃描全部低價股，回傳排序後的候選清單"""
    fetcher = _get_fetcher()
    if not fetcher:
        return []

    _ensure_universe_populated()

    candidates = []
    universe = list(LOW_PRICE_UNIVERSE.items())
    total = len(universe)
    start = time.time()

    print(f"[FIRE] 開始掃描 {total} 檔低價潛力股...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(analyze_low_price_candidate, sid, name): (sid, name)
            for sid, name in universe
        }
        for i, future in enumerate(as_completed(futures), 1):
            sid, name = futures[future]
            try:
                c = future.result()
                if c.score >= 20:
                    candidates.append(c)
            except Exception:
                pass
            if i % 20 == 0:
                elapsed = time.time() - start
                print(f"  進度: {i}/{total} | 已篩出 {len(candidates)} 檔 | {elapsed:.0f}s")

    candidates.sort(key=lambda x: x.score, reverse=True)
    grade_a = sum(1 for c in candidates if c.grade == "A")
    grade_b = sum(1 for c in candidates if c.grade == "B")
    grade_c = sum(1 for c in candidates if c.grade == "C")
    print(f"[OK] 掃描完成! {total} 檔中找到 {len(candidates)} 檔符合條件")
    print(f"  A級(90+): {grade_a} | B級(75+): {grade_b} | C級(60+): {grade_c}")

    return candidates[:top_n]


def format_candidates_table(candidates: list) -> pd.DataFrame:
    """將候選清單格式化為 DataFrame (含價格預測)"""
    rows = []
    for c in candidates:
        # 目標價標籤
        if c.confidence == "高":
            conf_tag = "!HIGH"
        elif c.confidence == "中":
            conf_tag = "-MED-"
        else:
            conf_tag = " low "

        rows.append({
            "等級": c.grade,
            "代號": c.stock_id,
            "名稱": c.stock_name,
            "股價": c.price,
            "評分": c.score,
            "短期目標": c.target_short,
            "中期目標": c.target_medium,
            "波段目標": c.target_peak,
            "漲幅%": c.upside_pct,
            "信心": conf_tag,
            "盈虧比": c.risk_reward,
            "停損": c.support_1,
            "漲跌幅%": f"{c.price_change_pct:+.1f}",
            "量比": c.volume_ratio,
            "RSI": c.rsi_14,
            "信號": ", ".join(c.signals[:4]),
        })
    return pd.DataFrame(rows)
