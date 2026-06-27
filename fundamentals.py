"""
基本面與籌碼分析模組
- 本益比、殖利率、淨值比 (TWSE BWIBBU_d)
- 三大法人買賣超 (TWSE T86)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})


def fetch_fundamentals(stock_id: str) -> dict:
    """
    取得基本面資料：本益比、殖利率、股價淨值比
    API 一次回傳全市場資料，在此過濾
    盤中查無今日資料時自動回退到最後交易日
    """
    # 從今天開始，往前試最多 5 個交易日
    base = datetime.now()
    for offset in range(7):
        d = base - timedelta(days=offset)
        if d.weekday() >= 5:
            continue  # 跳過週末
        date_str = d.strftime("%Y%m%d")
        url = (
            f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
            f"?date={date_str}&stockNo={stock_id}&response=json"
        )
        try:
            resp = SESSION.get(url, timeout=10)
            data = resp.json()
            if data.get("stat") != "OK":
                continue  # 這天沒資料，換前一天
            # 在回傳列表中找目標股票
            for row in data.get("data", []):
                if row[0] == stock_id:
                    # fields: 代號, 名稱, 收盤價, 殖利率(%), 股利年度, 本益比, 股價淨值比, 財報年度/季
                    return {
                        "stock_id": stock_id,
                        "name": row[1].strip(),
                        "close": float(row[2].replace(",", "")),
                        "dividend_yield": float(row[3]) if row[3] not in ["-", "0.00"] else None,
                        "dividend_year": row[4],
                        "pe_ratio": float(row[5]) if row[5] not in ["-", "0.00"] else None,
                        "pb_ratio": float(row[6]) if row[6] not in ["-", "0.00"] else None,
                        "report_season": row[7],
                    }
            return {"error": f"無 {stock_id} 的基本面資料"}
        except Exception:
            continue
    return {"error": "無法取得基本面資料（連試 7 天皆失敗）"}


def fetch_institutional_trading(stock_id: str, days: int = 5) -> pd.DataFrame:
    """
    取得三大法人買賣超
    API 回傳全市場資料，在此過濾
    盤中查無今日資料時自動回退到最後交易日
    注意: 必須加上 selectType=ALL 否則只回傳前 8 筆
    """
    base = datetime.now()
    for offset in range(7):
        d = base - timedelta(days=offset)
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        url = (
            f"https://www.twse.com.tw/rwd/zh/fund/T86"
            f"?date={date_str}&stockNo={stock_id}&response=json&selectType=ALL"
        )
        try:
            resp = SESSION.get(url, timeout=10)
            data = resp.json()
            if data.get("stat") != "OK":
                continue
            # 過濾出目標股票
            for row in data.get("data", []):
                if row[0] == stock_id:
                    return _parse_institutional_row(row)
            return pd.DataFrame()  # 有資料但沒找到這檔股票
        except Exception as e:
            print(f"Institutional fetch error ({date_str}): {e}")
            continue
    return pd.DataFrame()


def _parse_institutional_row(row: list) -> pd.DataFrame:
    """
    解析三大法人資料列
    fields: 代號, 名稱,
    外資買進(不含自營), 外資賣出(不含自營), 外資買賣超(不含自營),
    外資自營買進, 外資自營賣出, 外資自營買賣超,
    投信買進, 投信賣出, 投信買賣超,
    自營商買賣超,
    自營商買進(自行), 自營商賣出(自行), 自營商買賣超(自行),
    自營商買進(避險), 自營商賣出(避險), 自營商買賣超(避險),
    三大法人買賣超
    """
    if len(row) < 19:
        return pd.DataFrame()
    
    def to_int(val):
        try:
            return int(val.replace(",", ""))
        except (ValueError, AttributeError):
            return 0
    
    records = []
    
    # 外資
    foreign_buy = to_int(row[2])
    foreign_sell = to_int(row[3])
    foreign_net = to_int(row[4])
    records.append({"類別": "外資(不含自營)", "買進": foreign_buy, "賣出": foreign_sell, "買賣超": foreign_net})
    
    # 投信
    inv_buy = to_int(row[8])
    inv_sell = to_int(row[9])
    inv_net = to_int(row[10])
    records.append({"類別": "投信", "買進": inv_buy, "賣出": inv_sell, "買賣超": inv_net})
    
    # 自營商(合計)
    self_net = to_int(row[11])
    self_buy = to_int(row[12]) + to_int(row[15])
    self_sell = to_int(row[13]) + to_int(row[16])
    records.append({"類別": "自營商", "買進": self_buy, "賣出": self_sell, "買賣超": self_net})
    
    # 三大合計
    total_net = to_int(row[18])
    total_buy = foreign_buy + inv_buy + self_buy
    total_sell = foreign_sell + inv_sell + self_sell
    records.append({"類別": "三大法人合計", "買進": total_buy, "賣出": total_sell, "買賣超": total_net})
    
    df = pd.DataFrame(records)
    # 轉換為張 (除以1000)
    for col in ["買進", "賣出", "買賣超"]:
        df[col + "(張)"] = (df[col] / 1000).round(0).astype(int)
    return df


def generate_analysis_summary(stock_id: str, stock_name: str, tech_df: pd.DataFrame) -> list:
    """
    綜合技術+基本面分析摘要
    """
    from indicators import get_indicator_signals
    
    lines = []
    
    if tech_df.empty or len(tech_df) < 20:
        lines.append("⚠️ 技術資料不足，無法產生分析")
        return lines
    
    signals = get_indicator_signals(tech_df)
    latest = tech_df.iloc[-1]
    close = latest["Close"]
    
    # 價格趨勢
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    
    # 趨勢判斷
    if ma5 > ma20 > ma60:
        trend = "多頭排列 📈"
        trend_desc = "短中期均線多頭排列，趨勢偏多"
    elif ma5 < ma20 < ma60:
        trend = "空頭排列 📉"
        trend_desc = "短中期均線空頭排列，趨勢偏空"
    else:
        trend = "盤整 ⚖️"
        trend_desc = "均線交錯，盤整格局"
    
    lines.append(f"【趨勢】{trend} — {trend_desc}")
    
    # 價格位置
    if ma20 > 0:
        pct_from_ma20 = (close - ma20) / ma20 * 100
        if pct_from_ma20 > 10:
            lines.append(f"⚡ 股價高於月線 {pct_from_ma20:.1f}%，短線可能過熱")
        elif pct_from_ma20 < -10:
            lines.append(f"💡 股價低於月線 {abs(pct_from_ma20):.1f}%，短線可能超跌")
        else:
            lines.append(f"📊 股價在月線附近 ({pct_from_ma20:+.1f}%)")
    
    # RSI 解讀
    rsi = latest.get("RSI", 50)
    if rsi > 70:
        lines.append(f"⚠️ RSI {rsi:.0f} — 超買區，留意拉回風險")
    elif rsi < 30:
        lines.append(f"💡 RSI {rsi:.0f} — 超賣區，留意反彈機會")
    else:
        lines.append(f"✓ RSI {rsi:.0f} — 中性區間")
    
    # MACD 解讀
    macd_val = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    if macd_val > macd_signal:
        if macd_val > 0:
            lines.append("🟢 MACD 正値且站上訊號線，動能偏多")
        else:
            lines.append("🟡 MACD 負値但站上訊號線，動能改善中")
    else:
        if macd_val < 0:
            lines.append("🔴 MACD 負値且在訊號線下，動能偏空")
        else:
            lines.append("🟡 MACD 正値但跌破訊號線，動能減弱")
    
    # 布林通道
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    bb_mid = latest.get("BB_Mid", 0)
    if bb_upper > 0:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pos > 90:
            lines.append(f"🔴 股價觸及布林上軌 ({bb_pos:.0f}%)，短線壓力")
        elif bb_pos < 10:
            lines.append(f"🟢 股價觸及布林下軌 ({bb_pos:.0f}%)，短線支撐")
    
    # 基本面（如果有）
    fund = fetch_fundamentals(stock_id)
    pe = None
    dy = None
    pb_val = None
    if "error" not in fund:
        pe = fund.get("pe_ratio")
        dy = fund.get("dividend_yield")
        pb_val = fund.get("pb_ratio")
        if pe:
            pe_text = "偏低" if pe < 15 else "合理" if pe < 25 else "偏高"
            lines.append(f"📋 本益比 {pe:.1f} ({pe_text})")
        if dy:
            dy_text = "高" if dy > 5 else "尚可" if dy > 3 else "偏低"
            lines.append(f"📋 殖利率 {dy:.1f}% ({dy_text})")
        if pb_val:
            lines.append(f"📋 股價淨值比 {pb_val:.2f}")
    
    lines.append("\n" + "─" * 30)
    lines.append("⚠️ 以上為技術面客觀分析，不構成買賣建議")
    
    return lines


def assess_recommendation(stock_id: str, stock_name: str, tech_df: pd.DataFrame) -> dict:
    """
    評分並給出買賣建議
    回傳: {"rating": "買進"/"中立"/"賣出", "score": int, "details": [str,...]}
    """
    from indicators import get_indicator_signals
    
    result = {
        "rating": "中立",
        "score": 0,
        "bullish": [],
        "bearish": [],
        "details": [],
    }
    
    if tech_df.empty or len(tech_df) < 20:
        result["details"].append("⚠️ 資料不足，無法評估")
        return result
    
    score = 0
    latest = tech_df.iloc[-1]
    close = latest["Close"]
    
    # === 技術面評分（權重：-30 ~ +30）===
    # 均線排列
    ma5 = latest.get("MA5", 0)
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    if ma5 > ma20 > ma60:
        score += 20
        result["bullish"].append("均線多頭排列 (+20)")
    elif ma5 < ma20 < ma60:
        score -= 20
        result["bearish"].append("均線空頭排列 (-20)")
    else:
        # 檢查是否 partial bullish
        if ma5 > ma20:
            score += 5
            result["bullish"].append("短線站上月線 (+5)")
        if ma20 > ma60:
            score += 5
            result["bullish"].append("月線站上季線 (+5)")
    
    # RSI
    rsi = latest.get("RSI", 50)
    if rsi < 30:
        score += 15
        result["bullish"].append(f"RSI {rsi:.0f} 超賣區 (+15)")
    elif rsi > 70:
        score -= 15
        result["bearish"].append(f"RSI {rsi:.0f} 超買區 (-15)")
    elif rsi < 40:
        score += 5
        result["bullish"].append(f"RSI {rsi:.0f} 偏低 (+5)")
    elif rsi > 60:
        score -= 5
        result["bearish"].append(f"RSI {rsi:.0f} 偏高 (-5)")
    
    # MACD
    macd_val = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    if macd_val > macd_signal:
        score += 10
        result["bullish"].append("MACD 站上訊號線 (+10)")
    else:
        score -= 10
        result["bearish"].append("MACD 跌破訊號線 (-10)")
    
    # MACD 柱狀圖（動能增減）
    hist = latest.get("MACD_Hist", 0)
    if len(tech_df) > 1:
        prev_hist = tech_df["MACD_Hist"].iloc[-2]
        if hist > prev_hist and hist > 0:
            score += 5
            result["bullish"].append("MACD 動能增強 (+5)")
        elif hist < prev_hist and hist < 0:
            score -= 5
            result["bearish"].append("MACD 動能減弱 (-5)")
    
    # 布林通道位置
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if bb_upper > 0:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pos > 90:
            score -= 10
            result["bearish"].append(f"觸及布林上軌 ({bb_pos:.0f}%) 過熱 (-10)")
        elif bb_pos < 10:
            score += 10
            result["bullish"].append(f"觸及布林下軌 ({bb_pos:.0f}%) 超跌 (+10)")
    
    # 成交量
    if "Volume" in tech_df.columns and len(tech_df) > 5:
        avg_vol = tech_df["Volume"].iloc[-6:-1].mean()
        cur_vol = tech_df["Volume"].iloc[-1]
        if avg_vol > 0 and cur_vol > avg_vol * 1.5:
            # 價漲量增 / 價跌量增
            if close >= tech_df["Close"].iloc[-2]:
                score += 5
                result["bullish"].append("價漲量增 (+5)")
            else:
                score -= 5
                result["bearish"].append("價跌量增 (-5)")
    
    # === 基本面評分（權重：-15 ~ +15）===
    fund = fetch_fundamentals(stock_id)
    if "error" not in fund:
        pe = fund.get("pe_ratio")
        dy = fund.get("dividend_yield")
        if pe:
            if pe < 12:
                score += 10
                result["bullish"].append(f"本益比 {pe:.1f} 偏低 (+10)")
            elif pe < 18:
                score += 5
                result["bullish"].append(f"本益比 {pe:.1f} 合理偏低 (+5)")
            elif pe > 40:
                score -= 10
                result["bearish"].append(f"本益比 {pe:.1f} 過高 (-10)")
            elif pe > 25:
                score -= 5
                result["bearish"].append(f"本益比 {pe:.1f} 偏高 (-5)")
        if dy:
            if dy > 6:
                score += 5
                result["bullish"].append(f"殖利率 {dy:.1f}% 優異 (+5)")
            elif dy > 4:
                score += 3
                result["bullish"].append(f"殖利率 {dy:.1f}% 不錯 (+3)")
    
    # === 三大法人評分（權重：-10 ~ +10）===
    inst = fetch_institutional_trading(stock_id)
    if not inst.empty:
        total_row = inst[inst["類別"] == "三大法人合計"]
        if not total_row.empty:
            net = total_row["買賣超"].values[0]
            if net > 0:
                net_k = net / 1000
                if net_k > 10:
                    score += 10
                    result["bullish"].append(f"三大法人買超 {net_k:.0f} 張 (+10)")
                elif net_k > 1:
                    score += 5
                    result["bullish"].append(f"三大法人買超 {net_k:.0f} 張 (+5)")
                else:
                    score += 2
                    result["bullish"].append(f"三大法人小幅買超 (+2)")
            elif net < 0:
                net_k = abs(net) / 1000
                if net_k > 10:
                    score -= 10
                    result["bearish"].append(f"三大法人賣超 {net_k:.0f} 張 (-10)")
                elif net_k > 1:
                    score -= 5
                    result["bearish"].append(f"三大法人賣超 {net_k:.0f} 張 (-5)")
                else:
                    score -= 2
                    result["bearish"].append(f"三大法人小幅賣超 (-2)")
    
    # === 最終評級 ===
    score = max(-100, min(100, score))
    result["score"] = score
    
    if score >= 30:
        result["rating"] = "強烈買進"
    elif score >= 15:
        result["rating"] = "買進"
    elif score <= -30:
        result["rating"] = "強烈賣出"
    elif score <= -15:
        result["rating"] = "賣出"
    else:
        result["rating"] = "中立觀望"
    
    # === 價格目標計算 ===
    price_targets = _calculate_price_targets(tech_df, latest, close)
    result["price_targets"] = price_targets
    
    # 加總說明
    lines = []
    lines.append(f"📊 **綜合評分：{score:+.0f} / 100**")
    lines.append(f"🏆 **建議：{result['rating']}**")
    if score >= 30:
        lines.append("🔔 多項指標偏多，可考慮布局")
    elif score >= 15:
        lines.append("📈 部分指標偏多，可留意進場機會")
    elif score <= -30:
        lines.append("🔴 多項指標偏空，建議避開")
    elif score <= -15:
        lines.append("📉 部分指標偏空，建議減碼")
    else:
        lines.append("⚖️ 多空指標交錯，建議觀望等待方向")
    
    lines.append("")
    if result["bullish"]:
        lines.append("**🟢 多方訊號：**")
        for b in result["bullish"]:
            lines.append(f"  • {b}")
    if result["bearish"]:
        lines.append("**🔴 空方訊號：**")
        for b in result["bearish"]:
            lines.append(f"  • {b}")
    
    lines.append("")
    lines.append("**🎯 價格區間參考：**")
    if price_targets.get("buy_zones"):
        lines.append(f"  • 買進區間：{_fmt_price(price_targets['buy_zones'][0][0])} ~ {_fmt_price(price_targets['buy_zones'][-1][1])}")
    if price_targets.get("sell_zones"):
        lines.append(f"  • 賣出區間：{_fmt_price(price_targets['sell_zones'][0][0])} ~ {_fmt_price(price_targets['sell_zones'][-1][1])}")
    if price_targets.get("stop_loss"):
        lines.append(f"  • 停損參考：跌破 {_fmt_price(price_targets['stop_loss'])}")
    lines.append(f"  • 關鍵壓力：{_fmt_price(price_targets.get('strong_resistance', close))}")
    lines.append(f"  • 關鍵支撐：{_fmt_price(price_targets.get('strong_support', close))}")
    
    lines.append("\n" + "─" * 30)
    lines.append("⚠️ 此為量化指標評估，不構成買賣建議")
    
    result["details"] = lines
    return result


def _fmt_price(val: float) -> str:
    """格式化價格"""
    if val >= 100:
        return f"${val:.1f}"
    elif val >= 10:
        return f"${val:.2f}"
    else:
        return f"${val:.3f}"


def _calculate_price_targets(tech_df: pd.DataFrame, latest: pd.Series, close: float) -> dict:
    """
    計算建議的買進/賣出價格區間
    """
    result = {
        "current_price": close,
        "buy_zones": [],     # [(low, high), ...]
        "sell_zones": [],    # [(low, high), ...]
        "stop_loss": None,
        "strong_support": None,
        "strong_resistance": None,
    }
    
    if tech_df.empty or len(tech_df) < 10:
        return result
    
    # === 支撐位（越往下越強）===
    supports = []
    
    # 1. BB 下軌
    bb_l = latest.get("BB_Lower")
    if bb_l and bb_l > 0 and bb_l < close:
        supports.append(("布林下軌", bb_l))
    
    # 2. MA20
    ma20 = latest.get("MA20")
    if ma20 and ma20 > 0 and ma20 < close:
        supports.append(("月線(MA20)", ma20))
    
    # 3. MA60
    ma60 = latest.get("MA60")
    if ma60 and ma60 > 0 and ma60 < close:
        supports.append(("季線(MA60)", ma60))
    
    # 4. 近20日最低價
    recent_low = tech_df["Low"].tail(20).min()
    if recent_low < close:
        supports.append(("近20日低點", recent_low))
    
    # 5. 近60日最低價
    if len(tech_df) >= 60:
        low_60 = tech_df["Low"].tail(60).min()
        if low_60 < close and (not supports or low_60 < supports[-1][1]):
            supports.append(("近60日低點", low_60))
    
    # === 壓力位（越往上越強）===
    resistances = []
    
    # 1. BB 上軌
    bb_u = latest.get("BB_Upper")
    if bb_u and bb_u > 0 and bb_u > close:
        resistances.append(("布林上軌", bb_u))
    
    # 2. MA20 (如果股價在MA20之下)
    if ma20 and ma20 > 0 and ma20 > close:
        resistances.append(("月線(MA20)", ma20))
    
    # 3. MA60 (如果股價在MA60之下)
    if ma60 and ma60 > 0 and ma60 > close:
        resistances.append(("季線(MA60)", ma60))
    
    # 4. 近20日最高價
    recent_high = tech_df["High"].tail(20).max()
    if recent_high > close:
        resistances.append(("近20日高點", recent_high))
    
    # 5. 近60日最高價
    if len(tech_df) >= 60:
        high_60 = tech_df["High"].tail(60).max()
        if high_60 > close and (not resistances or high_60 > resistances[-1][1]):
            resistances.append(("近60日高點", high_60))
    
    # === 組合成買進/賣出區間 ===
    # 買進區間：介於支撐與現價之間
    buy_zones = []
    for name, level in supports:
        # 對每一支撐位，買進區間為支撐價 ~ 支撐價上方3%
        zone_top = level * 1.03
        if zone_top < close:
            buy_zones.append((level, zone_top))
    
    # 賣出區間：介於現價與壓力之間
    sell_zones = []
    for name, level in resistances:
        zone_bot = level * 0.97
        if zone_bot > close:
            sell_zones.append((zone_bot, level))
    
    # 停損：取最強支撐（最低的那個）下方2%
    if supports:
        strongest_support = min(s[1] for s in supports)
        result["stop_loss"] = strongest_support * 0.97
    
    # 關鍵支撐/壓力
    if supports:
        result["strong_support"] = min(s[1] for s in supports)
    if resistances:
        result["strong_resistance"] = max(r[1] for r in resistances)
    
    result["buy_zones"] = buy_zones
    result["sell_zones"] = sell_zones
    
    return result
