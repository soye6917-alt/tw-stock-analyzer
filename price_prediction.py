"""
📈 AI 自我意識價格預測模組 — 專業分析師多面向判斷
使用多因子加權模型預測半個月股價高低點
"""

import numpy as np
import pandas as pd


def _safe(val, default=0):
    return val if val is not None and not (isinstance(val, float) and np.isnan(val)) else default


def predict_price_path(df, days: int = 15):
    """
    專業多面向預測未來 days 個交易日股價路徑。
    綜合技術面 6 大因子加權判斷趨勢方向與強度。
    """
    if df.empty or len(df) < 30:
        return {"error": "歷史資料不足，需至少 30 筆交易資料"}

    close = df["Close"].values
    log_close = np.log(close)
    n = len(close)
    last = df.iloc[-1]
    current_price = float(close[-1])

    # ========================================================================
    # 因子 1：均線排列（確認趨勢方向與強度）
    # ========================================================================
    ma5 = _safe(last.get("MA5", 0))
    ma10 = _safe(last.get("MA10", 0))
    ma20 = _safe(last.get("MA20", 0))
    ma60 = _safe(last.get("MA60", 0))
    
    ma_score = 0
    ma_notes = []
    if ma5 > ma10 > ma20 and (ma60 == 0 or current_price > ma60):
        ma_score = 20
        ma_notes.append("均線多頭排列（MA5>MA10>MA20>MA60）")
    elif ma5 < ma10 < ma20 and (ma60 == 0 or current_price < ma60):
        ma_score = -20
        ma_notes.append("均線空頭排列（MA5<MA10<MA20<MA60）")
    elif current_price > ma20:
        ma_score = 5
        ma_notes.append("股價站上月線")
    elif current_price < ma20:
        ma_score = -5
        ma_notes.append("股價跌破月線")
    else:
        ma_notes.append("均線糾結，方向不明")
    
    # 均線乖離
    if ma20 > 0:
        ma20_dev = (current_price - ma20) / ma20 * 100
        if ma20_dev > 10:
            ma_score -= 5
            ma_notes.append(f"⚠️ 股價偏離月線 {ma20_dev:.1f}%（潛在修正壓力）")
        elif ma20_dev < -10:
            ma_score += 5
            ma_notes.append(f"🔄 股價低於月線 {abs(ma20_dev):.1f}%（潛在反彈機會）")
    
    # ========================================================================
    # 因子 2：RSI 動能分析
    # ========================================================================
    rsi = _safe(last.get("RSI", 50))
    rsi_score = 0
    rsi_notes = []
    
    # 短線 RSI 趨勢（最近 3 天 RSI 變化）
    recent_rsi = df["RSI"].dropna().values[-5:] if "RSI" in df.columns else None
    
    if rsi > 80:
        rsi_score = -15
        rsi_notes.append(f"RSI({rsi:.0f}) 嚴重超買區，短線過熱，拉回風險高")
    elif rsi > 70:
        rsi_score = -8
        rsi_notes.append(f"RSI({rsi:.0f}) 超買區，短線可能面臨獲利了結")
    elif rsi > 60:
        rsi_score = 5
        rsi_notes.append(f"RSI({rsi:.0f}) 偏強，動能仍在")
    elif rsi > 40:
        rsi_score = 0
        rsi_notes.append(f"RSI({rsi:.0f}) 中性區間")
    elif rsi > 30:
        rsi_score = 5
        rsi_notes.append(f"RSI({rsi:.0f}) 偏弱，短線有反彈空間")
    elif rsi > 20:
        rsi_score = 10
        rsi_notes.append(f"RSI({rsi:.0f}) 超賣區，醞釀反彈")
    else:
        rsi_score = 15
        rsi_notes.append(f"RSI({rsi:.0f}) 嚴重超賣，歷史經驗多為底部區")
    
    # RSI 背馳判斷
    if recent_rsi is not None and len(recent_rsi) >= 3:
        # 價格創新高但 RSI 沒創新高 → 頂背馳（看跌）
        price_high_last = close[-1] >= np.max(close[-10:-1])
        rsi_high_last = recent_rsi[-1] >= np.max(recent_rsi[:-1])
        if price_high_last and not rsi_high_last:
            rsi_score -= 10
            rsi_notes.append("⚠️ RSI 頂背馳：價格新高但動能衰減")
        # 價格創新低但 RSI 沒創新低 → 底背馳（看漲）
        price_low_last = close[-1] <= np.min(close[-10:-1])
        rsi_low_last = recent_rsi[-1] <= np.min(recent_rsi[:-1])
        if price_low_last and not rsi_low_last:
            rsi_score += 10
            rsi_notes.append("💡 RSI 底背馳：價格新低但動能轉強")
    
    # ========================================================================
    # 因子 3：MACD 趨勢確認
    # ========================================================================
    macd_score = 0
    macd_notes = []
    macd_line = _safe(last.get("MACD", 0))
    macd_signal = _safe(last.get("MACD_Signal", 0))
    macd_hist = _safe(last.get("MACD_Hist", 0))
    
    if macd_line > macd_signal and macd_hist > 0:
        # MACD 柱狀圖也翻正 → 多頭動能增強
        hist_trend = df["MACD_Hist"].dropna().values[-3:] if "MACD_Hist" in df.columns else None
        if hist_trend is not None and len(hist_trend) >= 2 and hist_trend[-1] > hist_trend[-2]:
            macd_score = 15
            macd_notes.append("MACD 黃金交叉 + 柱狀圖遞增，多頭動能轉強")
        else:
            macd_score = 8
            macd_notes.append("MACD 黃金交叉，偏多")
    elif macd_line < macd_signal and macd_hist < 0:
        hist_trend = df["MACD_Hist"].dropna().values[-3:] if "MACD_Hist" in df.columns else None
        if hist_trend is not None and len(hist_trend) >= 2 and hist_trend[-1] < hist_trend[-2]:
            macd_score = -15
            macd_notes.append("MACD 死亡交叉 + 柱狀圖遞減，空頭動能轉強")
        else:
            macd_score = -8
            macd_notes.append("MACD 死亡交叉，偏空")
    elif macd_line > macd_signal and macd_hist < 0:
        macd_score = 3
        macd_notes.append("MACD 線在訊號線之上但柱狀圖收斂，中性偏多")
    elif macd_line < macd_signal and macd_hist > 0:
        macd_score = -3
        macd_notes.append("MACD 線在訊號線之下但柱狀圖收斂，中性偏空")
    else:
        macd_notes.append("MACD 方向不明")
    
    # ========================================================================
    # 因子 4：布林通道位置
    # ========================================================================
    bb_score = 0
    bb_notes = []
    bb_upper = _safe(last.get("BB_Upper", 0))
    bb_lower = _safe(last.get("BB_Lower", 0))
    bb_mid = _safe(last.get("BB_Mid", 0) or ma20)
    
    if bb_upper > 0 and bb_lower > 0:
        bb_width = (bb_upper - bb_lower) / bb_mid * 100
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if bb_upper != bb_lower else 50
        
        if bb_position > 95:
            bb_score = -12
            bb_notes.append(f"股價觸及布林上緣（位置 {bb_position:.0f}%），短線過熱")
        elif bb_position > 80:
            bb_score = -5
            bb_notes.append(f"股價靠近布林上緣（{bb_position:.0f}%）")
        elif bb_position < 5:
            bb_score = 12
            bb_notes.append(f"股價觸及布林下緣（位置 {bb_position:.0f}%），超賣反彈機會")
        elif bb_position < 20:
            bb_score = 5
            bb_notes.append(f"股價靠近布林下緣（{bb_position:.0f}%）")
        else:
            bb_notes.append(f"布林通道中軌附近（{bb_position:.0f}%）")
        
        # 通道寬度：寬 = 趨勢明確，窄 = 噴出前兆
        if bb_width < 5:
            bb_score += 5
            bb_notes.append(f"⚡ 布林通道收窄（{bb_width:.1f}%），可能即將突破")
    else:
        bb_notes.append("布林通道資料不足")
    
    # ========================================================================
    # 因子 5：成交量分析
    # ========================================================================
    vol_score = 0
    vol_notes = []
    if "Volume" in df.columns:
        recent_vol = df["Volume"].dropna().values[-5:].mean() if len(df) >= 5 else 0
        avg_vol = df["Volume"].dropna().values[-20:].mean() if len(df) >= 20 else recent_vol
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            if vol_ratio > 1.5:
                vol_score = 8
                vol_notes.append(f"近 5 日均量為 20 日均量 {vol_ratio:.1f} 倍，量能擴增")
            elif vol_ratio > 1.2:
                vol_score = 3
                vol_notes.append(f"近 5 日量能微幅增溫（{vol_ratio:.1f}x）")
            elif vol_ratio < 0.6:
                vol_score = -5
                vol_notes.append(f"近 5 日量能萎縮至 {vol_ratio:.1f}x，動能不足")
            else:
                vol_notes.append(f"量能平穩（{vol_ratio:.1f}x 均值）")
            
            # 價量配合
            price_trend = (close[-1] / close[-5]) - 1 if len(close) >= 6 else 0
            if price_trend > 0.02 and vol_ratio > 1.2:
                vol_score += 5
                vol_notes.append("價量齊揚，多頭健康")
            elif price_trend < -0.02 and vol_ratio > 1.2:
                vol_score -= 5
                vol_notes.append("⚠️ 價跌量增，空頭壓力")
    else:
        vol_notes.append("無成交量數據")
    
    # ========================================================================
    # 因子 6：線性迴歸趨勢
    # ========================================================================
    lookback = min(60, n)
    x = np.arange(lookback)
    y = log_close[-lookback:]
    coeffs = np.polyfit(x, y, 1)
    trend_slope = coeffs[0]  # 每日 log 回報
    
    # R² 計算
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 斜率轉年化
    annualized = trend_slope * 252
    reg_score = np.clip(trend_slope * 1000, -15, 15)
    if annualized > 0.15:
        trend_label = "📈 強勢上漲"
    elif annualized > 0.08:
        trend_label = "📈 溫和上漲"
    elif annualized > -0.08:
        trend_label = "➡️ 盤整"
    elif annualized > -0.15:
        trend_label = "📉 溫和下跌"
    else:
        trend_label = "📉 強勢下跌"
    
    # ========================================================================
    # 綜合加權評分（-100 ~ +100）
    # ========================================================================
    total_score = ma_score * 0.20 + rsi_score * 0.20 + macd_score * 0.20 + bb_score * 0.15 + vol_score * 0.10 + reg_score * 0.15
    
    # 信心度：看因子共識度 （分歧大 -> 信心低）
    scores = [ma_score, rsi_score, macd_score, bb_score, vol_score, reg_score]
    consensus = 1 - (np.std(scores) / 30)  # 0~1, 越高越一致
    consensus = np.clip(consensus, 0, 1)
    
    # 結合 R² 作為信心基礎
    if r_squared > 0.5 and consensus > 0.6:
        confidence = "高"
    elif r_squared > 0.2 and consensus > 0.4:
        confidence = "中"
    else:
        confidence = "低"
    
    # ========================================================================
    # 價格路徑投影（因子調整後）
    # ========================================================================
    # ATR 計算
    high, low = df["High"].values[-lookback:], df["Low"].values[-lookback:]
    if n > lookback:
        prev_c = close[-lookback-1:-1]
    else:
        prev_c = np.full(lookback, close[0])
    tr = np.maximum(high - low, np.abs(high - prev_c), np.abs(low - prev_c))
    atr = float(np.mean(tr[-14:]))
    
    # 因子調整後的趨勢斜率
    adjusted_slope = trend_slope * (1 + total_score / 500)  # total_score 調幅
    
    # 投影 15 天
    predicted = []
    for i in range(1, days + 1):
        trend_return = adjusted_slope * i
        # 波動遞增
        vol_attenuation = 1.0 - (i / days) * 0.3  # 遠期信心衰減
        price = current_price * np.exp(trend_return) * (1 + np.random.normal(0, 0.005) * (1 - vol_attenuation))
        predicted.append(round(float(price), 2))
    
    # 用中位數去雜訊
    smooth_pred = []
    for i in range(days):
        window = predicted[max(0, i-2):min(days, i+3)]
        smooth_pred.append(round(float(np.median(window)), 2))
    
    pred_array = np.array(smooth_pred)
    buy_idx = int(np.argmin(pred_array))
    sell_idx = int(np.argmax(pred_array))
    
    expected_change_pct = round((pred_array[-1] / current_price - 1) * 100, 1)
    high_low_range_pct = round((pred_array.max() / pred_array.min() - 1) * 100, 1)
    
    # ========================================================================
    # 分析師專業判斷摘要
    # ========================================================================
    factor_lines = []
    factor_lines.append(f"**均線**（權重 20%）：{' + '.join(ma_notes)}")
    factor_lines.append(f"**RSI**（權重 20%）：{' + '.join(rsi_notes)}")
    factor_lines.append(f"**MACD**（權重 20%）：{' + '.join(macd_notes)}")
    factor_lines.append(f"**布林通道**（權重 15%）：{' + '.join(bb_notes)}")
    factor_lines.append(f"**成交量**（權重 10%）：{' + '.join(vol_notes)}")
    factor_lines.append(f"**線性趨勢**（權重 15%）：斜率 {trend_slope*100:.3f}%/日，R²={r_squared:.3f}")
    factor_lines.append(f"**綜合評分：** {total_score:+.0f}/100，信心度：{confidence}")
    
    # 買賣建議總結
    if total_score > 30:
        strategy_note = "✅ 多因子共振偏多，可考慮積極布局"
    elif total_score > 10:
        strategy_note = "📈 短線偏多，逢回可布局"
    elif total_score > -10:
        strategy_note = "⚖️ 多空交戰，建議觀望或小額試單"
    elif total_score > -30:
        strategy_note = "📉 短線偏空，反彈減碼"
    else:
        strategy_note = "⚠️ 多因子共振偏空，建議保守操作"
    
    return {
        "last_price": round(current_price, 2),
        "predicted_prices": smooth_pred,
        "high_point": {
            "day": f"第 {sell_idx + 1} 天",
            "price": round(float(pred_array[sell_idx]), 2),
        },
        "low_point": {
            "day": f"第 {buy_idx + 1} 天",
            "price": round(float(pred_array[buy_idx]), 2),
        },
        "expected_change_pct": expected_change_pct,
        "high_low_range_pct": high_low_range_pct,
        "trend_label": trend_label,
        "trend_slope": round(trend_slope * 100, 3),
        "atr": round(atr, 3),
        "total_score": round(total_score),
        "r_squared": round(r_squared, 3),
        "confidence": confidence,
        "strategy_note": strategy_note,
        "factor_analysis": factor_lines,
    }


def format_prediction_card(pred):
    """將預測結果轉為專業分析師 HTML 卡片"""
    if "error" in pred:
        return f'<div style="color:#ff8a80;">❌ {pred["error"]}</div>'

    hp = pred["high_point"]
    lp = pred["low_point"]
    arrow = "📈" if pred["expected_change_pct"] > 0 else "📉" if pred["expected_change_pct"] < 0 else "➡️"
    
    conf_color = "#69f0ae" if pred["confidence"] == "高" else "#ffd740" if pred["confidence"] == "中" else "#ff5252"
    conf_bg = "rgba(105,240,174,0.15)" if pred["confidence"] == "高" else "rgba(255,215,64,0.15)" if pred["confidence"] == "中" else "rgba(255,82,82,0.15)"

    # 因子條
    total_score = pred.get("total_score", 0)
    score_bar_width = (total_score + 100) / 2  # 0~200% 映射

    card = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a3e 0%, #2a1a4e 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 12px 0;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        border: 1px solid rgba(255,255,255,0.08);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 18px; font-weight: 700;">🧠 AI 分析師預測<span style="font-size:12px;opacity:0.6;margin-left:8px;">（半個月觀點 • 多面向加權）</span></span>
            <span style="font-size: 13px; background: {conf_bg}; color: {conf_color}; padding: 4px 14px; border-radius: 20px; font-weight:600;">
                信心度: {pred['confidence']}
            </span>
        </div>
        
        <!-- 綜合評分條 -->
        <div style="margin-bottom: 12px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; opacity:0.7; margin-bottom:2px;">
                <span>偏空</span>
                <span>綜合評分 <strong style="font-size:16px; color:{'#69f0ae' if total_score>0 else '#ff5252' if total_score<0 else '#fff'};">
                    {total_score:+.0f}</strong>/100</span>
                <span>偏多</span>
            </div>
            <div style="height:6px; background: rgba(255,255,255,0.15); border-radius:3px;">
                <div style="width: {score_bar_width:.0f}%; height:100%; 
                    background: linear-gradient(90deg, {'#ff5252' if total_score<0 else '#69f0ae'}, {'#ffd740' if abs(total_score)<15 else '#69f0ae'}); 
                    border-radius:3px; transition: width 0.3s;"></div>
            </div>
        </div>
        
        <!-- 高低點預測 -->
        <div style="display: flex; justify-content: space-around; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">
            <div style="text-align: center; background: rgba(0,200,83,0.15); border-radius: 12px; padding: 14px 20px; min-width: 130px; flex:1;">
                <div style="font-size: 12px; opacity: 0.7;">🟢 買點（低）</div>
                <div style="font-size: 28px; font-weight: 700; color: #69f0ae;">{lp["price"]}</div>
                <div style="font-size: 12px; opacity: 0.6;">{lp["day"]}</div>
            </div>
            <div style="text-align: center; background: rgba(255,23,68,0.15); border-radius: 12px; padding: 14px 20px; min-width: 130px; flex:1;">
                <div style="font-size: 12px; opacity: 0.7;">🔴 賣點（高）</div>
                <div style="font-size: 28px; font-weight: 700; color: #ff5252;">{hp["price"]}</div>
                <div style="font-size: 12px; opacity: 0.6;">{hp["day"]}</div>
            </div>
        </div>
        
        <!-- 數據列 -->
        <div style="margin-bottom: 10px; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; font-size: 13px; opacity: 0.8;">
            <span>{arrow} 預估漲跌: <strong>{pred['expected_change_pct']:+.1f}%</strong></span>
            <span>↕️ 振幅: <strong>{pred['high_low_range_pct']:.1f}%</strong></span>
            <span>📉 ATR: {pred['atr']}</span>
            <span>R²: {pred['r_squared']:.3f}</span>
        </div>
        
        <!-- 策略建議 -->
        <div style="
            margin-top: 8px; 
            padding: 10px 16px; 
            background: rgba(255,255,255,0.06); 
            border-radius: 10px;
            font-size: 14px;
            text-align: center;
            border-left: 3px solid {'#69f0ae' if total_score>10 else '#ff5252' if total_score<-10 else '#ffd740'};
        ">
            <strong>{pred['trend_label']}</strong> · {pred.get('strategy_note', '')}
        </div>
    </div>
    """
    return card
