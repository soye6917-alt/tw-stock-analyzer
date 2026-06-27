"""
professional_regime.py — 市場狀態分類器

目標：判斷大盤處在什麼階段，給出對應的交易策略建議

方法：
1. 均線排列 (MA5/MA20/MA60/MA120)
2. 波動率 (ATR/布林通道寬度)
3. 趨勢強度 (ADX)
4. 成交量趨勢
5. 相對強度 (創新高/新低比例)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

def classify_market_regime(df: pd.DataFrame) -> dict:
    """
    Classify market into one of 6 regimes.
    
    Requires: ohlcv data with at least 120 periods.
    """
    if df is None or len(df) < 120:
        return {'regime': 'unknown', 'confidence': 0, 'reason': '資料不足'}
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
    # Calculate MAs
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    ma120 = pd.Series(close).rolling(120).mean().values
    
    latest = {
        'close': close[-1],
        'ma5': ma5[-1],
        'ma20': ma20[-1],
        'ma60': ma60[-1],
        'ma120': ma120[-1],
    }
    
    # 1. Trend direction
    short_trend = close[-1] > ma20[-1]  # Price above 20MA
    mid_trend = close[-1] > ma60[-1]    # Price above 60MA
    long_trend = close[-1] > ma120[-1]  # Price above 120MA
    
    ma_bullish = ma5[-1] > ma20[-1] > ma60[-1]  # Perfect bullish alignment
    ma_bearish = ma5[-1] < ma20[-1] < ma60[-1]  # Perfect bearish alignment
    
    # 2. Volatility (Bollinger Band width)
    bb_mid = pd.Series(close).rolling(20).mean()
    bb_std = pd.Series(close).rolling(20).std()
    bb_width = ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std)) / bb_mid
    current_volatility = bb_width.iloc[-1]
    avg_volatility = bb_width.mean()
    vol_regime = 'high' if current_volatility > avg_volatility * 1.2 else 'low'
    
    # 3. Trend strength (ADX-like)
    price_change_20d = (close[-1] / close[-20] - 1) * 100 if len(close) > 20 else 0
    price_change_60d = (close[-1] / close[-60] - 1) * 100 if len(close) > 60 else 0
    
    # 4. Volume trend
    vol_ma20 = pd.Series(volume).rolling(20).mean().values
    vol_trend = 'increasing' if volume[-5:].mean() > vol_ma20[-5] * 1.1 else                 'decreasing' if volume[-5:].mean() < vol_ma20[-5] * 0.9 else 'neutral'
    
    # 5. Regime classification
    scores = {}
    
    # Strong Bull (多頭主升段)
    if ma_bullish and short_trend and price_change_20d > 3 and vol_trend != 'decreasing':
        if price_change_60d > 15:
            scores['bull_mature'] = 80  # 末升段
        else:
            scores['bull_early'] = 85   # 初升段
        scores['bull'] = 70
    
    # Bull with consolidation (多頭盤整)
    if short_trend and mid_trend and abs(price_change_20d) < 5:
        scores['bull_consolidation'] = 65
    
    # Sideways (盤整)
    if abs(price_change_20d) < 3 and abs(price_change_60d) < 10:
        if ma_bullish:
            scores['bull_consolidation'] = 60
        elif ma_bearish:
            scores['bear_consolidation'] = 55
        else:
            scores['sideways'] = 70
    
    # Correction (多頭修正)
    if long_trend and not short_trend and price_change_20d < -3:
        scores['correction'] = 75
    
    # Bear (空頭)
    if ma_bearish and price_change_20d < -5:
        scores['bear'] = 85
        if price_change_60d < -15:
            scores['bear_steep'] = 80  # 主跌段
    
    # Volatile (高波動盤整)
    if vol_regime == 'high' and abs(price_change_20d) < 5:
        scores['volatile'] = 60
    
    # Determine best match
    if not scores:
        return {
            'regime': 'neutral',
            'confidence': 30,
            'indicators': latest,
            'trend_strength': round(price_change_20d, 1),
            'volatility': round(float(current_volatility * 100), 1),
            'volume_trend': vol_trend,
            'advice': '盤勢不明，建議觀望或縮小部位'
        }
    
    best_regime = max(scores, key=scores.get)
    confidence = scores[best_regime]
    
    # Human-readable names
    regime_names = {
        'bull_early': '多頭初升段 📈',
        'bull_mature': '多頭末升段 🔥',
        'bull': '多頭行情 📈',
        'bull_consolidation': '多頭盤整 ⏸️',
        'correction': '多頭修正 📉',
        'sideways': '橫向盤整 ↔️',
        'bear_consolidation': '空頭盤整 ⏸️',
        'bear': '空頭行情 📉',
        'bear_steep': '主跌段 💀',
        'volatile': '高波動盤整 ⚡'
    }
    
    # Strategy advice per regime
    regime_advice = {
        'bull_early': '積極布局，拉回加碼。主流股優先，追突破。',
        'bull_mature': '持有但逐步減碼。移動停損收緊，不追高。',
        'bull': '順勢操作，拉回買進。停損設在20MA下方。',
        'bull_consolidation': '區間操作，低買高賣。突破區間再加碼。',
        'correction': '降低持股，現金為王。等止跌訊號再進場。',
        'sideways': '觀望或極小部位測試。不追突破，等明確方向。',
        'bear_consolidation': '反彈減碼。不摸底，等底部型態完成。',
        'bear': '現金為王。不接刀，不出手。',
        'bear_steep': '全力防守。任何反彈都是賣點。',
        'volatile': '縮小部位，擴大停損。等波動率下降。'
    }
    
    return {
        'regime': best_regime,
        'regime_name': regime_names.get(best_regime, best_regime),
        'confidence': confidence,
        'indicators': {
            'close': round(latest['close'], 2),
            'ma5': round(latest['ma5'], 2),
            'ma20': round(latest['ma20'], 2),
            'ma60': round(latest['ma60'], 2),
            'ma120': round(latest['ma120'], 2),
            'alignment': '多頭排列' if ma_bullish else '空頭排列' if ma_bearish else '交叉',
            'above_ma20': short_trend,
            'above_ma60': mid_trend,
            'above_ma120': long_trend,
        },
        'trend_strength': round(price_change_20d, 1),
        'volatility': round(float(current_volatility * 100), 1),
        'volume_trend': vol_trend,
        'advice': regime_advice.get(best_regime, '觀望')
    }


def get_market_breadth(stock_list: list) -> dict:
    """
    Analyze market breadth from a list of stock dataframes.
    Returns percentage of stocks in uptrend.
    """
    if not stock_list:
        return {'uptrend_pct': 0, 'total': 0}
    
    uptrend = 0
    for df in stock_list:
        if df is not None and len(df) > 20:
            if df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1]:
                uptrend += 1
    
    total = len(stock_list)
    return {
        'uptrend_stocks': uptrend,
        'total_stocks': total,
        'uptrend_pct': round(uptrend / total * 100, 1) if total > 0 else 0
    }


# =========== Quick Test ===========
if __name__ == '__main__':
    print('Market Regime Classifier 已載入')
    print('用法: classify_market_regime(df_with_ohlcv)')
