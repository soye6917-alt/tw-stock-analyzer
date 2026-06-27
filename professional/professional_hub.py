"""
professional_hub.py — 專業分析整合核心

整合 data/news/trading/regime 四大模組
產出完整的個股分析報告與市場快訊
"""
import sys, json, datetime, os
from typing import Optional
import pandas as pd

# Add parent to path for data_fetcher
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def professional_stock_analysis(stock_id: str, stock_name: str = '', capital: float = 100000):
    """
    Generate a complete professional analysis for any stock.
    Returns a dict with all analysis dimensions.
    """
    from data_fetcher import fetch_historical as fetch_stock_data, fetch_realtime_quote
    from professional_data import fetch_professional_data
    from professional_news import get_news_sentiment
    from professional_trading import calculate_entry_score, evaluate_exit
    from professional_trading import PositionSizing, calculate_position_size
    from professional_regime import classify_market_regime
    
    report = {
        'stock_id': stock_id,
        'stock_name': stock_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'sections': {}
    }
    
    # 1. 即時報價
    quote = fetch_realtime_quote(stock_id)
    if quote:
        report['sections']['realtime'] = {
            'price': float(quote.get('price', 0)),
            'open': float(quote.get('open', 0)),
            'high': float(quote.get('high', 0)),
            'low': float(quote.get('low', 0)),
            'volume': int(quote.get('volume', 0)),
            'bid': float(quote.get('bid', 0)),
            'ask': float(quote.get('ask', 0)),
            'range_pct': round((float(quote.get('high', 0)) - float(quote.get('low', 0))) / 
                              float(quote.get('price', 1)) * 100, 2)
        }
    
    # 2. 技術分析 + 市場狀態
    df = fetch_stock_data(stock_id, months=6)
    if df is not None and len(df) > 20:
        # Add technical indicators
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['volume_ma20'] = df['Volume'].rolling(20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Market regime
        regime = classify_market_regime(df)
        report['sections']['market_regime'] = regime
        
        # Technical snapshot
        latest = df.iloc[-1]
        report['sections']['technical'] = {
            'close': round(latest['Close'], 2),
            'ma5': round(latest['MA5'], 2),
            'ma20': round(latest['MA20'], 2),
            'ma60': round(latest['MA60'], 2),
            'rsi': round(latest['RSI'], 1),
            'volume_vs_avg': round(latest['Volume'] / df['Volume'].mean() * 100, 1) if df['Volume'].mean() > 0 else 0,
            'trend_20d': round((latest['Close'] / df['Close'].iloc[-20] - 1) * 100, 2) if len(df) >= 20 else 0,
        }
        
        # Entry checklist
        entry = calculate_entry_score(df)
        report['sections']['entry_analysis'] = entry
        
        # Exit evaluation (if we had entry price)
        if quote:
            exit_analysis = evaluate_exit(
                df, 
                entry_price=float(quote.get('price', 0)),
                current_price=float(quote.get('price', 0)),
                stop_loss=float(quote.get('price', 0)) * 0.95,
                target=float(quote.get('price', 0)) * 1.1
            )
            report['sections']['exit_analysis'] = exit_analysis
    
    # 3. 新聞情緒
    news = get_news_sentiment(stock_id, stock_name)
    report['sections']['news_sentiment'] = {
        'total_news': news['total_news'],
        'sentiment_score': news['sentiment_score'],
        'sentiment_label': news['sentiment_label'],
        'summary': news['summary']
    }
    
    # 4. 倉位建議
    if quote and df is not None:
        price = float(quote.get('price', 0))
        if price > 0:
            ps = PositionSizing(
                capital=capital,
                risk_per_trade=2,
                entry_price=price,
                stop_loss=round(price * 0.93, 2)  # 7% stop
            )
            sizing = calculate_position_size(ps, 'fixed_risk')
            report['sections']['position_sizing'] = sizing
    
    # 5. 綜合評語
    strengths = []
    weaknesses = []
    
    if report['sections'].get('entry_analysis', {}).get('total_score', 0) > 60:
        strengths.append('技術面進場訊號偏多')
    else:
        weaknesses.append('技術面進場訊號偏弱')
    
    if report['sections'].get('news_sentiment', {}).get('sentiment_score', 0) > 0.2:
        strengths.append('新聞情緒偏正面')
    elif report['sections'].get('news_sentiment', {}).get('sentiment_score', 0) < -0.2:
        weaknesses.append('新聞情緒偏負面')
    
    if report['sections'].get('market_regime', {}).get('confidence', 0) > 60:
        regime_name = report['sections']['market_regime'].get('regime_name', '')
        strengths.append(f'市場狀態明確：{regime_name}')
    
    report['summary'] = {
        'strengths': strengths,
        'weaknesses': weaknesses,
        'overall': '偏多' if len(strengths) >= len(weaknesses) else '偏空' if len(weaknesses) > len(strengths) else '中性'
    }
    
    return report


def generate_market_briefing(stock_ids: list = None) -> dict:
    """
    Generate a daily market briefing with key stocks.
    """
    if stock_ids is None:
        stock_ids = ['2618', '2382', '2330', '2603', '2303']
    
    briefing = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.datetime.now().strftime('%H:%M'),
        'stocks': {},
        'market_summary': ''
    }
    
    for sid in stock_ids:
        try:
            report = professional_stock_analysis(sid)
            briefing['stocks'][sid] = {
                'price': report['sections'].get('realtime', {}).get('price', 'N/A'),
                'sentiment': report['sections'].get('news_sentiment', {}).get('sentiment_label', 'neutral'),
                'regime': report['sections'].get('market_regime', {}).get('regime_name', 'unknown'),
                'entry_score': report['sections'].get('entry_analysis', {}).get('total_score', 0),
                'summary': report.get('summary', {})
            }
        except:
            continue
    
    return briefing


if __name__ == '__main__':
    # Test
    report = professional_stock_analysis('2618', '長榮航', capital=200000)
    print(json.dumps({k: v for k, v in report.items() if k != 'sections'}, 
                     ensure_ascii=False, indent=2))
    print(f'\nSections: {list(report["sections"].keys())}')
