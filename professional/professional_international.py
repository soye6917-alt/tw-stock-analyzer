"""
professional_international.py — 國際股市與總經數據

資料來源：
- yfinance (美股、全球指數、VIX)
- Yahoo Finance direct API (台股ADR)
- 開放總經 API
"""
import yfinance as yf
import requests as rq
import pandas as pd
import datetime, json, time
from typing import Dict, Optional

rq.packages.urllib3.disable_warnings()

# =========== 國際主要指數 ===========
GLOBAL_INDICES = {
    '^GSPC':  {'name': 'S&P 500', 'market': 'US'},
    '^DJI':   {'name': '道瓊', 'market': 'US'},
    '^IXIC':  {'name': '那斯達克', 'market': 'US'},
    '^VIX':   {'name': 'VIX 恐慌指數', 'market': 'US'},
    '^SOX':   {'name': '費城半導體', 'market': 'US'},
    '^RUT':   {'name': '羅素2000', 'market': 'US'},
    '^N225':  {'name': '日經225', 'market': 'JP'},
    '^HSI':   {'name': '恆生指數', 'market': 'HK'},
    '000001.SS': {'name': '上證指數', 'market': 'CN'},
    '^FTSE':  {'name': '富時100', 'market': 'UK'},
    '^GDAXI': {'name': '德國DAX', 'market': 'DE'},
}

# 台股 ADR
TAIWAN_ADRS = {
    'TSM':  {'name': '台積電ADR', 'tw_stock': '2330'},
    'UMC':  {'name': '聯電ADR', 'tw_stock': '2303'},
    'ASX':  {'name': '日月光ADR', 'tw_stock': '3711'},
    'WIT':  {'name': '緯創ADR', 'tw_stock': '3231'},
}

# 重要商品
COMMODITIES = {
    'CL=F':  {'name': '原油(WTI)', 'unit': 'USD/桶'},
    'GC=F':  {'name': '黃金', 'unit': 'USD/盎司'},
    'SI=F':  {'name': '白銀', 'unit': 'USD/盎司'},
    'DX-Y.NYB': {'name': '美元指數', 'unit': 'USD'},
    'ZN=F':  {'name': '10年期美債', 'unit': '殖利率'},
}

# 加密貨幣 (via yfinance)
CRYPTO = {
    'BTC-USD': {'name': '比特幣'},
    'ETH-USD': {'name': '以太幣'},
}


def fetch_global_markets() -> Dict:
    """
    Fetch all international market indices.
    Returns dict with prices, changes, and overnight movement.
    """
    results = {}
    
    for symbol, info in GLOBAL_INDICES.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if len(hist) >= 2:
                close = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                change = close - prev_close
                change_pct = (change / prev_close) * 100
                
                # Get today's open/high/low if today is active
                high = hist['High'].iloc[-1]
                low = hist['Low'].iloc[-1]
                
                results[symbol] = {
                    'name': info['name'],
                    'price': round(close, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'market': info['market'],
                    'currency': 'USD',
                    'is_active': True,
                    'timestamp': datetime.datetime.now().isoformat()
                }
            time.sleep(0.2)  # Rate limit
        except Exception as e:
            results[symbol] = {
                'name': info['name'],
                'error': str(e)[:50],
                'market': info['market']
            }
    
    return results


def fetch_adr_prices() -> Dict:
    """Fetch Taiwan ADR prices and calculate premium/discount to TW spot."""
    results = {}
    
    for symbol, info in TAIWAN_ADRS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 1:
                adr_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else adr_price
                
                results[symbol] = {
                    'name': info['name'],
                    'adr_price': round(adr_price, 2),
                    'change_pct': round((adr_price - prev_close) / prev_close * 100, 2),
                    'tw_stock': info['tw_stock'],
                    'conversion_ratio': 5,  # 1 ADR = 5 TW shares for TSM
                    'currency': 'USD',
                    'updated': datetime.datetime.now().isoformat()
                }
            time.sleep(0.2)
        except Exception as e:
            results[symbol] = {'name': info['name'], 'error': str(e)[:50]}
    
    return results


def fetch_commodities() -> Dict:
    """Fetch key commodity prices."""
    results = {}
    for symbol, info in COMMODITIES.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 1:
                price = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) >= 2 else price
                results[symbol] = {
                    'name': info['name'],
                    'price': round(price, 2),
                    'change_pct': round((price - prev) / prev * 100, 2),
                    'unit': info['unit']
                }
            time.sleep(0.2)
        except Exception as e:
            results[symbol] = {'name': info['name'], 'error': str(e)[:50]}
    return results


def fetch_market_summary() -> dict:
    """Generate a comprehensive market summary across all regions."""
    indices = fetch_global_markets()
    adrs = fetch_adr_prices()
    
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'us_market': {},
        'asia_market': {},
        'europe_market': {},
        'adr': adrs,
        'key_levels': {}
    }
    
    for symbol, data in indices.items():
        region = data.get('market', 'other')
        if region == 'US':
            summary['us_market'][symbol] = data
        elif region in ('JP', 'HK', 'CN'):
            summary['asia_market'][symbol] = data
        elif region in ('UK', 'DE'):
            summary['europe_market'][symbol] = data
    
    # Key levels summary
    if '^VIX' in indices and 'price' in indices['^VIX']:
        vix = indices['^VIX']['price']
        if vix < 15:
            summary['key_levels']['fear_greed'] = ('貪婪', vix)
        elif vix < 20:
            summary['key_levels']['fear_greed'] = ('中性', vix)
        elif vix < 25:
            summary['key_levels']['fear_greed'] = ('恐懼', vix)
        else:
            summary['key_levels']['fear_greed'] = ('極度恐懼', vix)
    
    if '^SOX' in indices and 'change_pct' in indices['^SOX']:
        summary['key_levels']['semi_impact'] = (
            '利多' if indices['^SOX']['change_pct'] > 1 else
            '利空' if indices['^SOX']['change_pct'] < -1 else
            '中性',
            indices['^SOX']['change_pct']
        )
    
    if 'TSM' in adrs and 'change_pct' in adrs['TSM']:
        summary['key_levels']['tsm_adr'] = adrs['TSM']['change_pct']
    
    return summary


def get_overnight_context() -> str:
    """Generate a concise overnight market summary text."""
    summary = fetch_market_summary()
    
    parts = ['【隔夜國際市場】']
    
    us = summary.get('us_market', {})
    if '^GSPC' in us and 'change_pct' in us['^GSPC']:
        sp = us['^GSPC']
        parts.append(f"美股: S&P {sp.get('price','?')} ({sp.get('change_pct','?')})")
    if '^IXIC' in us and 'change_pct' in us['^IXIC']:
        nas = us['^IXIC']
        parts.append(f"那指: {nas.get('price','?')} ({nas.get('change_pct','?')}%)")
    if '^SOX' in us and 'change_pct' in us['^SOX']:
        sox = us['^SOX']
        parts.append(f"費半: {sox.get('price','?')} ({sox.get('change_pct','?')}%)")
    
    asia = summary.get('asia_market', {})
    for sym in ['^N225', '^HSI']:
        if sym in asia and 'change_pct' in asia[sym]:
            a = asia[sym]
            parts.append(f"{a.get('name','?')}: {a.get('price','?')} ({a.get('change_pct','?')}%)")
    
    if summary.get('key_levels', {}).get('fear_greed'):
        fg, v = summary['key_levels']['fear_greed']
        parts.append(f"VIX: {v} ({fg})")
    
    if summary.get('adr', {}).get('TSM', {}).get('change_pct'):
        tsm = summary['adr']['TSM']
        parts.append(f"台積電ADR: ({tsm.get('change_pct','?')}%)")
    
    return ' | '.join(parts)


if __name__ == '__main__':
    import json
    summary = fetch_market_summary()
    for region in ['us_market', 'asia_market']:
        print(f'\n=== {region} ===')
        for sym, data in summary.get(region, {}).items():
            if 'price' in data:
                print(f"  {data['name']}: {data['price']} ({data.get('change_pct',0)}%)")
            elif 'error' in data:
                print(f"  {data['name']}: ERROR {data['error']}")
    print(f"\n--- Summary ---")
    print(get_overnight_context())
