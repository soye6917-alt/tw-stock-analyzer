"""
🌍 總經面數據整合模組
- 美國利率（Fed Funds Rate）
- 美元/台幣匯率
- 台指期貨資訊
- 國際股市連動
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict

# Requests session with retries
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


def fetch_usd_twd_rate() -> dict:
    """
    取得美元/台幣即時匯率（取自台銀）
    回傳買入/賣出匯率
    """
    try:
        # 台灣銀行牌告匯率
        url = "https://rate.bot.com.tw/xrt/flcsv/0/day"
        resp = _session.get(url, timeout=10)
        lines = resp.text.strip().split('\n')
        for line in lines:
            if line.startswith('USD'):
                parts = line.split(',')
                return {
                    "currency": "USD/TWD",
                    "cash_buy": float(parts[2]),
                    "cash_sell": float(parts[3]),
                    "spot_buy": float(parts[4]) if len(parts) > 4 else None,
                    "spot_sell": float(parts[5]) if len(parts) > 5 else None,
                    "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        return {"error": "無法解析匯率資料"}
    except Exception as e:
        # Fallback: 用 tw.rter.info API
        try:
            resp = _session.get("https://tw.rter.info/capi.php", timeout=10)
            data = resp.json()
            if "USD" in data and "TWD" in data["USD"]:
                rate = float(data["USD"]["TWD"]["Exrate"])
                return {
                    "currency": "USD/TWD",
                    "rate": rate,
                    "source": "rter.info",
                    "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
        except:
            pass
        return {"error": f"無法取得匯率：{str(e)}"}


def fetch_us_interest_rate() -> dict:
    """
    美國聯邦基準利率（FRED API 公開資料）
    """
    try:
        # 使用 FRED 公開 API（不需 API key）
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "FEDFUNDS",
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5,
        }
        resp = _session.get(url, params=params, timeout=10)
        data = resp.json()
        
        if "observations" in data and data["observations"]:
            latest = data["observations"][0]
            return {
                "rate": float(latest["value"]),
                "date": latest["date"],
                "source": "FRED (Federal Funds Effective Rate)",
            }
        return {"error": "無法取得利率資料"}
    except:
        # Fallback: FRED 公開 CSV
        try:
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FEDFUNDS&scale=left&cosd=2024-06-01&coed=2025-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2025-06-01&revision_date=2025-06-01&nd=1954-07-01"
            resp = _session.get(url, timeout=10)
            lines = resp.text.strip().split('\n')
            if len(lines) >= 2:
                last_line = lines[-1].strip().split(',')
                return {
                    "rate": float(last_line[1]),
                    "date": last_line[0],
                    "source": "FRED CSV",
                }
        except:
            pass
        return {"error": "無法從 FRED 取得利率資料"}


def fetch_taiwan_interest_rate() -> dict:
    """
    台灣央行重貼現率（公開資料）
    """
    try:
        # 央行利率決策公告（取自公開網頁）
        url = "https://www.cbc.gov.tw/tw/cp-1993-1-e0e3f-1.html"
        resp = _session.get(url, timeout=10)
        text = resp.text
        
        import re
        # 常見的利率數字模式
        patterns = [
            r'重貼現率[^0-9]*([0-9]+\.[0-9]+)',
            r'利率[^0-9]*([0-9]+\.[0-9]+)%',
        ]
        for p in patterns:
            match = re.search(p, text)
            if match:
                return {
                    "rate": float(match.group(1)),
                    "source": "中央銀行",
                    "note": "重貼現率",
                }
        
        # Fallback: 已知最新利率
        return {
            "rate": 2.0,  # 2024/3 之後的利率水準
            "source": "央行（參考值）",
            "note": "2024年3月起為2.0%",
        }
    except Exception as e:
        return {"error": f"無法取得台灣利率：{str(e)}"}


def fetch_taiwan_futures() -> dict:
    """
    台指期貨即時資訊（取自期交所 Mock API）
    """
    try:
        # 期交所即時行情
        url = "https://mis.taifex.com.tw/futures/api/getQuote"
        params = {"symbol": "TX"}
        resp = _session.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data:
            return {
                "futures_price": float(data.get("price", 0)),
                "change": float(data.get("change", 0)),
                "change_pct": float(data.get("changePercent", 0)),
                "volume": int(data.get("volume", 0)),
                "open": float(data.get("open", 0)),
                "high": float(data.get("high", 0)),
                "low": float(data.get("low", 0)),
                "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
    except:
        pass
    
    # API 無法使用時回傳說明
    return {
        "error": "台指期即時資料無法取得（盤後或無連線）",
        "note": "資料來源：台灣期交所公開資訊",
    }


def fetch_us_market_status() -> dict:
    """
    美國主要指數即時狀態
    """
    indices = {
        "^DJI": "道瓊工業", "^GSPC": "S&P 500", "^IXIC": "NASDAQ",
    }
    
    try:
        # 使用 Yahoo Finance (不用 API key)
        results = {}
        for symbol, name in indices.items():
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            resp = _session.get(url, timeout=10)
            data = resp.json()
            
            if "chart" in data and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                meta = result.get("meta", {})
                quotes = result.get("indicators", {}).get("quote", [{}])[0]
                close_prices = quotes.get("close", [])
                
                if close_prices and len(close_prices) >= 2:
                    current = close_prices[-1]
                    prev = close_prices[-2]
                    if current and prev:
                        change = current - prev
                        results[name] = {
                            "price": round(float(current), 2),
                            "change": round(float(change), 2),
                            "change_pct": round(float(change / prev * 100), 2),
                        }
        
        return results if results else {"error": "無法取得美國指數資料"}
    except Exception as e:
        return {"error": f"無法取得美國指數：{str(e)}"}


def get_macro_summary() -> dict:
    """
    總經綜合摘要（一次取得所有資訊）
    """
    usd_twd = fetch_usd_twd_rate()
    fed_rate = fetch_us_interest_rate()
    tw_rate = fetch_taiwan_interest_rate()
    us_market = fetch_us_market_status()
    
    summary = {
        "currency": usd_twd,
        "us_interest_rate": fed_rate,
        "tw_interest_rate": tw_rate,
        "us_market": us_market,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    return summary


def get_rate_direction() -> str:
    """
    判斷全球利率方向（寬鬆/緊縮/中性）
    """
    fed = fetch_us_interest_rate()
    if "error" in fed:
        return "無法判斷"
    
    rate = fed.get("rate", 0)
    if rate > 5:
        return "🔴 高利率環境（緊縮）"
    elif rate > 3:
        return "🟡 偏緊縮"
    elif rate > 1:
        return "🟢 中性"
    else:
        return "🟢 低利率環境（寬鬆）"
