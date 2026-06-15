"""
📡 選股篩選器
自訂條件篩選上市櫃股票：價位、本益比、殖利率、成交量、技術訊號
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from data_fetcher import fetch_stock_list, fetch_historical, POPULAR_STOCKS, SESSION
from indicators import add_all_indicators, get_indicator_signals
from fundamentals import fetch_fundamentals


def get_twse_listed_stocks():
    """取得所有上市股票代碼清單（TWSE API）"""
    try:
        # TWSE 上市股票清單
        url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        resp = SESSION.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            stocks = {}
            for item in data:
                sid = item.get("Code", "").strip()
                sname = item.get("Name", "").strip()
                if sid and sname:
                    stocks[sid] = sname
            return stocks
    except Exception:
        pass
    # fallback: 用 POPULAR_STOCKS + fetch_stock_list
    base = fetch_stock_list()
    base.update(POPULAR_STOCKS)
    return base


def filter_stocks(
    price_min=0, price_max=9999,
    pe_min=0, pe_max=999,
    dy_min=0, dy_max=100,
    vol_min=0,
    tech_signal=None,  # "多頭", "空頭", "中立"
    months=6,
    max_stocks=200,
):
    """
    篩選條件：
    - price_min ~ price_max: 股價區間
    - pe_min ~ pe_max: 本益比區間
    - dy_min ~ dy_max: 殖利率區間 (%)
    - vol_min: 最低成交量（張）
    - tech_signal: 技術型態（多頭/空頭/中立/None=全部）
    - max_stocks: 最多掃描檔數
    """
    stocks = list(get_twse_listed_stocks().items())
    results = []

    # 先取前 max_stocks 檔（含熱門股優先）
    priority = {sid: i for i, sid in enumerate(POPULAR_STOCKS.keys())}
    stocks.sort(key=lambda x: priority.get(x[0], 999))

    total = min(len(stocks), max_stocks)
    for idx, (sid, sname) in enumerate(stocks[:max_stocks]):
        try:
            df = fetch_historical(sid, months=months)
            if df.empty:
                continue

            df = add_all_indicators(df)
            last = df.iloc[-1]
            price = last.get("Close", 0)

            # 股價篩選
            if price < price_min or price > price_max:
                continue

            # 成交量篩選（張）
            volume = last.get("Volume", 0)
            if volume < vol_min:
                continue

            # 技術訊號篩選
            sig = get_indicator_signals(df)
            if tech_signal == "多頭" and sig.get("overall", "中立") != "多頭":
                continue
            if tech_signal == "空頭" and sig.get("overall", "中立") != "空頭":
                continue
            if tech_signal == "中立" and sig.get("overall", "中立") != "中立":
                continue

            # 基本面篩選（較慢，後置）
            fund = fetch_fundamentals(sid)
            pe = fund.get("pe_ratio", 0) or 0
            dy = fund.get("dividend_yield", 0) or 0

            if pe > 0 and (pe < pe_min or pe > pe_max):
                continue
            if dy > 0 and (dy < dy_min or dy > dy_max):
                continue

            # 漲跌幅
            chg = last.get("Change", 0)
            chg_pct = (chg / (price - chg)) * 100 if price - chg != 0 else 0

            results.append({
                "sid": sid,
                "name": sname,
                "price": round(price, 2),
                "change_pct": round(chg_pct, 2),
                "volume": int(volume),
                "pe": round(pe, 2) if pe else None,
                "div_yield": round(dy, 2) if dy else None,
                "signal": sig.get("overall", "中立"),
                "ma5_ma10": sig.get("ma5_ma10", ""),
                "rsi": round(last.get("RSI", 50), 1),
            })

            time.sleep(0.05)

        except Exception:
            continue

    # 排序：分數高的在前面
    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return pd.DataFrame(results)
