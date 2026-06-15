"""
台灣股市資料擷取模組
- 從 TWSE/TPEx 公開 API 取得歷史股價
- 從 Goodinfo/TWSE 取得即時報價
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Optional

# ============================================================
# 台股代號對照（常用股，可動態查詢）
# ============================================================
POPULAR_STOCKS = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科",
    "2412": "中華電", "2308": "台達電", "2881": "富邦金",
    "2882": "國泰金", "2891": "中信金", "2303": "聯電",
    "2002": "中鋼", "1301": "台塑", "1303": "南亞",
    "1326": "台化", "1216": "統一", "3008": "大立光",
    "3711": "日月光投控", "3034": "聯詠", "4904": "遠傳",
    "3045": "台灣大", "5880": "合庫金",
}

# 全域股票清單快取（代號→名稱）
_STOCK_LIST_CACHE = None


def fetch_stock_list() -> dict:
    """
    從 TWSE 取得全市場股票清單（上市），回傳 {代號: 名稱}
    結果快取避免重複請求。嘗試最近 5 個交易日。
    """
    global _STOCK_LIST_CACHE
    if _STOCK_LIST_CACHE is not None:
        return _STOCK_LIST_CACHE

    today = datetime.now()
    for day_offset in range(10):
        d = today - timedelta(days=day_offset)
        date_str = d.strftime("%Y%m%d")
        url = (
            f"https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
            f"?date={date_str}&response=json"
        )
        try:
            resp = SESSION.get(url, timeout=10)
            data = resp.json()
            if data.get("stat") == "OK" and data.get("data"):
                stock_list = {}
                for row in data["data"]:
                    code = row[0]
                    name = row[1].strip()
                    stock_list[code] = name
                stock_list.update(POPULAR_STOCKS)
                _STOCK_LIST_CACHE = stock_list
                return stock_list
        except Exception:
            continue

    # 全部失敗，退回到常用股
    print("Warning: could not fetch stock list from TWSE")
    _STOCK_LIST_CACHE = dict(POPULAR_STOCKS)
    return _STOCK_LIST_CACHE


def search_stocks_by_name(keyword: str) -> list:
    """
    根據關鍵字搜尋股票名稱，回傳 [(代號, 名稱), ...]
    """
    if not keyword or not keyword.strip():
        return []

    keyword = keyword.strip()
    stock_list = fetch_stock_list()
    results = []

    for code, name in stock_list.items():
        # 精確匹配名稱
        if keyword == name:
            results.insert(0, (code, name))
        elif keyword in name:
            results.append((code, name))
        elif keyword in code:
            results.append((code, name))

    # 限制最多顯示30筆
    return results[:30]


def get_stock_name(stock_id: str) -> str:
    """取得股票名稱（先查快取清單，再查熱門股）"""
    stock_list = fetch_stock_list()
    if stock_id in stock_list:
        return stock_list[stock_id]
    if stock_id in POPULAR_STOCKS:
        return POPULAR_STOCKS[stock_id]
    return stock_id

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
})


def get_stock_name(stock_id: str) -> str:
    """取得股票名稱"""
    if stock_id in POPULAR_STOCKS:
        return POPULAR_STOCKS[stock_id]
    return stock_id


def fetch_daily_twse(stock_id: str, year: int, month: int) -> pd.DataFrame:
    """
    從 TWSE 取得個股月曆日成交資料
    https://www.twse.com.tw/exchangeReport/STOCK_DAY
    """
    date_str = f"{year}{month:02d}01"
    url = (
        f"https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        f"?response=json&date={date_str}&stockNo={stock_id}"
    )
    try:
        resp = SESSION.get(url, timeout=15)
        data = resp.json()
        if data.get("stat") != "OK":
            return pd.DataFrame()
        rows = data["data"]
        # TWSE 回傳欄位：日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數,備註
        columns = ["日期", "成交股數", "成交金額", "開盤價",
                    "最高價", "最低價", "收盤價", "漲跌價差", "成交筆數", "備註"]
        df = pd.DataFrame(rows, columns=columns)
        # 轉換日期格式：TWSE用 113/01/02 (民國年)
        def parse_tw_date(d: str) -> str:
            parts = d.split("/")
            yr = int(parts[0]) + 1911
            return f"{yr}-{parts[1]}-{parts[2]}"
        df["日期"] = df["日期"].apply(parse_tw_date)
        df["日期"] = pd.to_datetime(df["日期"])
        for col in ["開盤價", "最高價", "最低價", "收盤價"]:
            df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # 成交量也有逗號
        df["成交股數"] = df["成交股數"].str.replace(",", "", regex=False)
        df["成交股數"] = pd.to_numeric(df["成交股數"], errors="coerce")
        df = df.rename(columns={
            "開盤價": "Open", "最高價": "High", "最低價": "Low",
            "收盤價": "Close", "成交股數": "Volume",
        })
        df = df[["日期", "Open", "High", "Low", "Close", "Volume"]].copy()
        return df.sort_values("日期")
    except Exception as e:
        print(f"TWSE fetch error ({stock_id}, {year}/{month}): {e}")
        return pd.DataFrame()


def fetch_daily_tpex(stock_id: str, year: int, month: int) -> pd.DataFrame:
    """
    從 TPEx (櫃買中心) 取得上櫃股票資料
    """
    date_str = f"{year}-{month:02d}-01"
    url = (
        f"https://www.tpex.org.tw/web/stock/aftertrading/daily_trading_info/st43_result.php"
        f"?l=zh-tw&d={date_str}&stkno={stock_id}"
    )
    try:
        resp = SESSION.get(url, timeout=15)
        data = resp.json()
        if data.get("report") != "一般":
            return pd.DataFrame()
        rows = data.get("aaData", [])
        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            # TPEx returns: 日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, 漲跌
            if len(row) < 7:
                continue
            d = row[0]  # 113/01/02
            parts = d.split("/")
            yr = int(parts[0]) + 1911
            date_str2 = f"{yr}-{parts[1]}-{parts[2]}"
            try:
                records.append({
                    "日期": pd.to_datetime(date_str2),
                    "Open": float(row[3].replace(",", "")),
                    "High": float(row[4].replace(",", "")),
                    "Low": float(row[5].replace(",", "")),
                    "Close": float(row[6].replace(",", "")),
                    "Volume": int(row[1].replace(",", "")),
                })
            except (ValueError, IndexError):
                continue
        df = pd.DataFrame(records)
        return df.sort_values("日期")
    except Exception as e:
        print(f"TPEx fetch error ({stock_id}): {e}")
        return pd.DataFrame()


def fetch_historical(stock_id: str, months: int = 12) -> pd.DataFrame:
    """
    取得歷史股價資料（自動判斷上市/上櫃）
    """
    today = datetime.now()
    all_dfs = []
    for i in range(months):
        m = today.month - i
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        df = fetch_daily_twse(stock_id, y, m)
        if df.empty:
            df = fetch_daily_tpex(stock_id, y, m)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(0.3)  # 避免打太兇
    if not all_dfs:
        return pd.DataFrame()
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.drop_duplicates(subset=["日期"])
    result = result.sort_values("日期").reset_index(drop=True)
    return result


def fetch_realtime_quote(stock_id: str) -> dict:
    """
    從 TWSE 即時 API 取得最新報價
    """
    # 先試上市
    url = (
        f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
        f"?ex_ch=tse_{stock_id}.tw&json=1&delay=0"
    )
    market = "tse"
    try:
        resp = SESSION.get(url, timeout=10)
        data = resp.json()
        if data.get("msgArray") and len(data["msgArray"]) > 0:
            market = "tse"
        else:
            # 試上櫃
            url = (
                f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
                f"?ex_ch=otc_{stock_id}.tw&json=1&delay=0"
            )
            resp = SESSION.get(url, timeout=10)
            data = resp.json()
            if data.get("msgArray") and len(data["msgArray"]) > 0:
                market = "otc"
            else:
                return {"error": "查無此股票代號"}
        item = data["msgArray"][0]
        # TWSE 有時會回傳多價格串接 (e.g. "105.2000_105.1500_105.1000_")
        raw_z = item.get("z", "0")
        if "_" in raw_z:
            raw_z = raw_z.split("_")[0]
        raw_o = item.get("o", "0")
        if "_" in raw_o:
            raw_o = raw_o.split("_")[0]
        raw_h = item.get("h", "0")
        if "_" in raw_h:
            raw_h = raw_h.split("_")[0]
        raw_l = item.get("l", "0")
        if "_" in raw_l:
            raw_l = raw_l.split("_")[0]
        raw_b = item.get("b", "0")
        if "_" in raw_b:
            raw_b = raw_b.split("_")[0]
        raw_a = item.get("a", "0")
        if "_" in raw_a:
            raw_a = raw_a.split("_")[0]
        
        return {
            "stock_id": stock_id,
            "name": item.get("n", stock_id),
            "market": market,
            "price": float(raw_z) if raw_z else 0,
            "open": float(raw_o) if raw_o else 0,
            "high": float(raw_h) if raw_h else 0,
            "low": float(raw_l) if raw_l else 0,
            "volume": int(item.get("v", "0")),
            "change": float(item.get("d", "0")),
            "change_percent": float(item.get("p", "0")),
            "bid": float(raw_b) if raw_b else 0,
            "ask": float(raw_a) if raw_a else 0,
            "updated_at": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception as e:
        return {"error": f"無法取得報價: {e}"}


def get_market_summary() -> pd.DataFrame:
    """
    取得大盤即時概況（加權指數）
    """
    url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_t00.tw&json=1"
    try:
        resp = SESSION.get(url, timeout=10)
        data = resp.json()
        items = data.get("msgArray", [])
        rows = []
        for item in items:
            rows.append({
                "代號": item.get("n", ""),
                "名稱": "加權指數",
                "成交價": float(item.get("z", "0")),
                "漲跌": float(item.get("d", "0")),
                "漲跌幅": float(item.get("p", "0")),
                "開盤": float(item.get("o", "0")),
                "最高": float(item.get("h", "0")),
                "最低": float(item.get("l", "0")),
                "成交量": int(item.get("v", "0")),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()
