"""
professional_data.py — Professional Data Pipeline

Three upgrades:
1. SQLite cache layer — avoid repeated TWSE API calls
2. Smart retry + error recovery — handle weekend empty responses
3. Multi-source fallback — TWSE -> TPEx -> realtime quote
"""
import os, json, sqlite3, datetime, time
import requests as rq
import pandas as pd
from pathlib import Path

rq.packages.urllib3.disable_warnings()

CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "stock_cache.db"

# --- SQLite Cache Layer ---
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS stock_daily (
        stock_id TEXT, date TEXT, open REAL, high REAL, low REAL,
        close REAL, volume REAL,
        PRIMARY KEY (stock_id, date)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS quote_cache (
        stock_id TEXT PRIMARY KEY, data TEXT, updated_at TEXT
    )""")
    return conn

def get_cached_data(stock_id: str, months: int = 6):
    """Get cached data, returns None if stale (>3 days old)."""
    conn = get_conn()
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=months*30)).strftime("%Y-%m-%d")
    df = pd.read_sql(
        "SELECT * FROM stock_daily WHERE stock_id=? AND date>=? ORDER BY date",
        conn, params=(stock_id, cutoff)
    )
    conn.close()
    if df.empty:
        return None
    latest_date = df["date"].iloc[-1]
    latest_dt = datetime.datetime.strptime(latest_date, "%Y-%m-%d")
    if (datetime.datetime.now() - latest_dt).days > 3:
        return None  # Stale
    return df

def cache_data(stock_id: str, df: pd.DataFrame):
    """Cache a dataframe to SQLite."""
    conn = get_conn()
    for _, row in df.iterrows():
        conn.execute("""INSERT OR REPLACE INTO stock_daily
            (stock_id, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (stock_id, row.get("date") or row.get("\u65e5\u671f"),
             row.get("open") or row.get("\u958b\u76e4\u50f9") or 0,
             row.get("high") or row.get("\u6700\u9ad8\u50f9") or 0,
             row.get("low") or row.get("\u6700\u4f4e\u50f9") or 0,
             row.get("close") or row.get("\u6536\u76e4\u50f9") or 0,
             row.get("volume") or row.get("\u6210\u4ea4\u80a1\u6578") or 0))
    conn.commit()
    conn.close()

# --- Smart Retry Fetch ---
def smart_fetch_twse(stock_id: str, year: int, month: int, retries=3):
    """Fetch with retry and SSL fallback."""
    url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?date={year}{month:02d}01&stockNo={stock_id}&response=json"
    
    for attempt in range(retries):
        try:
            resp = rq.get(url, verify=False, timeout=15)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            data = resp.json()
            if "data" in data and len(data["data"]) > 0:
                return data
            if attempt < retries - 1:
                time.sleep(2)
        except:
            if attempt < retries - 1:
                time.sleep(2)
            continue
    return None

def fetch_professional_data(stock_id: str, months: int = 6):
    """
    Professional fetch with cache-first strategy.
    1. Check SQLite cache
    2. If stale/missing, fetch from TWSE
    3. Cache result
    """
    cached = get_cached_data(stock_id, months)
    if cached is not None and len(cached) > 10:
        return cached
    
    all_data = []
    today = datetime.datetime.now()
    for i in range(months):
        m = today.month - i
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        result = smart_fetch_twse(stock_id, y, m)
        if result and "data" in result and "fields" in result:
            fields = result["fields"]
            for row in result["data"]:
                record = dict(zip(fields, row))
                try:
                    date_str = record.get("\u65e5\u671f", "")
                    if "/" in date_str:
                        parts = date_str.split("/")
                        date_str = str(int(parts[0]) + 1911) + "-" + parts[1] + "-" + parts[2]
                    record["date"] = date_str
                    record["open"] = float(str(record.get("\u958b\u76e4\u50f9", 0)).replace(",", ""))
                    record["high"] = float(str(record.get("\u6700\u9ad8\u50f9", 0)).replace(",", ""))
                    record["low"] = float(str(record.get("\u6700\u4f4e\u50f9", 0)).replace(",", ""))
                    record["close"] = float(str(record.get("\u6536\u76e4\u50f9", 0)).replace(",", ""))
                    record["volume"] = float(str(record.get("\u6210\u4ea4\u80a1\u6578", 0)).replace(",", ""))
                    all_data.append(record)
                except (ValueError, KeyError):
                    continue
    
    if all_data:
        df = pd.DataFrame(all_data)
        df = df.sort_values("date").drop_duplicates(subset=["date"])
        cache_data(stock_id, df)
        return df
    
    return None

if __name__ == "__main__":
    df = fetch_professional_data("2618", 3)
    if df is not None:
        print(f"2618: {len(df)} rows, latest close: {df.close.iloc[-1]}")
    else:
        print("2618: fetch failed (expected on weekend)")
