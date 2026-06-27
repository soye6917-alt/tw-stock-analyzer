import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, time
os.chdir(r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer")

from data_fetcher import fetch_daily_twse, get_stock_name
from datetime import datetime
import pandas as pd

# Scan all stocks on TWSE for breakout candidates
# Use TWSE API to get all listed stocks
from data_fetcher import SESSION

print("Getting all listed stocks...")
try:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    resp = SESSION.get(url, timeout=15)
    all_stocks = {}
    if resp.status_code == 200:
        for item in resp.json():
            sid = item.get("Code","").strip()
            sname = item.get("Name","").strip()
            if sid and sname:
                all_stocks[sid] = sname
except Exception as e:
    print(f"API error: {e}")
    # Fallback to popular stocks
    all_stocks = {
        "2330":"台積電","2317":"鴻海","2454":"聯發科","2303":"聯電",
        "2382":"廣達","2356":"英業達","2376":"技嘉","2357":"華碩",
        "3037":"欣興","8046":"南電","2881":"富邦金","2882":"國泰金",
        "2891":"中信金","2603":"長榮","2609":"陽明","2615":"萬海",
        "2618":"長榮航","2313":"華通","2368":"金像電","2383":"台光電",
        "3231":"緯創","3017":"奇鋐","4938":"和碩","2308":"台達電",
        "2327":"國巨","2002":"中鋼","1301":"台塑","1216":"統一",
        "3443":"創意","3661":"世芯","5269":"祥碩","6531":"愛普",
        "6515":"穎崴","3406":"玉晶光","2498":"宏達電","2353":"宏碁",
        "4904":"遠傳","3045":"台灣大哥大","2345":"智邦","6213":"聯茂",
        "3189":"景碩","3653":"健策","6770":"力積電","2884":"玉山金",
        "2886":"兆豐金","2912":"統一超","3711":"日月光","3034":"聯詠"
    }

print(f"Loaded {len(all_stocks)} stocks. Scanning for breakout candidates...")

now = datetime.now()
results = []

for idx, (sid, sname) in enumerate(all_stocks.items()):
    if idx % 20 == 0:
        print(f"  Scanned {idx}/{len(all_stocks)}...", end='\r')
    
    try:
        dfs = []
        for m_offset in range(3):
            m = now.month - m_offset
            y = now.year
            while m < 1:
                m += 12; y -= 1
            df = fetch_daily_twse(sid, y, m)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            continue
        
        price_df = pd.concat(dfs).drop_duplicates(subset=['日期']).sort_values('日期').reset_index(drop=True)
        if len(price_df) < 20:
            continue
        
        close = price_df['Close'].iloc[-1]
        if close < 10 or close > 1000:
            continue
        
        # Technical indicators
        ma20 = price_df['Close'].rolling(20).mean().iloc[-1]
        ma60 = price_df['Close'].rolling(60).mean().iloc[-1]
        vol_avg = price_df['Volume'].rolling(20).mean().iloc[-1]
        latest_vol = price_df['Volume'].iloc[-1]
        
        recent_high = price_df['High'].rolling(20).max().iloc[-1]
        
        # 1-week change
        if len(price_df) >= 6:
            week_change = (close / price_df['Close'].iloc[-6] - 1) * 100
        else:
            week_change = 0
        
        # Volume ratio
        vol_ratio = latest_vol / vol_avg if vol_avg > 0 else 0
        
        # Score: breakout candidates
        score = 0
        reasons = []
        
        # Bullish: above all MAs
        if close > ma20:
            score += 15
            if close > ma60:
                score += 15
            else:
                score += 5
        
        # Bullish: high volume
        if vol_ratio > 1.3:
            score += 15
        elif vol_ratio > 1.0:
            score += 8
        
        # Bullish: good momentum
        if week_change > 5:
            score += 10
        elif week_change > 2:
            score += 5
        
        # Near 52-week high (breakout)
        if close >= recent_high * 0.98:
            score += 10
        
        # Price above 50
        # if close > 50: score += 5
        
        if score >= 30:
            results.append({
                'code': sid,
                'name': sname,
                'price': close,
                'score': score,
                'ma20': ma20,
                'ma60': ma60,
                'vol_ratio': round(vol_ratio, 2),
                'week_chg': round(week_change, 2),
                'volume': int(latest_vol)
            })
    except:
        continue

print(f"\n\n=== BREAKOUT CANDIDATES (Score >= 30) ===")
results.sort(key=lambda x: x['score'], reverse=True)

for r in results[:30]:
    flag = '🔥' if r['vol_ratio'] > 1.5 and r['week_chg'] > 5 else '✅'
    print(f"  {flag} {r['code']} {r['name']:6s} ${r['price']:<8.2f} Score:{r['score']:2d} "
          f"Vol:{r['vol_ratio']:.1f}x Wk:{r['week_chg']:+.1f}% "
          f"MA20:{r['ma20']:.1f} MA60:{r['ma60']:.1f}")

print(f"\nTotal candidates found: {len(results)}")
