"""
🚀 近期飆股預測模組 — 專家分析師等級
專門預測即將大漲的股票：爆量突破 + 技術共振 + 新聞催化 + 籌碼助攻

兩階段掃描:
  Stage 1: 快速技術篩選（200 檔 → 取前 20）
  Stage 2: 完整分析（含多源新聞）取前 10

三階段信心評分:
  🟢 強烈突破信號（高機率飆漲）
  🟡 潛在突破觀察
  🔴 暫不建議（風險偏高）
"""

import pandas as pd
import numpy as np
import time
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import sys

# 所有對 data_fetcher 的引用都用函數內動態 import，
# 避免 USB 檔案系統毀損導致 module-level import 失敗。
# data_fetcher 在 app.py 中已被優先載入到 sys.modules，
# 動態 import 會直接取用快取，繞過損毀的檔案讀取。
def _get_data_fetcher():
    return sys.modules.get("data_fetcher")

def _get_indicators():
    import indicators
    return indicators

def _get_fundamentals():
    import fundamentals
    return fundamentals


# ─────────────────────────────────────────────
# 掃描標的清單（200 檔：熱門 + 各產業 + 高流動性）
# ─────────────────────────────────────────────

# 核心掃描池（一定要掃的）
CORE_UNIVERSE = {
    # 半導體龍頭
    "2330": "台積電", "2454": "聯發科", "2303": "聯電",
    "3034": "聯詠", "3711": "日月光投控", "3443": "創意",
    "3661": "世芯-KY", "5269": "祥碩", "6643": "M31",
    "3532": "台勝科", "6239": "力成", "3260": "威剛",
    "2401": "凌陽", "2458": "義隆", "4968": "立積",
    "3035": "智原", "3227": "原相", "6233": "旺玖",
    # AI / HPC 相關
    "2382": "廣達", "2356": "英業達", "2357": "華碩",
    "2376": "技嘉", "2377": "微星", "3231": "緯創",
    "3706": "神達", "4938": "和碩", "2324": "仁寶",
    "2353": "宏碁", "3017": "奇鋐", "3324": "雙鴻",
    "3653": "健策", "6213": "聯茂", "2383": "台光電",
    # 電子零組件
    "2308": "台達電", "2327": "國巨", "2492": "華新科",
    "3023": "信邦", "3679": "新至陞", "3380": "明泰",
    # PCB / 載板
    "3037": "欣興", "8046": "南電", "6278": "台表科",
    # 面板 / 光電
    "2409": "友達", "3481": "群創", "6116": "彩晶",
    "3008": "大立光", "3406": "玉晶光",
    # 網通 / 電信
    "4904": "遠傳", "3045": "台灣大", "2412": "中華電",
    "2345": "智邦", "3704": "合勤控", "3596": "智易",
    # 金融
    "2881": "富邦金", "2882": "國泰金", "2891": "中信金",
    "2886": "兆豐金", "2892": "第一金", "5880": "合庫金",
    "2884": "玉山金", "2885": "元大金", "2887": "台新金",
    "2890": "永豐金", "2888": "新光金", "2880": "華南金",
    "2883": "開發金", "2834": "臺企銀", "2812": "台中銀",
    # 傳產龍頭
    "2002": "中鋼", "1301": "台塑", "1303": "南亞",
    "1326": "台化", "1216": "統一", "2912": "統一超",
    "1101": "台泥", "1102": "亞泥", "2207": "和泰車",
    "2201": "裕隆", "2105": "正新", "2101": "南港",
    "1402": "遠東新", "9904": "寶成", "9910": "豐泰",
    # 航運
    "2603": "長榮", "2609": "陽明", "2610": "華航",
    "2618": "長榮航", "2637": "慧洋-KY",
    # 鋼鐵 / 原物料
    "2014": "中鴻", "2027": "大成鋼", "2031": "新光鋼",
    "1304": "台聚", "1305": "華夏", "1314": "中石化",
    # 生技 / 醫療
    "4137": "麗豐-KY", "4105": "東洋", "4736": "泰博",
    "4126": "太醫", "6469": "大樹",
    # 電機 / 機械
    "2049": "上銀", "1590": "亞德客-KY", "1536": "和大",
    # 綠能 / 儲能
    "2305": "全友", "3708": "上緯投控", "1519": "華城",
    "1513": "中興電", "1504": "東元",
    # 營建 / 資產
    "2501": "國建", "2548": "華固", "2542": "興富發",
    "5522": "遠雄", "9945": "潤泰新",
    # 紡織 / 百貨
    "1476": "儒鴻", "1477": "聚陽", "2915": "潤泰全",
    # 半導體設備
    "2464": "盟立", "3413": "京鼎", "3131": "弘塑",
    "3587": "閎康", "3680": "家登",
    # 其他重要
    "2317": "鴻海", "6515": "穎崴", "6531": "愛普",
    "6669": "緯穎", "8210": "勤誠", "2351": "順德",
    "5274": "信驊", "6104": "創惟", "2439": "美律",
    "2329": "華泰", "2360": "致茂", "3010": "華立",
    # ETF
    "0050": "元大台灣50", "0056": "元大高股息",
    "00878": "國泰永續高股息",
}

# 應用全部 POPULAR_STOCKS（懶載入繞過 USB 檔案毀損）
UNIVERSE = dict(CORE_UNIVERSE)
def _ensure_universe_populated():
    if len(UNIVERSE) > len(CORE_UNIVERSE):
        return
    df_mod = _get_data_fetcher()
    if df_mod:
        extra = getattr(df_mod, "POPULAR_STOCKS", {})
        for k, v in extra.items():
            if k not in UNIVERSE:
                UNIVERSE[k] = v


# ─────────────────────────────────────────────
# 多源新聞情緒分析
# ─────────────────────────────────────────────

def fetch_yahoo_news(stock_id: str, stock_name: str) -> dict:
    """
    從 Yahoo 奇摩股市取得最新新聞
    回傳: {score, headlines, source}
    """
    score = 0.0
    headlines = []
    try:
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml",
        }
        resp = _get_data_fetcher().SESSION.get(url, timeout=8, headers=headers)
        if resp.status_code != 200:
            return {"score": 0, "headlines": [], "source": "yahoo", "error": "HTTP"}

        text = resp.text
        found = []

        # 抓 h3 內文
        for m in re.findall(r'<h3[^>]*>(.*?)</h3>', text, re.DOTALL):
            clean = re.sub(r'<[^>]+>', '', m).strip()
            if len(clean) > 10 and len(re.findall(r'[\u4e00-\u9fff]', clean)) >= 4:
                found.append(clean)

        if len(found) < 3:
            for m in re.findall(r'<a[^>]*title=["\'](.*?)["\']', text):
                if len(m) > 10 and len(re.findall(r'[\u4e00-\u9fff]', m)) >= 4 and m not in found:
                    found.append(m)

        headlines = list(dict.fromkeys(found))[:8]
        score = compute_news_mood(headlines, stock_id, stock_name)
    except Exception:
        pass
    return {"score": score, "headlines": headlines, "source": "yahoo"}


def fetch_google_news(stock_id: str, stock_name: str) -> dict:
    """
    從 Google News 搜尋該股票的最新新聞（中文）
    """
    score = 0.0
    headlines = []
    try:
        query = f"{stock_id} {stock_name} 股票"
        url = "https://news.google.com/rss/search?q=" + requests.utils.quote(query) + "&hl=zh-TW&gl=TW"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _get_data_fetcher().SESSION.get(url, timeout=8, headers=headers)
        if resp.status_code != 200:
            return {"score": 0, "headlines": [], "source": "google"}

        # 從 RSS 提取標題
        text = resp.text
        for m in re.findall(r'<title>(.*?)</title>', text):
            clean = m.strip()
            if len(clean) > 8 and len(re.findall(r'[\u4e00-\u9fff]', clean)) >= 3:
                if clean not in headlines:
                    headlines.append(clean)

        # 關鍵字搜索新聞頁面
        url2 = f"https://news.google.com/search?q={requests.utils.quote(f'{stock_id} {stock_name}')}&hl=zh-TW&gl=TW"
        resp2 = _get_data_fetcher().SESSION.get(url2, timeout=8, headers=headers)
        if resp2.status_code == 200:
            for m in re.findall(r'<[^>]*role="heading"[^>]*>(.*?)</', resp2.text, re.DOTALL):
                clean = re.sub(r'<[^>]+>', '', m).strip()
                if len(clean) > 10 and clean not in headlines:
                    headlines.append(clean)

        headlines = [h for h in headlines if len(h) > 10][:6]
        score = compute_news_mood(headlines, stock_id, stock_name)
    except Exception:
        pass
    return {"score": score, "headlines": headlines, "source": "google"}


def fetch_udn_news(stock_id: str, stock_name: str) -> dict:
    """從 UDN 聯合新聞網搜尋"""
    score = 0.0
    headlines = []
    try:
        url = f"https://udn.com/search/word/2/{requests.utils.quote(stock_name)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _get_data_fetcher().SESSION.get(url, timeout=6, headers=headers)
        if resp.status_code == 200:
            for m in re.findall(r'<h3[^>]*>.*?<a[^>]*>(.*?)</a>', resp.text, re.DOTALL):
                clean = re.sub(r'<[^>]+>', '', m).strip()
                if len(clean) > 10 and clean not in headlines:
                    headlines.append(clean)
            for m in re.findall(r'<a[^>]*title=["\'](.*?)["\']', resp.text):
                if len(m) > 10 and m not in headlines:
                    headlines.append(m)
        headlines = headlines[:5]
        score = compute_news_mood(headlines, stock_id, stock_name)
    except Exception:
        pass
    return {"score": score, "headlines": headlines, "source": "udn"}


def fetch_moneydj_news(stock_id: str, stock_name: str) -> dict:
    """從 MoneyDJ 財經新聞搜尋"""
    score = 0.0
    headlines = []
    try:
        url = f"https://www.moneydj.com/KMDJ/News/NewsSearch.aspx?q={requests.utils.quote(stock_name)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = _get_data_fetcher().SESSION.get(url, timeout=6, headers=headers)
        if resp.status_code == 200:
            for m in re.findall(r'<a[^>]*href="[^"]*News[^"]*"[^>]*>(.*?)</a>', resp.text, re.DOTALL):
                clean = re.sub(r'<[^>]+>', '', m).strip()
                if len(clean) > 8 and len(re.findall(r'[\u4e00-\u9fff]', clean)) >= 3 and clean not in headlines:
                    headlines.append(clean)
        headlines = headlines[:5]
        score = compute_news_mood(headlines, stock_id, stock_name)
    except Exception:
        pass
    return {"score": score, "headlines": headlines, "source": "moneydj"}


def compute_news_mood(headlines: list, stock_id: str, stock_name: str) -> float:
    """
    新聞情緒評分（-10 ~ +10）
    用更豐富的關鍵字庫 + 強度權重
    """
    if not headlines:
        return 0

    # 強正向（一次+2）
    strong_pos = ["創高", "突破", "漲停", "大漲", "飆漲", "爆發", "噴出",
                  "目標價", "調升", "利多", "併購", "整併", "擴大投資",
                  "轉盈", "翻倍", "獨家", "領先", "大單", "急單", "搶單"]
    # 中正向（一次+1）
    mid_pos = ["買進", "成長", "受惠", "擴產", "營收", "年增", "獲利",
               "合作", "訂單", "轉機", "回溫", "復甦", "反彈", "走強",
               "法說", "亮眼", "加碼", "配息", "股利", "補漲", "超車",
               "布局", "AI", "HPC", "先進", "獨角獸"]
    # 強負向（一次-2）
    strong_neg = ["跌停", "大跌", "暴跌", "重挫", "利空", "虧損", "裁員",
                  "關廠", "掏空", "作假", "違約", "下市", "地雷", "破產",
                  "清算", "停工", "被罰", "訴訟", "調查"]
    # 中負向（一次-1）
    mid_neg = ["下跌", "賣壓", "調降", "降評", "減碼", "賣出", "衰退",
               "下滑", "年減", "月減", "庫存", "降溫", "匯損", "警戒",
               "壓力", "換手", "出貨", "套牢"]

    score = 0
    for title in headlines:
        for kw in strong_pos:
            if kw in title:
                score += 2
                break
        for kw in strong_neg:
            if kw in title:
                score -= 2
                break
        # 中級關鍵字（只在沒被強級抓到時檢查）
        if not any(kw in title for kw in strong_pos):
            for kw in mid_pos:
                if kw in title:
                    score += 1
                    break
        if not any(kw in title for kw in strong_neg):
            for kw in mid_neg:
                if kw in title:
                    score -= 1
                    break

    # 標準化到 -10 ~ +10
    score = max(-10, min(10, score / max(len(headlines), 1) * 3))

    # 加分：明確提到該股
    for h in headlines:
        if stock_id in h or stock_name in h[:15]:
            score += 0.5
            break

    return round(max(-10, min(10, score)), 1)


def fetch_multi_source_news(stock_id: str, stock_name: str) -> dict:
    """
    多源新聞收集 + 情緒分析
    併行爬取 Yahoo + Google + UDN + MoneyDJ
    回傳: {score, headlines, sources, summary}
    """
    results = {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(fetch_yahoo_news, stock_id, stock_name): "yahoo",
            ex.submit(fetch_google_news, stock_id, stock_name): "google",
            ex.submit(fetch_udn_news, stock_id, stock_name): "udn",
            ex.submit(fetch_moneydj_news, stock_id, stock_name): "moneydj",
        }
        for f in as_completed(futures):
            try:
                res = f.result(timeout=10)
                src = futures[f]
                results[src] = res
            except Exception:
                pass

    # 合併
    all_headlines = []
    total_score = 0.0
    source_count = {}

    for src, data in results.items():
        if data and data.get("headlines"):
            all_headlines.extend(data["headlines"])
            total_score += data.get("score", 0)
            source_count[src] = len(data["headlines"])

    # 去重
    seen = set()
    unique_headlines = []
    for h in all_headlines:
        # 用前30字比對避免微小差異
        key = h[:30]
        if key not in seen:
            seen.add(key)
            unique_headlines.append(h)

    # 綜合情緒分(考慮來源數加權)
    num_sources = len(source_count)
    avg_score = total_score / num_sources if num_sources > 0 else 0
    source_bonus = min(num_sources * 0.5, 2)  # 最多+2
    final_score = max(-10, min(10, avg_score + source_bonus))

    # 來源摘要
    sources_summary = ", ".join([f"{s}({c})" for s, c in source_count.items()])

    return {
        "score": round(final_score, 1),
        "headlines": unique_headlines[:10],
        "source_count": source_count,
        "sources_summary": sources_summary,
        "total_headlines": len(unique_headlines),
    }


# ─────────────────────────────────────────────
# 飆股偵測核心
# ─────────────────────────────────────────────

@dataclass
class SurgeCandidate:
    stock_id: str
    stock_name: str
    # 總評分
    total_score: float = 0.0
    surge_score: float = 0.0    # 飆漲潛力總分 (0~100)
    conviction: str = "⚪ 待確認"  # 預測信心
    # 六因子細項
    vol_score: float = 0.0       # 爆量 (0~20)
    breakout_score: float = 0.0  # 突破型態 (0~20)
    tech_score: float = 0.0      # 技術共振 (0~20)
    news_score: float = 0.0      # 新聞催化 (0~15)
    momentum_score: float = 0.0  # 動能加速 (0~15)
    inst_score: float = 0.0      # 籌碼助攻 (0~10)
    # 訊號標記
    surge_signals: list = field(default_factory=list)
    risk_warnings: list = field(default_factory=list)
    # 市場數據
    current_price: float = 0.0
    change_pct: float = 0.0
    volume_ratio: float = 0.0    # 量比 (當日/20日均)
    high_break_pct: float = 0.0  # 距近期高點 %
    # 預測資訊
    news_headlines: list = field(default_factory=list)
    news_sources: str = ""
    estimated_timing: str = ""    # 預估發動時機
    analysis: list = field(default_factory=list)
    entry_note: str = ""
    # 狀態
    error: str = None


def detect_volume_surge(df: pd.DataFrame) -> tuple:
    """
    爆量偵測（最高 20 分）
    檢查是否有異於常態的成交量爆發
    """
    if df.empty or len(df) < 25 or "Volume" not in df.columns:
        return 0, [], []

    score = 0.0
    signals = []
    warnings = []

    close = df["Close"].values
    volume = df["Volume"].values

    # 20日均量
    avg_vol_20 = volume[-21:-1].mean() if len(volume) > 21 else volume.mean()
    recent_vol = volume[-5:].mean()
    cur_vol = volume[-1]

    vol_ratio_avg = recent_vol / avg_vol_20 if avg_vol_20 > 0 else 1
    vol_ratio_cur = cur_vol / avg_vol_20 if avg_vol_20 > 0 else 1

    # 量能爆發評分
    if vol_ratio_cur > 3.0:
        score += 12
        signals.append(f"⚡ 單日爆量 {vol_ratio_cur:.1f}x（巨量!）")
    elif vol_ratio_cur > 2.0:
        score += 8
        signals.append(f"🔥 明顯放量 {vol_ratio_cur:.1f}x")
    elif vol_ratio_cur > 1.5:
        score += 5
        signals.append(f"📊 量增 {vol_ratio_cur:.1f}x")

    if vol_ratio_avg > 1.8:
        score += 8
        signals.append(f"📈 近5日均量持續高檔 {vol_ratio_avg:.1f}x")
    elif vol_ratio_avg > 1.3:
        score += 4
        signals.append(f"近5日量能增溫 {vol_ratio_avg:.1f}x")

    # 量價配合檢驗
    price_chg_1d = (close[-1] / close[-2] - 1) * 100
    if vol_ratio_cur > 1.5 and price_chg_1d > 2:
        score += 5
        signals.append(f"✅ 價量齊揚（漲 {price_chg_1d:+.1f}% + 量 {vol_ratio_cur:.1f}x）")
    elif vol_ratio_cur > 2.0 and price_chg_1d < -2:
        score -= 8
        warnings.append(f"⚠️ 爆量下跌 {price_chg_1d:.1f}%（出貨疑慮）")

    # 量能是否持續上升趨勢
    if len(volume) >= 20:
        vol_trend = (volume[-5:].mean() - volume[-10:-5].mean()) / (volume[-10:-5].mean() + 1)
        if vol_trend > 0.3:
            score += 3
            signals.append("量能趨勢遞增")

    score = max(-5, min(20, score))
    return score, signals, warnings


def detect_breakout(df: pd.DataFrame) -> tuple:
    """
    突破型態偵測（最高 20 分）
    檢查股價是否突破近期的整理區間 / 壓力線
    """
    if df.empty or len(df) < 30:
        return 0, [], []

    score = 0.0
    signals = []
    warnings = []
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    # 近期高點
    high_20 = np.max(high[-20:])
    high_60 = np.max(high[-60:]) if len(high) >= 60 else high_20
    close_now = close[-1]

    breake_pct_20 = (close_now / high_20 - 1) * 100
    breake_pct_60 = (close_now / high_60 - 1) * 100

    # 突破20日高點
    if close_now >= high_20:
        score += 8
        signals.append(f"🟢 突破20日高點（{close_now:.1f} > {high_20:.1f}）")
    elif close_now >= high_20 * 0.98:
        score += 4
        signals.append(f"逼近20日高點（差 {high_20 - close_now:.1f}）")

    # 突破60日高點
    if close_now >= high_60:
        score += 8
        signals.append(f"🟢 突破60日高點（{close_now:.1f} > {high_60:.1f}）")
    elif close_now >= high_60 * 0.98:
        score += 4
        signals.append(f"📊 逼近60日高點")

    # 整理區間突破（N字突破）
    if len(close) >= 40:
        recent_highs = high[-40:]
        recent_lows = low[-40:]
        range_width = (recent_highs.max() - recent_lows.min()) / recent_lows.min() * 100

        if range_width < 15:
            # 窄幅整理
            if close_now >= np.max(recent_highs[-10:]):
                score += 6
                signals.append(f"📐 窄幅整理突破（振幅僅 {range_width:.1f}%）")
                signals.append("窄幅整理越久，突破後爆發力越強")

    # 價格位於高檔區
    price_pos = (close_now - low[-20:].min()) / (high[-20:].max() - low[-20:].min() + 0.01) * 100
    if price_pos > 90:
        score += 3
        signals.append(f"📍 價格處在高檔區（{price_pos:.0f}% 位置）")

    # 反轉風險
    if breake_pct_20 > 8:
        warnings.append(f"⚠️ 短期已漲 {breake_pct_20:.1f}%，留意追高風險")

    score = max(0, min(20, score))
    return score, signals, warnings


def detect_tech_convergence(df: pd.DataFrame) -> tuple:
    """
    技術因子共振（最高 20 分）
    多項技術指標同時轉多
    """
    if df.empty or len(df) < 30:
        return 0, [], []

    score = 0.0
    signals = []
    warnings = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = latest["Close"]

    # 1. 短均線突破長均線（黃金交叉）
    for s, l in [("MA5", "MA10"), ("MA10", "MA20"), ("MA20", "MA60")]:
        if s in latest and l in latest and l in prev:
            if latest[s] > latest[l] and prev[s] <= prev[l]:
                score += 4
                signals.append(f"🟢 {s} 突破 {l}（黃金交叉）")
            elif latest[s] > latest[l]:
                score += 2
                signals.append(f"{s} 在 {l} 之上")

    # 2. MACD 狀態
    macd = latest.get("MACD", 0)
    macd_sig = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    prev_hist = prev.get("MACD_Hist", 0)

    if macd > macd_sig and macd > 0:
        score += 4
        signals.append("📊 MACD 多頭 + 正值")
    elif macd > macd_sig and macd < 0 and macd_hist > prev_hist:
        score += 3
        signals.append("🔄 MACD 負轉正（動能翻多）")

    if macd_hist > 0 and macd_hist > prev_hist:
        score += 2
        signals.append("MACD 柱狀圖遞增（動能增強）")

    # 3. RSI 位置
    rsi = latest.get("RSI", 50)
    if 50 < rsi < 70:
        score += 3
        signals.append(f"RSI {rsi:.0f} 偏多區（未過熱）")
    elif 40 < rsi <= 50:
        score += 2
        signals.append(f"RSI {rsi:.0f} 中性偏多（有空間）")

    # 4. KDJ 金叉
    k = latest.get("K", 50)
    d = latest.get("D", 50)
    prev_k = prev.get("K", 50)
    prev_d = prev.get("D", 50)
    if k > d and prev_k <= prev_d:
        score += 3
        signals.append(f"🟢 KDJ 黃金交叉（K:{k:.0f} D:{d:.0f}）")
    elif k > d:
        score += 1
        signals.append(f"KDJ K線在D上")

    # 5. 布林位置
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    if bb_upper > 0 and bb_lower > 0:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if 50 < bb_pos < 80:
            score += 3
            signals.append(f"布林中段偏上（{bb_pos:.0f}%，有空間）")
        elif bb_pos > 90 and macd > macd_sig:
            score += 1
            signals.append("布林上緣但動能配合")
        elif bb_pos < 30:
            score += 2
            signals.append(f"布林低檔（{bb_pos:.0f}%，反彈空間）")

    # 整體超買警告
    if rsi > 75:
        warnings.append(f"⚠️ RSI {rsi:.0f} 超買區")
    if bb_upper > 0 and close > bb_upper:
        warnings.append("⚠️ 股價超出布林上軌")
    if macd_hist < 0 and macd < 0:
        warnings.append("⚠️ MACD 全面空頭")

    score = max(-5, min(20, score))
    return score, signals, warnings


def detect_momentum(df: pd.DataFrame) -> tuple:
    """
    動能加速偵測（最高 15 分）
    短期漲速是否正在加快
    """
    if df.empty or len(df) < 15:
        return 0, [], []

    score = 0.0
    signals = []
    warns = []
    close = df["Close"].values

    # 不同週期漲幅
    ret_1d = (close[-1] / close[-2] - 1) * 100
    ret_3d = (close[-1] / close[-3] - 1) * 100 if len(close) >= 3 else 0
    ret_5d = (close[-1] / close[-5] - 1) * 100 if len(close) >= 5 else 0
    ret_10d = (close[-1] / close[-10] - 1) * 100 if len(close) >= 10 else 0
    ret_20d = (close[-1] / close[-20] - 1) * 100 if len(close) >= 20 else 0

    # 動能加速度（近3日 vs 之前3日）
    if len(close) >= 6:
        speed_1 = (close[-1] / close[-3] - 1) * 100
        speed_2 = (close[-4] / close[-6] - 1) * 100 if len(close) >= 6 else 0
        accel = speed_1 - speed_2
        if accel > 3:
            score += 6
            signals.append(f"🚀 動能加速 {accel:+.1f}%（速度加快）")
        elif accel > 1:
            score += 3
            signals.append(f"📈 動能微增 {accel:+.1f}%")

    # 短線強度評分
    if ret_5d > 5:
        score += 4
        signals.append(f"短線強勢（5日+{ret_5d:.1f}%）")
    elif ret_5d > 2:
        score += 2
        signals.append(f"5日漲幅 {ret_5d:+.1f}%")

    if ret_20d > 0 and ret_10d > ret_20d:
        score += 3
        signals.append("短線漲速 > 長線漲速（加速上漲）")

    # 連漲天數獎勵
    consecutive_up = 0
    for i in range(min(10, len(close) - 1)):
        if close[-(i + 1)] > close[-(i + 2)]:
            consecutive_up += 1
        else:
            break
    if consecutive_up >= 3:
        score += 2
        signals.append(f"連漲 {consecutive_up} 日")

    # 漲多風險
    if ret_5d > 12:
        signals.append("⚠️ 短期漲幅過大，留意獲利了結")

    if ret_1d > 6:
        warns.append(f"⚠️ 單日大漲 {ret_1d:.1f}%，隔日可能震盪")

    score = max(-5, min(15, score))
    return score, signals, warns


def detect_institutional_surge(stock_id: str) -> tuple:
    """
    籌碼異常偵測（最高 10 分）
    三大法人是否出現異常買超
    """
    score = 0.0
    signals = []
    warnings = []

    try:
        inst_df = _get_fundamentals().fetch_institutional_trading(stock_id)
        if inst_df.empty:
            return 0, [], []

        for label, max_s in [("外資", 5), ("投信", 3), ("自營商", 2)]:
            row = inst_df[inst_df["類別"].str.contains(label)]
            if not row.empty:
                net = row["買賣超"].values[0]
                net_k = net / 1000
                if net_k > 5:
                    s = max_s
                    signals.append(f"🏢 {label}大買 {net_k:.0f}張")
                elif net_k > 1:
                    s = max_s - 1
                    signals.append(f"{label}買超 {net_k:.0f}張")
                elif net_k > 0:
                    s = max(1, max_s - 2)
                elif net_k > -1:
                    s = 0
                elif net_k < -5:
                    s = -max_s
                    warnings.append(f"{label}大賣 {abs(net_k):.0f}張")
                else:
                    s = -1
                    warnings.append(f"{label}小賣")
                score += s
    except Exception:
        pass

    return max(-5, min(10, score)), signals, warnings


# ─────────────────────────────────────────────
# 主評分函式
# ─────────────────────────────────────────────

def surge_score_stock(stock_id: str, stock_name: str, months: int = 6) -> SurgeCandidate:
    """
    對單一股票進行飆股潛力完整評分
    快速版：僅技術面（用於 Stage 1 篩選）
    """
    candidate = SurgeCandidate(stock_id=stock_id, stock_name=stock_name)

    try:
        df = _get_data_fetcher().fetch_historical(stock_id, months=months)
        if df.empty:
            candidate.error = "無資料"
            return candidate

        df = _get_indicators().add_all_indicators(df)
        candidate.current_price = df["Close"].iloc[-1]
        if len(df) > 1:
            candidate.change_pct = (candidate.current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100

        # 六因子評分
        vol_score, vol_sig, vol_warn = detect_volume_surge(df)
        breakout_score, brk_sig, brk_warn = detect_breakout(df)
        tech_score, tech_sig, tech_warn = detect_tech_convergence(df)
        momentum_score, mom_sig, mom_warn = detect_momentum(df)

        candidate.vol_score = vol_score
        candidate.breakout_score = breakout_score
        candidate.tech_score = tech_score
        candidate.momentum_score = momentum_score

        candidate.surge_signals.extend(vol_sig + brk_sig + tech_sig + mom_sig)
        candidate.risk_warnings.extend(vol_warn + brk_warn + tech_warn + mom_warn)

        # 基礎資訊
        if "Volume" in df.columns:
            vol = df["Volume"].values
            avg_vol = vol[-21:-1].mean() if len(vol) > 21 else vol.mean()
            candidate.volume_ratio = vol[-1] / avg_vol if avg_vol > 0 else 0

        if len(df) > 20:
            candidate.high_break_pct = (candidate.current_price / df["High"].iloc[-20:].max() - 1) * 100

        # 初評（暫不含新聞/籌碼）
        candidate.surge_score = vol_score + breakout_score + tech_score + momentum_score

    except Exception as e:
        candidate.error = str(e)

    return candidate


def surge_score_stock_full(stock_id: str, stock_name: str, months: int = 6) -> SurgeCandidate:
    """
    對單一股票進行飆股潛力完整評分（含新聞+籌碼）
    用於 Stage 2 深入分析
    """
    candidate = SurgeCandidate(stock_id=stock_id, stock_name=stock_name)

    try:
        df = _get_data_fetcher().fetch_historical(stock_id, months=months)
        if df.empty:
            candidate.error = "無資料"
            return candidate

        df = _get_indicators().add_all_indicators(df)
        candidate.current_price = df["Close"].iloc[-1]
        if len(df) > 1:
            candidate.change_pct = (candidate.current_price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100

        # 六因子全面評分
        vol_score, vol_sig, vol_warn = detect_volume_surge(df)
        breakout_score, brk_sig, brk_warn = detect_breakout(df)
        tech_score, tech_sig, tech_warn = detect_tech_convergence(df)
        momentum_score, mom_sig, mom_warn = detect_momentum(df)
        inst_score, inst_sig, inst_warn = detect_institutional_surge(stock_id)

        candidate.vol_score = vol_score
        candidate.breakout_score = breakout_score
        candidate.tech_score = tech_score
        candidate.momentum_score = momentum_score
        candidate.inst_score = inst_score

        candidate.surge_signals.extend(vol_sig + brk_sig + tech_sig + mom_sig + inst_sig)
        candidate.risk_warnings.extend(vol_warn + brk_warn + tech_warn + mom_warn + inst_warn)

        # 多源新聞
        news_data = fetch_multi_source_news(stock_id, stock_name)
        candidate.news_score = max(0, news_data["score"])  # 只取正向
        candidate.news_headlines = news_data["headlines"]
        candidate.news_sources = news_data["sources_summary"]
        if news_data["headlines"]:
            candidate.surge_signals.append(f"🗞️ 多源新聞×{news_data['total_headlines']}則")

        # 基礎資訊
        if "Volume" in df.columns:
            vol = df["Volume"].values
            avg_vol = vol[-21:-1].mean() if len(vol) > 21 else vol.mean()
            candidate.volume_ratio = vol[-1] / avg_vol if avg_vol > 0 else 0

        if len(df) > 20:
            candidate.high_break_pct = (candidate.current_price / df["High"].iloc[-20:].max() - 1) * 100

        # 總分計算
        candidate.surge_score = (
            candidate.vol_score +
            candidate.breakout_score +
            candidate.tech_score +
            candidate.momentum_score +
            candidate.news_score +
            candidate.inst_score
        )

        # 信心評級
        candidate = _assign_conviction(candidate)
        candidate = _gen_analysis(candidate)
        candidate = _gen_estimated_timing(candidate)

    except Exception as e:
        candidate.error = str(e)

    return candidate


def _assign_conviction(c: SurgeCandidate) -> SurgeCandidate:
    """賦予信心評級"""
    s = c.surge_score

    if s >= 70:
        c.conviction = "🟢 強烈突破信號"
    elif s >= 55:
        c.conviction = "🟢 突破信號明確"
    elif s >= 40:
        c.conviction = "🟡 潛在突破觀察"
    elif s >= 25:
        c.conviction = "🟡 初步徵兆"
    else:
        c.conviction = "⚪ 暫不建議"

    # 有重大風險時降級
    if len(c.risk_warnings) >= 3:
        if c.conviction.startswith("🟢"):
            c.conviction = "🟡 " + c.conviction[2:]
        elif c.conviction.startswith("🟡"):
            c.conviction = "⚪ 風險偏高"

    return c


def _gen_estimated_timing(c: SurgeCandidate) -> SurgeCandidate:
    """預估發動時機"""
    s = c.surge_score
    signals = c.surge_signals

    # 檢查是否有爆量+突破同時發生
    has_volume = any("爆量" in sig or "放量" in sig for sig in signals)
    has_breakout = any("突破" in sig or "逼近" in sig for sig in signals)
    has_momentum = any("動能加速" in sig or "連漲" in sig for sig in signals)

    if has_volume and has_breakout and has_momentum:
        c.estimated_timing = "⚡ 1~3 日內（爆量+突破+動能，噴出前期）"
    elif has_volume and has_breakout:
        c.estimated_timing = "🔥 1~5 日內（量價突破，隨時發動）"
    elif has_breakout and has_momentum:
        c.estimated_timing = "📈 3~7 日內（突破+動能，醞釀攻勢）"
    elif has_volume:
        c.estimated_timing = "📊 5~10 日內（量能先行，觀察突破）"
    elif s >= 40:
        c.estimated_timing = "👀 1~2 週內（逐步轉強，逢回布局）"
    else:
        c.estimated_timing = "⏳ 需更多確認訊號"

    return c


def _gen_analysis(c: SurgeCandidate) -> SurgeCandidate:
    """生成分析文字"""
    lines = []
    s = c.surge_score

    if s >= 70:
        lines.append("🚀 **強烈噴出信號！爆量+突破+技術共振，多頭合力強勁**")
    elif s >= 55:
        lines.append("🔥 **明確突破信號，短線爆發機率高，可積極關注**")
    elif s >= 40:
        lines.append("📈 **潛在突破標的，部分因子轉強，納入觀察清單**")
    elif s >= 25:
        lines.append("👀 **初步技術徵兆，需更多確認訊號後再進場**")
    else:
        lines.append("⚪ **訊號雜亂，暫時不具突破條件**")

    lines.append("")
    lines.append(f"**📊 飆股潛力評分：{s:.0f}/100**")
    lines.append(f"  🔥 爆量 {c.vol_score:.0f}/20 | 🎯 突破 {c.breakout_score:.0f}/20")
    lines.append(f"  📊 技術 {c.tech_score:.0f}/20 | 🚀 動能 {c.momentum_score:.0f}/15")
    lines.append(f"  🗞️ 新聞 {c.news_score:.0f}/15 | 🏢 籌碼 {c.inst_score:.0f}/10")

    if c.surge_signals:
        lines.append("")
        lines.append("**💡 關鍵訊號：**")
        for sig in c.surge_signals[:6]:
            lines.append(f"  • {sig}")

    if c.risk_warnings:
        lines.append("")
        lines.append("**⚠️ 風險警示：**")
        for warn in c.risk_warnings[:4]:
            lines.append(f"  • {warn}")

    if c.entry_note:
        lines.append("")
        lines.append(f"**📌 進場建議：** {c.entry_note}")

    lines.append("")
    lines.append(f"⏱️ **預估發動：** {c.estimated_timing}")
    lines.append(f"📌 現價 {c.current_price:.2f} | 量比 {c.volume_ratio:.1f}x")
    lines.append("⚠️ 飆股預測僅供參考，請嚴設停損")

    c.analysis = lines
    return c


# ─────────────────────────────────────────────
# 批量掃描
# ─────────────────────────────────────────────

def scan_surge_candidates(
    top_n: int = 10,
    months: int = 6,
    stage1_max: int = 200,
) -> list:
    """
    兩階段飆股掃描

    Stage 1: 快速掃描 200 檔之技術面，取前 20~30 名
    Stage 2: 對候選做完整分析（含多源新聞+籌碼），排前 top_n
    """
    # ─── Stage 1：快速技術篩選 ───
    stage1_results = []
    total = min(len(UNIVERSE), stage1_max)

    with ThreadPoolExecutor(max_workers=12) as ex:
        fut_map = {
            ex.submit(surge_score_stock, sid, sname, months=months): sid
            for sid, sname in list(UNIVERSE.items())[:stage1_max]
        }
        for f in as_completed(fut_map):
            try:
                r = f.result(timeout=25)
                if r.error is None and r.surge_score > 0:
                    stage1_results.append(r)
            except Exception:
                pass

    # 排序取前30名進 Stage 2
    stage1_results.sort(key=lambda x: x.surge_score, reverse=True)
    candidates = stage1_results[:min(25, len(stage1_results))]

    # ─── Stage 2：完整分析（含新聞+籌碼） ───
    final_results = []

    for c in candidates:
        try:
            full = surge_score_stock_full(c.stock_id, c.stock_name, months=months)
            if full.error is None:
                final_results.append(full)
        except Exception:
            continue
        time.sleep(0.1)  # 避免打太兇

    final_results.sort(key=lambda x: x.surge_score, reverse=True)
    return final_results[:top_n]


def get_market_surge_context() -> dict:
    """市場背景對飆股環境的影響"""
    ctx = {
        "environment": "未知",
        "suitability": "⚪ 待判斷",
        "note": "",
    }
    try:
        df = _get_data_fetcher().fetch_historical("0050", months=12)
        if df.empty or len(df) < 60:
            return ctx

        df = _get_indicators().add_all_indicators(df)
        latest = df.iloc[-1]
        cur = latest["Close"]
        ma20 = latest.get("MA20", 0)
        ma60 = latest.get("MA60", 0)

        # 多頭市場：飆股容易發動
        if cur > ma20 > ma60 and ma60 > 0:
            ctx["environment"] = "🟢 多頭市場（有利飆股）"
            ctx["suitability"] = "🟢 非常適合"
            ctx["note"] = "多頭格局，資金充沛，飆股容易形成連續噴出走勢"
        # 盤整市場：個股表現
        elif cur > ma20:
            ctx["environment"] = "🟡 短多盤整（個股表現）"
            ctx["suitability"] = "🟡 適中"
            ctx["note"] = "大盤震盪，有題材的個股仍有機會突破"
        # 空頭市場：飆股難度提高
        else:
            ctx["environment"] = "🔴 空頭格局（飆股難度高）"
            ctx["suitability"] = "🔴 不適合"
            ctx["note"] = "空頭市場資金退潮，飆股持續性較差，應嚴格控制倉位"

        # 成交量環境
        if "Volume" in df.columns:
            vol = df["Volume"].values
            recent_vol = vol[-5:].mean()
            avg_vol = vol[-21:-1].mean() if len(vol) > 21 else vol.mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
            if vol_ratio > 1.2:
                ctx["note"] += f"，市場量能增溫（{vol_ratio:.1f}x）"
            elif vol_ratio < 0.7:
                ctx["note"] += f"，市場量能不足（{vol_ratio:.1f}x）"

    except Exception:
        pass
    return ctx
