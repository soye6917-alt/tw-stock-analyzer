"""
📊 專家級單股分析系統
- 綜合技術分析（多週期、多指標）
- 新聞情緒/消息面解讀
- 內線/異常訊號偵測（爆量、跳空、大戶籌碼）
- 投資專家綜合建議（含風險評估、進出策略）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import re
import json
import urllib.parse

from data_fetcher import fetch_historical, fetch_realtime_quote, get_stock_name
from indicators import add_all_indicators, get_indicator_signals
from fundamentals import fetch_fundamentals, fetch_institutional_trading
from daily_picks import fetch_news_sentiment
import pattern_recognition as pr


# ============================================================
# 0. 跨平台 HTTP 請求工具
# ============================================================
def _safe_http_get(url: str, timeout: int = 10, headers: dict = None) -> Optional[str]:
    """安全發送 HTTP GET 請求，回傳 text 或 None
    自動偵測編碼：先用 apparent_encoding，若為非中文編碼則嘗試 utf-8 / big5
    """
    import requests as req
    try:
        default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
        }
        if headers:
            default_headers.update(headers)
        resp = req.get(url, headers=default_headers, timeout=timeout)
        # 自動偵測編碼，避免台灣網站 Big5/UTF-8 混用導致亂碼
        apparent = (resp.apparent_encoding or "").lower()
        if apparent and any(c in apparent for c in ["utf", "big5", "cp950"]):
            resp.encoding = resp.apparent_encoding
        else:
            # 嘗試 UTF-8，失敗再試 Big5
            try:
                resp.content.decode("utf-8")
                resp.encoding = "utf-8"
            except (UnicodeDecodeError, LookupError):
                resp.encoding = "big5"
        resp.raise_for_status()
        return resp.text
    except ImportError:
        return None
    except Exception:
        return None


def _apply_encoding(resp):
    """自動偵測並設定 requests Response 的編碼（UTF-8 / Big5），避免台灣網站亂碼"""
    apparent = (resp.apparent_encoding or "").lower()
    if apparent and any(c in apparent for c in ["utf", "big5", "cp950"]):
        resp.encoding = resp.apparent_encoding
    else:
        try:
            resp.content.decode("utf-8")
            resp.encoding = "utf-8"
        except (UnicodeDecodeError, LookupError):
            resp.encoding = "big5"


# ============================================================
# 0a. 公開資訊觀測站 (MOPS) 資料
# ============================================================
def fetch_mops_monthly_revenue(stock_id: str) -> list:
    """
    從公開資訊觀測站取得近期月營收資料
    回傳 list of dict: [{month, revenue_yoy, revenue_mom, cumulative_yoy}, ...]
    """
    results = []
    try:
        import requests as req
        url = "https://mops.twse.com.tw/mops/web/ajax_t51sb01"
        form_data = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "TYPEK": "sii",
            "year": str(datetime.now().year - 1911),  # 民國年
            "month": str(datetime.now().month),
            "co_id": stock_id,
        }
        # 先試上市
        resp = req.post(url, data=form_data, timeout=10,
                        headers={"User-Agent": "Mozilla/5.0"})
        _apply_encoding(resp)
        html = resp.text

        # 如果上市找不到，試上櫃
        if "查無資料" in html:
            form_data["TYPEK"] = "otc"
            resp = req.post(url, data=form_data, timeout=10,
                            headers={"User-Agent": "Mozilla/5.0"})
            _apply_encoding(resp)
            html = resp.text

        # 解析 HTML 表格 (MOPS 回傳統表格)
        # 尋找 <table> 中的月營收資料
        import re
        # 找所有行
        rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL)
        for row in rows[1:]:  # 跳過 header
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cells) >= 7 and cells[1].strip() == stock_id:
                entry = {
                    "month": cells[0].strip(),
                    "revenue": cells[2].strip(),
                    "revenue_yoy": cells[5].strip().replace(",", ""),
                    "revenue_mom": cells[6].strip().replace(",", ""),
                    "cumulative_yoy": cells[8].strip().replace(",", "") if len(cells) > 8 else "",
                }
                results.append(entry)
    except Exception:
        pass
    return results


def fetch_mops_financial_highlights(stock_id: str) -> dict:
    """
    從公開資訊觀測站取得簡易財務亮點
    """
    result = {"eps": None, "book_value": None, "revenue_growth": None}
    try:
        import requests as req
        now = datetime.now()
        roc_year = now.year - 1911
        season = (now.month - 1) // 3  # 0,1,2,3

        url = "https://mops.twse.com.tw/mops/web/ajax_t163sb06"
        form_data = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "TYPEK": "sii",
            "year": str(roc_year),
            "season": str(season + 1),  # 1-4
            "co_id": stock_id,
        }
        resp = req.post(url, data=form_data, timeout=10,
                        headers={"User-Agent": "Mozilla/5.0"})
        _apply_encoding(resp)
        html = resp.text

        # 基本 EPS
        eps_match = re.search(r'基本每股盈餘[^0-9]*([0-9,.]+)', html)
        if eps_match:
            result["eps"] = eps_match.group(1).replace(",", "")

        # 每股淨值
        bv_match = re.search(r'每股淨值[^0-9]*([0-9,.]+)', html)
        if bv_match:
            result["book_value"] = bv_match.group(1).replace(",", "")

    except Exception:
        pass
    return result


# ============================================================
# 0b. UDN 聯合新聞網 新聞爬取
# ============================================================
def fetch_udn_news(stock_id: str, stock_name: str) -> list:
    """從 UDN 搜尋該股票相關新聞（最多30則，多關鍵字交叉搜尋）"""
    news_list = []
    try:
        import requests as req
        # 多重查詢關鍵字以獲取更多結果
        queries = [
            f"{stock_id} {stock_name} 股票",
            f"{stock_id} {stock_name} 股市",
            f"{stock_id} {stock_name} 投資",
        ]
        seen_titles = set()
        for q in queries:
            query = urllib.parse.quote(q)
            url = f"https://udn.com/search/word/query/{query}/searchtype/1"
            resp = req.get(url, timeout=10,
                           headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            _apply_encoding(resp)
            html = resp.text

            titles = re.findall(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
            if not titles:
                titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html, re.DOTALL)
            if not titles:
                titles = re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.DOTALL)
            for t in titles:
                clean = re.sub(r'<[^>]+>', '', t).strip()
                if clean and len(clean) > 5 and clean not in seen_titles:
                    seen_titles.add(clean)
                    news_list.append(clean)
                if len(news_list) >= 30:
                    break
            if len(news_list) >= 30:
                break
    except Exception:
        pass
    return news_list[:30]


# ============================================================
# 0c. CMoney 籌碼面輔助資料
# ============================================================
def fetch_cmoney_institutional_summary(stock_id: str) -> dict:
    """
    從 CMoney 籌碼面取得法人買賣超彙整
    (使用 cmoney.tw 公開頁面)
    """
    result = {
        "foreign_net": None,
        "sitc_net": None,
        "dealer_net": None,
        "total_net": None,
    }
    try:
        import requests as req
        url = f"https://www.cmoney.tw/finance/html/financialreport/InstitutionalInvestors.aspx?StockId={stock_id}"
        resp = req.get(url, timeout=10,
                       headers={"User-Agent": "Mozilla/5.0"})
        _apply_encoding(resp)
        html = resp.text

        # 法人買賣超數值 (找買超/賣超數字)
        numbers = re.findall(r'<td[^>]*class="[^"]*num[^"]*"[^>]*>([^<]+)</td>', html)
        if numbers and len(numbers) >= 3:
            for i, n in enumerate(numbers[:3]):
                n_clean = n.replace(",", "").strip()
                try:
                    val = int(n_clean) if n_clean.lstrip('-').isdigit() else 0
                except ValueError:
                    val = 0
                if i == 0:
                    result["foreign_net"] = val
                elif i == 1:
                    result["sitc_net"] = val
                elif i == 2:
                    result["dealer_net"] = val
            result["total_net"] = (result["foreign_net"] or 0) + (result["sitc_net"] or 0) + (result["dealer_net"] or 0)
    except Exception:
        pass
    return result


# ============================================================
# 0d. Google 新聞搜尋
# ============================================================
def fetch_google_news_sentiment(stock_id: str, stock_name: str) -> list:
    """使用 Google News RSS 搜尋股票相關新聞，最多取 30 則"""
    headlines = []
    try:
        import requests as req
        import urllib.parse
        # 多關鍵字查詢以增加結果
        queries = [
            f"{stock_id} {stock_name}",
            f"{stock_id} 股票",
            f"{stock_id} 台股",
        ]
        seen = set()
        for q in queries:
            query = urllib.parse.quote(q)
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&num=30"
            resp = req.get(url, timeout=10,
                           headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            _apply_encoding(resp)
            titles = re.findall(r'<title>(.*?)</title>', resp.text, re.DOTALL)
            for t in titles[1:]:  # 跳過 RSS feed title
                clean = t.strip()
                if clean and len(clean) > 5 and clean not in seen:
                    seen.add(clean)
                    headlines.append(clean)
                if len(headlines) >= 30:
                    break
            if len(headlines) >= 30:
                break
    except Exception:
        pass
    return headlines[:30]


# ============================================================
# 0d. 其他財經新聞來源（補充 50 則目標）
# ============================================================

def fetch_moneydj_news(stock_id: str, stock_name: str) -> list:
    """從 MoneyDJ 財經新聞搜尋該股票相關新聞"""
    news = []
    try:
        import requests as req
        import urllib.parse
        query = urllib.parse.quote(f"{stock_id} {stock_name}")
        url = f"https://www.moneydj.com/search/news/{query}"
        resp = req.get(url, timeout=10,
                       headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        _apply_encoding(resp)
        html = resp.text
        # 擷取新聞標題
        titles = re.findall(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
        if not titles:
            titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html, re.DOTALL)
        for t in titles:
            clean = re.sub(r'<[^>]+>', '', t).strip()
            if clean and len(clean) > 5:
                news.append(clean)
                if len(news) >= 20:
                    break
    except Exception:
        pass
    return news


def fetch_anue_news(stock_id: str, stock_name: str) -> list:
    """從鉅亨網搜尋該股票相關新聞"""
    news = []
    try:
        import requests as req
        import urllib.parse
        query = urllib.parse.quote(f"{stock_id} {stock_name}")
        url = f"https://news.cnyes.com/search?keyword={query}&exp=da"
        resp = req.get(url, timeout=10,
                       headers={
                           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                           "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                       })
        _apply_encoding(resp)
        html = resp.text
        # 擷取新聞標題
        titles = re.findall(r'<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
        if not titles:
            titles = re.findall(r'<h3[^>]*>(.*?)</h3>', html, re.DOTALL)
        if not titles:
            titles = re.findall(r'<h2[^>]*>(.*?)</h2>', html, re.DOTALL)
        for t in titles:
            clean = re.sub(r'<[^>]+>', '', t).strip()
            if clean and len(clean) > 5:
                news.append(clean)
                if len(news) >= 20:
                    break
    except Exception:
        pass
    return news



# ============================================================
# 0e. 產業分析
# ============================================================

# 台灣股票產業對照表（常用股之產業映射，作為 Yahoo API 容錯）
INDUSTRY_FALLBACK = {
    "2330": "半導體", "2454": "半導體", "2303": "半導體", "5347": "半導體",
    "3711": "半導體", "6770": "半導體", "3034": "半導體",
    "2317": "其他電子", "2382": "其他電子", "3231": "其他電子",
    "2357": "電腦週邊", "2376": "電腦週邊", "2356": "電腦週邊",
    "2383": "電子零組件", "2327": "電子零組件", "8046": "電子零組件",
    "2308": "電子零組件", "3037": "電子零組件",
    "2498": "通訊網路", "2412": "通訊網路", "3045": "通訊網路", "4904": "通訊網路",
    "3008": "光電", "3673": "光電",
    "2002": "鋼鐵", "2027": "鋼鐵", "2031": "鋼鐵",
    "1301": "塑膠", "1303": "塑膠", "1326": "塑膠", "6505": "塑膠",
    "1216": "食品", "1210": "食品", "1215": "食品", "1232": "食品",
    "2881": "金融業", "2882": "金融業", "2883": "金融業",
    "2884": "金融業", "2885": "金融業", "2886": "金融業",
    "2887": "金融業", "2890": "金融業", "2891": "金融業", "2892": "金融業",
    "5880": "金融業", "2888": "金融業",
    "2603": "航運", "2609": "航運", "2615": "航運", "2618": "航運", "2606": "航運",
    "9910": "其他", "2912": "其他",
    "1101": "水泥", "1102": "水泥",
    "1402": "紡織", "1476": "紡織",
    "2207": "汽車", "2227": "汽車",
    "2105": "橡膠", "2101": "橡膠",
    "9904": "百貨", "2915": "百貨",
    "3049": "營建", "2534": "營建",
}


def get_stock_industry(stock_id: str, stock_name: str) -> str:
    """
    從 Yahoo 奇摩股市取得股票所屬產業，失敗時回傳靜態映射表
    回傳：產業中文名稱（如"半導體"、"金融業"）
    """
    # 先試 Yahoo
    try:
        import requests as req
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = req.get(url, timeout=10, headers=headers)
        if resp.status_code == 200:
            # 找到產業分類：出現在「XX產業漲跌排行」的 XX 部分
            import re
            matches = re.findall(r'<span class="Fz\(20px\) Fw\(600\)">(.+?)產業', resp.text)
            if not matches:
                matches = re.findall('<span class="Fz\\(20px\\) Fw\\(600\\)">(.+?)產業', resp.text)
            for m in matches:
                clean = re.sub(r'<[^>]+>', '', m).strip()
                # 過濾掉非產業的文字（如「網友也在看」這類sidebar其他區塊）
                valid_industries = [
                    "半導體", "金融", "航運", "鋼鐵", "塑膠", "食品", "電子零組件",
                    "其他電子", "電腦週邊", "通訊網路", "光電", "水泥", "紡織",
                    "汽車", "橡膠", "百貨", "營建", "生技醫療", "化學", "電機",
                    "貿易百貨", "油電燃氣", "運動休閒", "其他", "資訊服務",
                ]
                for vi in valid_industries:
                    if vi in clean:
                        return vi
                # 如果上面沒匹配到但文字合理，直接用
                if clean and len(clean) <= 6:
                    return clean
    except Exception:
        pass

    # 容錯：靜態映射表
    return INDUSTRY_FALLBACK.get(stock_id, "其他")


def fetch_industry_news(industry: str, max_items: int = 15) -> list:
    """搜尋該產業的相關新聞，評估產業景氣趨勢"""
    news = []
    try:
        import requests as req
        import urllib.parse
        # Google News 搜尋產業名稱 + "前景"
        query = urllib.parse.quote(f"{industry} 產業 前景")
        url = f"https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&num=15"
        resp = req.get(url, timeout=10,
                       headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        _apply_encoding(resp)
        import re
        titles = re.findall(r'<title>(.*?)</title>', resp.text, re.DOTALL)
        seen = set()
        for t in titles[1:]:
            clean = t.strip()
            if clean and len(clean) > 5 and clean not in seen:
                seen.add(clean)
                news.append(clean)
                if len(news) >= max_items:
                    break
    except Exception:
        pass
    return news


def analyze_industry_outlook(stock_id: str, stock_name: str) -> dict:
    """
    產業分析綜合判斷
    回傳：{industry, outlook_score, trends, summary_lines}
    """
    result = {
        "industry": "未分類",
        "outlook_score": 0,
        "trends": [],
        "news": [],
        "summary_lines": [],
    }

    industry = get_stock_industry(stock_id, stock_name)
    result["industry"] = industry

    lines = []
    positive_keywords = ["成長", "擴增", "需求", "復甦", "突破", "利多", "調升", "擴產"]
    negative_keywords = ["衰退", "庫存", "降溫", "利空", "緊縮", "供過於求", "關廠", "裁員"]

    # 產業新聞
    try:
        industry_news = fetch_industry_news(industry)
        result["news"] = industry_news
        if industry_news:
            pos = sum(1 for h in industry_news if any(kw in h for kw in positive_keywords))
            neg = sum(1 for h in industry_news if any(kw in h for kw in negative_keywords))
            net = pos - neg
            # 標準化到 -10 ~ +10
            total = len(industry_news)
            if total > 0:
                result["outlook_score"] = round((net / total) * 10, 1)
    except Exception:
        pass

    # 產業景氣判斷
    score = result["outlook_score"]
    lines.append(f"🏭 **所屬產業：{industry}**")
    lines.append(f"📰 **產業相關新聞（{len(result['news'])} 則）**")

    if score >= 4:
        lines.append(f"📈 **產業景氣展望：正向**（評分 {score:+.1f}）")
        lines.append("  → 該產業近期利多消息為主，產業處於成長/復甦階段")
    elif score <= -4:
        lines.append(f"📉 **產業景氣展望：保守**（評分 {score:+.1f}）")
        lines.append("  → 該產業近期利空消息較多，可能需要留意產業下行風險")
    else:
        lines.append(f"⚖️ **產業景氣展望：中性**（評分 {score:+.1f}）")
        lines.append("  → 產業訊息多空交錯，需獨立判斷")

    # 補充產業常識（通用分析）
    lines.append("")
    lines.append("💡 **產業分析補充：**")
    lines.append("  • 個股表現會受所屬產業景氣循環影響")
    lines.append("  • 產業龍頭股的動向常是產業風向球")
    lines.append("  • 需關注上中下游供應鏈變化、政策法規、國際競爭")
    lines.append(f"  • 綜合技術面與消息面已反映產業資訊於股價中")

    result["summary_lines"] = lines
    return result

# ============================================================
# 1. 綜合技術分析
# ============================================================
def comprehensive_technical_analysis(df: pd.DataFrame) -> dict:
    """
    全方位技術分析 — 涵蓋所有指標、多週期解讀
    回傳詳細的技術面報告 dict
    """
    result = {
        "trend": {},           # 趨勢判斷
        "indicators": {},      # 各指標狀態
        "patterns": {},        # 型態辨識
        "support_resistance": {},  # 支撐壓力
        "volume_analysis": {}, # 成交量分析
        "score": 0,            # 技術分數 -100~100
        "summary": [],         # 摘要文字
    }
    if df.empty or len(df) < 30:
        result["summary"].append("⚠️ 資料不足，技術分析無法進行")
        return result

    latest = df.iloc[-1]
    close = latest["Close"]
    score = 0
    lines = []

    # ── 均線趨勢判定 ──
    ma5 = latest.get("MA5", 0)
    ma10 = latest.get("MA10", 0)
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    ma120 = latest.get("MA120", 0)
    valid_mas = [m for m in [ma5, ma10, ma20, ma60, ma120] if m > 0]

    if len(valid_mas) >= 3:
        # 多頭排列: 短 > 中 > 長
        if ma5 > ma20 > ma60:
            trend = "多頭排列 📈"
            trend_strength = "強勁"
            score += 25
            lines.append("🟢 **強力多頭排列** — 5日>20日>60日，多頭格局明確")
        elif ma5 > ma20 and ma20 < ma60:
            trend = "短多中空 ⚡"
            trend_strength = "分歧"
            lines.append("🟡 **短多中空** — 站上月線但季線仍在頭上，反彈格局")
        elif ma5 < ma20 and ma20 > ma60:
            trend = "短空中多 ⚡"
            trend_strength = "分歧"
            lines.append("🟡 **短空中多** — 跌破月線但季線仍向上，漲多拉回")
            score -= 5
        elif ma5 < ma20 < ma60:
            trend = "空頭排列 📉"
            trend_strength = "弱勢"
            score -= 25
            lines.append("🔴 **空頭排列** — 5日<20日<60日，空頭格局明確")
        else:
            trend = "盤整 ⚖️"
            trend_strength = "中性"
            lines.append("⚪ **盤整格局** — 均線交錯，方向不明確")
    else:
        trend = "資料不足"
        trend_strength = ""
        lines.append("⚠️ 均線資料不足")

    # 股價與月線距離
    if ma20 > 0:
        dist_ma20 = (close - ma20) / ma20 * 100
        if abs(dist_ma20) < 1:
            lines.append(f"💡 股價與月線幾乎貼齊 ({dist_ma20:+.2f}%)，方向即將表態")
        elif dist_ma20 > 5:
            lines.append(f"⚡ 股價在月線上 {dist_ma20:+.1f}%，短線偏強但留意乖離過大")
            score += 5
        elif dist_ma20 > 0:
            lines.append(f"📊 股價在月線上 {dist_ma20:+.1f}%，短線偏多")
            score += 3
        elif dist_ma20 < -5:
            lines.append(f"💥 股價在月線下 {abs(dist_ma20):.1f}%，短線偏弱，留意超跌反彈")
            score -= 5
        else:
            lines.append(f"📊 股價在月線下 {abs(dist_ma20):.1f}%，短線偏弱")
            score -= 3

    # ── RSI 深層解讀 ──
    rsi = latest.get("RSI", 50)
    if len(df) > 14:
        prev_rsi = df["RSI"].iloc[-2]
        rsi_div = rsi - prev_rsi
        # 一般區間
        if rsi > 80:
            lines.append(f"🔴 **RSI {rsi:.1f}** — 嚴重超買區！短線過熱風險極高，建議勿追高")
            score -= 20
        elif rsi > 70:
            lines.append(f"🟡 **RSI {rsi:.1f}** — 超買區，短線可能有拉回壓力，留意是否背離")
            score -= 10
        elif rsi > 60:
            lines.append(f"📈 **RSI {rsi:.1f}** — 偏強區間，動能仍佳但未過熱")
            score += 5
        elif rsi > 40:
            lines.append(f"⚪ **RSI {rsi:.1f}** — 中性區間")
        elif rsi > 30:
            lines.append(f"📉 **RSI {rsi:.1f}** — 偏弱區間")
            score -= 5
        elif rsi > 20:
            lines.append(f"🟢 **RSI {rsi:.1f}** — 超賣區，留意反彈機會")
            score += 15
        else:
            lines.append(f"💚 **RSI {rsi:.1f}** — 嚴重超賣！可能超跌反彈")
            score += 20

        # RSI 背離偵測（簡化版）
        if len(df) > 20:
            rsi_14d_ago = df["RSI"].iloc[-15:-1].mean()
            close_14d_ago = df["Close"].iloc[-15:-1].mean()
            if close > close_14d_ago * 1.05 and rsi < rsi_14d_ago * 0.95:
                lines.append("⚠️ **RSI 頂背離!** 股價創高但RSI沒跟上，可能即將反轉")
                score -= 15
            elif close < close_14d_ago * 0.95 and rsi > rsi_14d_ago * 1.05:
                lines.append("💡 **RSI 底背離!** 股價創新低但RSI已回升，可能即將反彈")
                score += 15

    # ── MACD 深層解讀 ──
    macd_val = latest.get("MACD", 0)
    macd_signal = latest.get("MACD_Signal", 0)
    macd_hist = latest.get("MACD_Hist", 0)
    if len(df) > 2:
        prev_macd = df["MACD"].iloc[-2]
        prev_hist = df["MACD_Hist"].iloc[-2]

        # MACD 位階
        if macd_val > macd_signal:
            if prev_macd <= prev_hist:  # 剛黃金交叉
                lines.append(f"🟢 **MACD 剛黃金交叉!** DIF({macd_val:.2f})上穿訊號線({macd_signal:.2f})，動能轉多")
                score += 15
            else:
                lines.append(f"🟢 MACD 在訊號線上方(DIF:{macd_val:.2f})，持續偏多")
                score += 8

            # 柱狀圖動能
            if macd_hist > prev_hist:
                lines.append(f"📈 MACD柱狀圖擴大中 ({prev_hist:.2f}→{macd_hist:.2f})，多頭動能增強")
                score += 5
            elif macd_hist < prev_hist:
                lines.append(f"📊 MACD柱狀圖收斂中 ({prev_hist:.2f}→{macd_hist:.2f})，多頭動能減弱")
                score -= 5
        else:
            if prev_macd >= prev_hist:  # 剛死亡交叉
                lines.append(f"🔴 **MACD 剛死亡交叉!** DIF({macd_val:.2f})跌破訊號線({macd_signal:.2f})，動能轉空")
                score -= 15
            else:
                lines.append(f"🔴 MACD 在訊號線下方(DIF:{macd_val:.2f})，持續偏空")
                score -= 8

            if macd_hist < prev_hist:
                lines.append(f"📉 MACD柱狀圖擴大中 ({prev_hist:.2f}→{macd_hist:.2f})，空頭動能增強")
                score -= 5
            elif macd_hist > prev_hist:
                lines.append(f"📊 MACD柱狀圖收斂中 ({prev_hist:.2f}→{macd_hist:.2f})，空頭動能減弱")
                score += 5

        # DIF 在零軸上下
        if macd_val > 0:
            lines.append(f"📊 MACD DIF 在零軸之上({macd_val:.2f})，中長期偏多")
            score += 5
        else:
            lines.append(f"📊 MACD DIF 在零軸之下({macd_val:.2f})，中長期偏空")
            score -= 5

    # ── 布林通道深層解讀 ──
    bb_upper = latest.get("BB_Upper", 0)
    bb_lower = latest.get("BB_Lower", 0)
    bb_mid = latest.get("BB_Mid", 0)
    bb_width = latest.get("BB_Width", 0)
    if bb_upper > 0:
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pos > 95:
            lines.append(f"🔴 **股價觸及布林上軌！** ({bb_pos:.0f}%)，短線壓力沉重，勿追高")
            score -= 12
        elif bb_pos > 80:
            lines.append(f"🟡 股價在布林上軌附近 ({bb_pos:.0f}%)，短線有壓")
            score -= 5
        elif bb_pos < 5:
            lines.append(f"🟢 **股價觸及布林下軌！** ({bb_pos:.0f}%)，短線超跌，可能反彈")
            score += 12
        elif bb_pos < 20:
            lines.append(f"🟢 股價在布林下軌附近 ({bb_pos:.0f}%)，接近支撐")
            score += 5
        else:
            lines.append(f"⚪ 股價在布林通道中部 ({bb_pos:.0f}%)")

        # 布林通道寬度（波動率變化）
        if len(df) > 20:
            avg_width = df["BB_Width"].iloc[-20:].mean()
            if bb_width > avg_width * 1.3:
                lines.append(f"⚠️ **波動率放大！** 布林通道拓寬中，可能出現趨勢行情")
            elif bb_width < avg_width * 0.7:
                lines.append(f"💤 波動率收縮(布林通道變窄)，盤整待變")

    # ── KDJ 解讀 ──
    k_val = latest.get("K", 50)
    d_val = latest.get("D", 50)
    j_val = latest.get("J", 50)
    if len(df) > 9:
        if k_val > d_val:
            if k_val > 80:
                lines.append(f"🔴 KDJ K值{k_val:.1f}>D值{d_val:.1f} 且>80，超買區交叉向上")
                score -= 8
            else:
                lines.append(f"🟢 KDJ K值{k_val:.1f}>D值{d_val:.1f}，短線偏多")
                score += 5
        else:
            if k_val < 20:
                lines.append(f"🟢 KDJ K值{k_val:.1f}<D值{d_val:.1f} 且<20，超賣區，留意反彈")
                score += 8
            else:
                lines.append(f"🔴 KDJ K值{k_val:.1f}<D值{d_val:.1f}，短線偏空")
                score -= 5

    # ── 成交量分析 ──
    if "Volume" in df.columns and len(df) > 10:
        avg_vol_5 = df["Volume"].iloc[-6:-1].mean()
        avg_vol_20 = df["Volume"].iloc[-21:-1].mean()
        cur_vol = df["Volume"].iloc[-1]
        if avg_vol_5 > 0:
            vol_ratio_5 = cur_vol / avg_vol_5
            if vol_ratio_5 > 2:
                lines.append(f"⚡ **爆量！** 今日成交量為近5日均量的{vol_ratio_5:.1f}倍，市場關注度急升")
                # 量價關係判斷
                if close >= df["Close"].iloc[-2]:
                    lines.append("✅ 價漲量增，多頭攻擊訊號")
                    score += 10
                else:
                    lines.append("❌ 價跌量增，出貨或恐慌賣壓")
                    score -= 15
            elif vol_ratio_5 > 1.5:
                lines.append(f"📊 成交量放大至近5日均量的{vol_ratio_5:.1f}倍")
                if close >= df["Close"].iloc[-2]:
                    score += 5
                else:
                    score -= 5
            elif vol_ratio_5 < 0.5:
                lines.append(f"💤 量縮至近5日平均的{vol_ratio_5:.1f}倍，市場觀望")
            else:
                lines.append(f"📊 成交量正常 ({vol_ratio_5:.1f}x近5日均量)")

        # 20日均量趨勢
        if avg_vol_20 > 0 and len(df) > 25:
            vol_20_ago = df["Volume"].iloc[-26:-6].mean()
            if vol_20_ago > 0:
                vol_trend = avg_vol_5 / vol_20_ago
                if vol_trend > 1.2:
                    lines.append(f"📈 **近5日成交量較前20日增加{((vol_trend-1)*100):.0f}%，市場關注度提升**")
                elif vol_trend < 0.8:
                    lines.append(f"📉 近5日成交量較前20日減少{((1-vol_trend)*100):.0f}%，人氣退潮")

    # 裝填結果
    result["trend"] = {
        "state": trend,
        "strength": trend_strength,
        "price_vs_ma20": dist_ma20 if ma20 > 0 else None,
    }
    result["indicators"] = {
        "rsi": round(rsi, 1),
        "macd": round(macd_val, 3),
        "macd_signal": round(macd_signal, 3),
        "macd_hist": round(macd_hist, 3),
        "bb_position": round(bb_pos, 1) if bb_upper > 0 else None,
        "kdj": {"k": round(k_val, 1), "d": round(d_val, 1), "j": round(j_val, 1)},
    }
    result["score"] = max(-100, min(100, score))
    result["summary"] = lines
    return result


# ============================================================
# 2. 內線/異常訊號偵測
# ============================================================
def detect_abnormal_signals(stock_id: str, df: pd.DataFrame) -> dict:
    """
    偵測非正常市場訊號：
    - 爆量/異常量
    - 跳空缺口
    - 大戶/主力籌碼變化
    - 連續異常走勢
    """
    result = {
        "signals": [],
        "risk_level": "低",
        "details": [],
    }
    lines = []
    risk_count = 0

    if df.empty or len(df) < 30:
        return result

    latest = df.iloc[-1]
    close = latest["Close"]

    # ── 跳空缺口偵測 ──
    gaps = []
    for i in range(max(1, len(df)-20), len(df)):
        prev_high = df["High"].iloc[i-1]
        prev_low = df["Low"].iloc[i-1]
        curr_open = df["Open"].iloc[i]
        curr_low = df["Low"].iloc[i]
        curr_high = df["High"].iloc[i]

        gap_up = curr_low > prev_high  # 向上跳空
        gap_down = curr_high < prev_low  # 向下跳空

        if gap_up:
            gap_size = (curr_low - prev_high) / prev_high * 100
            gaps.append({
                "date": str(df.index[i])[:10],
                "type": "向上跳空",
                "size": round(gap_size, 2),
            })
        elif gap_down:
            gap_size = (prev_low - curr_high) / prev_high * 100
            gaps.append({
                "date": str(df.index[i])[:10],
                "type": "向下跳空",
                "size": round(gap_size, 2),
            })

    if gaps:
        recent_gaps = [g for g in gaps if g["size"] > 1]
        if recent_gaps:
            up_gaps = [g for g in recent_gaps if g["type"] == "向上跳空"]
            down_gaps = [g for g in recent_gaps if g["type"] == "向下跳空"]
            if len(up_gaps) >= 2:
                lines.append(f"⚡ **近期出現{len(up_gaps)}次向上跳空！** 可能有利多消息或主力拉抬")
                result["signals"].append("向上跳空")
            if len(down_gaps) >= 2:
                lines.append(f"🔴 **近期出現{len(down_gaps)}次向下跳空！** 可能有利空消息")
                result["signals"].append("向下跳空")
                risk_count += 1
            for g in recent_gaps[:5]:
                lines.append(f"  • {g['date']} {g['type']} {g['size']:.1f}%")

    # ── 爆量偵測 ──
    if "Volume" in df.columns and len(df) > 20:
        avg_vol = df["Volume"].iloc[-21:-1].mean()
        recent_vols = df["Volume"].iloc[-5:]
        if avg_vol > 0:
            for i in range(len(recent_vols)):
                vol_ratio = recent_vols.iloc[i] / avg_vol
                if vol_ratio > 2.5:
                    date_str = str(recent_vols.index[i])[:10]
                    price = df["Close"].iloc[-5+i]
                    lines.append(f"⚡ **{date_str} 爆量 {vol_ratio:.1f}x 均量！** (收盤 {price:.2f})")
                    result["signals"].append("異常爆量")
                    # 判斷爆量方向
                    if price >= df["Close"].iloc[-6+i]:
                        lines.append(f"  → 爆量上漲，可能是主力買進或利多發動")
                    else:
                        lines.append(f"  → 爆量下跌，可能是主力出貨或恐慌賣壓")
                        risk_count += 2

    # ── 連續漲跌走勢 ──
    if len(df) >= 10:
        recent_returns = df["Close"].pct_change().iloc[-11:]
        # 連漲/連跌
        consecutive_up = 0
        consecutive_down = 0
        for i in range(len(recent_returns)-1, -1, -1):
            if recent_returns.iloc[i] > 0:
                consecutive_up += 1
                consecutive_down = 0
            elif recent_returns.iloc[i] < 0:
                consecutive_down += 1
                consecutive_up = 0
            else:
                break
            if i == 0:
                break

        if consecutive_up >= 5:
            lines.append(f"🔥 **連續{consecutive_up}日上漲！** 短線強勢但留意獲利了結賣壓")
            result["signals"].append(f"連{consecutive_up}漲")
            risk_count += 1
        elif consecutive_up >= 3:
            lines.append(f"📈 連{consecutive_up}漲，短線偏強")
        elif consecutive_down >= 5:
            lines.append(f"🧊 **連續{consecutive_down}日下跌！** 短線弱勢但可能超跌反彈")
            result["signals"].append(f"連{consecutive_down}跌")
            risk_count += 2
        elif consecutive_down >= 3:
            lines.append(f"📉 連{consecutive_down}跌，短線偏弱")

    # ── 價格波動率異常 ──
    if len(df) > 30:
        recent_std = df["Close"].pct_change().iloc[-10:].std()
        long_std = df["Close"].pct_change().iloc[-30:].std()
        if long_std > 0 and recent_std > long_std * 2:
            lines.append(f"⚠️ **近期波動率異常放大！** (近10日波動為30日平均的{(recent_std/long_std):.1f}倍)")
            result["signals"].append("波動率異常")
            risk_count += 2

    # ── 三大法人累計流向(簡化) ──
    try:
        inst = fetch_institutional_trading(stock_id)
        if not inst.empty:
            total_net = inst[inst["類別"] == "三大法人合計"]["買賣超"].values
            if len(total_net) > 0:
                net = total_net[0]
                net_k = net / 1000
                if abs(net_k) > 10:
                    direction = "買超" if net > 0 else "賣超"
                    signal_type = f"三大法人{direction}{abs(net_k):.0f}張"
                    lines.append(f"🏢 **{signal_type}**！法人明顯{'偏多' if net > 0 else '偏空'}")
                    result["signals"].append(signal_type)
                    if net < 0:
                        risk_count += 2
                elif abs(net_k) > 3:
                    direction = "買超" if net > 0 else "賣超"
                    lines.append(f"🏢 三大法人{direction}{abs(net_k):.0f}張")
    except Exception:
        pass

    # 綜合風險評級
    if risk_count >= 5:
        result["risk_level"] = "高"
    elif risk_count >= 3:
        result["risk_level"] = "中"
    elif risk_count >= 1:
        result["risk_level"] = "中低"

    result["details"] = lines
    return result


# ============================================================
# 3. 新聞與市場消息
# ============================================================
def analyze_news_and_market(stock_id: str, stock_name: str) -> dict:
    """
    多來源新聞情緒分析：
    - Yahoo 奇摩股市（既有）
    - UDN 聯合新聞網
    - Google News 搜尋
    - 公開資訊觀測站營收數據
    僅看事實、數據；主動交叉比對
    """
    result = {
        "sentiment_score": 0,
        "headlines": [],
        "summary": [],
        "market_context": [],
        "sources": [],
        "revenue_data": [],
    }
    lines = []
    all_headlines = []
    positive_keywords = ["成長", "創高", "利多", "突破", "調升", "擴產", "獲利", "訂單",
                          "買超", "增資", "聯盟", "得標", "補助", "市占", "認證",
                          "漲停", "創新高", "年增", "轉盈", "配息", "股利", "併購",
                          "回溫", "復甦", "走強", "優於", "領先", "布局AI"]
    negative_keywords = ["衰退", "利空", "跌破", "調降", "裁員", "虧損", "訴訟",
                          "賣超", "違約", "下市", "警示", "處分", "跌停",
                          "暴跌", "重挫", "賣壓", "降評", "下滑", "年減",
                          "破底", "罰款", "庫存", "降溫", "終止"]

    # ---- 多來源新聞收集（目標 50+ 則） ----

    # 1. Yahoo 奇摩新聞 (既有)
    try:
        yahoo_score, yahoo_headlines, msg = fetch_news_sentiment(stock_id, stock_name)
        for h in yahoo_headlines:
            all_headlines.append(("Yahoo", h))
        positive_count = sum(1 for h in yahoo_headlines if any(kw in h for kw in positive_keywords))
        negative_count = sum(1 for h in yahoo_headlines if any(kw in h for kw in negative_keywords))
        yahoo_net = positive_count - negative_count
    except Exception:
        yahoo_score = 0
        yahoo_headlines = []
        yahoo_net = 0

    # 2. UDN 新聞（最多30則）
    try:
        udn_news = fetch_udn_news(stock_id, stock_name)
        seen_udn = set()
        for h in udn_news:
            if h not in seen_udn:
                seen_udn.add(h)
                all_headlines.append(("UDN", h))
        udn_positive = sum(1 for h in udn_news if any(kw in h for kw in positive_keywords))
        udn_negative = sum(1 for h in udn_news if any(kw in h for kw in negative_keywords))
        udn_net = udn_positive - udn_negative
    except Exception:
        udn_news = []
        udn_net = 0

    # 3. Google News（最多30則）
    try:
        google_news = fetch_google_news_sentiment(stock_id, stock_name)
        seen_gg = set()
        for h in google_news:
            if h not in seen_gg:
                seen_gg.add(h)
                all_headlines.append(("Google", h))
        gg_positive = sum(1 for h in google_news if any(kw in h for kw in positive_keywords))
        gg_negative = sum(1 for h in google_news if any(kw in h for kw in negative_keywords))
        gg_net = gg_positive - gg_negative
    except Exception:
        google_news = []
        gg_net = 0

    # 4. MoneyDJ 財經新聞（最多20則）
    try:
        moneydj_news = fetch_moneydj_news(stock_id, stock_name)
        seen_md = set()
        for h in moneydj_news:
            if h not in seen_md:
                seen_md.add(h)
                all_headlines.append(("MoneyDJ", h))
        md_positive = sum(1 for h in moneydj_news if any(kw in h for kw in positive_keywords))
        md_negative = sum(1 for h in moneydj_news if any(kw in h for kw in negative_keywords))
        md_net = md_positive - md_negative
    except Exception:
        moneydj_news = []
        md_net = 0

    # 5. 鉅亨網 Anue 財經新聞（最多20則）
    try:
        anue_news = fetch_anue_news(stock_id, stock_name)
        seen_an = set()
        for h in anue_news:
            if h not in seen_an:
                seen_an.add(h)
                all_headlines.append(("鉅亨網", h))
        an_positive = sum(1 for h in anue_news if any(kw in h for kw in positive_keywords))
        an_negative = sum(1 for h in anue_news if any(kw in h for kw in negative_keywords))
        an_net = an_positive - an_negative
    except Exception:
        anue_news = []
        an_net = 0

    # 4. MOPS 月營收(客觀事實)
    revenue_data = []
    try:
        revenue_data = fetch_mops_monthly_revenue(stock_id)
        if revenue_data:
            result["revenue_data"] = revenue_data
    except Exception:
        pass

    # 綜合情緒評分
    total_headlines = len([h for _, h in all_headlines])
    result["headlines"] = [h for _, h in all_headlines]
    net_sentiment = yahoo_net + udn_net + gg_net + md_net + an_net

    # 客觀評分：Yahoo 情緒占 30%，其餘各自按比例
    scores_sum = yahoo_score * 0.3 + net_sentiment * 0.7
    score = max(-10, min(10, scores_sum))
    result["sentiment_score"] = round(score, 1)

    # 來源統計
    source_count = {}
    for src, _ in all_headlines:
        source_count[src] = source_count.get(src, 0) + 1
    result["sources"] = source_count

    # ---- 生成分析摘要 ----
    if total_headlines > 0:
        lines.append(f"🗞️ **多來源新聞分析（共 {total_headlines} 則）**")
        for src, cnt in sorted(source_count.items(), key=lambda x: -x[1]):
            lines.append(f"  • {src}: {cnt} 則")
        if total_headlines < 20:
            lines.append(f"  ⚠️ 新聞量偏少（僅{total_headlines}則），可自行搜尋更多資訊")
    else:
        lines.append("📰 目前無近期相關新聞")

    # 營收事實
    if revenue_data:
        latest = revenue_data[0]
        yoy = latest.get("revenue_yoy", "")
        mom = latest.get("revenue_mom", "")
        lines.append(f"📊 **公開資訊觀測站月營收**（最新月份: {latest.get('month', 'N/A')}）")
        if yoy:
            yoy_val = float(yoy) if yoy.replace('-', '').replace('.', '').isdigit() else 0
            if yoy_val > 0:
                lines.append(f"  ✅ 年增率 {yoy_val:+.1f}% — 營收成長中")
                result["market_context"].append(f"營收年增{yoy_val:+.1f}%")
            elif yoy_val < 0:
                lines.append(f"  ❌ 年增率 {yoy_val:+.1f}% — 營收衰退")
                result["market_context"].append(f"營收年減{abs(yoy_val):.1f}%")
            else:
                lines.append(f"  • 年增率 {yoy}")

    # 情緒判斷（客觀、基於事實）
    if score >= 3:
        lines.append(f"📈 **綜合訊息偏多**（評分 {score:+.1f}）")
        if score >= 6:
            lines.append("  → 利多訊息集中，留意是否已反映在股價")
        result["market_context"].append("訊息偏多")
    elif score <= -3:
        lines.append(f"📉 **綜合訊息偏空**（評分 {score:+.1f}）")
        if score <= -6:
            lines.append("  → 利空訊息集中，留意是否過度反應")
        result["market_context"].append("訊息偏空")
    else:
        lines.append(f"⚖️ **綜合訊息中性**（評分 {score:+.1f}）— 多空訊息交錯")

    # 獨立思考提醒
    lines.append("")
    lines.append("💡 **獨立思考提醒：**")
    lines.append("  • 以上訊息來自 Yahoo、UDN、Google News、MoneyDJ、鉅亨網、公開資訊觀測站")
    lines.append("  • 僅呈現客觀數據與事實，不納入個人臆測")
    lines.append("  • 新聞情緒為關鍵字輔助分析，請自行釐清邏輯與因果關係")
    lines.append("  • 股價已反映已知訊息，真正影響市場的是「預期之外的變化」")

    result["summary"] = lines
    return result


# ============================================================
# 4. 專家級綜合建議
# ============================================================
def expert_recommendation(
    stock_id: str,
    stock_name: str,
    df: pd.DataFrame,
    tech_result: dict,
    abnormal_signals: dict,
    news_result: dict,
    industry_result: dict = None,
) -> dict:
    """
    綜合所有分析，產出專家級投資建議
    """
    result = {
        "overall_score": 0,
        "rating": "中立觀望",
        "rating_emoji": "⚖️",
        "rating_color": "#7f8c8d",
        "strategy": "",
        "entry_exit": {},
        "risk_assessment": {},
        "position_sizing": {},
        "detailed_reasoning": [],
        "final_words": [],
    }

    lines = []
    tech_score = tech_result.get("score", 0)
    news_score = news_result.get("sentiment_score", 0) * 3  # 新聞權重放大
    abnormal_risk = len(abnormal_signals.get("signals", [])) * (-3) if abnormal_signals.get("risk_level") in ("高", "中") else 0

    # 產業面評分（權重 20%）
    industry_score = 0
    if industry_result:
        industry_outlook = industry_result.get("outlook_score", 0)
        # outlook_score 範圍 -10~+10，線性映射到 -20~+20
        industry_score = industry_outlook * 2
    else:
        industry_score = 0

    # 計算總分（含產業分析權重）
    overall = tech_score * 0.55 + news_score * 0.2 + industry_score * 0.15 + abnormal_risk * 0.1
    overall = max(-100, min(100, round(overall)))
    result["overall_score"] = overall

    # ── 評級與顏色 ──
    if overall >= 40:
        result["rating"] = "強烈買進"
        result["rating_emoji"] = "🚀"
        result["rating_color"] = "#00a86b"
        result["strategy"] = "積極布局，可考慮分批建立部位"
    elif overall >= 20:
        result["rating"] = "買進"
        result["rating_emoji"] = "📈"
        result["rating_color"] = "#27ae60"
        result["strategy"] = "偏多操作，逢回布局"
    elif overall >= 5:
        result["rating"] = "偏多"
        result["rating_emoji"] = "👍"
        result["rating_color"] = "#82c91e"
        result["strategy"] = "小幅偏多，適合短線操作"
    elif overall <= -40:
        result["rating"] = "強烈賣出"
        result["rating_emoji"] = "☠️"
        result["rating_color"] = "#e74c3c"
        result["strategy"] = "積極減碼或出清，避開風險"
    elif overall <= -20:
        result["rating"] = "賣出"
        result["rating_emoji"] = "📉"
        result["rating_color"] = "#c0392b"
        result["strategy"] = "偏空操作，逢高減碼"
    elif overall <= -5:
        result["rating"] = "偏空"
        result["rating_emoji"] = "👎"
        result["rating_color"] = "#f39c12"
        result["strategy"] = "小幅偏空，適合放空或避開"
    else:
        result["rating"] = "中立觀望"
        result["rating_emoji"] = "⚖️"
        result["rating_color"] = "#7f8c8d"
        result["strategy"] = "多空交錯，建議觀望等待明確方向"

    # ── 詳細解說 ──
    lines.append(f"## 📊 綜合評分：{overall:+.0f} / 100")
    lines.append(f"### 🏆 建議：{result['rating_emoji']} {result['rating']}")
    lines.append(f"**策略：** {result['strategy']}")
    lines.append("")

    # 技術面摘要
    lines.append("### 📈 技術面分析")
    for l in tech_result.get("summary", []):
        lines.append(l)

    lines.append("")
    lines.append("### 📰 消息面分析")
    for l in news_result.get("summary", []):
        lines.append(l)

    # 產業分析報告
    if industry_result:
        lines.append("")
        lines.append("### 🏭 產業前景分析")
        for l in industry_result.get("summary_lines", []):
            lines.append(l)

    # 異常訊號
    if abnormal_signals.get("details"):
        lines.append("")
        lines.append("### 🔍 異常/內線訊號分析")
        for l in abnormal_signals["details"]:
            lines.append(l)
        risk_lvl = abnormal_signals.get("risk_level", "低")
        lines.append(f"**異常風險評估：{risk_lvl}**")

    # ── 進出場策略 ──
    lines.append("")
    lines.append("### 🎯 進出場策略建議")
    close_price = df["Close"].iloc[-1] if not df.empty and len(df) > 0 else 0

    # ── 改進版：多重支撐/壓力計算 ──
    if tech_result.get("indicators") and not df.empty and len(df) > 30:
        bb_u = df["BB_Upper"].iloc[-1] if "BB_Upper" in df.columns else None
        bb_l = df["BB_Lower"].iloc[-1] if "BB_Lower" in df.columns else None
        bb_m = df["BB_Mid"].iloc[-1] if "BB_Mid" in df.columns else None
        ma20v = df["MA20"].iloc[-1] if "MA20" in df.columns else None
        ma60v = df["MA60"].iloc[-1] if "MA60" in df.columns else None
        ma120v = df["MA120"].iloc[-1] if "MA120" in df.columns else None

        # ATR 計算（動態波動幅度）
        atr = None
        if "ATR" in df.columns:
            atr = df["ATR"].iloc[-1]
        elif all(c in df.columns for c in ["High", "Low", "Close"]):
            # 手動算 14 日 ATR
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"] - df["Close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            if len(tr) > 14:
                atr = tr.iloc[-14:].mean()

        # ── 支撐位計算（優先級排序）──
        supports = []

        # 1. 均線支撐
        if ma20v and close_price > ma20v:
            supports.append(("月線(MA20)", ma20v, 5))
        if ma60v and close_price > ma60v:
            supports.append(("季線(MA60)", ma60v, 4))
        if ma120v and close_price > ma120v:
            supports.append(("半年線(MA120)", ma120v, 3))

        # 2. 布林下軌
        if bb_l and close_price > bb_l:
            supports.append(("布林下軌", bb_l, 3))

        # 3. 近期低點 (20日/60日)
        low_20 = df["Low"].iloc[-20:].min()
        low_60 = df["Low"].iloc[-60:].min() if len(df) >= 60 else None
        if low_20 < close_price:
            supports.append(("近20日低點", low_20, 4))
        if low_60 and low_60 < close_price:
            supports.append(("近60日低點", low_60, 2))

        # 4. 前波低點(型態)
        patterns_in = tech_result.get("patterns", [])
        if isinstance(patterns_in, list) and patterns_in:
            for p in patterns_in:
                if isinstance(p, dict):
                    pl = p.get("low") or p.get("price_low") or p.get("target")
                    if pl and pl < close_price:
                        supports.append((p.get("type", "型態"), pl, 2))

        # 去重、排序：優先級越高(數字越大)越優先，同級取價高
        seen_levels = set()
        unique_supports = []
        for name, val, priority in supports:
            level_key = round(val, 1)
            if level_key not in seen_levels:
                seen_levels.add(level_key)
                unique_supports.append((name, val, priority))

        # 排序：優先級高→低，同級價高→低（保守估算）
        unique_supports.sort(key=lambda x: (-x[2], -x[1]))
        supports_top = unique_supports[:4]

        # ── 壓力位計算 ──
        resistances = []
        if ma20v and close_price < ma20v:
            resistances.append(("月線(MA20)", ma20v, 5))
        if ma60v and close_price < ma60v:
            resistances.append(("季線(MA60)", ma60v, 4))
        if ma120v and close_price < ma120v:
            resistances.append(("半年線(MA120)", ma120v, 3))
        if bb_u and close_price < bb_u:
            resistances.append(("布林上軌", bb_u, 3))

        high_20 = df["High"].iloc[-20:].max()
        high_60 = df["High"].iloc[-60:].max() if len(df) >= 60 else None
        if high_20 > close_price:
            resistances.append(("近20日高點", high_20, 4))
        if high_60 and high_60 > close_price:
            resistances.append(("近60日高點", high_60, 2))

        # 去重排序
        seen_res = set()
        unique_res = []
        for name, val, priority in resistances:
            level_key = round(val, 1)
            if level_key not in seen_res:
                seen_res.add(level_key)
                unique_res.append((name, val, priority))
        unique_res.sort(key=lambda x: (-x[2], x[1]))  # 優先級高→低，同級價低→高
        res_top = unique_res[:4]

        # ── Fibonacci 回撤計算 ──
        fib_levels = {}
        if len(df) >= 60:
            recent_high_fib = df["High"].iloc[-60:].max()
            recent_low_fib = df["Low"].iloc[-60:].min()
            fib_range = recent_high_fib - recent_low_fib
            if fib_range > close_price * 0.02:  # 至少 2% 波動
                for ratio, name in [(0.236, "23.6%"), (0.382, "38.2%"), (0.5, "50%"), (0.618, "61.8%"), (0.786, "78.6%")]:
                    level = recent_high_fib - fib_range * ratio
                    fib_levels[name] = round(level, 2)

        # ── 買進區間 ──
        entry_exit = {}
        buy_zone_note = []

        if supports_top:
            # 主要支撐 = 權重最高的 valid 支撐
            primary_support = supports_top[0][1]
            # 次級支撐
            secondary_support = supports_top[1][1] if len(supports_top) > 1 else primary_support * 0.97

            # 動態買進區間
            if atr and atr > 0:
                atr_mult = 0.5 if tech_score > 0 else 1.0  # 多頭趨勢下區間緊一些
                buy_zone_low = primary_support - atr * atr_mult
                buy_zone_high = min(close_price, primary_support + atr * 0.5)
            else:
                buy_zone_low = secondary_support * 0.98
                buy_zone_high = primary_support * 1.02

            # 買進區間不能離現價太遠（最多 ±15%）
            max_dist = close_price * 0.15
            buy_zone_low = max(buy_zone_low, close_price - max_dist)
            buy_zone_high = min(buy_zone_high, close_price + max_dist * 0.5)

            # 如果區間不合理（下限>上限），調整
            if buy_zone_low > buy_zone_high:
                mid = (buy_zone_low + buy_zone_high) / 2
                half_range = atr * 0.5 if atr and atr > 0 else close_price * 0.01
                buy_zone_low = mid - half_range
                buy_zone_high = mid + half_range

            entry_exit["buy_zone"] = (round(buy_zone_low, 2), round(buy_zone_high, 2))

            # 多層支撐說明
            lines.append(f"**買進區間：** {buy_zone_low:.2f} ~ {buy_zone_high:.2f}")
            lines.append(f"  📍 現價：{close_price:.2f} | 建議買進區距現價 "
                         f"{((buy_zone_low+buy_zone_high)/2 - close_price)/close_price*100:+.1f}%")
            lines.append("")
            lines.append("**支撐參考：**")
            for i, (name, val, pri) in enumerate(supports_top[:3], 1):
                gap = (val - close_price) / close_price * 100
                lines.append(f"  {i}. {name}：{val:.2f}（距現價 {gap:+.1f}%）")
        else:
            lines.append("**買進區間：** 暫無明確支撐參考，建議觀望")
            entry_exit["buy_zone"] = (0, 0)

        # ── Fibonacci 買進機會提示 ──
        if fib_levels and bb_l and bb_l < close_price:
            # 找出最近的 Fib 支撐
            nearest_fib = None
            nearest_dist = float("inf")
            for name, level in fib_levels.items():
                if level < close_price and level > close_price * 0.85:
                    dist = abs(level - close_price)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_fib = (name, level)
            if nearest_fib:
                lines.append(f"  📐 Fib回撤 {nearest_fib[0]}：{nearest_fib[1]:.2f}（可作為參考支撐）")

        # ── 目標價（更精準、多階層）──
        # 收集所有壓力位（由近到遠）
        all_res = []
        for name, val, pri in res_top:
            if val > close_price and val < close_price * 1.30:
                all_res.append((name, val, pri))

        # 加入 Fibonacci 延伸壓力
        fib_ext = {}
        if len(df) >= 60:
            fl = df["Low"].iloc[-60:].min()
            fh = df["High"].iloc[-60:].max()
            fr = fh - fl
            if fr > close_price * 0.02:
                for ratio, fname in [(0.618, "0.618"), (1.0, "1.000"), (1.272, "1.272"), (1.618, "1.618")]:
                    fib_ext[fname] = round(fh + fr * ratio, 2)

        for fib_name, level in fib_ext.items():
            if level > close_price and level < close_price * 1.30:
                all_res.append((f"Fib({fib_name})", level, 2))

        # 去重排序
        seen_set = set()
        unique_res_p = []
        for name, val, pri in all_res:
            k = round(val, 1)
            if k not in seen_set:
                seen_set.add(k)
                unique_res_p.append((name, val, pri))
        unique_res_p.sort(key=lambda x: x[1])

        if unique_res_p:
            # 多階層目標
            target_1_val = unique_res_p[0][1]
            target_1_label = unique_res_p[0][0]
            tp1_pct = (target_1_val / close_price - 1) * 100

            # 第二目標
            target_2_val = 0
            target_2_label = ""
            for name, val, pri in unique_res_p[1:]:
                if val - target_1_val >= (atr * 0.5 if atr else close_price * 0.01):
                    target_2_val = val
                    target_2_label = name
                    break
            if target_2_val == 0:
                fib_1272 = fib_ext.get("1.272", 0)
                if fib_1272 > target_1_val + (atr if atr else close_price * 0.01):
                    target_2_val = fib_1272
                    target_2_label = "Fib(1.272)"

            # 第三目標（強勢突破時）
            target_3_val = 0
            target_3_label = ""
            if target_2_val > 0:
                for name, val, pri in unique_res_p[2:]:
                    if val - target_2_val >= (atr * 0.5 if atr else close_price * 0.01):
                        target_3_val = val
                        target_3_label = name
                        break
                if target_3_val == 0:
                    fib_1618 = fib_ext.get("1.618", 0)
                    if fib_1618 > target_2_val + (atr if atr else close_price * 0.01):
                        target_3_val = fib_1618
                        target_3_label = "Fib(1.618)"
                    elif atr and atr > 0:
                        target_3_val = round(target_2_val + atr * 1.5, 2)
                        target_3_label = "強勢突破目標"

            # 合理範圍限制
            target_1_val = max(target_1_val, close_price * 1.01)
            target_1_val = min(target_1_val, close_price * 1.25)
            if target_2_val > 0:
                target_2_val = max(target_2_val, target_1_val * 1.01)
                target_2_val = min(target_2_val, close_price * 1.30)
            if target_3_val > 0:
                target_3_val = max(target_3_val, target_2_val * 1.01)
                target_3_val = min(target_3_val, close_price * 1.50)

            # 賣出區間（以目標1為主，做區間寬度）
            sell_zone_low = target_1_val * 0.98
            sell_zone_high = target_1_val * 1.03

            entry_exit["sell_zone"] = (round(sell_zone_low, 2), round(sell_zone_high, 2))
            entry_exit["targets"] = {
                1: {"price": round(target_1_val, 2), "source": target_1_label, "gain_pct": round(tp1_pct, 1)},
            }
            if target_2_val > 0:
                tp2_pct = (target_2_val / close_price - 1) * 100
                entry_exit["targets"][2] = {"price": round(target_2_val, 2), "source": target_2_label, "gain_pct": round(tp2_pct, 1)}
            if target_3_val > 0:
                tp3_pct = (target_3_val / close_price - 1) * 100
                entry_exit["targets"][3] = {"price": round(target_3_val, 2), "source": target_3_label, "gain_pct": round(tp3_pct, 1)}

            lines.append(f"")
            lines.append(f"**多階層目標價 (依壓力位階):**")
            for tidx, tinfo in entry_exit["targets"].items():
                gain = tinfo["gain_pct"]
                if gain >= 10:
                    icon = "🚀"
                elif gain >= 5:
                    icon = "📈"
                else:
                    icon = "📊"
                lines.append(f"  {tidx}. {icon} **目標{tidx}：{tinfo['price']:.2f}** ({gain:+.1f}%) ← {tinfo['source']}")
            if len(entry_exit["targets"]) >= 2:
                lines.append(f"")
                lines.append(f"**📋 分批獲利建議：**")
                lines.append(f"  • T1 ({entry_exit['targets'][1]['price']:.2f}) 出 1/3")
                if 2 in entry_exit["targets"]:
                    lines.append(f"  • T2 ({entry_exit['targets'][2]['price']:.2f}) 再出 1/3")
                if 3 in entry_exit["targets"]:
                    lines.append(f"  • T3 ({entry_exit['targets'][3]['price']:.2f}) 出剩餘 1/3")
                else:
                    lines.append(f"  • 剩餘部分移動停損保護")

            lines.append("")
            lines.append("**壓力參考：**")
            for i, (name, val, pri) in enumerate(res_top[:4], 1):
                gap = (val - close_price) / close_price * 100
                lines.append(f"  {i}. {name}：{val:.2f}（距現價 {gap:+.1f}%）")

            # 時間預估
            lines.append("")
            if atr and atr > 0:
                bars_to_t1 = int((target_1_val - close_price) / atr) + 1
                days_est = bars_to_t1
                if days_est <= 5:
                    time_horizon = "短線（約1週）"
                elif days_est <= 20:
                    time_horizon = f"短中線（約{days_est}個交易日 ≈ {days_est//5}週）"
                elif days_est <= 60:
                    time_horizon = f"中線（約{days_est}個交易日 ≈ {days_est//20}月）"
                else:
                    time_horizon = "中長線（1個月以上）"
                lines.append(f"⏱ **預估時間：** {time_horizon}（基於ATR={atr:.2f} / {bars_to_t1}根K棒）")
        else:
            lines.append("**目標價區間：** 暫無明確壓力參考")
            entry_exit["sell_zone"] = (0, 0)

        # ── 動態停損 ──
        if atr and atr > 0:
            # ATR-based 停損
            stop_atr = atr * 2.0  # 2倍ATR
            stop_loss = close_price - stop_atr

            # 不能跌破近20日低點的 3% 以下
            low_20 = df["Low"].iloc[-20:].min() if len(df) >= 20 else None
            if low_20 and stop_loss < low_20 * 0.97:
                stop_loss = low_20 * 0.97

            # 至少要有合理停損空間
            stop_pct = abs(stop_loss - close_price) / close_price * 100
            if stop_pct < 2:
                stop_loss = close_price * 0.97  # 最少3%停損空間
            elif stop_pct > 15:
                stop_loss = close_price * 0.9  # 最多10%停損

            entry_exit["stop_loss"] = round(stop_loss, 2)
            lines.append(f"")
            stop_pct_actual = abs(stop_loss - close_price) / close_price * 100
            lines.append(f"**停損價：** {stop_loss:.2f}（-{stop_pct_actual:.1f}%")
            if atr:
                lines.append(f"  📏 ATR={atr:.2f}（ATR乘數：{stop_atr/atr if atr > 0 else 0:.1f}x）")
        else:
            # 傳統停損
            if supports_top:
                stop_loss = supports_top[0][1] * 0.95
                entry_exit["stop_loss"] = round(stop_loss, 2)
                lines.append(f"")
                lines.append(f"**停損價：** {stop_loss:.2f}（跌破{round((1-0.95)*100)}%")

        # 多階層風險報酬比（各目標分別計算）
        buy_center = (entry_exit["buy_zone"][0] + entry_exit["buy_zone"][1]) / 2 if entry_exit.get("buy_zone", (0,0))[1] > 0 else close_price
        stop_val = entry_exit.get("stop_loss", close_price * 0.95)
        targets_dict = entry_exit.get("targets", {})

        if buy_center > 0 and stop_val < buy_center:
            lines.append(f"")
            lines.append(f"**⚖️ 多階層風報比：**")
            potential_loss = (buy_center - stop_val) / buy_center * 100

            # 逐層計算風報比
            for tidx in sorted(targets_dict.keys()):
                tp = targets_dict[tidx]["price"]
                gain_pct = targets_dict[tidx]["gain_pct"]
                rr = (tp - buy_center) / (buy_center - stop_val) if (buy_center - stop_val) > 0 else 0
                if rr >= 3:
                    grade = "✅ 優良"
                elif rr >= 2:
                    grade = "👍 尚可"
                elif rr >= 1:
                    grade = "⚠️ 普通"
                else:
                    grade = "❌ 不佳"

                lines.append(f"  T{tidx} ({tp:.2f}, {gain_pct:+.1f}%) → **1:{rr:.2f}** {grade}")

            # 混合風報比（加權平均）
            if 1 in targets_dict:
                rr1 = (targets_dict[1]["price"] - buy_center) / (buy_center - stop_val)
                rr2 = (targets_dict[2]["price"] - buy_center) / (buy_center - stop_val) if 2 in targets_dict else rr1 * 1.5
                weighted_rr = rr1 * 0.5 + rr2 * 0.3 + (rr1 * 2) * 0.2 if 3 not in targets_dict else rr1 * 0.4 + rr2 * 0.35 + (targets_dict[3]["price"] - buy_center) / (buy_center - stop_val) * 0.25
                entry_exit["risk_reward"] = round(weighted_rr, 2)

            lines.append(f"")
            first_rr = (targets_dict[1]["price"] - buy_center) / (buy_center - stop_val) if 1 in targets_dict else 0
            if first_rr < 1:
                lines.append("  ❌ **整體風報不理想（<1:1），建議嚴格控制倉位或等待更佳進場點**")
            elif weighted_rr >= 2.5:
                lines.append("  ✅ **整體風報結構優良，各層目標均有合理空間**")

        result["entry_exit"] = entry_exit

    # ── 風險評估 ──
    lines.append("")
    lines.append("### ⚠️ 風險評估")
    risk_items = []
    risk_level = "低"

    if tech_score < -30:
        risk_items.append("🔴 技術面全面偏空")
    if abnormal_signals.get("risk_level") in ("高", "中"):
        risk_items.append(f"🔴 異常訊號風險：{abnormal_signals['risk_level']}")
    if news_score < -10:
        risk_items.append("🔴 新聞面偏空")

    if len(risk_items) >= 3:
        risk_level = "高"
    elif len(risk_items) >= 1:
        risk_level = "中"

    result["risk_assessment"] = {
        "level": risk_level,
        "items": risk_items or ["⚪ 未發現明顯異常風險"],
    }
    for item in result["risk_assessment"]["items"]:
        lines.append(item)
    lines.append(f"**整體風險等級：{risk_level}**")

    # ── 資金配置建議 ──
    lines.append("")
    lines.append("### 💰 資金配置參考")
    if overall >= 30:
        suggested_position = "40~60%"
        lines.append(f"建議倉位：{suggested_position}")
        lines.append("分批進場，每拉回2~3%加碼一次")
    elif overall >= 10:
        suggested_position = "20~40%"
        lines.append(f"建議倉位：{suggested_position}")
        lines.append("逢回檔分批布局，勿追高")
    elif overall <= -30:
        suggested_position = "0~10%（或空手）"
        lines.append(f"建議倉位：{suggested_position}")
        lines.append("風險偏高，建議保留現金等待機會")
    elif overall <= -10:
        suggested_position = "10~20%"
        lines.append(f"建議倉位：{suggested_position}")
        lines.append("偏空看待，降低曝險")
    else:
        suggested_position = "10~30%"
        lines.append(f"建議倉位：{suggested_position}")
        lines.append("方向不明，輕倉操作")
    result["position_sizing"]["suggested"] = suggested_position

    # ── 總結與提醒 ──
    lines.append("")
    lines.append("### 💬 投資專家提醒")
    reminders = [
        "📌 本分析基於技術面+新聞面+籌碼面客觀數據，不構成買賣建議",
        "📌 股市有風險，投資需謹慎，請依個人風險承受度決策",
        "📌 建議結合基本面研究，長期投資需關注公司營運體質",
        "📌 短線操作請嚴格執行停損紀律",
    ]
    if "內線消息" in str(abnormal_signals):
        reminders.append("⚠️ 近期有異常交易訊號，建議留意是否為內線交易或主力操控")
    for r in reminders:
        lines.append(r)

    result["detailed_reasoning"] = lines
    result["final_words"] = reminders[:2]

    return result


# ============================================================
# 5. 主入口
# ============================================================
def run_expert_analysis(stock_id: str, stock_name: str, months: int = 12) -> dict:
    """
    專家級單股分析主入口
    整合所有分析模組，回傳完整報告
    """
    # 取得歷史資料
    df = fetch_historical(stock_id, months=months)
    # 防禦：確認 df 為 DataFrame
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    if df.empty:
        return {"error": f"無法取得 {stock_id} 的歷史資料"}

    # 加入技術指標
    df = add_all_indicators(df)
    patterns = pr.detect_all_patterns(df)

    # 各項分析
    tech = comprehensive_technical_analysis(df)
    tech["patterns"] = patterns  # 附上型態辨識結果
    abnormal = detect_abnormal_signals(stock_id, df)
    news = analyze_news_and_market(stock_id, stock_name)
    # 產業分析
    industry = analyze_industry_outlook(stock_id, stock_name)

    rec = expert_recommendation(stock_id, stock_name, df, tech, abnormal, news, industry)

    # 基本面 + MOPS 輔助
    fundamental = fetch_fundamentals(stock_id)
    try:
        mops_revenue = fetch_mops_monthly_revenue(stock_id)
        mops_finance = fetch_mops_financial_highlights(stock_id)
    except Exception:
        mops_revenue = []
        mops_finance = {}

    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "current_price": df["Close"].iloc[-1],
        "data": df,
        "technical": tech,
        "abnormal_signals": abnormal,
        "news_analysis": news,
        "industry_analysis": industry,
        "fundamentals": fundamental,
        "mops": {"revenue": mops_revenue, "financial_highlights": mops_finance},
        "recommendation": rec,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

