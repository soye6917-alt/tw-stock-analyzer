"""
📦 ETF 投資評估分析模組
提供 ETF 評分、績效比較、風險評估功能
資料來源：TWSE/TPEx 歷史股價 + 即時報價
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_fetcher import fetch_historical, fetch_realtime_quote, SESSION

# ─────────────────────────────────────────────
# ETF 資料庫（手動維護熱門標的）
# ─────────────────────────────────────────────
ETF_UNIVERSE = {
    # --- 市值型 ---
    "0050": {"name": "元大台灣50", "category": "市值型", "desc": "追蹤台灣50指數，全市場指標ETF"},
    "006208": {"name": "富邦台50", "category": "市值型", "desc": "追蹤台灣50指數，費用率更低"},
    "00692": {"name": "富邦公司治理", "category": "市值型", "desc": "公司治理評鑑前20%成分股"},
    "00888": {"name": "永豐台灣ESG", "category": "市值型", "desc": "ESG評分篩選成分股"},
    "00923": {"name": "群益台ESG低碳", "category": "市值型", "desc": "ESG低碳轉型主題ETF"},
    "00905": {"name": "FT臺灣SMART", "category": "市值型", "desc": "多因子Smart Beta策略"},
    
    # --- 高股息 ---
    "0056": {"name": "元大高股息", "category": "高股息", "desc": "高股息成分股，傳統高息ETF"},
    "00878": {"name": "國泰永續高股息", "category": "高股息", "desc": "ESG+高股息，季配息"},
    "00919": {"name": "群益台灣精選高息", "category": "高股息", "desc": "精選高息股，季配息"},
    "00929": {"name": "復華台灣科技優息", "category": "高股息", "desc": "科技股+月配息"},
    "00940": {"name": "元大台灣價值高息", "category": "高股息", "desc": "低波動高息，月配息"},
    "00900": {"name": "富邦特選高股息", "category": "高股息", "desc": "高股息精選30"},
    "00713": {"name": "元大台灣高息低波", "category": "高股息", "desc": "高股息+低波動防禦型"},
    "00936": {"name": "台新永續高息中小", "category": "高股息", "desc": "中小型高息股"},
    "00915": {"name": "凱基優選高股息", "category": "高股息", "desc": "多因子高股息篩選"},
    
    # --- 科技/半導體 ---
    "00881": {"name": "國泰台灣5G+", "category": "科技", "desc": "台灣5G科技領先股"},
    "00891": {"name": "中信關鍵半導體", "category": "科技", "desc": "半導體產業鏈關鍵股"},
    "00904": {"name": "新光臺灣半導體", "category": "科技", "desc": "半導體30指數"},
    "00892": {"name": "富邦台灣半導體", "category": "科技", "desc": "半導體產業ETF"},
    "00733": {"name": "富邦臺灣中小", "category": "科技", "desc": "中小型A級動能選股"},
    
    # --- 其他主題 ---
    "00679B": {"name": "元大美債20年", "category": "債券", "desc": "美國長期公債ETF"},
    "00751B": {"name": "群益25年美債", "category": "債券", "desc": "美國25年期以上公債"},
    "00687B": {"name": "國泰20年美債", "category": "債券", "desc": "美國20年期公債"},
    "00864B": {"name": "中信美國公債", "category": "債券", "desc": "美國公債ETF"},
    "00772B": {"name": "中信高評級債", "category": "債券", "desc": "投資級公司債ETF"},
    "00720B": {"name": "元大投資級公司債", "category": "債券", "desc": "投資級公司債ETF"},
    
    # --- 國際 ---
    "00646": {"name": "元大S&P500", "category": "國際", "desc": "追蹤S&P500指數"},
    "00662": {"name": "富邦NASDAQ", "category": "國際", "desc": "追蹤NASDAQ-100"},
    "00830": {"name": "國泰費城半導體", "category": "國際", "desc": "費城半導體指數"},
    "00757": {"name": "統一FANG+", "category": "國際", "desc": "FANG+尖牙股指數"},
    
    # --- 傳產/金融 ---
    "0055": {"name": "元大MSCI金融", "category": "產業", "desc": "金融類股指數"},
    "00907": {"name": "永豐優息存股", "category": "產業", "desc": "金融+傳產高息"},
}

# 分類顏色
CATEGORY_COLORS = {
    "市值型": "#4CAF50",
    "高股息": "#FF9800",
    "科技": "#2196F3",
    "債券": "#9C27B0",
    "國際": "#009688",
    "產業": "#795548",
}


# ─────────────────────────────────────────────
# ETF 評分結果
# ─────────────────────────────────────────────
@dataclass
class ETFScore:
    stock_id: str
    name: str
    category: str = ""
    desc: str = ""
    
    # 績效 (權重 30%)
    perf_score: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_1y: float = 0.0
    return_3y: float = 0.0
    
    # 風險 (權重 25%)
    risk_score: float = 0.0
    volatility: float = 0.0       # 年化波動率
    max_drawdown: float = 0.0     # 最大回撤
    sharpe_ratio: float = 0.0     # 夏普比率
    
    # 殖利率 (權重 25%)
    yield_score: float = 0.0
    est_dividend_yield: float = 0.0
    year1_return: float = 0.0
    
    # 趨勢動能 (權重 20%)
    momentum_score: float = 0.0
    ma20_pct: float = 0.0         # 距離月線%
    ma60_pct: float = 0.0         # 距離季線%
    rsi: float = 50.0
    trend_status: str = "中性"
    
    # 總評
    total_score: float = 0.0
    rating: str = "中立"
    analysis: list = field(default_factory=list)
    current_price: float = 0.0
    error: str = None


def fetch_etf_price_data(stock_id: str, months: int = 36) -> pd.DataFrame:
    """取得 ETF 歷史價格資料 (最多36個月)"""
    from indicators import add_all_indicators
    df = fetch_historical(stock_id, months=months)
    if df.empty:
        return pd.DataFrame()
    df = add_all_indicators(df)
    return df


def calc_returns(df: pd.DataFrame) -> dict:
    """計算各週期報酬率 (%)"""
    if df.empty or len(df) < 5:
        return {"1m": 0, "3m": 0, "6m": 0, "1y": 0, "3y": 0}
    
    close = df["Close"]
    latest = close.iloc[-1]
    
    def r(n):
        if len(close) > n:
            return (latest / close.iloc[-(n+1)] - 1) * 100
        return 0
    
    return {
        "1m": r(21),    # 約21個交易日
        "3m": r(63),
        "6m": r(126),
        "1y": r(252),
        "3y": r(756),
    }


def calc_risk_metrics(df: pd.DataFrame) -> dict:
    """計算風險指標"""
    result = {
        "volatility": 0.0,
        "max_drawdown": 0.0,
        "sharpe_ratio": 0.0,
    }
    
    if df.empty or len(df) < 20:
        return result
    
    close = df["Close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    
    if len(log_ret) > 20:
        result["volatility"] = log_ret.tail(252).std() * np.sqrt(252) * 100
    
    if len(close) > 20:
        roll_max = close.expanding().max()
        drawdowns = (close - roll_max) / roll_max * 100
        result["max_drawdown"] = drawdowns.min()
    
    if result["volatility"] > 0 and len(log_ret) > 252:
        avg_return = log_ret.tail(252).mean() * 252
        rf_rate = 0.015  # 假設無風險利率 1.5%
        result["sharpe_ratio"] = (avg_return - rf_rate / 252) / log_ret.tail(252).std() * np.sqrt(252)
    
    return result


def estimate_dividend_yield(df: pd.DataFrame) -> float:
    """
    從歷史價格估算殖利率
    方法：利用除息日價格跳空來估算
    如果資料不足則回傳 0
    """
    if df.empty or len(df) < 100:
        return 0.0
    
    close = df["Close"]
    # 找出單日跌幅超過 0.5% 且隔天反彈不到 50% 的交易日 → 可能是除息
    daily_ret = close.pct_change()
    est_dividends = []
    
    for i in range(1, len(close) - 2):
        if daily_ret.iloc[i] < -0.005:  # 跌超過 0.5%
            drop = abs(daily_ret.iloc[i])
            recovery = daily_ret.iloc[i+1] if i+1 < len(daily_ret) else 0
            if recovery < drop * 0.5:  # 沒有立即回補 → 可能是除息
                est_dividends.append(close.iloc[i-1] * drop)
    
    if est_dividends:
        total_div = sum(est_dividends[-6:])  # 最近6次
        avg_price = close.iloc[-253:].mean() if len(close) > 252 else close.mean()
        if avg_price > 0:
            return (total_div / avg_price) * 100
    
    return 0.0


def calc_momentum(df: pd.DataFrame) -> dict:
    """計算技術動能指標"""
    result = {
        "ma20_pct": 0.0,
        "ma60_pct": 0.0,
        "rsi": 50.0,
        "trend_status": "⚪ 中性",
    }
    
    if df.empty or len(df) < 30:
        return result
    
    latest = df.iloc[-1]
    close = latest["Close"]
    
    ma20 = latest.get("MA20", 0)
    ma60 = latest.get("MA60", 0)
    rsi = latest.get("RSI", 50)
    
    if ma20 > 0:
        result["ma20_pct"] = (close - ma20) / ma20 * 100
    if ma60 > 0:
        result["ma60_pct"] = (close - ma60) / ma60 * 100
    
    result["rsi"] = rsi
    
    # 趨勢判斷
    ma5 = latest.get("MA5", 0)
    if ma5 > ma20 > ma60 and ma60 > 0:
        result["trend_status"] = "🟢 多頭"
    elif ma5 < ma20 < ma60 and ma60 > 0:
        result["trend_status"] = "🔴 空頭"
    else:
        result["trend_status"] = "⚪ 盤整"
    
    return result


def compute_perf_score(returns: dict) -> float:
    """績效評分 (最高30分)"""
    score = 0.0
    # 短期 (1m) 權重 5
    if returns["1m"] > 5: score += 5
    elif returns["1m"] > 2: score += 3
    elif returns["1m"] > 0: score += 1
    elif returns["1m"] < -5: score -= 3
    # 中期 (3m, 6m) 權重各5
    for period in ["3m", "6m"]:
        if returns[period] > 10: score += 5
        elif returns[period] > 5: score += 3
        elif returns[period] > 0: score += 1
        elif returns[period] < -10: score -= 2
    # 長期 (1y) 權重10
    if returns["1y"] > 20: score += 10
    elif returns["1y"] > 10: score += 7
    elif returns["1y"] > 5: score += 4
    elif returns["1y"] > 0: score += 2
    elif returns["1y"] < -15: score -= 5
    # 3年 權重5
    if returns["3y"] > 40: score += 5
    elif returns["3y"] > 20: score += 3
    elif returns["3y"] > 0: score += 1
    return max(-10, min(30, score))


def compute_risk_score(metrics: dict) -> float:
    """風險評分 (最高25分)"""
    score = 15.0  # 基準分
    
    vol = metrics.get("volatility", 0)
    if vol < 12: score += 5
    elif vol < 18: score += 3
    elif vol < 25: score += 1
    elif vol > 35: score -= 5
    elif vol > 40: score -= 8
    
    mdd = metrics.get("max_drawdown", 0)
    if mdd > -10: score += 5
    elif mdd > -20: score += 3
    elif mdd > -30: score += 0
    elif mdd > -40: score -= 3
    else: score -= 5
    
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe > 1.5: score += 5
    elif sharpe > 1.0: score += 3
    elif sharpe > 0.5: score += 1
    elif sharpe < 0: score -= 3
    
    return max(0, min(25, score))


def compute_yield_score(est_yield: float) -> float:
    """殖利率評分 (最高25分)"""
    if est_yield <= 0:
        return 5  # 沒資料給基本分
    if est_yield > 8: return 25
    if est_yield > 6: return 22
    if est_yield > 5: return 19
    if est_yield > 4: return 15
    if est_yield > 3: return 11
    if est_yield > 2: return 7
    return 5


def compute_momentum_score(mom: dict) -> float:
    """動能評分 (最高20分)"""
    score = 10.0
    
    rsi = mom.get("rsi", 50)
    if 40 < rsi < 60: score += 3
    elif 30 <= rsi <= 40: score += 4
    elif 60 <= rsi <= 70: score += 2
    elif rsi > 75: score -= 3
    elif rsi < 25: score += 2
    
    ma20_pct = mom.get("ma20_pct", 0)
    if ma20_pct > 0 and ma20_pct < 5: score += 4
    elif ma20_pct > 5: score += 2
    elif ma20_pct < -5: score -= 2
    elif ma20_pct < -10: score -= 4
    
    ma60_pct = mom.get("ma60_pct", 0)
    if ma60_pct > 0 and ma60_pct < 10: score += 3
    elif ma60_pct > 10: score += 1
    elif ma60_pct < -10: score -= 3
    
    if mom.get("trend_status", "").startswith("🟢"):
        score += 3
    elif mom.get("trend_status", "").startswith("🔴"):
        score -= 3
    
    return max(0, min(20, score))


def score_etf(stock_id: str, months: int = 36) -> ETFScore:
    """對單一 ETF 進行完整評分"""
    info = ETF_UNIVERSE.get(stock_id, {"name": stock_id, "category": "其他", "desc": ""})
    result = ETFScore(
        stock_id=stock_id,
        name=info["name"],
        category=info.get("category", ""),
        desc=info.get("desc", ""),
    )
    
    try:
        df = fetch_etf_price_data(stock_id, months=months)
        if df.empty:
            result.error = "無資料"
            return result
        
        result.current_price = df["Close"].iloc[-1]
        
        # 各項評分
        returns = calc_returns(df)
        result.return_1m = returns["1m"]
        result.return_3m = returns["3m"]
        result.return_6m = returns["6m"]
        result.return_1y = returns["1y"]
        result.return_3y = returns["3y"]
        result.perf_score = compute_perf_score(returns)
        
        risk = calc_risk_metrics(df)
        result.volatility = risk["volatility"]
        result.max_drawdown = risk["max_drawdown"]
        result.sharpe_ratio = risk["sharpe_ratio"]
        result.risk_score = compute_risk_score(risk)
        
        est_yield = estimate_dividend_yield(df)
        result.est_dividend_yield = est_yield
        result.yield_score = compute_yield_score(est_yield)
        
        mom = calc_momentum(df)
        result.ma20_pct = mom["ma20_pct"]
        result.ma60_pct = mom["ma60_pct"]
        result.rsi = mom["rsi"]
        result.trend_status = mom["trend_status"]
        result.momentum_score = compute_momentum_score(mom)
        
        # 總分
        result.total_score = result.perf_score + result.risk_score + result.yield_score + result.momentum_score
        
        # 評級
        ts = result.total_score
        if ts >= 75: result.rating = "⭐ 優質"
        elif ts >= 60: result.rating = "📈 推薦"
        elif ts >= 45: result.rating = "👀 關注"
        elif ts >= 30: result.rating = "⚖️ 中立"
        else: result.rating = "⏳ 觀望"
        
        # 分析摘要
        result.analysis = generate_etf_analysis(result)
        
    except Exception as e:
        result.error = str(e)
    
    return result


def generate_etf_analysis(etf: ETFScore) -> list:
    """生成 ETF 分析摘要"""
    lines = []
    
    lines.append(f"**{etf.name} ({etf.stock_id})** — {etf.category}")
    if etf.desc:
        lines.append(f"📝 {etf.desc}")
    
    lines.append("")
    lines.append("**📊 評分明細:**")
    lines.append(f"  📈 績效 {etf.perf_score:.0f}/30 | 🛡️ 風險 {etf.risk_score:.0f}/25 | 💰 殖利率 {etf.yield_score:.0f}/25 | 📶 動能 {etf.momentum_score:.0f}/20")
    lines.append(f"  **總分: {etf.total_score:.0f}/100** → {etf.rating}")
    
    lines.append("")
    lines.append("**📈 各期報酬:**")
    lines.append(f"  1月:{etf.return_1m:+.1f}% | 3月:{etf.return_3m:+.1f}% | 6月:{etf.return_6m:+.1f}% | 1年:{etf.return_1y:+.1f}% | 3年:{etf.return_3y:+.1f}%")
    
    lines.append("")
    lines.append("**📊 風險指標:**")
    lines.append(f"  波動率 {etf.volatility:.1f}% | 最大回撤 {etf.max_drawdown:.1f}% | 夏普比率 {etf.sharpe_ratio:.2f}")
    
    lines.append("")
    lines.append("**💰 殖利率估算:**")
    if etf.est_dividend_yield > 0:
        lines.append(f"  預估殖利率 {etf.est_dividend_yield:.2f}%")
    else:
        lines.append("  資料不足，無法估算殖利率")
    
    lines.append("")
    lines.append(f"**📌 技術面:** {etf.trend_status}")
    lines.append(f"  距月線 {etf.ma20_pct:+.1f}% | 距季線 {etf.ma60_pct:+.1f}% | RSI {etf.rsi:.0f}")
    
    lines.append("")
    lines.append("─" * 25)
    lines.append(f"現價:{etf.current_price:.2f} | 非投資建議")
    
    return lines


def get_etf_picks(categories: list = None, top_n: int = 5, months: int = 36) -> list:
    """
    掃描 ETF 並回傳評分排名
    categories: 過濾分類（None=全部）
    """
    results = []
    target_etfs = ETF_UNIVERSE
    
    if categories:
        target_etfs = {k: v for k, v in ETF_UNIVERSE.items() if v.get("category") in categories}
    
    for sid, info in target_etfs.items():
        result = score_etf(sid, months=months)
        if result.error is None:
            results.append(result)
    
    results.sort(key=lambda x: x.total_score, reverse=True)
    return results[:top_n]


def get_category_stats(months: int = 36) -> dict:
    """計算各分類 ETF 平均分數"""
    stats = {}
    for cat in set(v["category"] for v in ETF_UNIVERSE.values()):
        cat_etfs = {k: v for k, v in ETF_UNIVERSE.items() if v["category"] == cat}
        scores = []
        avg_metrics = {"perf": 0, "risk": 0, "yield": 0, "momentum": 0}
        count = 0
        for sid in cat_etfs:
            s = score_etf(sid, months=months)
            if s.error is None:
                scores.append(s.total_score)
                avg_metrics["perf"] += s.perf_score
                avg_metrics["risk"] += s.risk_score
                avg_metrics["yield"] += s.yield_score
                avg_metrics["momentum"] += s.momentum_score
                count += 1
        if count > 0:
            stats[cat] = {
                "avg_score": sum(scores) / count,
                "count": count,
                **{k: v / count for k, v in avg_metrics.items()},
            }
    return stats


def compare_etfs(stock_ids: list, months: int = 36) -> pd.DataFrame:
    """比較多檔 ETF 的關鍵指標"""
    rows = []
    for sid in stock_ids:
        s = score_etf(sid, months=months)
        if s.error is None:
            rows.append({
                "代號": s.stock_id,
                "名稱": s.name,
                "類型": s.category,
                "總分": round(s.total_score, 0),
                "評級": s.rating,
                "現價": s.current_price,
                "1月": f"{s.return_1m:+.1f}%",
                "3月": f"{s.return_3m:+.1f}%",
                "1年": f"{s.return_1y:+.1f}%",
                "波動率": f"{s.volatility:.1f}%",
                "最大回撤": f"{s.max_drawdown:.1f}%",
                "夏普": f"{s.sharpe_ratio:.2f}",
                "殖利率": f"{s.est_dividend_yield:.2f}%" if s.est_dividend_yield > 0 else "N/A",
                "趨勢": s.trend_status,
            })
    return pd.DataFrame(rows)
