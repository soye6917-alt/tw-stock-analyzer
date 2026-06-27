"""
📐 統計套利與配對交易分析模組
- Cointegration (共整合) 測試：找出長期相關的股票對
- 價差回歸均值 (Mean Reversion) 策略
- Cross-Asset 比較 (台積電ADR溢價)
- 類股價差分析
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. Cointegration 共整合測試 (Engle-Granger)
# ─────────────────────────────────────────────
def cointegration_test(price_a: np.ndarray, price_b: np.ndarray) -> Dict:
    """
    Engle-Granger 共整合測試 (手寫實作)

    回傳:
        is_cointegrated: 是否共整合
        spread: 價差序列
        zscore: 當前 zscore
        hedge_ratio: 避險比率
        half_life: 均值回歸半衰期
    """
    n = min(len(price_a), len(price_b))
    if n < 30:
        return {"error": f"資料不足 (需≥30, 實際{n})"}

    y = price_a[-n:]
    x = price_b[-n:]

    # ── 第一步：回歸 y = α + β·x + ε ──
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha, hedge_ratio = beta[0], beta[1]
    spread = y - hedge_ratio * x - alpha

    # ── 第二步：對殘差做 ADF 檢定（簡化版） ──
    adf_result = _adf_test(spread)
    is_cointegrated = adf_result.get("is_stationary", False)

    # ── 均值回歸參數 ──
    current_zscore = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-8)

    # 半衰期 (Ornstein-Uhlenbeck)
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    if len(spread_lag) > 5:
        coeff = np.polyfit(spread_lag, spread_diff, 1)[0]
        half_life = -np.log(2) / coeff if coeff < 0 else None
    else:
        half_life = None

    # 當前是否偏離
    if abs(current_zscore) > 2:
        signal = "🔴 極度偏離 (>2σ)"
    elif abs(current_zscore) > 1.5:
        signal = "🟡 偏離 (>1.5σ)"
    elif abs(current_zscore) > 1:
        signal = "⚪ 輕微偏離"
    else:
        signal = "🟢 正常範圍"

    direction = "short_pair" if current_zscore > 0 else "long_pair"

    return {
        "is_cointegrated": is_cointegrated,
        "correlation": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "hedge_ratio": round(float(hedge_ratio), 4),
        "alpha": round(float(alpha), 4),
        "current_spread": round(float(spread[-1]), 4),
        "spread_mean": round(float(np.mean(spread)), 4),
        "spread_std": round(float(np.std(spread)), 4),
        "current_zscore": round(float(current_zscore), 3),
        "zscore_signal": signal,
        "half_life_days": round(float(half_life), 1) if half_life else None,
        "position_direction": direction,
        "adf_statistic": round(float(adf_result.get("statistic", 0)), 4),
        "adf_pvalue": round(float(adf_result.get("pvalue", 1)), 4),
    }


def _adf_test(series: np.ndarray, max_lag: int = 5) -> Dict:
    """
    簡化版 Augmented Dickey-Fuller 檢定

    使用 OLS 回歸殘差序列來判斷是否定態 (stationary)
    從 MacKinnon (1994) 取簡化臨界值
    H0: 有單根 (非定態)
    """
    n = len(series)
    if n < 10:
        return {"is_stationary": False, "statistic": 0, "pvalue": 1}

    # ADF 回歸: Δy_t = α + β·t + γ·y_{t-1} + Σδ·Δy_{t-i} + ε
    dy = np.diff(series)
    y_lag = series[:-1]

    # 簡單版本: 只測 AR(1) 係數
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        coeff = np.linalg.lstsq(X, dy, rcond=None)[0]
        gamma = coeff[1]  # y_{t-1} 的係數

        # t-statistic for gamma
        residuals = dy - X @ coeff
        se = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2) / np.sum((y_lag - np.mean(y_lag)) ** 2))
        t_stat = gamma / (se + 1e-8)

        # 簡化臨界值 (n≈100 時的 5% 水準 ≈ -2.89)
        # 保守起見用 -3.0
        is_stationary = t_stat < -3.0

        return {
            "is_stationary": is_stationary,
            "statistic": t_stat,
            "pvalue": None,  # 需查表才能給精確 p-value
        }
    except Exception:
        return {"is_stationary": False, "statistic": 0, "pvalue": 1}


# ─────────────────────────────────────────────
# 2. 類股價差分析
# ─────────────────────────────────────────────
def sector_spread_analysis(
    df_dict: Dict[str, pd.DataFrame],
    benchmark_id: str = None,
) -> Dict:
    """
    類股相對強度與價差分析

    df_dict: {stock_id: DataFrame with Close}
    benchmark_id: 基準股票（如 0050）
    """
    if not df_dict or len(df_dict) < 2:
        return {"error": "需要至少 2 檔股票"}

    # 獲取基準
    if benchmark_id and benchmark_id in df_dict:
        benchmark = df_dict[benchmark_id]['Close'].values
    else:
        # 用第一支作為基準
        first_key = list(df_dict.keys())[0]
        benchmark = df_dict[first_key]['Close'].values
        benchmark_id = first_key

    results = {}
    for sid, df in df_dict.items():
        if sid == benchmark_id:
            continue
        if df.empty or len(df) < 20:
            continue

        close = df['Close'].values
        min_len = min(len(close), len(benchmark))
        if min_len < 20:
            continue

        a = close[-min_len:]
        b = benchmark[-min_len:]

        # 相對強度
        rel_strength = (a[-1] / a[0]) / (b[-1] / b[0])
        # 價差
        spread = a - b * (a[-1] / (b[-1] + 1e-8))

        coint = cointegration_test(a, b)

        results[sid] = {
            "relative_strength": round(float(rel_strength), 4),
            "relative_performance": f"{(rel_strength - 1) * 100:+.2f}%",
            "correlation_with_benchmark": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "cointegration": coint,
        }

    return {
        "benchmark": benchmark_id,
        "pairs": results,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }


# ─────────────────────────────────────────────
# 3. 台積電 ADR 溢價分析
# ─────────────────────────────────────────────
def tsm_arbitrage_analysis() -> Dict:
    """
    台積電 (2330) vs 台積電 ADR (TSM) 溢價分析
    TSM ADR = 1 ADR 換算 5 股台積電
    """
    result = {
        "tsm_adr_price": None,
        "tsm_spot_price": None,
        "implied_twd_price": None,
        "premium_pct": None,
        "usd_twd_rate": None,
        "arbitrage_signal": "⚪ 無法計算",
        "error": None,
    }

    try:
        import requests as req
        session = req.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        })

        # TSM ADR 價格 (NASDAQ)
        url = "https://query1.finance.yahoo.com/v8/finance/chart/TSM"
        resp = session.get(url, timeout=10)
        tsm_data = resp.json()
        if tsm_data.get("chart", {}).get("result"):
            quotes = tsm_data["chart"]["result"][0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            if quotes and quotes[-1]:
                tsm_adr = float(quotes[-1])
            else:
                tsm_adr = None
        else:
            tsm_adr = None

        # 台積電現貨 (從 Goodinfo)
        from data_fetcher import fetch_realtime_quote
        quote = fetch_realtime_quote("2330")
        tsm_tw = quote.get("price", None)

        # 匯率
        from macro_data import fetch_usd_twd_rate
        fx = fetch_usd_twd_rate()
        fx_rate = fx.get("rate") or fx.get("spot_buy")

        if tsm_adr and tsm_tw and fx_rate:
            # 1 ADR = 5 股台積電
            implied_twd = tsm_adr * fx_rate / 5
            premium = (implied_twd / tsm_tw - 1) * 100

            if premium > 5:
                signal = "🔴 台積電溢價 >5% (ADR 比現貨貴，可能有套利空間)"
            elif premium > 2:
                signal = "🟡 輕微溢價"
            elif premium > -2:
                signal = "🟢 價格合理 (折溢價 < 2%)"
            elif premium > -5:
                signal = "🟡 輕微折價"
            else:
                signal = "🟢 台積電折價 >5% (ADR 比現貨便宜，外資偏空)"

            result["tsm_adr_price"] = round(tsm_adr, 2)
            result["tsm_spot_price"] = round(tsm_tw, 2)
            result["implied_twd_price"] = round(implied_twd, 2)
            result["premium_pct"] = round(premium, 2)
            result["usd_twd_rate"] = round(fx_rate, 4)
            result["arbitrage_signal"] = signal

    except Exception as e:
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────
# 4. 配對交易建議生成
# ─────────────────────────────────────────────
def suggest_pair_trades(
    df_dict: Dict[str, pd.DataFrame],
    min_correlation: float = 0.7,
) -> List[Dict]:
    """
    在股票池中找共整合對，產生配對交易建議

    回傳: 共整合對列表，按 zscore 偏離度排序
    """
    pairs = []
    stock_ids = list(df_dict.keys())

    if len(stock_ids) < 2:
        return [{"error": "需要至少 2 檔股票"}]

    for i in range(len(stock_ids)):
        for j in range(i + 1, len(stock_ids)):
            sid_a = stock_ids[i]
            sid_b = stock_ids[j]

            df_a = df_dict[sid_a]
            df_b = df_dict[sid_b]

            if df_a.empty or df_b.empty or len(df_a) < 30 or len(df_b) < 30:
                continue

            close_a = df_a['Close'].values
            close_b = df_b['Close'].values

            # 需要等長
            min_len = min(len(close_a), len(close_b))
            if min_len < 30:
                continue

            corr = np.corrcoef(close_a[-min_len:], close_b[-min_len:])[0, 1]
            if corr < min_correlation:
                continue

            coint = cointegration_test(close_a[-min_len:], close_b[-min_len:])
            if coint.get("error"):
                continue

            if coint.get("is_cointegrated", False) or abs(coint.get("current_zscore", 0)) > 2:
                pairs.append({
                    "stock_a": sid_a,
                    "stock_b": sid_b,
                    "correlation": corr,
                    "cointegration": coint,
                    "zscore": coint.get("current_zscore", 0),
                    "is_cointegrated": coint.get("is_cointegrated", False),
                })

    # 按 zscore 絕對值排序
    pairs.sort(key=lambda x: abs(x.get("zscore", 0)), reverse=True)
    return pairs[:10]


# ─────────────────────────────────────────────
# 5. 主入口
# ─────────────────────────────────────────────
def run_statistical_arbitrage(
    stock_id: str,
    df_main: pd.DataFrame,
    all_stocks_data: Dict[str, pd.DataFrame] = None,
) -> Dict:
    """
    統計套利主入口

    對指定股票進行：
    1. 與其他股票共整合測試
    2. 與大盤相關性分析
    3. 台積電 ADR 分析（如果股票是 2330）

    注意: 需要多支股票資料
    """
    if df_main.empty or len(df_main) < 30:
        return {"error": "資料不足"}

    result = {
        "stock_id": stock_id,
        "pairs": [],
        "tsm_adr": None,
        "benchmark_analysis": None,
        "summary_lines": [],
        "error": None,
    }

    lines = []

    # 與大盤 (0050) 的分析
    try:
        from data_fetcher import fetch_historical
        df_0050 = fetch_historical("0050", months=6)
        if not df_0050.empty and len(df_0050) >= 30:
            min_l = min(len(df_main), len(df_0050))
            c_a = df_main['Close'].values[-min_l:]
            c_b = df_0050['Close'].values[-min_l:]
            corr = np.corrcoef(c_a, c_b)[0, 1]
            beta_est = np.polyfit(c_b, c_a, 1)[0]
            result["benchmark_analysis"] = {
                "correlation_with_0050": round(float(corr), 4),
                "beta_to_0050": round(float(beta_est), 4),
            }
            lines.append(f"**📊 與大盤 (0050) 對比**")
            lines.append(f"  相關係數: {corr:.4f}")
            lines.append(f"  β (Beta): {beta_est:.4f}")
    except Exception as e:
        pass

    # 共整合分析 (如果有多檔資料)
    if all_stocks_data and len(all_stocks_data) >= 2:
        try:
            pairs = suggest_pair_trades(all_stocks_data, min_correlation=0.5)
            result["pairs"] = [p for p in pairs if p.get("stock_a") == stock_id or p.get("stock_b") == stock_id][:5]
            if result["pairs"]:
                lines.append(f"")
                lines.append(f"**🔄 共整合配對發現**")
                for pair in result["pairs"]:
                    other = pair["stock_b"] if pair["stock_a"] == stock_id else pair["stock_a"]
                    z = pair.get("zscore", 0)
                    coint_mark = "✅" if pair.get("is_cointegrated") else "🔶"
                    lines.append(f"  {coint_mark} 與 {other} 價差 z-score: {z:.2f} ({pair.get('cointegration', {}).get('zscore_signal', 'N/A')})")
        except Exception:
            pass

    # 台積電 ADR 分析
    if stock_id == "2330":
        tsm = tsm_arbitrage_analysis()
        result["tsm_adr"] = tsm
        if tsm.get("error") is None and tsm.get("premium_pct") is not None:
            lines.append(f"")
            lines.append(f"**🌎 台積電 ADR 溢價分析**")
            lines.append(f"  TSM ADR: ${tsm.get('tsm_adr_price', 'N/A')}")
            lines.append(f"  台積電現貨: {tsm.get('tsm_spot_price', 'N/A')}")
            lines.append(f"  隱含台幣價格: {tsm.get('implied_twd_price', 'N/A')}")
            lines.append(f"  折溢價: {tsm.get('premium_pct', 0):+.2f}%")
            lines.append(f"  訊號: {tsm.get('arbitrage_signal', 'N/A')}")

    result["summary_lines"] = lines
    return result
