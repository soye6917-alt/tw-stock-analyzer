"""
🎲 蒙地卡羅模擬 + 參數最佳化模組
- 蒙地卡羅模擬未來股價走勢
- 參數最佳化（網格搜尋最佳策略參數）
- 多策略組合績效比較
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


def monte_carlo_simulation(df: pd.DataFrame, 
                           n_simulations: int = 1000,
                           n_days: int = 30,
                           seed: int = 42) -> dict:
    """
    蒙地卡羅模擬：根據歷史報酬率與波動率，模擬未來股價路徑
    """
    if df.empty or len(df) < 30:
        return {"error": "歷史資料不足，需至少 30 筆"}
    
    prices = df['Close'].values
    returns = np.diff(np.log(prices))
    
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    current_price = prices[-1]
    
    np.random.seed(seed)
    
    # 幾何布朗運動模擬
    simulation_results = np.zeros((n_days, n_simulations))
    
    for i in range(n_simulations):
        daily_returns = np.random.normal(mu, sigma, n_days)
        price_path = current_price * np.exp(np.cumsum(daily_returns))
        simulation_results[:, i] = price_path
    
    # 統計結果
    final_prices = simulation_results[-1, :]
    median_price = np.median(final_prices)
    mean_price = np.mean(final_prices)
    p5 = np.percentile(final_prices, 5)
    p25 = np.percentile(final_prices, 25)
    p75 = np.percentile(final_prices, 75)
    p95 = np.percentile(final_prices, 95)
    
    # 獲勝機率（模擬結束時價格 > 目前價格）
    win_prob = np.mean(final_prices > current_price)
    
    # 期望報酬率
    expected_return = (mean_price / current_price - 1) * 100
    
    # 風險指標
    worst_case = (p5 / current_price - 1) * 100
    best_case = (p95 / current_price - 1) * 100
    
    return {
        "current_price": round(float(current_price), 2),
        "n_simulations": n_simulations,
        "n_days": n_days,
        "median_price": round(float(median_price), 2),
        "mean_price": round(float(mean_price), 2),
        "p5_bear_case": round(float(p5), 2),
        "p25": round(float(p25), 2),
        "p75": round(float(p75), 2),
        "p95_bull_case": round(float(p95), 2),
        "win_probability": round(float(win_prob), 4),
        "win_prob_pct": f"{win_prob*100:.1f}%",
        "expected_return_pct": round(float(expected_return), 2),
        "worst_case_pct": round(float(worst_case), 2),
        "best_case_pct": round(float(best_case), 2),
        "annualized_volatility": round(float(sigma * np.sqrt(252) * 100), 2),
        "simulation_paths": simulation_results,  # 保留最後幾條路徑畫圖用
    }


def monte_carlo_risk_analysis(df: pd.DataFrame, 
                              n_simulations: int = 10000,
                              confidence: float = 0.99) -> dict:
    """
    蒙地卡羅風險分析：包含 CVaR、最大回撤機率
    """
    result = monte_carlo_simulation(df, n_simulations=n_simulations, n_days=20)
    if "error" in result:
        return result
    
    final_prices = result.get("simulation_paths", np.zeros((20, n_simulations)))[-1, :]
    current_price = result["current_price"]
    
    # CVaR (Conditional VaR)
    var_val = np.percentile(final_prices, (1 - confidence) * 100)
    cvar = final_prices[final_prices <= var_val].mean()
    
    # 最大回撤機率（20天內下跌超過某比例）
    max_drop = 0
    for path_idx in range(min(1000, result.get("n_simulations", 1000))):
        path = result["simulation_paths"][:, path_idx]
        peak = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        max_drop = min(max_drop, drawdown.min())
    
    return {
        "current_price": result["current_price"],
        "cvar_99": round(float((cvar / current_price - 1) * 100), 2),
        "max_simulated_drawdown": round(float(max_drop * 100), 2),
        "simulated_win_prob": result["win_prob_pct"],
        "simulated_expected_return": result["expected_return_pct"],
    }


# ─────────────────────────────────────────────
# 參數最佳化（網格搜尋）
# ─────────────────────────────────────────────
def optimize_ma_crossover(df: pd.DataFrame,
                          fast_range: List[int] = [5, 10, 15, 20, 30],
                          slow_range: List[int] = [20, 30, 45, 60, 90]) -> dict:
    """
    均線交叉策略參數最佳化
    掃描所有 (fast, slow) 組合，找出夏普比率最高的參數
    """
    from backtest import backtest_ma_crossover
    
    if df.empty or len(df) < 100:
        return {"error": "資料不足，需 ≥100 筆"}
    
    results = []
    best_sharpe = -999
    best_params = None
    
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            try:
                bt = backtest_ma_crossover(df, fast, slow)
                if "error" not in bt and bt.get("sharpe_ratio", -999) > -999:
                    sr = bt["sharpe_ratio"]
                    results.append({
                        "fast": fast, "slow": slow,
                        "total_return": bt.get("total_return_pct", 0),
                        "sharpe_ratio": sr,
                        "max_drawdown": bt.get("max_drawdown_pct", 0),
                        "trade_count": bt.get("trade_count", 0),
                        "win_rate": bt.get("win_rate", 0),
                    })
                    if sr > best_sharpe:
                        best_sharpe = sr
                        best_params = (fast, slow)
            except:
                continue
    
    if not results:
        return {"error": "無有效的回測結果"}
    
    df_results = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    
    return {
        "best_params": {"fast": best_params[0], "slow": best_params[1]},
        "best_sharpe": round(float(best_sharpe), 4),
        "top_5": df_results.head(5).to_dict('records'),
        "total_combinations": len(results),
    }


def optimize_rsi(df: pd.DataFrame,
                 period_range: List[int] = [7, 9, 14, 21],
                 oversold_range: List[int] = [20, 25, 30, 35],
                 overbought_range: List[int] = [65, 70, 75, 80]) -> dict:
    """
    RSI 策略參數最佳化
    """
    from backtest import backtest_rsi
    
    if df.empty or len(df) < 100:
        return {"error": "資料不足"}
    
    results = []
    best_sharpe = -999
    best_params = None
    
    for period in period_range:
        for oversold in oversold_range:
            for overbought in overbought_range:
                if oversold >= overbought:
                    continue
                try:
                    bt = backtest_rsi(df, period, oversold, overbought)
                    if "error" not in bt and bt.get("sharpe_ratio", -999) > -999:
                        sr = bt["sharpe_ratio"]
                        results.append({
                            "period": period, "oversold": oversold, "overbought": overbought,
                            "total_return": bt.get("total_return_pct", 0),
                            "sharpe_ratio": sr,
                            "max_drawdown": bt.get("max_drawdown_pct", 0),
                            "trade_count": bt.get("trade_count", 0),
                        })
                        if sr > best_sharpe:
                            best_sharpe = sr
                            best_params = (period, oversold, overbought)
                except:
                    continue
    
    if not results:
        return {"error": "無有效的回測結果"}
    
    df_results = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    
    return {
        "best_params": {
            "period": best_params[0], "oversold": best_params[1], "overbought": best_params[2]
        },
        "best_sharpe": round(float(best_sharpe), 4),
        "top_5": df_results.head(5).to_dict('records'),
        "total_combinations": len(results),
    }


# ─────────────────────────────────────────────
# 多策略組合分析
# ─────────────────────────────────────────────
def strategy_comparison(df: pd.DataFrame) -> dict:
    """
    比較多種策略在同一段歷史的績效
    """
    from backtest import backtest_ma_crossover, backtest_rsi, backtest_macd, backtest_bollinger
    
    strategies = {
        "MA5/20 交叉": lambda: backtest_ma_crossover(df, 5, 20),
        "MA10/60 交叉": lambda: backtest_ma_crossover(df, 10, 60),
        "RSI(14,30/70)": lambda: backtest_rsi(df, 14, 30, 70),
        "RSI(7,25/75)": lambda: backtest_rsi(df, 7, 25, 75),
        "MACD": lambda: backtest_macd(df),
        "布林通道": lambda: backtest_bollinger(df),
    }
    
    results = []
    for name, fn in strategies.items():
        try:
            bt = fn()
            if "error" not in bt:
                results.append({
                    "strategy": name,
                    "total_return_pct": bt.get("total_return_pct", 0),
                    "sharpe_ratio": bt.get("sharpe_ratio", 0),
                    "max_drawdown_pct": bt.get("max_drawdown_pct", 0),
                    "win_rate": bt.get("win_rate", 0),
                    "trade_count": bt.get("trade_count", 0),
                })
        except:
            continue
    
    if not results:
        return {"error": "所有策略回測皆失敗"}
    
    return pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False).to_dict('records')
