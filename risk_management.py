"""
📊 資金管理與風險控管模組
- 凱利公式（Kelly Criterion）
- VaR（風險價值）
- 最大回撤分析
- 夏普比率 / 索提諾比率 / 卡瑪比率
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> dict:
    """
    凱利公式：計算最佳下注比例
    f* = (p * b - q) / b
    p = 勝率, b = 賠率（平均獲利/平均虧損）, q = 1-p
    
    回傳建議的資金配置比例（0~1）
    """
    if avg_loss == 0:
        return {"error": "平均虧損不能為0"}
    b = avg_win / abs(avg_loss)
    q = 1 - win_rate
    f = (win_rate * b - q) / b if b != 0 else 0
    
    # 保守版：使用半凱利（Half Kelly）
    half_kelly = f / 2
    
    return {
        "full_kelly": round(max(0, min(f, 1)), 4),
        "half_kelly": round(max(0, min(half_kelly, 1)), 4),
        "quarter_kelly": round(max(0, min(f / 4, 1)), 4),
        "win_rate": round(win_rate, 4),
        "odds_ratio": round(b, 4),
        "suggestion": _kelly_suggestion(f)
    }


def _kelly_suggestion(f: float) -> str:
    if f <= 0:
        return "🔴 目前策略無正期望值，建議停止交易或調整策略"
    elif f < 0.05:
        return "🔶 建議配置 5% 以下資金（低信心策略）"
    elif f < 0.15:
        return "🟡 建議配置 5%~15% 資金（半凱利為佳）"
    elif f < 0.30:
        return "🟢 建議配置 15%~30% 資金（穩健策略）"
    else:
        return "🟢 高信心策略，建議仍以半凱利控制風險"


def calculate_var(returns: np.ndarray, confidence: float = 0.95,
                  method: str = "historical") -> dict:
    """
    VaR（風險價值）：在給定信心水準下，一段期間內最大預期損失
    method: "historical" / "parametric" / "monte_carlo"
    """
    if len(returns) < 30:
        return {"error": "至少需要 30 筆資料"}
    
    result = {"confidence": confidence}
    
    if method == "historical" or method == "all":
        var_hist = np.percentile(returns, (1 - confidence) * 100)
        result["historical_var"] = round(float(var_hist), 4)
        result["historical_var_pct"] = f"{abs(var_hist)*100:.2f}%"
    
    if method == "parametric" or method == "all":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        from scipy.stats import norm
        z = norm.ppf(1 - confidence)
        var_para = mu + z * sigma
        result["parametric_var"] = round(float(var_para), 4)
        result["parametric_var_pct"] = f"{abs(var_para)*100:.2f}%"
    
    if method == "monte_carlo" or method == "all":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        np.random.seed(42)
        sim_returns = np.random.normal(mu, sigma, 100000)
        var_mc = np.percentile(sim_returns, (1 - confidence) * 100)
        result["monte_carlo_var"] = round(float(var_mc), 4)
        result["monte_carlo_var_pct"] = f"{abs(var_mc)*100:.2f}%"
    
    # CVaR（條件VaR）：超過VaR的平均損失
    if method == "all":
        var_val = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var_val].mean()
        result["cvar"] = round(float(cvar), 4)
        result["cvar_pct"] = f"{abs(cvar)*100:.2f}%"
    
    return result


def max_drawdown(equity_curve: pd.Series) -> dict:
    """最大回撤分析"""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    
    # 回撤持續時間
    dd_start = None
    dd_durations = []
    is_in_drawdown = False
    for i, val in enumerate(drawdown):
        if val < 0 and not is_in_drawdown:
            dd_start = i
            is_in_drawdown = True
        elif val >= 0 and is_in_drawdown:
            dd_durations.append(i - dd_start)
            is_in_drawdown = False
    if is_in_drawdown:
        dd_durations.append(len(drawdown) - dd_start)
    
    avg_dd_duration = np.mean(dd_durations) if dd_durations else 0
    
    return {
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "max_drawdown_date": str(max_dd_idx.date()) if hasattr(max_dd_idx, 'date') else str(max_dd_idx),
        "avg_drawdown_duration_days": round(float(avg_dd_duration), 1),
        "drawdown_count": len(dd_durations),
        "current_drawdown": round(float(drawdown.iloc[-1] * 100), 2),
    }


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                 periods_per_year: int = 252) -> float:
    """夏普比率：每單位風險的超額報酬"""
    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(returns, ddof=1) == 0:
        return 0.0
    sr = np.mean(excess_returns) / np.std(returns, ddof=1)
    return round(float(sr * np.sqrt(periods_per_year)), 4)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                  periods_per_year: int = 252) -> float:
    """索提諾比率：只考慮下行風險"""
    excess_returns = returns - risk_free_rate / periods_per_year
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside, ddof=1) == 0:
        return 0.0
    sortino = np.mean(excess_returns) / np.std(downside, ddof=1)
    return round(float(sortino * np.sqrt(periods_per_year)), 4)


def calmar_ratio(returns: np.ndarray, equity_curve: pd.Series,
                 periods_per_year: int = 252) -> float:
    """卡瑪比率：年化報酬 / 最大回撤"""
    annual_return = np.mean(returns) * periods_per_year
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd = abs(drawdown.min())
    if max_dd == 0:
        return 0.0
    return round(float(annual_return / max_dd), 4)


def calculate_position_size(portfolio_value: float, risk_per_trade: float,
                            entry_price: float, stop_loss: float,
                            max_position_pct: float = 0.2) -> dict:
    """
    根據固定風險比例計算部位大小
    risk_per_trade: 每筆交易願意承受的資金損失比例 (ex: 0.02 = 2%)
    """
    risk_amount = portfolio_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss)
    if price_risk == 0:
        return {"error": "停損價與進場價相同"}
    
    shares = int(risk_amount / price_risk / 1000) * 1000  # 台股以張為單位
    cost = shares * entry_price
    max_allowed = portfolio_value * max_position_pct
    
    if cost > max_allowed:
        shares = int(max_allowed / entry_price / 1000) * 1000
        cost = shares * entry_price
    
    return {
        "shares": shares,
        "position_cost": round(cost),
        "portfolio_pct": round(cost / portfolio_value * 100, 2),
        "risk_amount": round(risk_amount),
        "risk_pct": risk_per_trade * 100,
        "stop_loss_price": stop_loss,
    }


def full_risk_report(returns: np.ndarray, equity_curve: pd.Series,
                     portfolio_value: float = 1000000) -> dict:
    """完整的風險報告"""
    sr = sharpe_ratio(returns)
    sor = sortino_ratio(returns)
    dd_info = max_drawdown(equity_curve)
    var_95 = calculate_var(returns, 0.95, "all")
    var_99 = calculate_var(returns, 0.99, "all")
    cr = calmar_ratio(returns, equity_curve)
    
    # Win rate
    win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    
    return {
        "sharpe_ratio": sr,
        "sortino_ratio": sor,
        "calmar_ratio": cr,
        "win_rate": round(float(win_rate), 4),
        "max_drawdown": dd_info,
        "var_95": var_95,
        "var_99": var_99,
        "annual_return": round(float(np.mean(returns) * 252 * 100), 2),
        "annual_volatility": round(float(np.std(returns, ddof=1) * np.sqrt(252) * 100), 2),
        "total_return": round(float((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100), 2),
    }
