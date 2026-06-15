"""
回測引擎模組
支援多種策略：均線交叉、RSI、MACD、布林通道
"""

import pandas as pd
import numpy as np
from typing import Callable


def backtest_ma_crossover(df: pd.DataFrame,
                          fast_period: int = 5,
                          slow_period: int = 20,
                          initial_capital: float = 1_000_000) -> dict:
    """
    均線交叉策略
    - 快線向上突破慢線 → 買入
    - 快線向下跌破慢線 → 賣出
    """
    data = df.copy()
    if len(data) < slow_period + 5:
        return {"error": "資料不足"}
    data["FAST"] = data["Close"].rolling(window=fast_period).mean()
    data["SLOW"] = data["Close"].rolling(window=slow_period).mean()
    data["Signal"] = 0
    data.loc[data["FAST"] > data["SLOW"], "Signal"] = 1
    data["Position"] = data["Signal"].diff()
    return _run_backtest(data, initial_capital, "均線交叉策略")


def backtest_rsi(df: pd.DataFrame,
                 period: int = 14,
                 oversold: float = 30,
                 overbought: float = 70,
                 initial_capital: float = 1_000_000) -> dict:
    """
    RSI 策略
    - RSI < oversold → 買入
    - RSI > overbought → 賣出
    """
    from indicators import add_rsi
    data = add_rsi(df.copy(), period)
    data["Signal"] = 0
    # RSI 從超賣區反彈 → 買；從超買區回落 → 賣
    data["Signal"] = np.where(data["RSI"] < oversold, 1, data["Signal"])
    data["Signal"] = np.where(data["RSI"] > overbought, -1, data["Signal"])
    # 計算持倉變化：只有真正的交叉才算
    prev_rsi = data["RSI"].shift(1)
    data["Buy"] = ((data["RSI"] > oversold) & (prev_rsi <= oversold)).astype(int)
    data["Sell"] = ((data["RSI"] < overbought) & (prev_rsi >= overbought)).astype(int)
    data["Signal2"] = 0
    data.loc[data["Buy"] == 1, "Signal2"] = 1
    data.loc[data["Sell"] == 1, "Signal2"] = -1
    data["Position"] = data["Signal2"].diff()
    return _run_backtest(data, initial_capital, f"RSI({period}) 策略")


def backtest_macd(df: pd.DataFrame,
                  initial_capital: float = 1_000_000) -> dict:
    """
    MACD 策略
    - MACD 突破訊號線 → 買入
    - MACD 跌破訊號線 → 賣出
    """
    from indicators import add_macd
    data = add_macd(df.copy())
    # 使用 MACD_Hist 由負轉正 / 由正轉負
    data["Hist_pos"] = data["MACD_Hist"] > 0
    data["Buy"] = (data["Hist_pos"] & ~data["Hist_pos"].shift(1).fillna(False)).astype(int)
    data["Sell"] = (~data["Hist_pos"] & data["Hist_pos"].shift(1).fillna(False)).astype(int)
    data["Signal"] = 0
    data.loc[data["Buy"] == 1, "Signal"] = 1
    data.loc[data["Sell"] == 1, "Signal"] = -1
    data["Position"] = data["Signal"].diff()
    return _run_backtest(data, initial_capital, "MACD 策略")


def backtest_bollinger(df: pd.DataFrame,
                       period: int = 20,
                       std: float = 2,
                       initial_capital: float = 1_000_000) -> dict:
    """
    布林通道策略
    - 跌破下軌 → 買入（反彈）
    - 突破上軌 → 賣出（獲利）
    """
    from indicators import add_bollinger
    data = add_bollinger(df.copy(), period, std)
    data["Buy"] = (data["Close"] <= data["BB_Lower"]).astype(int)
    data["Sell"] = (data["Close"] >= data["BB_Upper"]).astype(int)
    # 確保訊號交替
    data["Signal"] = 0
    in_position = False
    for i in range(len(data)):
        if not in_position and data["Buy"].iloc[i] == 1:
            data.loc[data.index[i], "Signal"] = 1
            in_position = True
        elif in_position and data["Sell"].iloc[i] == 1:
            data.loc[data.index[i], "Signal"] = -1
            in_position = False
    data["Position"] = data["Signal"].diff()
    return _run_backtest(data, initial_capital, "布林通道策略")


def _run_backtest(data: pd.DataFrame,
                  initial_capital: float,
                  strategy_name: str) -> dict:
    """
    執行回測計算
    """
    result = data.copy()
    result["Returns"] = result["Close"].pct_change()
    result["Strategy_Returns"] = 0.0

    capital = initial_capital
    shares = 0
    cash = capital
    trades = []
    portfolio_values = []

    for i in range(1, len(result)):
        price = result["Close"].iloc[i]
        signal = result["Position"].iloc[i]
        date = result["日期"].iloc[i]

        # 買入訊號
        if signal == 1 and cash > 0:
            # 投入 95% 現金買入（預留手續費空間）
            invest = cash * 0.95
            shares_to_buy = invest / price
            if shares_to_buy >= 1:
                cost = shares_to_buy * price
                cash -= cost
                shares += shares_to_buy
                trades.append({
                    "日期": date, "類型": "買入",
                    "價格": price, "數量": round(shares_to_buy, 2),
                    "金額": cost, "剩餘現金": cash,
                })

        # 賣出訊號
        elif signal == -1 and shares > 0:
            revenue = shares * price
            cash += revenue
            trades.append({
                "日期": date, "類型": "賣出",
                "價格": price, "數量": round(shares, 2),
                "金額": revenue, "剩餘現金": cash,
            })
            shares = 0

        portfolio_values.append(cash + shares * price)

    # 最後平倉
    if shares > 0:
        final_price = result["Close"].iloc[-1]
        cash += shares * final_price
        trades.append({
            "日期": result["日期"].iloc[-1], "類型": "強制平倉",
            "價格": final_price, "數量": round(shares, 2),
            "金額": shares * final_price, "剩餘現金": cash,
        })
        shares = 0

    final_value = cash
    total_return = (final_value - initial_capital) / initial_capital * 100

    # 計算最大回撤
    portfolio_series = pd.Series(portfolio_values, index=result["日期"].iloc[1:])
    if len(portfolio_series) > 0:
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
    else:
        max_drawdown = 0

    # 計算 Sharpe Ratio (概略)
    if len(portfolio_series) > 1:
        daily_returns = portfolio_series.pct_change().dropna()
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    # 勝率
    if len(trades) >= 2:
        buy_trades = [t for t in trades if t["類型"] == "買入"]
        sell_trades = [t for t in trades if t["類型"] == "賣出"]
        wins = 0
        total_closed = 0
        for sell in sell_trades:
            for buy in buy_trades:
                if buy["日期"] < sell["日期"]:
                    profit = sell["金額"] - buy["金額"]
                    if profit > 0:
                        wins += 1
                    total_closed += 1
                    break
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
    else:
        win_rate = 0

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    return {
        "strategy": strategy_name,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "trades": trades_df,
        "portfolio_values": portfolio_series if len(portfolio_series) > 0 else pd.Series(),
        "buy_hold_return_pct": round(
            (result["Close"].iloc[-1] - result["Close"].iloc[0]) / result["Close"].iloc[0] * 100, 2
        ),
    }
