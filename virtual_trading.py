"""
🎮 虛擬交易模組 — 新手練習用模擬股市
支援兩種儲存模式：
  - 本機模式：JSON 檔持久化（virtual_portfolio.json）
  - 雲端模式：傳入 portfolio dict（由 app.py 管理 session state）
功能：買賣、庫存、損益、交易紀錄、績效統計
"""

import json
import os
import copy
from datetime import datetime
from typing import Optional

from data_fetcher import fetch_realtime_quote, get_stock_name

# ─────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────
PORTFOLIO_FILE = "virtual_portfolio.json"
STARTING_CASH = 1_000_000
BROKER_FEE_RATE = 0.001425
TAX_RATE = 0.003
MIN_FEE = 20


# ─────────────────────────────────────────────
# 初始 / 預設 portfolio 結構
# ─────────────────────────────────────────────

def new_portfolio() -> dict:
    """建立一個全新的空 portfolio"""
    return {
        "cash": STARTING_CASH,
        "holdings": {},
        "orders": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ─────────────────────────────────────────────
# 本機檔案 I/O
# ─────────────────────────────────────────────

def _portfolio_path() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, PORTFOLIO_FILE)


def _load_portfolio() -> dict:
    path = _portfolio_path()
    if not os.path.exists(path):
        return new_portfolio()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_portfolio(pf: dict):
    pf["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(_portfolio_path(), "w", encoding="utf-8") as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 核心交易邏輯（純函數，不直接讀寫檔案）
# ─────────────────────────────────────────────

def buy_stock(
    stock_id: str,
    shares: int,
    price: float = None,
    portfolio: dict = None,
) -> dict:
    """
    模擬買入股票
    - portfolio: 傳入則操作此 dict（不回寫檔案）；不傳則讀寫 JSON
    回傳: {success, message, portfolio (if input provided)}
    """
    if shares < 1:
        return _result(False, "至少買入 1 股")

    # 取得 portfolio
    pf, from_file = _resolve_portfolio(portfolio)

    # 取得報價
    if price is None:
        quote = fetch_realtime_quote(stock_id)
        if "error" in quote:
            return _result(False, f"無法取得報價: {quote['error']}", pf if portfolio else None)
        price = quote["price"]
        if price <= 0:
            return _result(False, "報價異常", pf if portfolio else None)

    stock_name = get_stock_name(stock_id)

    # 計算成本
    total_cost = price * shares
    fee = max(total_cost * BROKER_FEE_RATE, MIN_FEE)
    total_paid = total_cost + fee

    if pf["cash"] < total_paid:
        return _result(
            False,
            f"餘額不足！需要 ${total_paid:,.0f}，可用 ${pf['cash']:,.0f}",
            pf if portfolio else None,
        )

    # 更新庫存
    holding = pf["holdings"].get(stock_id)
    if holding:
        old_shares = holding["shares"]
        old_cost = holding["total_cost"]
        new_shares = old_shares + shares
        new_total_cost = old_cost + total_paid
        holding["shares"] = new_shares
        holding["total_cost"] = new_total_cost
        holding["avg_cost"] = new_total_cost / new_shares
        holding["stock_name"] = stock_name
    else:
        pf["holdings"][stock_id] = {
            "stock_id": stock_id,
            "stock_name": stock_name,
            "shares": shares,
            "avg_cost": total_paid / shares,
            "total_cost": total_paid,
        }

    pf["cash"] -= total_paid

    # 交易紀錄
    order = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_type": "buy",
        "stock_id": stock_id,
        "stock_name": stock_name,
        "shares": shares,
        "price": round(price, 2),
        "total": round(total_cost, 2),
        "fee": round(fee, 0),
        "tax": 0,
        "cash_after": round(pf["cash"], 0),
    }
    pf["orders"].append(order)

    if from_file:
        _save_portfolio(pf)

    msg = (
        f"✅ 買入 {stock_name}({stock_id}) {shares} 股 @ ${price:.2f}，"
        f"總成本 ${total_paid:,.0f}（含手續費 ${fee:.0f}）"
    )
    return _result(True, msg, pf if portfolio else None, order)


def sell_stock(
    stock_id: str,
    shares: int,
    price: float = None,
    portfolio: dict = None,
) -> dict:
    """模擬賣出股票，參數同 buy_stock"""
    if shares < 1:
        return _result(False, "至少賣出 1 股")

    pf, from_file = _resolve_portfolio(portfolio)

    holding = pf["holdings"].get(stock_id)
    if not holding or holding["shares"] < shares:
        return _result(
            False,
            f"庫存不足！持有 {holding['shares'] if holding else 0} 股",
            pf if portfolio else None,
        )

    stock_name = holding["stock_name"]

    if price is None:
        quote = fetch_realtime_quote(stock_id)
        if "error" in quote:
            return _result(False, f"無法取得報價: {quote['error']}", pf if portfolio else None)
        price = quote["price"]
        if price <= 0:
            return _result(False, "報價異常", pf if portfolio else None)

    total_revenue = price * shares
    fee = max(total_revenue * BROKER_FEE_RATE, MIN_FEE)
    tax = total_revenue * TAX_RATE
    net_received = total_revenue - fee - tax

    old_shares = holding["shares"]
    new_shares = old_shares - shares
    cost_of_sold = holding["avg_cost"] * shares
    realized_pnl = net_received - cost_of_sold
    realized_pnl_pct = (net_received / cost_of_sold - 1) * 100 if cost_of_sold > 0 else 0

    if new_shares == 0:
        del pf["holdings"][stock_id]
    else:
        cost_ratio = shares / old_shares
        holding["total_cost"] -= holding["total_cost"] * cost_ratio
        holding["shares"] = new_shares
        holding["avg_cost"] = holding["total_cost"] / new_shares

    pf["cash"] += net_received

    order = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_type": "sell",
        "stock_id": stock_id,
        "stock_name": stock_name,
        "shares": shares,
        "price": round(price, 2),
        "total": round(total_revenue, 2),
        "fee": round(fee, 0),
        "tax": round(tax, 0),
        "net_received": round(net_received, 0),
        "realized_pnl": round(realized_pnl, 0),
        "realized_pnl_pct": round(realized_pnl_pct, 2),
        "cash_after": round(pf["cash"], 0),
    }
    pf["orders"].append(order)

    if from_file:
        _save_portfolio(pf)

    msg = (
        f"✅ 賣出 {stock_name}({stock_id}) {shares} 股 @ ${price:.2f}，"
        f"實收 ${net_received:,.0f}（手續${fee:.0f}+稅${tax:.0f}），"
        f"損益 ${realized_pnl:+,.0f} ({realized_pnl_pct:+.2f}%)"
    )
    return _result(True, msg, pf if portfolio else None, order)


# ─────────────────────────────────────────────
# 查詢功能
# ─────────────────────────────────────────────

def get_portfolio(portfolio: dict = None) -> dict:
    """取得完整 portfolio"""
    pf, from_file = _resolve_portfolio(portfolio)
    return pf


def get_holdings_with_prices(portfolio: dict = None) -> list:
    """取得庫存即時報價與損益"""
    pf, _ = _resolve_portfolio(portfolio)
    results = []

    for sid, h in pf["holdings"].items():
        quote = fetch_realtime_quote(sid)
        current_price = quote.get("price", 0)
        change_pct = quote.get("change_percent", 0)

        market_value = current_price * h["shares"]
        cost = h["total_cost"]
        pnl = market_value - cost
        pnl_pct = (market_value / cost - 1) * 100 if cost > 0 else 0

        results.append({
            "stock_id": sid,
            "stock_name": h["stock_name"],
            "shares": h["shares"],
            "avg_cost": round(h["avg_cost"], 2),
            "total_cost": round(cost, 0),
            "current_price": current_price,
            "market_value": round(market_value, 0),
            "unrealized_pnl": round(pnl, 0),
            "unrealized_pnl_pct": round(pnl_pct, 2),
            "day_change_pct": change_pct,
        })

    results.sort(key=lambda x: abs(x["unrealized_pnl"]), reverse=True)
    return results


def get_portfolio_summary(portfolio: dict = None) -> dict:
    """取得資產摘要"""
    pf, _ = _resolve_portfolio(portfolio)
    cash = pf["cash"]

    holdings_val = 0
    total_cost = 0
    for sid, h in pf["holdings"].items():
        holdings_val += h["avg_cost"] * h["shares"]  # 以成本估算
        total_cost += h["total_cost"]

    total_value = cash + holdings_val
    overall_pnl = total_value - STARTING_CASH
    overall_pnl_pct = (total_value / STARTING_CASH - 1) * 100

    realized_pnl = sum(
        o.get("realized_pnl", 0) for o in pf["orders"]
        if o["order_type"] == "sell"
    )

    return {
        "cash": round(cash, 0),
        "holdings_value": round(holdings_val, 0),
        "total_value": round(total_value, 0),
        "total_cost": round(total_cost, 0),
        "overall_pnl": round(overall_pnl, 0),
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "realized_pnl": round(realized_pnl, 0),
        "holdings_count": len(pf["holdings"]),
        "order_count": len(pf["orders"]),
    }


def get_order_history(limit: int = 50, portfolio: dict = None) -> list:
    """交易紀錄（最新在前）"""
    pf, _ = _resolve_portfolio(portfolio)
    return list(reversed(pf["orders"]))[:limit]


def reset_portfolio(portfolio: dict = None) -> dict:
    """重設資產組合"""
    pf, from_file = _resolve_portfolio(portfolio)
    new_pf = new_portfolio()

    if from_file:
        _save_portfolio(new_pf)
        return {"success": True, "message": f"✅ 已重設！初始資金 ${STARTING_CASH:,.0f}"}

    # 雲端模式：回傳新的 portfolio
    return {"success": True, "message": f"✅ 已重設！初始資金 ${STARTING_CASH:,.0f}", "portfolio": new_pf}


# ─────────────────────────────────────────────
# 內部工具函式
# ─────────────────────────────────────────────

def _resolve_portfolio(portfolio: dict = None):
    """解析 portfolio 來源 — 傳入 dict 就用它，否則讀檔案"""
    if portfolio is not None:
        return portfolio, False
    return _load_portfolio(), True


def _result(success: bool, message: str, portfolio=None, order=None):
    d = {"success": success, "message": message}
    if portfolio is not None:
        d["portfolio"] = portfolio
    if order is not None:
        d["order"] = order
    return d
