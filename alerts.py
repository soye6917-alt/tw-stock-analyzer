"""
🔔 盤中即時監控與通知模組
- 價格警示（突破/跌破指定價位）
- 技術面觸發（RSI 超買/超賣、均線交叉）
- 條件設定與觸發記錄
- Line Notify 通知
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────

@dataclass
class AlertCondition:
    """警示條件"""
    id: str
    stock_id: str
    stock_name: str = ""
    alert_type: str = "price_above"  # price_above, price_below, rsi_oversold, rsi_overbought, ma_cross
    target_value: float = 0.0        # 觸發價位/數值
    note: str = ""
    enabled: bool = True
    created_at: str = ""
    last_triggered: str = ""


@dataclass
class AlertEvent:
    """觸發事件記錄"""
    id: str
    condition_id: str
    stock_id: str
    alert_type: str
    triggered_value: float
    triggered_at: str
    message: str
    notified: bool = False


# ─────────────────────────────────────────────
# 警示管理
# ─────────────────────────────────────────────

_ALERTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.json")
_EVENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_events.json")


def _load_json(filepath: str, default: list = None) -> list:
    if default is None:
        default = []
    if not os.path.exists(filepath):
        return default
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return default
    except:
        return default


def _save_json(filepath: str, data: list):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_alert(stock_id: str, alert_type: str, target_value: float,
              stock_name: str = "", note: str = "") -> dict:
    """新增警示條件"""
    alerts = _load_json(_ALERTS_FILE)
    
    alert = AlertCondition(
        id=f"alert_{int(time.time())}_{stock_id}",
        stock_id=stock_id,
        stock_name=stock_name,
        alert_type=alert_type,
        target_value=target_value,
        note=note,
        enabled=True,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    
    alerts.append(asdict(alert))
    _save_json(_ALERTS_FILE, alerts)
    
    return asdict(alert)


def remove_alert(alert_id: str) -> bool:
    """刪除警示條件"""
    alerts = _load_json(_ALERTS_FILE)
    new_alerts = [a for a in alerts if a.get("id") != alert_id]
    if len(new_alerts) < len(alerts):
        _save_json(_ALERTS_FILE, new_alerts)
        return True
    return False


def toggle_alert(alert_id: str) -> bool:
    """啟用/停用警示"""
    alerts = _load_json(_ALERTS_FILE)
    for a in alerts:
        if a.get("id") == alert_id:
            a["enabled"] = not a.get("enabled", True)
            _save_json(_ALERTS_FILE, alerts)
            return a["enabled"]
    return False


def get_alerts(stock_id: str = "") -> List[dict]:
    """取得警示列表，可依股票過濾"""
    alerts = _load_json(_ALERTS_FILE)
    if stock_id:
        return [a for a in alerts if a.get("stock_id") == stock_id]
    return alerts


def check_price_alert(current_price: float, stock_id: str, stock_name: str = "") -> List[str]:
    """
    檢查價格警示是否觸發
    回傳觸發的訊息列表
    """
    alerts = _load_json(_ALERTS_FILE)
    events = _load_json(_EVENTS_FILE)
    triggered = []
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for a in alerts:
        if not a.get("enabled", True):
            continue
        if a.get("stock_id") != stock_id:
            continue
        
        alert_type = a.get("alert_type", "")
        target = a.get("target_value", 0)
        triggered_flag = False
        message = ""
        
        if alert_type == "price_above" and current_price >= target:
            triggered_flag = True
            message = (f"🚨 {stock_name}({stock_id}) 價格突破！"
                       f"{current_price:.2f} ≥ 目標 {target:.2f}")
        elif alert_type == "price_below" and current_price <= target:
            triggered_flag = True
            message = (f"🚨 {stock_name}({stock_id}) 價格跌破！"
                       f"{current_price:.2f} ≤ 目標 {target:.2f}")
        
        if triggered_flag:
            # 檢查是否在短時間內重複觸發（24小時內不重複）
            last = a.get("last_triggered", "")
            if last:
                try:
                    last_dt = datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
                    if (datetime.now() - last_dt).total_seconds() < 86400:
                        continue  # 24小時內不重複通知
                except:
                    pass
            
            # 更新最後觸發時間
            a["last_triggered"] = now
            
            # 記錄事件
            event = AlertEvent(
                id=f"evt_{int(time.time())}",
                condition_id=a["id"],
                stock_id=stock_id,
                alert_type=alert_type,
                triggered_value=current_price,
                triggered_at=now,
                message=message,
            )
            events.append(asdict(event))
            
            triggered.append(message)
    
    # 只保留最近100筆事件
    if len(events) > 100:
        events = events[-100:]
    
    _save_json(_ALERTS_FILE, alerts)
    _save_json(_EVENTS_FILE, events)
    
    return triggered


def get_recent_events(limit: int = 20) -> List[dict]:
    """取得最近觸發事件"""
    events = _load_json(_EVENTS_FILE)
    return events[-limit:][::-1]


def clear_events():
    """清除事件記錄"""
    _save_json(_EVENTS_FILE, [])


# ─────────────────────────────────────────────
# Line Notify 通知（可選）
# ─────────────────────────────────────────────

def send_line_notify(token: str, message: str) -> bool:
    """
    透過 Line Notify 發送通知
    需先在 https://notify-bot.line.me 申請 token
    """
    if not token:
        return False
    
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        return resp.status_code == 200
    except:
        return False
