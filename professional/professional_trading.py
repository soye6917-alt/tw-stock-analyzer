"""
professional_trading.py — 專業交易系統

功能：
1. 進場檢查清單 (Entry Checklist) — 多因子評分
2. 出場檢查清單 (Exit Checklist) — 停損移動
3. 倉位計算機 (Position Sizer) — 凱利公式 + 固定比例
4. 交易評分卡 (Trade Scorecard) — 每筆交易回顧
"""
import json, math
from typing import Optional, Dict, List
from dataclasses import dataclass
import pandas as pd

# =========== 1. Entry Checklist ===========
ENTRY_FACTORS = {
    'trend': {
        'name': '趨勢方向',
        'weight': 20,
        'description': '均線多頭排列(MA5>MA20>MA60)',
        'check': lambda df: 'MA5' in df.columns and 'MA20' in df.columns and 
                df['MA5'].iloc[-1] > df['MA20'].iloc[-1]
    },
    'volume': {
        'name': '量能確認',
        'weight': 15,
        'description': '成交量 > 20日均量 * 1.2',
        'check': lambda df: 'volume_ma20' in df.columns and 
                df['Volume'].iloc[-1] > df['volume_ma20'].iloc[-1] * 1.2
    },
    'rsi': {
        'name': 'RSI 位置',
        'weight': 10,
        'description': 'RSI(14) 在 40-70 之間 (非超買區追高)',
        'check': lambda df: 'rsi' in df.columns and 40 <= df['RSI'].iloc[-1] <= 70
    },
    'support': {
        'name': '支撐測試',
        'weight': 15,
        'description': '價格在關鍵支撐附近 (前低或均線)',
        'check': lambda df: True  # Simplified, real check needs price levels
    },
    'divergence': {
        'name': '無背離',
        'weight': 10,
        'description': '價量無明顯背離，MACD無背離',
        'check': lambda df: True
    },
    'stop_loss': {
        'name': '有明確停損點',
        'weight': 20,
        'description': '能在進場價格的 3-8% 內設停損',
        'check': lambda df: True  # User decision
    },
    'risk_reward': {
        'name': '盈虧比 > 2:1',
        'weight': 10,
        'description': '預期獲利 / 停損距離 >= 2',
        'check': lambda df: True  # User calculation
    }
}

def calculate_entry_score(df: pd.DataFrame, entry_price: float = None) -> dict:
    """Calculate entry checklist score (0-100)."""
    scores = {}
    total_score = 0
    max_possible = 0
    
    for key, factor in ENTRY_FACTORS.items():
        max_possible += factor['weight']
        try:
            if factor['check'](df):
                scores[key] = {'pass': True, 'weight': factor['weight'], 'score': factor['weight']}
                total_score += factor['weight']
            else:
                scores[key] = {'pass': False, 'weight': factor['weight'], 'score': 0}
        except:
            scores[key] = {'pass': False, 'weight': factor['weight'], 'score': 0, 'error': True}
    
    # Convert to percentage
    pct = round(total_score / max_possible * 100, 1) if max_possible > 0 else 0
    
    # Grade
    if pct >= 80:
        grade = 'A — 強烈建議進場'
    elif pct >= 60:
        grade = 'B — 可考慮進場'
    elif pct >= 40:
        grade = 'C — 謹慎，需更多確認'
    else:
        grade = 'D — 不建議進場'
    
    passed = sum(1 for v in scores.values() if v.get('pass'))
    total = len(scores)
    
    return {
        'total_score': pct,
        'grade': grade,
        'passed': f'{passed}/{total}',
        'details': scores
    }

# =========== 2. Position Sizer ===========
@dataclass
class PositionSizing:
    capital: float          # Total capital
    risk_per_trade: float   # Risk per trade (%)
    entry_price: float
    stop_loss: float
    win_rate: float = 0.5  # Estimated win rate (for Kelly)
    avg_win: float = 0.15  # Average win (%)
    avg_loss: float = 0.07  # Average loss (%)

def calculate_position_size(ps: PositionSizing, method: str = 'fixed_risk') -> dict:
    """Calculate position size using different methods."""
    
    risk_per_share = abs(ps.entry_price - ps.stop_loss)
    risk_amount = ps.capital * (ps.risk_per_trade / 100)
    
    if method == 'fixed_risk':
        # Fixed fractional: risk fixed % of capital per trade
        shares = math.floor(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * ps.entry_price
        risk_pct = round(risk_amount / ps.capital * 100, 2)
    
    elif method == 'kelly':
        # Kelly Criterion: f* = (p*b - q) / b
        # where p = win rate, q = lose rate, b = win/loss ratio
        b = ps.avg_win / ps.avg_loss if ps.avg_loss > 0 else 0
        q = 1 - ps.win_rate
        kelly_f = (ps.win_rate * b - q) / b if b > 0 else 0
        # Conservative: use 25% of Kelly
        kelly_f = max(0, min(0.25, kelly_f * 0.25))
        position_value = ps.capital * kelly_f
        shares = math.floor(position_value / ps.entry_price) if ps.entry_price > 0 else 0
        risk_pct = round(risk_amount / ps.capital * 100, 2)
    
    elif method == 'volatility':
        # ATR-based sizing
        shares = math.floor(risk_amount / (risk_per_share * 1.5)) if risk_per_share > 0 else 0
        position_value = shares * ps.entry_price
        risk_pct = round(risk_amount / ps.capital * 100, 2)
    else:
        return {'error': f'Unknown method: {method}'}
    
    return {
        'method': method,
        'capital': ps.capital,
        'entry_price': ps.entry_price,
        'stop_loss': ps.stop_loss,
        'risk_per_share': round(risk_per_share, 2),
        'risk_amount': round(risk_amount, 0),
        'risk_pct': risk_pct,
        'shares': shares,
        'position_value': round(position_value, 0),
        'position_pct': round(position_value / ps.capital * 100, 1),
        'potential_loss': round(shares * risk_per_share, 0) if shares > 0 else 0
    }

# =========== 3. Exit Checklist ===========
EXIT_REASONS = {
    'stop_hit': '停損觸發',
    'target_hit': '目標價達成',
    'trailing_stop': '移動停損被觸發',
    'trend_reversal': '趨勢反轉確認',
    'fundamental_change': '基本面改變',
    'time_stop': '時間停損（持有過久無進展）',
    'better_opportunity': '換股到更佳機會'
}

def evaluate_exit(df: pd.DataFrame, entry_price: float, current_price: float,
                  stop_loss: float, target: float) -> dict:
    """Evaluate whether to exit a position."""
    signals = []
    
    # Current P&L
    pnl_pct = (current_price - entry_price) / entry_price * 100
    
    # Technical exit signals
    if 'MA5' in df.columns and 'MA20' in df.columns:
        if current_price < df['MA20'].iloc[-1]:
            signals.append({'reason': '跌破20日均線', 'severity': 'warning'})
    
    if 'rsi' in df.columns:
        if df['RSI'].iloc[-1] > 80:
            signals.append({'reason': 'RSI超買區(>80)', 'severity': 'warning'})
        elif df['RSI'].iloc[-1] < 30:
            signals.append({'reason': 'RSI超賣區(<30)', 'severity': 'info'})
    
    # Volume
    if 'volume_ma20' in df.columns:
        if df['Volume'].iloc[-1] > df['volume_ma20'].iloc[-1] * 2 and pnl_pct < -2:
            signals.append({'reason': '爆量下跌', 'severity': 'danger'})
    
    return {
        'entry_price': entry_price,
        'current_price': current_price,
        'pnl_pct': round(pnl_pct, 2),
        'pnl_abs': round((current_price - entry_price), 2),
        'stop_distance': round((current_price - stop_loss) / current_price * 100, 2),
        'target_distance': round((target - current_price) / current_price * 100, 2),
        'signals': signals,
        'action': '持有' if pnl_pct > -5 else '考慮減碼' if pnl_pct > -10 else '建議停損'
    }

# =========== 4. Trade Scorecard ===========
def create_trade_record(stock_id: str, stock_name: str) -> dict:
    """Create a trade record template."""
    return {
        'stock_id': stock_id,
        'stock_name': stock_name,
        'entry_date': '',
        'entry_price': 0,
        'exit_date': '',
        'exit_price': 0,
        'shares': 0,
        'position_size': 0,
        'stop_loss': 0,
        'target': 0,
        'win': None,
        'pnl_pct': 0,
        'pnl_abs': 0,
        'entry_reasons': [],
        'exit_reason': '',
        'followed_plan': None,
        'lesson': '',
        'rating': 0  # 1-5
    }

def rate_trade(trade: dict) -> dict:
    """Rate a completed trade 1-5 stars."""
    score = 3  # Start at neutral
    
    # Followed plan?
    if trade.get('followed_plan'):
        score += 1
    else:
        score -= 1
    
    # Good risk management?
    if trade.get('stop_loss', 0) > 0:
        score += 0.5
    
    # Good entry reason?
    if len(trade.get('entry_reasons', [])) >= 3:
        score += 0.5
    
    # Outcome (process > outcome)
    if trade.get('win'):
        score += 0.5
    else:
        score -= 0.5
    
    score = max(1, min(5, round(score)))
    stars = '⭐' * score + '☆' * (5 - score)
    
    return {'score': score, 'stars': stars}

if __name__ == '__main__':
    # Test position sizing
    ps = PositionSizing(
        capital=100000,
        risk_per_trade=2,
        entry_price=43.8,
        stop_loss=41.5
    )
    result = calculate_position_size(ps, 'fixed_risk')
    print(json.dumps(result, ensure_ascii=False, indent=2))
