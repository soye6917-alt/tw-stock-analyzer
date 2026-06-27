#!/usr/bin/env python3
"""
professional_workflow.py — 多 Agent 並行分析工作流

靈感來自 Claude Code Workflow 概念，但實作成 Python 版本：
1. 平行分析層：技術分析 / 新聞情緒 / 國際市場 / 市場狀態 同時啟動
2. 聚合層：加權評分 → 最終建議
3. 可復用：定義好的 workflow 可反覆執行不同股票

對比原本 sequential 的 professional_hub.py：
  原本：抓資料→算指標→爬新聞→評分→倉位→總結（串行，耗時約 15-30 秒）
  現在：抓資料後 → 4 個 agent 平行跑 → 聚合（平行，耗時約 5-10 秒）
"""
import sys, os, json, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ====================================================================
# Agent 定義
# ====================================================================
@dataclass
class AgentSpec:
    """單一分析 agent 的規格定義"""
    name: str                              # agent 名稱
    description: str                       # 職責描述
    schema: Dict                           # 輸出 schema（期望回傳的鍵）
    function: Callable                     # 執行函式
    weight: float = 1.0                    # 最終決策權重
    timeout: int = 30                      # 超時秒數

    def to_dict(self):
        return {
            'name': self.name,
            'description': self.description,
            'schema_keys': list(self.schema.keys()),
            'weight': self.weight
        }


@dataclass
class AnalysisResult:
    """平行分析結果的封裝"""
    agent_name: str
    success: bool
    data: Dict = field(default_factory=dict)
    error: Optional[str] = None
    elapsed: float = 0.0
    
    def to_dict(self):
        return {
            'agent': self.agent_name,
            'success': self.success,
            'data_keys': list(self.data.keys()) if self.data else [],
            'error': self.error,
            'elapsed': round(self.elapsed, 2)
        }


# ====================================================================
# Agent 函式實現
# ====================================================================
def _technical_agent(df, stock_id, stock_name=""):
    """Agent 1: 技術面分析師 — 均線、RSI、成交量、趨勢"""
    from professional_trading import calculate_entry_score
    
    if df is None or len(df) < 20:
        return {
            'status': 'insufficient_data',
            'close': 0,
            'ma5': 0, 'ma20': 0, 'ma60': 0,
            'rsi': 50,
            'trend_20d': 0,
            'volume_vs_avg': 0,
            'score': 50,
            'grade': 'N/A'
        }
    
    latest = df.iloc[-1]
    
    # 均線多空判斷
    ma_alignment = ''
    if latest['MA5'] > latest['MA20'] > latest['MA60']:
        ma_alignment = '多頭排列'
    elif latest['MA5'] < latest['MA20'] < latest['MA60']:
        ma_alignment = '空頭排列'
    else:
        ma_alignment = '糾結/盤整'
    
    # 價格 vs 均線位置
    price_vs_ma = ''
    if latest['Close'] > latest['MA20']:
        price_vs_ma = '站上MA20'
    elif latest['Close'] < latest['MA20']:
        price_vs_ma = '跌破MA20'
    else:
        price_vs_ma = '貼近MA20'
    
    # RSI 狀態
    rsi = round(latest.get('RSI', 50), 1)
    if rsi > 70:
        rsi_state = '超買⚠️'
    elif rsi > 50:
        rsi_state = '偏多'
    elif rsi > 30:
        rsi_state = '偏空'
    else:
        rsi_state = '超賣💡'
    
    # 成交量分析
    vol_ma20 = df['Volume'].rolling(20).mean().iloc[-1]
    vol_today = latest['Volume']
    vol_ratio = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 0
    
    # 趨勢
    trend_20 = round((latest['Close'] / df['Close'].iloc[-20] - 1) * 100, 2) if len(df) >= 20 else 0
    
    # Entry score from trading module
    entry = calculate_entry_score(df)
    
    return {
        'status': 'ok',
        'close': round(latest['Close'], 2),
        'ma5': round(latest['MA5'], 2),
        'ma20': round(latest['MA20'], 2),
        'ma60': round(latest['MA60'], 2),
        'ma_alignment': ma_alignment,
        'price_vs_ma': price_vs_ma,
        'rsi': rsi,
        'rsi_state': rsi_state,
        'volume_ratio': vol_ratio,
        'volume_vs_avg': round(vol_today / df['Volume'].mean() * 100, 1) if df['Volume'].mean() > 0 else 0,
        'trend_20d': trend_20,
        'entry_score': entry.get('total_score', 50),
        'entry_grade': entry.get('grade', 'N/A')
    }


def _news_agent(stock_id, stock_name=""):
    """Agent 2: 新聞分析師 — 三源爬取 + 情緒評分"""
    from professional_news import get_news_sentiment, detect_hot_topics, fetch_cnyes_news
    
    news = get_news_sentiment(stock_id, stock_name)
    
    # 額外抓熱門題材
    raw_news = fetch_cnyes_news(stock_id, pages=1)
    topics = detect_hot_topics(raw_news)
    
    return {
        'status': 'ok',
        'total_news': news['total_news'],
        'sentiment_score': news['sentiment_score'],
        'sentiment_label': news['sentiment_label'],
        'positive_count': news.get('positive_news', 0),
        'negative_count': news.get('negative_news', 0),
        'top_positive': news.get('top_positive', [])[:2],
        'top_negative': news.get('top_negative', [])[:2],
        'topics': list(topics.keys())[:3],
        'summary': news['summary']
    }


def _regime_agent(df, stock_id=""):
    """Agent 3: 市場判讀師 — 整體盤勢分類"""
    from professional_regime import classify_market_regime
    
    if df is None or len(df) < 60:
        return {
            'status': 'insufficient_data',
            'regime_name': '未知',
            'confidence': 0,
            'advice': '資料不足，無法判斷'
        }
    
    regime = classify_market_regime(df)
    
    return {
        'status': 'ok',
        'regime_name': regime.get('regime_name', '未知'),
        'confidence': regime.get('confidence', 0),
        'advice': regime.get('advice', '觀望'),
        'indicators': regime.get('indicators', {}),
        'trend_direction': 'up' if df['Close'].iloc[-1] > df['MA20'].iloc[-1] else 'down'
    }


def _position_agent(quote, df, capital=100000):
    """Agent 4: 倉位管理師 — 凱利 + 固定風險計算"""
    from professional_trading import PositionSizing, calculate_position_size
    
    if not quote or df is None:
        return {
            'status': 'no_data',
            'shares': 0,
            'position_value': 0,
            'risk_pct': 0,
            'stop_loss': 0
        }
    
    price = float(quote.get('price', 0))
    if price <= 0:
        return {'status': 'invalid_price'}
    
    # 根據波動率動態停損
    atr = df['Close'].rolling(14).std().iloc[-1] if len(df) >= 14 else price * 0.03
    atr_pct = atr / price * 100
    
    # 波動大停損放寬，波動小停損收緊
    stop_pct = max(5, min(12, atr_pct * 1.5))
    
    ps = PositionSizing(
        capital=capital,
        risk_per_trade=2,
        entry_price=price,
        stop_loss=round(price * (1 - stop_pct/100), 2)
    )
    
    fixed_risk = calculate_position_size(ps, 'fixed_risk')
    kelly = calculate_position_size(ps, 'kelly')
    
    return {
        'status': 'ok',
        'price': price,
        'capital': capital,
        'stop_loss': round(price * (1 - stop_pct/100), 2),
        'stop_pct': round(stop_pct, 1),
        'fixed_risk': {
            'shares': fixed_risk.get('shares', 0),
            'position_value': fixed_risk.get('position_value', 0),
            'risk_amount': fixed_risk.get('risk_amount', 0),
            'risk_pct': capital
        },
        'kelly': {
            'fraction': kelly.get('shares', 0),
        },
        'recommended': 'fixed_risk'
    }


def _international_agent(stock_id=""):
    """Agent 5: 國際分析師 — 隔夜市場 + 台股ADR"""
    try:
        from professional_international import fetch_market_summary
        
        summary = fetch_market_summary()
        
        us = {}
        for sym, data in summary.get('us_market', {}).items():
            if 'price' in data:
                us[sym] = {
                    'name': data['name'],
                    'price': data['price'],
                    'change_pct': data.get('change_pct', 0)
                }
        
        adr = {}
        for sym, data in summary.get('adr', {}).items():
            if 'adr_price' in data:
                adr[sym] = {
                    'name': data['name'],
                    'price': data['adr_price'],
                    'change_pct': data.get('change_pct', 0)
                }
        
        fear_greed = summary.get('key_levels', {}).get('fear_greed', ('unknown', 0))
        semi = summary.get('key_levels', {}).get('semi_impact', ('unknown', 0))
        
        return {
            'status': 'ok',
            'us_indices': us,
            'adr': adr,
            'fear_greed': f'{fear_greed[0]} (VIX={fear_greed[1]})',
            'semi_impact': f'{semi[0]} ({semi[1]:+.2f}%)',
            'tsm_adr_change': adr.get('TSM', {}).get('change_pct', 0)
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)[:60]}


def _aggregator(results: Dict[str, AnalysisResult], stock_id: str, stock_name: str = "") -> Dict:
    """
    Aggregator — 匯總所有 agent 結果，加權評分，產出最終建議
    
    靈感：Workflow 的 aggregator 角色——不是簡單平均，
    而是根據各 agent 的 confidence 做動態加權。
    """
    # Extract data
    tech = results.get('technical', AnalysisResult('', False)).data
    news = results.get('news', AnalysisResult('', False)).data
    regime = results.get('regime', AnalysisResult('', False)).data
    position = results.get('position', AnalysisResult('', False)).data
    internatl = results.get('international', AnalysisResult('', False)).data
    
    # 計分系統（權重可調整）
    scores = {}
    reasons = {'bullish': [], 'bearish': [], 'neutral': []}
    
    # 技術面權重（40%）
    if tech.get('status') == 'ok':
        entry = tech.get('entry_score', 50)
        scores['technical'] = entry / 100 * 40  # max 40
        if entry > 60:
            reasons['bullish'].append(f"技術評分{entry}分")
        elif entry < 40:
            reasons['bearish'].append(f"技術評分僅{entry}分")
        
        if tech.get('ma_alignment') == '多頭排列':
            scores['technical'] += 10
            reasons['bullish'].append("均線多頭排列")
        elif tech.get('ma_alignment') == '空頭排列':
            scores['technical'] -= 5
        
        rsi = tech.get('rsi', 50)
        if 30 <= rsi <= 40:
            reasons['bullish'].append(f"RSI={rsi} 超賣區")
        elif rsi > 70:
            reasons['bearish'].append(f"RSI={rsi} 超買區")
    
    # 新聞權重（20%）
    if news.get('status') == 'ok':
        sent = news.get('sentiment_score', 0)
        scores['news'] = (sent + 1) / 2 * 20  # map [-1,1] to [0,20]
        if sent > 0.2:
            reasons['bullish'].append("新聞情緒正面")
        elif sent < -0.2:
            reasons['bearish'].append("新聞情緒負面")
    
    # 市場狀態權重（20%）
    if regime.get('status') == 'ok':
        conf = regime.get('confidence', 0) / 100  # normalize
        name = regime.get('regime_name', '')
        if '多頭' in name:
            scores['regime'] = 15 * conf
            reasons['bullish'].append(f"市場狀態: {name}")
        elif '空頭' in name:
            scores['regime'] = -5 * conf
            reasons['bearish'].append(f"市場狀態: {name}")
        elif '末升段' in name:
            scores['regime'] = 5 * conf
            reasons['bearish'].append("末升段 風險漸增")
        elif '整理' in name:
            scores['regime'] = 10 * conf
    
    # 國際市場權重（10%）
    if internatl.get('status') == 'ok':
        fg = internatl.get('fear_greed', '')
        semi = internatl.get('semi_impact', '')
        tsm = internatl.get('tsm_adr_change', 0)
        
        if '極度恐懼' in fg:
            reasons['neutral'].append("VIX極高，全球避險")
        elif '貪婪' in fg:
            scores['international'] = 5
            reasons['bullish'].append("VIX偏低，風險偏好高")
        
        if tsm > 1:
            scores['international'] += 5
            reasons['bullish'].append(f"台積電ADR +{tsm}%")
        elif tsm < -2:
            reasons['bearish'].append(f"台積電ADR {tsm}%")
    
    # 倉位結果（直接輸出，不計分）
    
    # 總分
    total = sum(scores.values())
    total = max(0, min(100, total))
    
    # 綜合評級
    if total >= 70:
        overall = '[多] 偏多'
    elif total >= 50:
        overall = '[平多] 中性偏多'
    elif total >= 30:
        overall = '[平空] 中性偏空'
    else:
        overall = '[空] 偏空'
    
    return {
        'stock_id': stock_id,
        'stock_name': stock_name,
        'timestamp': datetime.datetime.now().isoformat(),
        'overall_score': round(total, 1),
        'overall_grade': overall,
        'score_breakdown': {k: round(v, 1) for k, v in scores.items()},
        'reasons': reasons,
        'summary': {
            'bullish': len(reasons['bullish']),
            'bearish': len(reasons['bearish']),
            'decision': overall,
            'top_reason': (reasons['bullish'] + reasons['bearish'] + reasons['neutral'])[:3]
        }
    }


# ====================================================================
# 工作流引擎
# ====================================================================
class StockAnalysisWorkflow:
    """
    股票分析工作流引擎
    
    使用方法：
        workflow = StockAnalysisWorkflow()
        result = workflow.run('2618', capital=200000)
        print(result['final_grade'])
    """
    
    def __init__(self, parallel: bool = True):
        self.parallel = parallel
        self.agents = {}  # Registered agents
        self._register_default_agents()
    
    def _register_default_agents(self):
        """註冊預設 agent"""
        self.register_agent(
            'technical', AgentSpec(
                name='技術分析師',
                description='K線、均線、RSI、成交量趨勢判讀',
                schema={'close', 'ma5', 'ma20', 'ma60', 'rsi', 'volume_ratio', 'entry_score'},
                function=_technical_agent,
                weight=4.0
            )
        )
        self.register_agent(
            'news', AgentSpec(
                name='新聞情緒分析師',
                description='三源新聞爬取 + 情緒辭典評分 + 題材偵測',
                schema={'sentiment_score', 'sentiment_label', 'topics'},
                function=_news_agent,
                weight=2.0
            )
        )
        self.register_agent(
            'regime', AgentSpec(
                name='市場判讀師',
                description='6種市場狀態分類 + 信心指標',
                schema={'regime_name', 'confidence', 'advice'},
                function=_regime_agent,
                weight=2.0
            )
        )
        self.register_agent(
            'position', AgentSpec(
                name='倉位管理師',
                description='動態停損 + 凱利公式 + 固定風險',
                schema={'stop_loss', 'fixed_risk', 'kelly'},
                function=_position_agent,
                weight=0  # 不計分，僅資訊
            )
        )
        self.register_agent(
            'international', AgentSpec(
                name='國際分析師',
                description='11國指數 + 台股ADR + VIX恐慌指數',
                schema={'us_indices', 'adr', 'fear_greed', 'semi_impact'},
                function=_international_agent,
                weight=1.0
            )
        )
    
    def register_agent(self, name: str, spec: AgentSpec):
        """動態註冊新 agent"""
        self.agents[name] = spec
    
    def run(self, stock_id: str, stock_name: str = '', capital: float = 100000) -> Dict:
        """
        執行完整工作流
        
        流程：
        1. Fetch phase：取得原始資料（順序，共享依賴）
        2. Agent phase：多個 agent 平行或順序執行
        3. Aggregate phase：聚合評分
        """
        from data_fetcher import fetch_historical, fetch_realtime_quote
        
        # Phase 1: 資料準備（必要的共用資料）
        phase1_start = datetime.datetime.now()
        
        # 並行抓 quote 和歷史資料
        with ThreadPoolExecutor(max_workers=2) as pool:
            hist_future = pool.submit(fetch_historical, stock_id, 6)
            quote_future = pool.submit(fetch_realtime_quote, stock_id)
            
            df = hist_future.result()
            quote = quote_future.result()
        
        # 計算技術指標（在 df 上做一次就好）
        if df is not None and len(df) > 20:
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA60'] = df['Close'].rolling(60).mean()
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        phase1_time = (datetime.datetime.now() - phase1_start).total_seconds()
        
        # Phase 2: 平行執行 agents
        phase2_start = datetime.datetime.now()
        agent_results = {}
        
        def run_agent(name, spec):
            start = datetime.datetime.now()
            try:
                # 依 agent 類型傳入不同參數
                if name == 'technical':
                    data = spec.function(df, stock_id, stock_name)
                elif name == 'news':
                    data = spec.function(stock_id, stock_name)
                elif name == 'regime':
                    data = spec.function(df, stock_id)
                elif name == 'position':
                    data = spec.function(quote, df, capital)
                elif name == 'international':
                    data = spec.function(stock_id)
                else:
                    data = spec.function(stock_id)
                
                elapsed = (datetime.datetime.now() - start).total_seconds()
                return name, AnalysisResult(
                    agent_name=spec.name,
                    success=True,
                    data=data,
                    elapsed=elapsed
                )
            except Exception as e:
                elapsed = (datetime.datetime.now() - start).total_seconds()
                return name, AnalysisResult(
                    agent_name=spec.name,
                    success=False,
                    error=str(e)[:100],
                    elapsed=elapsed
                )
        
        if self.parallel and len(self.agents) > 1:
            with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
                futures = {pool.submit(run_agent, n, s): n for n, s in self.agents.items()}
                for future in as_completed(futures):
                    name, result = future.result()
                    agent_results[name] = result
        else:
            for name, spec in self.agents.items():
                _, result = run_agent(name, spec)
                agent_results[name] = result
        
        phase2_time = (datetime.datetime.now() - phase2_start).total_seconds()
        
        # Phase 3: 聚合
        phase3_start = datetime.datetime.now()
        aggregation = _aggregator(agent_results, stock_id, stock_name)
        phase3_time = (datetime.datetime.now() - phase3_start).total_seconds()
        
        return {
            'workflow': 'parallel_multi_agent' if self.parallel else 'sequential',
            'stock_id': stock_id,
            'stock_name': stock_name,
            'timestamp': aggregation['timestamp'],
            'final_score': aggregation['overall_score'],
            'final_grade': aggregation['overall_grade'],
            'score_breakdown': aggregation['score_breakdown'],
            'reasons': aggregation['reasons'],
            'summary': aggregation['summary'],
            'agents': {n: r.to_dict() for n, r in agent_results.items()},
            'agent_data': {n: r.data for n, r in agent_results.items()},
            'performance': {
                'phase1_data_fetch': round(phase1_time, 2),
                'phase2_agents': round(phase2_time, 2),
                'phase3_aggregation': round(phase3_time, 2),
                'total': round(phase1_time + phase2_time + phase3_time, 2),
                'parallel': self.parallel
            }
        }


# ====================================================================
# 快速入口：舊版相容
# ====================================================================
def quick_analysis(stock_id: str, stock_name: str = '', capital: float = 100000,
                   parallel: bool = True) -> Dict:
    """快速分析：一鍵執行完整工作流"""
    workflow = StockAnalysisWorkflow(parallel=parallel)
    return workflow.run(stock_id, stock_name, capital)


def compare_analysis(stock_ids: list, capital: float = 100000) -> Dict:
    """多股比較分析：對多檔股票同時跑 workflow"""
    results = {}
    workflow = StockAnalysisWorkflow(parallel=True)
    
    for sid in stock_ids:
        try:
            results[sid] = workflow.run(sid, capital=capital)
        except Exception as e:
            results[sid] = {'error': str(e)[:60]}
    
    # 排名
    ranked = sorted(
        [(sid, r.get('final_score', 0)) for sid, r in results.items() if 'final_score' in r],
        key=lambda x: -x[1]
    )
    
    return {
        'results': results,
        'ranking': [{'stock': s, 'score': sc} for s, sc in ranked]
    }


if __name__ == '__main__':
    import sys as _sys
    _sys.stdout.reconfigure(encoding='utf-8')
    import json
    
    # Test
    print('=== Single Stock Workflow Test (2618) ===')
    result = quick_analysis('2618', '', capital=200000, parallel=True)
    
    if 'final_score' in result:
        print(f'Final Grade: {result["final_grade"]} (Score: {result["final_score"]})')
        print(f'Score Detail: {result["score_breakdown"]}')
        print(f'Bullish: {result["reasons"]["bullish"]}')
        print(f'Bearish: {result["reasons"]["bearish"]}')
        print(f'Timing: {result["performance"]["total"]}s total | {result["performance"]["phase2_agents"]}s agents')
        
        print('')
        print('=== Multi Stock Compare ===')
        compare = compare_analysis(['2618', '2330', '2382'])
        print(f'Rank: {[r["stock"] + " (" + str(r["score"]) + ")" for r in compare["ranking"]]}')
    else:
        print('Error:', str(result)[:200])
