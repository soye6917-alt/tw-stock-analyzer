#!/usr/bin/env python3
"""
global_knowledge_graph.py — 全球市場知識圖譜 (Global Stock Nexus)

擴展原本的 stock_knowledge_graph.py，加入：
- 全球指數節點（S&P500/費半/日經/恆生…）
- 台股 ADR 節點（TSM/UMC/ASX）
- 商品/貨幣節點（原油/黃金/美元指數）
- 加密貨幣節點（BTC/ETH）
- 全球 ↔ 台股相關性邊
- ADR 溢價監控邊
- 隔夜風險傳導分析

Usage:
    from professional.global_knowledge_graph import GlobalKnowledgeGraph

    gkg = GlobalKnowledgeGraph()
    gkg.build_correlations()      # Fetch data and compute edges
    gkg.save("global_graph.json")
    gkg.global_impact("^GSPC")    # "如果S&P500跌，哪些台股受影響？"
    gkg.adr_premium("TSM")        # "台積電ADR溢價多少？"
"""
import sys, os, json, math
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Use the same urllib3 setup as other modules
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ============================================================
# Global Node Definitions
# ============================================================

GLOBAL_INDICES = {
    '^GSPC':  {'name': 'S&P 500', 'group': 'US_index', 'market': 'USA'},
    '^DJI':   {'name': '道瓊', 'group': 'US_index', 'market': 'USA'},
    '^IXIC':  {'name': '那斯達克', 'group': 'US_index', 'market': 'USA'},
    '^SOX':   {'name': '費城半導體', 'group': 'US_semiconductor', 'market': 'USA'},
    '^RUT':   {'name': '羅素2000', 'group': 'US_index', 'market': 'USA'},
    '^N225':  {'name': '日經225', 'group': 'asia_index', 'market': 'Japan'},
    '^HSI':   {'name': '恆生指數', 'group': 'asia_index', 'market': 'HongKong'},
    '000001.SS': {'name': '上證指數', 'group': 'asia_index', 'market': 'China'},
    '^FTSE':  {'name': '富時100', 'group': 'europe_index', 'market': 'UK'},
    '^GDAXI': {'name': '德國DAX', 'group': 'europe_index', 'market': 'Germany'},
}

ADRS = {
    'TSM':  {'name': '台積電ADR', 'group': 'adr', 'tw_stock': '2330'},
    'UMC':  {'name': '聯電ADR', 'group': 'adr', 'tw_stock': '2303'},
    'ASX':  {'name': '日月光ADR', 'group': 'adr', 'tw_stock': '3711'},
}

COMMODITIES = {
    'CL=F':  {'name': '原油WTI', 'group': 'commodity'},
    'GC=F':  {'name': '黃金', 'group': 'commodity'},
    'SI=F':  {'name': '白銀', 'group': 'commodity'},
    'DX-Y.NYB': {'name': '美元指數', 'group': 'currency'},
    'ZN=F':  {'name': '10年美債', 'group': 'bond'},
}

CRYPTO = {
    'BTC-USD': {'name': '比特幣', 'group': 'crypto'},
    'ETH-USD': {'name': '以太幣', 'group': 'crypto'},
}

# Taiwan stock IDs that are in our knowledge graph (for correlation)
TW_STOCK_IDS = [
    '2330', '2317', '2454', '2308', '2412',
    '2881', '2882', '2891', '2303', '2002',
    '1301', '1303', '1326', '1216', '3008',
    '3711', '3034', '4904', '3045', '5880',
    '2603', '2609', '2618', '2605',
    '2382', '3231',
]

# ============================================================
# Data Fetchers (lightweight, no yfinance dependency for base)
# ============================================================

def _fetch_yf_history(ticker: str, period: str = '3mo') -> pd.DataFrame:
    """Try yfinance, fallback to Yahoo CSV."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        if not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass

    # Fallback: Yahoo Finance CSV
    try:
        import requests as rq
        end = datetime.now()
        start = end - timedelta(days=90)
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={int(start.timestamp())}"
            f"&period2={int(end.timestamp())}"
            f"&interval=1d&events=history"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = rq.get(url, headers=headers, verify=False, timeout=15)
        if resp.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            if 'Close' in df.columns:
                df['close'] = df['Close']
            elif 'close' in df.columns:
                pass
            else:
                raise ValueError("No Close column")
            return df
    except Exception:
        pass
    return pd.DataFrame()


class GlobalKnowledgeGraph:
    """
    Global market knowledge graph.

    Nodes: global indices, ADRs, commodities, crypto, FX
    Edges:
      - index_index: correlation between two global indices
      - global_tw: correlation between a global index and a Taiwan stock
      - adr_premium: TSM ADR premium vs 2330 TW
      - sector_peer: indices in same group
      - commodity_corr: commodity-index correlation
    """

    def __init__(self):
        self.graph = {}  # {node_id: {neighbor: {weight, relation_type}}}
        self.metadata = {}  # {node_id: {name, group, ...}}
        self._built = False

    def add_node(self, node_id: str, **attrs):
        """Add a node to the graph."""
        if node_id not in self.graph:
            self.graph[node_id] = {}
        self.metadata[node_id] = attrs

    def add_edge(self, u: str, v: str, weight: float, relation: str):
        """Add undirected weighted edge."""
        if u not in self.graph:
            self.graph[u] = {}
        if v not in self.graph:
            self.graph[v] = {}
        self.graph[u][v] = {'weight': weight, 'relation': relation}
        self.graph[v][u] = {'weight': weight, 'relation': relation}

    # -------- Build Methods --------

    def _add_all_nodes(self):
        """Add all predefined global nodes."""
        for nid, attrs in GLOBAL_INDICES.items():
            self.add_node(nid, **attrs)
        for nid, attrs in ADRS.items():
            self.add_node(nid, **attrs)
        for nid, attrs in COMMODITIES.items():
            self.add_node(nid, **attrs)
        for nid, attrs in CRYPTO.items():
            self.add_node(nid, **attrs)

        # Also add TW stocks as weak nodes (for edges)
        from data_fetcher import get_stock_name
        for sid in TW_STOCK_IDS:
            name = get_stock_name(sid) or f'TW_{sid}'
            self.add_node(sid, name=name, group='tw_stock', market='Taiwan')

    def _add_index_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Compute correlation between global indices."""
        ids = [k for k in GLOBAL_INDICES.keys()]
        returns = {}
        for nid in ids:
            df = price_data.get(nid)
            if df is not None and not df.empty:
                col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col:
                    r = df[col].pct_change().dropna()
                    if len(r) >= 20:
                        returns[nid] = r

        common_idx = None
        for r in returns.values():
            if common_idx is None:
                common_idx = set(r.index)
            else:
                common_idx &= set(r.index)
        if common_idx is None:
            return

        keys = list(returns.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                ki, kj = keys[i], keys[j]
                ri = returns[ki].loc[list(common_idx & set(returns[ki].index) & set(returns[kj].index))]
                rj = returns[kj].loc[list(common_idx & set(returns[ki].index) & set(returns[kj].index))]
                if len(ri) >= 20 and len(rj) >= 20:
                    corr = ri.corr(rj)
                    if abs(corr) >= 0.3:
                        rel = 'index_corr_high' if abs(corr) >= 0.7 else 'index_corr_medium'
                        self.add_edge(ki, kj, round(corr, 4), rel)

        # Sector peers: same market/group
        for nid, attrs in GLOBAL_INDICES.items():
            for nid2, attrs2 in GLOBAL_INDICES.items():
                if nid < nid2 and attrs.get('group') == attrs2.get('group'):
                    if nid2 not in self.graph.get(nid, {}):
                        grp = attrs.get('group', '')
                        if grp == 'US_index':
                            w = 0.9
                        elif grp == 'asia_index':
                            w = 0.8
                        else:
                            w = 0.7
                        self.add_edge(nid, nid2, w, 'sector_peer')

    def _add_global_tw_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Compute correlation between each global index and each TW stock.

        TW stock data from TWSE has '\u65e5\u671f' (date) column with RangeIndex.
        Global index data from yfinance has DatetimeIndex.
        Both must be aligned by date for correlation.
        """
        from data_fetcher import fetch_historical

        def _tw_returns(sid):
            """Get daily returns with DatetimeIndex from TW stock data."""
            try:
                df = fetch_historical(sid, months=3)
                if df is None or df.empty or 'Close' not in df.columns:
                    return None
                # Convert date column to DatetimeIndex
                if '\u65e5\u671f' in df.columns:
                    df['_date'] = pd.to_datetime(df['\u65e5\u671f'])
                    df = df.set_index('_date')
                elif df.index.name != 'Date' and not isinstance(df.index, pd.DatetimeIndex):
                    return None
                r = df['Close'].pct_change().dropna()
                if len(r) >= 15:
                    return r
            except Exception:
                pass
            return None

        def _global_returns(nid, price_data):
            """Get daily returns from global index data (DatetimeIndex from yfinance)."""
            df = price_data.get(nid)
            if df is None or df.empty:
                return None
            col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
            if col is None:
                return None
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    return None
            r = df[col].pct_change().dropna()
            if len(r) >= 15:
                return r
            return None

        # Build returns dicts
        tw_returns = {}
        for sid in TW_STOCK_IDS:
            r = _tw_returns(sid)
            if r is not None:
                tw_returns[sid] = r

        global_returns = {}
        for nid in GLOBAL_INDICES:
            r = _global_returns(nid, price_data)
            if r is not None:
                global_returns[nid] = r

        if not tw_returns or not global_returns:
            return

        # Cross-correlation: for each global index vs each TW stock
        edge_count = 0
        for gid, gr in global_returns.items():
            # Strip tz from global returns (yfinance has tz-aware, TW data is tz-naive)
            if hasattr(gr.index, 'tz') and gr.index.tz is not None:
                gr_index = gr.index.tz_localize(None)
            else:
                gr_index = gr.index
            gr_vals = gr.values  # Keep values, just use tz-naive index for matching
            gr_tznaive = pd.Series(gr_vals, index=gr_index)

            for sid, sr in tw_returns.items():
                common_dates = set(gr_tznaive.index.strftime('%Y-%m-%d')) & \
                               set(sr.index.strftime('%Y-%m-%d'))
                if len(common_dates) >= 10:
                    g_vals = gr_tznaive[gr_tznaive.index.strftime('%Y-%m-%d').isin(common_dates)]
                    s_vals = sr[sr.index.strftime('%Y-%m-%d').isin(common_dates)]
                    if len(g_vals) >= 10 and len(s_vals) >= 10:
                        corr = g_vals.corr(s_vals)
                        if abs(corr) >= 0.25:
                            rel = 'global_tw_high' if abs(corr) >= 0.5 else 'global_tw_medium'
                            self.add_edge(gid, sid, round(corr, 4), rel)
                            edge_count += 1

        if edge_count == 0:
            print("  Warning: no global-TW edges found (threshold=0.25)")
        print(f"  -> {edge_count} global-TW edges added")

    def _add_adr_premium(self, price_data: Dict[str, pd.DataFrame]):
        """Compute ADR premium: (ADR_price - TW_price) / TW_price using last close.

        ADR prices are in USD, TW prices in TWD. Premium = (ADR_TWD - TW) / TW.
        TSM ADR is 1:5 ratio vs 2330 (5 TSM ADR = 1 2330 share).
        """
        from data_fetcher import fetch_historical

        # ADR ratios: how many ADR shares = 1 TW share
        ADR_RATIOS = {'TSM': 5, 'UMC': 5, 'ASX': 5}
        # Approximate TWD/USD rate (or fetch if possible)
        twd_per_usd = 32.8  # rough estimate; update as needed

        for adr_id, adr_info in ADRS.items():
            tw_id = adr_info['tw_stock']
            if tw_id not in TW_STOCK_IDS:
                continue

            adr_df = price_data.get(adr_id)
            if adr_df is None or adr_df.empty:
                continue

            try:
                tw_df = fetch_historical(tw_id, months=2)
                if tw_df is None or tw_df.empty:
                    continue
            except Exception:
                continue

            col = 'close' if 'close' in adr_df.columns else ('Close' if 'Close' in adr_df.columns else None)
            if col is None:
                continue

            adr_price_usd = adr_df[col].iloc[-1]
            tw_price_twd = tw_df['Close'].iloc[-1]

            ratio = ADR_RATIOS.get(adr_id, 5)
            # ADR converted to TWD per TW share: adr_price_usd * ratio * twd_per_usd
            adr_converted_twd = adr_price_usd * twd_per_usd / ratio

            if tw_price_twd > 0 and adr_price_usd > 0:
                premium = (adr_converted_twd - tw_price_twd) / tw_price_twd
                premium = round(premium * 100, 3)  # as percentage (per-mille precision)
                edge_weight = min(abs(premium) / 20, 1.0)
                rel = 'adr_premium_high' if abs(premium) > 5 else 'adr_premium_low'
                self.add_edge(adr_id, tw_id, round(edge_weight, 4), rel)
                self.metadata[adr_id]['adr_premium_pct'] = premium
                self.metadata[adr_id]['adr_usd'] = round(adr_price_usd, 2)
                self.metadata[adr_id]['tw_twd'] = round(tw_price_twd, 2)
                self.metadata[adr_id]['adr_converted_twd'] = round(adr_converted_twd, 2)

    def _add_commodity_correlations(self, price_data: Dict[str, pd.DataFrame]):
        """Add commodity ↔ global index correlations."""
        commodity_ids = list(COMMODITIES.keys()) + list(CRYPTO.keys())
        index_ids = list(GLOBAL_INDICES.keys())

        comm_returns = {}
        for nid in commodity_ids:
            df = price_data.get(nid)
            if df is not None and not df.empty:
                col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col:
                    r = df[col].pct_change().dropna()
                    if len(r) >= 20:
                        comm_returns[nid] = r

        idx_returns = {}
        for nid in index_ids:
            df = price_data.get(nid)
            if df is not None and not df.empty:
                col = 'close' if 'close' in df.columns else ('Close' if 'Close' in df.columns else None)
                if col:
                    r = df[col].pct_change().dropna()
                    if len(r) >= 20:
                        idx_returns[nid] = r

        # Normalize tz for all returns
        def _normalize(s):
            if hasattr(s.index, 'tz') and s.index.tz is not None:
                return pd.Series(s.values, index=s.index.tz_localize(None))
            return s

        for cid, cr in comm_returns.items():
            cr = _normalize(cr)
            for iid, ir in idx_returns.items():
                ir = _normalize(ir)
                common = set(cr.index.strftime('%Y-%m-%d')) & set(ir.index.strftime('%Y-%m-%d'))
                if len(common) >= 10:
                    cv = cr[cr.index.strftime('%Y-%m-%d').isin(common)]
                    iv = ir[ir.index.strftime('%Y-%m-%d').isin(common)]
                    if len(cv) >= 10 and len(iv) >= 10:
                        corr = cv.corr(iv)
                        if abs(corr) >= 0.25:
                            self.add_edge(cid, iid, round(corr, 4), 'commodity_index')

    def build(self, fetch_prices: bool = True) -> Dict:
        """
        Build the global knowledge graph.

        Returns summary dict with nodes/edges counts.
        """
        self._add_all_nodes()

        if fetch_prices:
            import time as _time
            price_data = {}
            all_tickers = list(GLOBAL_INDICES.keys()) + list(ADRS.keys()) + \
                          list(COMMODITIES.keys()) + list(CRYPTO.keys())

            for i, ticker in enumerate(all_tickers):
                try:
                    df = _fetch_yf_history(ticker)
                    if not df.empty:
                        price_data[ticker] = df
                except Exception:
                    continue
                if (i + 1) % 5 == 0:
                    _time.sleep(0.5)  # Rate limiting

            self._add_index_correlations(price_data)
            self._add_global_tw_correlations(price_data)
            self._add_adr_premium(price_data)
            self._add_commodity_correlations(price_data)

        self._built = True
        return self.stats()

    def stats(self) -> Dict:
        """Return graph statistics."""
        nodes = len(self.graph)
        edges = sum(len(neighbors) for neighbors in self.graph.values()) // 2
        by_group = {}
        for nid, meta in self.metadata.items():
            g = meta.get('group', 'unknown')
            by_group[g] = by_group.get(g, 0) + 1
        by_relation = {}
        for u in self.graph:
            for v, data in self.graph[u].items():
                if u < v:
                    r = data.get('relation', 'unknown')
                    by_relation[r] = by_relation.get(r, 0) + 1
        return {
            'nodes': nodes,
            'edges': edges,
            'node_groups': by_group,
            'edge_relations': by_relation,
            'is_built': self._built,
        }

    # -------- Query Methods --------

    def global_impact(self, index_id: str, top_n: int = 10) -> List[Dict]:
        """
        "如果這個全球指數跌，哪些台股受影響？"
        Returns sorted list of (stock_id, correlation, relation_type).
        """
        if index_id not in self.graph:
            return []
        results = []
        for neighbor, data in self.graph[index_id].items():
            if data.get('relation', '').startswith('global_tw'):
                name = self.metadata.get(neighbor, {}).get('name', neighbor)
                results.append({
                    'id': neighbor,
                    'name': name,
                    'correlation': data['weight'],
                    'relation': data['relation'],
                })
        results.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return results[:top_n]

    def adr_premium(self, adr_id: str) -> Optional[Dict]:
        """Get ADR premium info for a given ADR ticker."""
        if adr_id not in self.graph:
            return None
        tw_stock = self.metadata.get(adr_id, {}).get('tw_stock')
        premium = self.metadata.get(adr_id, {}).get('adr_premium_pct', 'N/A')

        connected_tw = [n for n in self.graph.get(adr_id, {}).keys()
                        if n in TW_STOCK_IDS]
        return {
            'adr': adr_id,
            'name': self.metadata.get(adr_id, {}).get('name', adr_id),
            'tw_stock': tw_stock,
            'tw_name': self.metadata.get(tw_stock, {}).get('name', tw_stock) if tw_stock else None,
            'adr_premium_pct': premium,
            'connected_tw': connected_tw,
        }

    def overnight_risk_assessment(self) -> Dict:
        """
        Assess overnight risk from US market to specific TW stocks.
        Returns dict with high/medium/low risk stocks.
        """
        us_indices = [k for k, v in GLOBAL_INDICES.items()
                      if v.get('market') == 'USA']
        assessment = {'high_risk': [], 'medium_risk': [], 'low_risk': []}

        for sid in TW_STOCK_IDS:
            corrs = []
            for uid in us_indices:
                if uid in self.graph and sid in self.graph.get(uid, {}):
                    corrs.append(self.graph[uid][sid]['weight'])
            if corrs:
                avg_corr = sum(corrs) / len(corrs)
                name = self.metadata.get(sid, {}).get('name', sid)
                entry = {'stock_id': sid, 'name': name, 'avg_correlation': round(avg_corr, 4)}
                if avg_corr >= 0.4:
                    assessment['high_risk'].append(entry)
                elif avg_corr >= 0.2:
                    assessment['medium_risk'].append(entry)
                else:
                    assessment['low_risk'].append(entry)

        for k in assessment:
            assessment[k].sort(key=lambda x: abs(x['avg_correlation']), reverse=True)
        return assessment

    def sector_exposure(self, sector: str = 'US_semiconductor') -> List[Dict]:
        """
        "費半跌，哪些台股最受傷？"
        Returns stocks sorted by correlation to a sector (average of indices in that group).
        """
        indices = [k for k, v in GLOBAL_INDICES.items()
                   if v.get('group') == sector]
        if not indices:
            return []

        scores = {}
        for sid in TW_STOCK_IDS:
            corrs = []
            for idx in indices:
                if idx in self.graph and sid in self.graph.get(idx, {}):
                    corrs.append(self.graph[idx][sid]['weight'])
            if corrs:
                avg = sum(corrs) / len(corrs)
                name = self.metadata.get(sid, {}).get('name', sid)
                scores[sid] = {'stock_id': sid, 'name': name, 'avg_correlation': round(avg, 4)}

        results = sorted(scores.values(), key=lambda x: abs(x['avg_correlation']), reverse=True)
        return results[:15]

    def correlation_matrix(self, nodes: List[str] = None) -> pd.DataFrame:
        """Build correlation matrix for given nodes (or all global indices)."""
        if nodes is None:
            nodes = list(GLOBAL_INDICES.keys())[:8]  # Limit to avoid huge matrix
        valid = [n for n in nodes if n in self.graph]
        matrix = pd.DataFrame(index=valid, columns=valid, dtype=float)
        for u in valid:
            for v in valid:
                if u == v:
                    matrix.loc[u, v] = 1.0
                elif v in self.graph.get(u, {}):
                    matrix.loc[u, v] = self.graph[u][v]['weight']
                else:
                    matrix.loc[u, v] = 0.0
        return matrix

    def neighbors(self, node_id: str) -> List[Dict]:
        """Get all neighbors of a node with edge info."""
        if node_id not in self.graph:
            return []
        results = []
        for nid, data in self.graph[node_id].items():
            meta = self.metadata.get(nid, {})
            results.append({
                'id': nid,
                'name': meta.get('name', nid),
                'group': meta.get('group', 'unknown'),
                'weight': data['weight'],
                'relation': data['relation'],
            })
        results.sort(key=lambda x: abs(x['weight']), reverse=True)
        return results

    def suggest_hedges(self, tw_stock_id: str, top_n: int = 5) -> List[Dict]:
        """
        給定台股持倉，建議避險工具（負相關的全球資產）。
        e.g., 「持有台積電，建議用什麼避險？」
        """
        if tw_stock_id not in self.graph:
            return []
        hedges = []
        for nid, data in self.graph[tw_stock_id].items():
            if nid not in self.graph.get('^VIX', {}):  # Filter to global assets
                continue
            weight = data['weight']
            if weight < 0:  # Negative correlation = hedge
                name = self.metadata.get(nid, {}).get('name', nid)
                hedges.append({
                    'asset': nid,
                    'name': name,
                    'correlation': weight,
                    'hedge_score': round(abs(weight), 4),
                })
        # Also check inverse ETFs / VIX
        vix_data = self.graph.get('^VIX', {})
        if tw_stock_id in vix_data:
            vix_corr = vix_data[tw_stock_id]['weight']
            if vix_corr < 0:
                hedges.append({
                    'asset': '^VIX',
                    'name': 'VIX恐慌指數',
                    'correlation': vix_corr,
                    'hedge_score': round(abs(vix_corr), 4),
                })
        hedges.sort(key=lambda x: x['hedge_score'], reverse=True)
        return hedges[:top_n]

    # -------- Persistence --------

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON."""
        edges_list = []
        seen = set()
        for u in self.graph:
            for v, data in self.graph[u].items():
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    edges_list.append({
                        'source': u,
                        'target': v,
                        'weight': data['weight'],
                        'relation': data['relation'],
                    })
        return {
            'metadata': self.metadata,
            'edges': edges_list,
            'built': self._built,
            'built_at': datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GlobalKnowledgeGraph':
        """Deserialize from dict."""
        gkg = cls()
        gkg.metadata = data.get('metadata', {})
        gkg._built = data.get('built', False)
        for edge in data.get('edges', []):
            u, v = edge['source'], edge['target']
            w = edge['weight']
            r = edge['relation']
            if u not in gkg.graph:
                gkg.graph[u] = {}
            if v not in gkg.graph:
                gkg.graph[v] = {}
            gkg.graph[u][v] = {'weight': w, 'relation': r}
            gkg.graph[v][u] = {'weight': w, 'relation': r}
        return gkg

    def save(self, path: str):
        """Save to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'GlobalKnowledgeGraph':
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# ============================================================
# CLI / Interactive
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Global Knowledge Graph Builder")
    print("=" * 60)

    gkg = GlobalKnowledgeGraph()
    print("\n[1/5] Adding nodes...")
    gkg._add_all_nodes()
    print(f"  → {len(gkg.graph)} nodes added")

    print("\n[2/5] Fetching global price data (yfinance)...")
    import time as _t
    all_tickers = list(GLOBAL_INDICES.keys()) + list(ADRS.keys()) + \
                  list(COMMODITIES.keys()) + list(CRYPTO.keys())
    price_data = {}
    for i, ticker in enumerate(all_tickers):
        try:
            df = _fetch_yf_history(ticker)
            if not df.empty:
                price_data[ticker] = df
                print(f"  OK {ticker} ({len(df)} days)")
            else:
                print(f"  XX {ticker} (no data)")
        except Exception as e:
            print(f"  XX {ticker} ({e})")
        if (i + 1) % 5 == 0:
            _t.sleep(0.5)

    print(f"\n[3/5] Computing correlations...")
    gkg._add_index_correlations(price_data)
    print(f"  → index-index correlations done")

    gkg._add_global_tw_correlations(price_data)
    print(f"  → global-TW correlations done")

    gkg._add_adr_premium(price_data)
    print(f"  → ADR premium done")

    gkg._add_commodity_correlations(price_data)
    print(f"  → commodity-index correlations done")

    gkg._built = True

    stats = gkg.stats()
    print(f"\n[4/5] Graph Stats:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  By group: {json.dumps(stats['node_groups'], ensure_ascii=False)}")
    print(f"  By relation: {json.dumps(stats['edge_relations'], ensure_ascii=False)}")

    # Save
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'global_graph.json')
    gkg.save(graph_path)
    print(f"\n[5/5] Saved: {graph_path}")

    # Demo queries
    print("\n" + "=" * 60)
    print("  Demo: Overnight Risk Assessment")
    print("=" * 60)
    risk = gkg.overnight_risk_assessment()
    for k, v in risk.items():
        print(f"\n  [{k.upper()}]")
        for item in v[:5]:
            print(f"    {item['stock_id']} {item['name']}: {item['avg_correlation']:.4f}")

    print("\n" + "=" * 60)
    print("  Demo: 費半對台股影響")
    print("=" * 60)
    sox = gkg.sector_exposure('US_semiconductor')
    for item in sox[:8]:
        print(f"  {item['stock_id']} {item['name']}: {item['avg_correlation']:.4f}")

    print("\n  ✅ Done!")
