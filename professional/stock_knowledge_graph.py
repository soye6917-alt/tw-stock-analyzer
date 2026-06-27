#!/usr/bin/env python3
"""
stock_knowledge_graph.py — Stock Knowledge Graph

靈感：GitNexus 將程式碼庫索引為知識圖譜，
讓 AI 工具「真正理解程式碼結構」。

這裡反過來：將台股生態系建立知識圖譜，
讓分析工具「真正理解股票之間的關係」。

連連看：
  ┌─ 產業鏈 ──→ 同一產業、上下游
  ├─ 共整合 ──→ 配對交易夥伴
  ├─ 技術共振 ─→ 相關係數高的股票
  ├─ 法人鎖股 ─→ 被同一法人大量持有的
  └─ 新聞連動 ─→ 常被一起報導的

提供查詢：
  - 「如果台積電跌，誰會被波及？」（影響範圍分析）
  - 「這個族群裡哪一檔最強？」（cluster 排序）
  - 「2618 長榮航跟誰最像？」（相似度搜尋）
"""
import sys, os, json, datetime, math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# ====================================================================
# 產業分類 (TWSE 產業別 + 自訂)
# ====================================================================
SECTOR_MAP = {
    # 半導體
    '2330': '半導體-晶圓代工', '2303': '半導體-晶圓代工',
    '2454': '半導體-IC設計', '3034': '半導體-IC設計', '3661': '半導體-IC設計',
    '3443': '半導體-IC設計', '6735': '半導體-IC設計',
    '3711': '半導體-封測', '2329': '半導體-封測',
    # 航運
    '2603': '航運-貨櫃', '2609': '航運-貨櫃', '2615': '航運-貨櫃',
    '2618': '航運-航空', '2610': '航運-航空',
    '2637': '航運-散裝', '2606': '航運-散裝',
    '5608': '航運-物流',
    # 金融
    '2881': '金融-金控', '2882': '金融-金控', '2886': '金融-金控',
    '2891': '金融-金控', '2884': '金融-金控', '2885': '金融-金控',
    '2887': '金融-金控', '2890': '金融-金控', '2888': '金融-金控',
    '2883': '金融-金控',
    # 電子代工
    '2317': '電子-代工', '2354': '電子-代工', '2382': '電子-代工',
    '3231': '電子-代工',
    # 面板
    '2409': '電子-面板', '3481': '電子-面板', '6116': '電子-面板',
    # 電信
    '2412': '電信', '3045': '電信', '4904': '電信',
    # 傳產
    '1301': '傳產-塑膠', '1303': '傳產-塑膠', '1326': '傳產-塑膠',
    '2002': '傳產-鋼鐵', '2015': '傳產-鋼鐵',
    '1101': '傳產-水泥', '1102': '傳產-水泥',
    '1216': '傳產-食品', '1210': '傳產-食品',
    '2912': '傳產-百貨',
    # 電子通路
    '2347': '電子-通路', '3036': '電子-通路', '3702': '電子-通路',
    # 網通
    '3380': '電子-網通', '2345': '電子-網通', '3596': '電子-網通',
    # 被動元件
    '2327': '電子-被動', '2492': '電子-被動',
    # 光學
    '3008': '電子-光學', '3406': '電子-光學',
    # 電源
    '2308': '電子-電源', '3017': '電子-電源', '2301': '電子-電源',
    # 生技
    '4126': '生技', '4743': '生技', '4164': '生技',
    # 汽車
    '2207': '汽車', '2204': '汽車',
    # 其他
    '2498': '其他-手機', '2357': '其他-手機',
    '4938': '其他-貿易',
    '6515': '其他-機電',
    '6239': '其他-IC通路',
}


def _get_sector(stock_id: str) -> str:
    """Get sector for a stock, with fallback to general parent."""
    if stock_id in SECTOR_MAP:
        return SECTOR_MAP[stock_id]
    # Return general category based on prefix
    prefix = stock_id[0]
    if prefix == '2':
        return '電子-其他'
    elif prefix == '3':
        return '電子-其他'
    return '其他'


# ====================================================================
# Knowledge Graph Builder
# ====================================================================
class StockKnowledgeGraph:
    """
    Stock Knowledge Graph — 股票關係知識圖譜
    
    Nodes: each stock ticker
    Edges with weights:
        sector_peer       1.0   — 同一產業
        co_integration    0.8   — 共整合配對
        correlation_high  0.6   — 高相關性
        news_co_mention   0.3   — 新聞共同出現
        same_institution  0.5   — 被同一法人持有
    """
    
    def __init__(self):
        self.graph = nx.Graph() if HAS_NX else None
        self.nodes = set()
        self.edges = {}  # (sid1, sid2) -> {'type': ..., 'weight': ...}
        self.metadata = {
            'built_at': None,
            'stock_count': 0,
            'edge_count': 0,
            'clusters': 0
        }
    
    def add_stock(self, stock_id: str, name: str = ''):
        """Add a stock node."""
        sector = _get_sector(stock_id)
        attrs = {'name': name, 'sector': sector, 'type': 'stock'}
        if HAS_NX:
            self.graph.add_node(stock_id, **attrs)
        self.nodes.add(stock_id)
    
    def add_edge(self, sid1: str, sid2: str, etype: str, weight: float, 
                 metadata: dict = None):
        """Add an edge between two stocks."""
        key = tuple(sorted([sid1, sid2]))
        if key not in self.edges or self.edges[key]['weight'] < weight:
            self.edges[key] = {
                'type': etype,
                'weight': weight,
                'metadata': metadata or {}
            }
        if HAS_NX:
            existing = self.graph.get_edge_data(sid1, sid2)
            if existing:
                existing['weight'] = max(existing['weight'], weight)
                existing['types'] = existing.get('types', []) + [etype]
            else:
                self.graph.add_edge(sid1, sid2, weight=weight, types=[etype])
    
    def build_sector_edges(self):
        """Add sector-based edges: stocks in same sector are connected."""
        sector_groups = defaultdict(list)
        for sid in self.nodes:
            sector = _get_sector(sid)
            sector_groups[sector].append(sid)
        
        for sector, members in sector_groups.items():
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    self.add_edge(members[i], members[j], 'sector_peer', 1.0)
    
    def build_correlation_edges(self, df_dict: dict = None):
        """
        Build correlation edges from historical data.
        df_dict: {stock_id: DataFrame with 'Close' column}
        """
        if not df_dict:
            return
            
        import pandas as pd
        # Build price matrix
        prices = {}
        for sid, df in df_dict.items():
            if sid in self.nodes and df is not None and len(df) > 20:
                prices[sid] = df['Close']
        
        if len(prices) < 2:
            return
            
        price_df = pd.DataFrame(prices)
        corr = price_df.corr()
        
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                c = corr.iloc[i, j]
                if abs(c) > 0.7:
                    sid1, sid2 = corr.columns[i], corr.columns[j]
                    self.add_edge(sid1, sid2, 'correlation_high', 
                                 weight=round(abs(c), 2),
                                 metadata={'correlation': round(c, 3)})
    
    def build_from_stat_arb(self, pairs: list = None):
        """Add co-integration edges from stat_arb module."""
        if not pairs:
            return
        
        for pair in pairs:
            sid1, sid2 = pair.get('stock1'), pair.get('stock2')
            score = pair.get('coint_score', 0)  # 0-100
            if score > 50 and sid1 and sid2:
                weight = max(0.5, score / 100)
                self.add_edge(sid1, sid2, 'co_integration', weight,
                             metadata={'score': score})
    
    def build_news_edges(self, news_data: dict = None):
        """Build edges from news co-mentions."""
        if not news_data:
            return
        
        # news_data: {stock_id: [news_items]}
        # Find pairs that appear in same news
        from collections import defaultdict
        mention_pairs = defaultdict(int)
        
        for sid, items in news_data.items():
            if sid not in self.nodes:
                continue
            # Check which other stocks are mentioned in news about this stock
            for item in items:
                title = (item.get('title', '') or '') + ' ' + (item.get('summary', '') or '')
                for other in self.nodes:
                    if other != sid and other in title:
                        mention_pairs[tuple(sorted([sid, other]))] += 1
        
        for pair, count in mention_pairs.items():
            if count >= 2:  # At least 2 co-mentions
                self.add_edge(pair[0], pair[1], 'news_co_mention',
                             weight=min(0.3 + count * 0.1, 0.8),
                             metadata={'co_mention_count': count})
    
    def build(self, additional_stocks=None, correlation_data=None, 
              stat_arb_pairs=None, news_data=None):
        """Build the complete graph."""
        # Add stocks
        if additional_stocks:
            for sid, name in additional_stocks:
                self.add_stock(sid, name)
        
        # Build edges
        self.build_sector_edges()
        self.build_correlation_edges(correlation_data)
        self.build_from_stat_arb(stat_arb_pairs)
        self.build_news_edges(news_data)
        
        # Update metadata
        self.metadata['built_at'] = datetime.datetime.now().isoformat()
        self.metadata['stock_count'] = len(self.nodes)
        self.metadata['edge_count'] = len(self.edges)
        
        if HAS_NX:
            import networkx.algorithms.community as nx_comm
            try:
                # Detect communities
                communities = nx_comm.greedy_modularity_communities(self.graph)
                self.metadata['clusters'] = len(communities)
                self.metadata['communities'] = [
                    list(c) for c in communities
                ]
            except:
                self.metadata['clusters'] = 0
                self.metadata['communities'] = []
        
        return self
    
    # ================================================================
    # Query Methods — Impact Analysis, Similarity Search
    # ================================================================
    
    def impact_analysis(self, stock_id: str, min_weight: float = 0.3) -> Dict:
        """
        Impact Analysis (靈感: GitNexus impact tool)
        
        如果 stock_id 大跌，誰會被波及？
        回傳波及列表 + 波及強度 + 波及理由
        """
        if not HAS_NX:
            # Fallback: sector-based
            results = []
            sector = _get_sector(stock_id)
            for other in self.nodes:
                if other == stock_id:
                    continue
                other_sector = _get_sector(other)
                if other_sector == sector:
                    # Check if there's a direct edge
                    key = tuple(sorted([stock_id, other]))
                    edge = self.edges.get(key)
                    edge_type = edge['type'] if edge else 'sector_only'
                    weight = edge['weight'] if edge else 0.5
                    if weight >= min_weight:
                        results.append({
                            'stock': other,
                            'sector': other_sector,
                            'impact_weight': weight,
                            'reason': f'同產業({sector})' if not edge else f'{edge_type}({weight})'
                        })
            return {
                'target': stock_id,
                'sector': sector,
                'total_potential_impact': len(results),
                'high_impact': [r for r in results if r['impact_weight'] >= 0.7],
                'medium_impact': [r for r in results if 0.4 <= r['impact_weight'] < 0.7],
                'affected_stocks': sorted(results, key=lambda x: -x['impact_weight'])
            }
        
        # Use NetworkX
        if stock_id not in self.graph:
            return {'target': stock_id, 'error': 'Stock not in graph'}
        
        neighbors = list(self.graph.neighbors(stock_id))
        affected = []
        for nb in neighbors:
            edge_data = self.graph.get_edge_data(stock_id, nb)
            w = edge_data.get('weight', 0.5)
            if w >= min_weight:
                affected.append({
                    'stock': nb,
                    'name': self.graph.nodes[nb].get('name', ''),
                    'sector': self.graph.nodes[nb].get('sector', ''),
                    'impact_weight': w,
                    'relation': edge_data.get('types', ['unknown']),
                })
        
        # Sort by weight
        affected = sorted(affected, key=lambda x: -x['impact_weight'])
        
        return {
            'target': stock_id,
            'target_sector': self.graph.nodes[stock_id].get('sector', ''),
            'total_potential_impact': len(affected),
            'high_impact': [a for a in affected if a['impact_weight'] >= 0.7],
            'medium_impact': [a for a in affected if 0.4 <= a['impact_weight'] < 0.7],
            'affected_stocks': affected
        }
    
    def similar_stocks(self, stock_id: str, top_n: int = 5) -> List:
        """
        Find most similar stocks to the target.
        Uses weighted combination of sector, correlation, and co-integration.
        """
        impact = self.impact_analysis(stock_id)
        results = impact.get('affected_stocks', [])[:top_n]
        
        # Add sector peers if not enough results
        if len(results) < top_n:
            sector = _get_sector(stock_id)
            for other in sorted(self.nodes):
                if other == stock_id:
                    continue
                if other not in [r['stock'] for r in results]:
                    if _get_sector(other) == sector:
                        results.append({
                            'stock': other,
                            'sector': sector,
                            'impact_weight': 0.5,
                            'reason': '同產業'
                        })
                if len(results) >= top_n:
                    break
        
        return results[:top_n]
    
    def cluster_summary(self) -> Dict:
        """Returns cluster/community summary of the graph."""
        if not HAS_NX or not self.metadata.get('communities'):
            return {'clusters': 0, 'note': 'NetworkX required for community detection'}
        
        communities = self.metadata['communities']
        summary = {}
        for i, community in enumerate(communities):
            sectors_in_community = Counter()
            names = []
            for sid in community:
                sec = _get_sector(sid)
                sectors_in_community[sec] += 1
                if 'name' in self.graph.nodes[sid]:
                    names.append(self.graph.nodes[sid]['name'][:4])
            
            top_sector = sectors_in_community.most_common(1)[0][0] if sectors_in_community else 'unknown'
            
            summary[f'Cluster_{i+1}'] = {
                'size': len(community),
                'dominant_sector': top_sector,
                'sectors': dict(sectors_in_community.most_common(3)),
                'members': community[:10]  # Top 10
            }
        
        return {'cluster_count': len(communities), 'clusters': summary}
    
    def to_dict(self) -> Dict:
        """Export graph to dict (JSON-serializable)."""
        nodes_dict = {}
        for sid in sorted(self.nodes):
            sector = _get_sector(sid)
            nodes_dict[sid] = {'sector': sector}
        
        edges_list = []
        for (s1, s2), data in sorted(self.edges.items(), key=lambda x: -x[1]['weight']):
            edges_list.append({
                'source': s1, 'target': s2,
                'type': data['type'],
                'weight': data['weight']
            })
        
        return {
            'metadata': self.metadata,
            'nodes': nodes_dict,
            'edges': edges_list,
            'query': {
                'impact_analysis': 'Use impact_analysis(stock_id) to see "what breaks"',
                'similar_stocks': 'Use similar_stocks(stock_id) to find peer stocks'
            }
        }
    
    def save(self, path: str = None):
        """Save graph to JSON."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..', 'stock_graph.json')
        data = self.to_dict()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    
    @classmethod
    def load(cls, path: str = None):
        """Load graph from JSON."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..', 'stock_graph.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls()
        kg.nodes = set(data.get('nodes', {}).keys())
        kg.metadata = data.get('metadata', {})
        
        # Rebuild edges
        for edge in data.get('edges', []):
            key = tuple(sorted([edge['source'], edge['target']]))
            kg.edges[key] = {
                'type': edge['type'],
                'weight': edge['weight'],
                'metadata': {}
            }
        
        if HAS_NX:
            for sid, attrs in data.get('nodes', {}).items():
                kg.graph.add_node(sid, **attrs)
            for edge in data.get('edges', []):
                kg.graph.add_edge(edge['source'], edge['target'],
                                 weight=edge['weight'], types=[edge['type']])
        
        return kg


# ====================================================================
# 快速建構
# ====================================================================
def build_default_graph(stock_ids: list = None) -> StockKnowledgeGraph:
    """
    Build a default knowledge graph with common TW stocks.
    
    This is the GitNexus-inspired "index" step:
    Locally builds a graph with zero API calls (sector data is built-in).
    """
    if stock_ids is None:
        stock_ids = [
            # 你的持股
            ('2618', '長榮航'), ('2382', '廣達'), ('2330', '台積電'),
            # 航運
            ('2603', '長榮'), ('2609', '陽明'), ('2615', '萬海'),
            ('2610', '華航'), ('2637', '慧洋'),
            # 半導體
            ('2303', '聯電'), ('2454', '聯發科'), ('3034', '聯詠'),
            ('3711', '日月光'), ('3443', '創意'), ('6735', '系微'),
            # 金融
            ('2881', '富邦金'), ('2882', '國泰金'), ('2886', '兆豐金'),
            ('2891', '中信金'),
            # 傳產
            ('1301', '台塑'), ('2002', '中鋼'), ('1216', '統一'),
            # 其他關注
            ('2317', '鴻海'), ('2412', '中華電'), ('3008', '大立光'),
            ('2409', '友達'), ('3481', '群創'),
            ('2357', '華碩'), ('3231', '緯創'),
        ]
    
    kg = StockKnowledgeGraph()
    for sid, name in stock_ids:
        kg.add_stock(sid, name)
    kg.build_sector_edges()
    kg.metadata.update({
        'built_at': datetime.datetime.now().isoformat(),
        'stock_count': len(kg.nodes),
        'edge_count': len(kg.edges),
    })
    
    return kg


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print('=== Building Stock Knowledge Graph ===')
    kg = build_default_graph()
    print(f'  Nodes: {kg.metadata["stock_count"]}')
    print(f'  Edges: {kg.metadata["edge_count"]}')
    
    print()
    print('=== Impact Analysis: 2618 長榮航 ===')
    impact = kg.impact_analysis('2618')
    print(f'  Potential impact: {impact["total_potential_impact"]} stocks')
    print(f'  High impact: {len(impact["high_impact"])}')
    for s in impact['high_impact'][:5]:
        print(f'    {s["stock"]} ({s.get("reason", s.get("sector",""))}) - weight {s["impact_weight"]}')
    
    print()
    print('=== Similar Stocks: 2618 長榮航 ===')
    similar = kg.similar_stocks('2618', 5)
    for s in similar:
        print(f'  {s["stock"]} - {s.get("sector", "")} (weight: {s["impact_weight"]})')
    
    print()
    print('=== Saving graph ===')
    path = kg.save()
    print(f'  Saved to: {path}')
