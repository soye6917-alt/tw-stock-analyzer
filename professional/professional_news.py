"""
professional_news.py — 專業財經新聞分析引擎

功能：
1. 多源爬取：鉅亨網 + Moneydj + Yahoo奇摩股市
2. 情緒評分：基於繁體中文情緒辭典
3. 個股新聞追蹤：自動彙整每檔股票相關新聞
4. 熱門題材偵測：從新聞標題提取熱門主題
"""
import requests as rq
import re, json, datetime, time
from typing import List, Dict, Optional
import pandas as pd

rq.packages.urllib3.disable_warnings()

# =========== 繁體中文情緒辭典 ===========
POSITIVE_WORDS = {
    # 強正面 (score +3)
    '飆漲', '漲停', '創高', '歷史新高', '突破', '爆發', '大賺', '井噴',
    '利多', '重磅', '超預期', '噴出', '轉機', '復甦', '擴張',
    '獲利倍增', '營收創高', '併購', '策略聯盟',
    # 中正面 (score +2)
    '上漲', '成長', '回升', '反彈', '增加', '調升', '看好',
    '布局', '擴產', '新單', '訂單', '合作', '進軍',
    # 弱正面 (score +1)
    '穩定', '回穩', '注意', '觀察', '可望', '有撐', '平穩'
}

NEGATIVE_WORDS = {
    # 強負面 (score -3)
    '崩跌', '跌停', '破底', '新低', '暴跌', '重挫', '慘賠', '腰斬',
    '利空', '違約', '掏空', '地雷', '警示', '危機', '裁員',
    '虧損', '營收衰退', '下市', '暫停交易',
    # 中負面 (score -2)
    '下跌', '衰退', '下滑', '調降', '賣壓', '套牢', '追繳',
    '降評', '減資', '增資', '借款', '負債', '訴訟',
    # 弱負面 (score -1)
    '震盪', '整理', '觀望', '保守', '疲弱', '停滯', '回檔'
}

def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of Chinese financial text."""
    score = 0
    matches = {'positive': [], 'negative': []}
    
    for word in POSITIVE_WORDS:
        if word in text:
            if word in ('飆漲', '漲停', '創高', '歷史新高', '突破', '爆發',
                       '大賺', '井噴', '利多', '重磅', '超預期'):
                score += 3
            elif word in ('上漲', '成長', '回升', '反彈', '增加', '調升', '看好',
                         '布局', '擴產', '新單', '訂單', '合作', '進軍'):
                score += 2
            else:
                score += 1
            matches['positive'].append(word)
    
    for word in NEGATIVE_WORDS:
        if word in text:
            if word in ('崩跌', '跌停', '破底', '新低', '暴跌', '重挫', '慘賠',
                       '腰斬', '利空', '違約', '掏空', '地雷', '警示', '危機'):
                score -= 3
            elif word in ('下跌', '衰退', '下滑', '調降', '賣壓', '套牢', '追繳',
                         '降評', '減資'):
                score -= 2
            else:
                score -= 1
            matches['negative'].append(word)
    
    # Normalize to [-1, 1]
    normalized = max(-1, min(1, score / 10))
    
    # Determine label
    if normalized > 0.3:
        label = 'positive'
    elif normalized < -0.3:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'score': round(normalized, 3),
        'label': label,
        'raw_score': score,
        'matches': matches
    }

# =========== 新聞爬取 ===========
def fetch_cnyes_news(stock_id: str, pages: int = 2) -> List[Dict]:
    """Fetch news from 鉅亨網 (cnyes.com)."""
    news_list = []
    session = rq.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for page in range(1, pages + 1):
        url = f'https://api.cnyes.com/search?q={stock_id}&page={page}&source=news'
        try:
            resp = session.get(url, timeout=10, verify=False)
            data = resp.json()
            items = data.get('data', {}).get('items', [])
            for item in items[:10]:
                news_list.append({
                    'source': '鉅亨網',
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'date': item.get('publishedAt', ''),
                    'stock_id': stock_id
                })
            time.sleep(0.5)
        except:
            break
    
    return news_list

def fetch_yahoo_news(stock_id: str) -> List[Dict]:
    """Fetch news from Yahoo奇摩股市."""
    news_list = []
    try:
        url = f'https://tw.stock.yahoo.com/quote/{stock_id}/news'
        resp = rq.get(url, verify=False, timeout=10,
                      headers={'User-Agent': 'Mozilla/5.0'})
        # Yahoo pages are JS-rendered, but we can get some data from meta
        if resp.status_code == 200:
            # Extract news links from HTML
            titles = re.findall(r'<h3[^>]*>([^<]+)</h3>', resp.text)
            for t in titles[:5]:
                news_list.append({
                    'source': 'Yahoo奇摩',
                    'title': t.strip(),
                    'summary': '',
                    'url': '',
                    'date': '',
                    'stock_id': stock_id
                })
    except:
        pass
    return news_list

def fetch_moneydj_news(stock_id: str) -> List[Dict]:
    """Fetch news from Moneydj."""
    news_list = []
    try:
        url = f'https://www.moneydj.com/KMDJ/News/NewsRealList.aspx?a=G&b={stock_id}'
        resp = rq.get(url, verify=False, timeout=10,
                      headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code == 200:
            titles = re.findall(r'<a[^>]*title="([^"]+)"', resp.text)
            for t in titles[:5]:
                news_list.append({
                    'source': 'Moneydj',
                    'title': t.strip(),
                    'summary': '',
                    'url': '',
                    'date': '',
                    'stock_id': stock_id
                })
    except:
        pass
    return news_list

def get_news_sentiment(stock_id: str, stock_name: str = '') -> dict:
    """Get comprehensive news sentiment for a stock."""
    all_news = []
    
    # Fetch from all sources
    all_news.extend(fetch_cnyes_news(stock_id, pages=1))
    all_news.extend(fetch_yahoo_news(stock_id))
    all_news.extend(fetch_moneydj_news(stock_id))
    
    if not all_news:
        return {
            'stock_id': stock_id,
            'total_news': 0,
            'sentiment_score': 0,
            'sentiment_label': 'neutral',
            'news_items': [],
            'summary': '無近期新聞'
        }
    
    # Analyze sentiment for each
    for item in all_news:
        text = item['title'] + ' ' + item['summary']
        sentiment = analyze_sentiment(text)
        item['sentiment'] = sentiment
    
    # Aggregate
    scores = [n['sentiment']['score'] for n in all_news if n.get('sentiment')]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score > 0.2:
        label = 'positive'
    elif avg_score < -0.2:
        label = 'negative'
    else:
        label = 'neutral'
    
    # Generate summary
    pos_count = sum(1 for s in scores if s > 0)
    neg_count = sum(1 for s in scores if s < 0)
    
    return {
        'stock_id': stock_id,
        'total_news': len(all_news),
        'sentiment_score': round(avg_score, 3),
        'sentiment_label': label,
        'positive_news': pos_count,
        'negative_news': neg_count,
        'top_positive': [n['title'] for n in all_news[:3] if n.get('sentiment',{}).get('label') == 'positive'][:3],
        'top_negative': [n['title'] for n in all_news[:3] if n.get('sentiment',{}).get('label') == 'negative'][:3],
        'summary': f'共{len(all_news)}則新聞, 正面{pos_count}, 負面{neg_count}'
    }

# =========== 熱門題材偵測 ===========
HOT_TOPICS = {
    'AI': ['AI', '人工智慧', '機器人', 'ChatGPT', '輝達', 'NVIDIA', 'GPU'],
    '半導體': ['半導體', '晶圓', '晶片', 'IC設計', '台積電', '先進製程'],
    '航運': ['航運', '貨櫃', '散裝', '長榮', '陽明', '萬海'],
    '電動車': ['電動車', 'EV', '電池', '充電樁', '特斯拉'],
    '能源': ['綠能', '太陽能', '風電', '儲能', '重電'],
    '金融': ['金融', '銀行', '金控', '升息', '降息'],
    '生技': ['生技', '藥品', '醫材', '疫苗', '檢測'],
    '資安': ['資安', '資訊安全', '駭客', '網路安全'],
    '軍工': ['軍工', '國防', '航太', '軍用'],
    '碳權': ['碳權', '碳費', 'ESG', '淨零', '永續'],
}

def detect_hot_topics(news_items: List[Dict]) -> Dict[str, int]:
    """Detect hot investment themes from news articles."""
    topic_counts = {}
    for item in news_items:
        text = (item.get('title', '') + ' ' + item.get('summary', '')).lower()
        for topic, keywords in HOT_TOPICS.items():
            for kw in keywords:
                if kw.lower() in text:
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    break
    return dict(sorted(topic_counts.items(), key=lambda x: -x[1]))

if __name__ == '__main__':
    # Test
    result = get_news_sentiment('2618', '長榮航')
    print(f'2618 新聞情緒: {json.dumps(result, ensure_ascii=False, indent=2)}')
