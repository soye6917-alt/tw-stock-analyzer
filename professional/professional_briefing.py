"""
professional_briefing.py — 每日盤前/盤後快訊

功能：
1. 盤前快訊：隔夜國際市場 + 今日關注 + 持股提醒
2. 盤後回顧：今日表現 + 技術變化 + 新聞摘要
3. 一鍵產出完整市場日報
"""
import sys, os, datetime, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from professional_international import fetch_market_summary, get_overnight_context
from professional_news import get_news_sentiment, detect_hot_topics, fetch_cnyes_news
from professional_trading import calculate_entry_score
from professional_regime import classify_market_regime

def generate_premarket_briefing(stock_ids: list = None) -> str:
    """
    Generate pre-market briefing text.
    
    Args:
        stock_ids: List of (stock_id, stock_name) to include in briefing
    """
    if stock_ids is None:
        stock_ids = [('2618', '長榮航'), ('2382', '廣達'), ('2330', '台積電'), ('2603', '長榮')]
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y/%m/%d')
    day_name = ['一', '二', '三', '四', '五', '六', '日'][now.weekday()]
    
    lines = []
    lines.append(f"╔══════════════════════════════════════╗")
    lines.append(f"║      🌅 盤前快訊 · {date_str} (週{day_name})      ║")
    lines.append(f"╚══════════════════════════════════════╝")
    lines.append("")
    
    # 1. Overnight markets
    lines.append("📊 隔夜國際市場")
    lines.append("─" * 40)
    try:
        overnight = get_overnight_context()
        lines.append(overnight)
    except Exception as e:
        lines.append(f"  無法取得: {e}")
    lines.append("")
    
    # 2. 台股市場狀態
    from data_fetcher import fetch_historical
    try:
        taiex = fetch_historical('0000', months=3)  # 加權指數
        if taiex is not None and len(taiex) > 60:
            regime = classify_market_regime(taiex)
            lines.append(f"🎯 台股盤勢: {regime.get('regime_name', '待確認')}")
            lines.append(f"  信心: {regime.get('confidence', 0)}%")
            lines.append(f"  建議: {regime.get('advice', '')}")
    except:
        lines.append("🎯 台股盤勢: 分析中(資料載入)")
    lines.append("")
    
    # 3. 重點持股分析
    lines.append("📈 你的持股")
    lines.append("─" * 40)
    
    for sid, sname in stock_ids:
        try:
            from data_fetcher import fetch_historical, fetch_realtime_quote
            df = fetch_historical(sid, months=2)
            quote = fetch_realtime_quote(sid)
            if df is not None and len(df) > 10:
                close = df['Close'].iloc[-1]
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                signal = "📈 多頭" if close > ma20 else "📉 空頭" if close < ma20 else "➡️ 盤整"
                price_str = f"${quote.get('price', close)}" if quote and quote.get('price') else f"${close:.2f}"
                lines.append(f"  {sname}({sid}): {price_str} {signal}")
        except:
            lines.append(f"  {sname}({sid}): 資料載入中")
    
    lines.append("")
    
    # 4. 今日關注
    lines.append("🔍 今日關注")
    lines.append("─" * 40)
    try:
        # Hot topics from recent news
        all_news = []
        for sid, sname in stock_ids[:2]:
            try:
                news = fetch_cnyes_news(sid, pages=1)
                all_news.extend(news)
            except:
                pass
        topics = detect_hot_topics(all_news)
        if topics:
            for topic, count in list(topics.items())[:5]:
                lines.append(f"  🔥 {topic}: {count} 則新聞")
        else:
            lines.append("  暫無顯著熱門題材")
    except:
        lines.append("  題材分析載入中")
    lines.append("")
    
    # 5. 交易提醒
    lines.append("⚡ 今日提醒")
    lines.append("─" * 40)
    lines.append(f"  • 盤前確認持股停損點是否更新")
    lines.append(f"  • 注意今日重要經濟數據公布")
    lines.append(f"  • 遵守交易計畫，不衝動交易")
    
    return "\n".join(lines)


def generate_postmarket_review(stock_ids: list = None) -> str:
    """Generate post-market review text."""
    if stock_ids is None:
        stock_ids = [('2618', '長榮航'), ('2382', '廣達'), ('2330', '台積電'), ('2603', '長榮')]
    
    now = datetime.datetime.now()
    date_str = now.strftime('%Y/%m/%d')
    
    lines = []
    lines.append(f"╔══════════════════════════════════════╗")
    lines.append(f"║      🌇 盤後回顧 · {date_str}                 ║")
    lines.append(f"╚══════════════════════════════════════╝")
    lines.append("")
    
    lines.append("📊 今日表現")
    lines.append("─" * 40)
    
    for sid, sname in stock_ids:
        try:
            from data_fetcher import fetch_historical, fetch_realtime_quote
            df = fetch_historical(sid, months=1)
            quote = fetch_realtime_quote(sid)
            if df is not None and len(df) > 1:
                today_close = df['Close'].iloc[-1]
                yesterday_close = df['Close'].iloc[-2] if len(df) > 1 else today_close
                change_pct = (today_close - yesterday_close) / yesterday_close * 100
                signal = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️"
                lines.append(f"  {signal} {sname}({sid}): ${today_close:.2f} ({change_pct:+.2f}%)")
        except:
            pass
    
    lines.append("")
    lines.append("📝 今日學到什麼")
    lines.append("─" * 40)
    lines.append("  (盤後檢討待補充)")
    lines.append("")
    lines.append("📋 明日計畫")
    lines.append("─" * 40)
    lines.append("  (明日策略待規劃)")
    
    return "\n".join(lines)


if __name__ == '__main__':
    print(generate_premarket_briefing())
    print()
    print(generate_postmarket_review())
