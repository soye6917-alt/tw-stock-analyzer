#!/usr/bin/env python3
"""
auto_briefing.py — 自動盤前快訊產生器

用於 Cron Job 自動產生每日盤前/盤後快訊。
輸出置於 memory/briefing/ 供查閱。
"""
import sys, os, json, datetime

sys.stdout.reconfigure(encoding='utf-8')
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

BRIEFING_DIR = os.path.join(_project_dir, 'memory', 'briefing')
os.makedirs(BRIEFING_DIR, exist_ok=True)

# —— 持股列表（來自 MEMORY.md） ——
HOLDINGS = [
    ('2618', '長榮航'),
    ('2382', '廣達'),
]

WATCHLIST = [
    ('2330', '台積電'), ('2603', '長榮'), ('2609', '陽明'),
    ('2605', '華航'), ('3034', '聯詠'), ('1301', '台塑'),
    ('2317', '鴻海'), ('2303', '聯電'),
]


def generate_premarket_briefing(holdings=HOLDINGS, watchlist=WATCHLIST) -> str:
    """
    產生盤前快訊（隔夜市場 + 持股狀態 + 今日關注）
    可在 cron job 中呼叫
    """
    from data_fetcher import fetch_realtime_quote, get_stock_name
    
    now = datetime.datetime.now()
    lines = []
    lines.append(f"# 🌅 不響老師 盤前快訊 — {now.strftime('%Y-%m-%d')}")
    lines.append(f"更新時間：{now.strftime('%H:%M')}")
    lines.append("")
    
    # 先拿持股即時報價
    lines.append("## 📌 目前持股")
    for sid, name in holdings:
        try:
            q = fetch_realtime_quote(sid)
            if q and q.get('price', 0) > 0:
                p = q['price']
                lines.append(f"- **{sid} {name}**: ${p:.2f}")
            else:
                lines.append(f"- **{sid} {name}**: 尚無報價")
        except Exception as e:
            lines.append(f"- **{sid} {name}**: 錯誤({e})")
    
    lines.append("")
    lines.append("## 📋 觀察清單")
    for sid, name in watchlist:
        try:
            q = fetch_realtime_quote(sid)
            if q and q.get('price', 0) > 0:
                p = q['price']
                chg = q.get('change_pct', 0)
                emoji = '🟢' if chg >= 1 else ('🔴' if chg <= -1 else '⚪')
                lines.append(f"- **{sid} {name}**: ${p:.2f} ({chg:+.2f}%) {emoji}")
            else:
                lines.append(f"- **{sid} {name}**: 尚無報價")
        except:
            lines.append(f"- **{sid} {name}**: 錯誤")
    
    lines.append("")
    lines.append("## 🌍 國際市場摘要")
    try:
        from professional.professional_international import get_overnight_context
        ctx = get_overnight_context()
        if isinstance(ctx, str):
            # get_overnight_context returns a formatted string
            for line in ctx.split('\n'):
                lines.append(f"- {line.strip()}")
        else:
            lines.append(f"- S&P 500: {ctx.get('sp500_change', 'N/A')}")
            lines.append(f"- 費半: {ctx.get('sox_change', 'N/A')}")
            lines.append(f"- 台積電 ADR: {ctx.get('tsm_adr_change', 'N/A')}")
            lines.append(f"- VIX 恐慌: {ctx.get('vix_label', 'N/A')}")
    except Exception as e:
        lines.append(f"- 國際市場數據暫不可用: {e}")
    
    lines.append("")
    lines.append("---")
    lines.append("*⚠️ 本快訊由不響老師自動產生，僅供分析參考*")
    
    return '\n'.join(lines)


def save_briefing(kind: str = 'premarket'):
    """Generate and save briefing to file."""
    text = generate_premarket_briefing()
    filename = f'{kind}_{datetime.datetime.now().strftime("%Y%m%d")}.md'
    path = os.path.join(BRIEFING_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f'✅ {kind.capitalize()} briefing saved: {path}')
    print()
    print(text)
    return path, text


if __name__ == '__main__':
    import sys as _sys
    kind = _sys.argv[1] if len(_sys.argv) > 1 else 'premarket'
    save_briefing(kind)
