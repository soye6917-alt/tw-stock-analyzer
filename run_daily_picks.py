"""
🏆 每日推薦股排程執行器
由 Windows Task Scheduler 觸發，盤前執行
輸出結果到 output/daily_picks.json
"""
import sys
import os
import json
import logging
from datetime import datetime

# 確保從專案目錄執行
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

# 設定 logging
os.makedirs("logs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 使用支援 emoji 的 stream handler（Task Scheduler 環境無 console，檔案 log 為主）
class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            pass  # cp950 console 無法顯示 emoji，跳過不影響檔案 log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/daily_picks.log", encoding="utf-8"),
        SafeStreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("daily_picks")

log.info("=" * 50)
log.info("🏆 每日推薦股排程開始")

try:
    from daily_picks import get_daily_picks_with_context

    result = get_daily_picks_with_context(top_n=5, months=6, include_news=True)

    # 轉換 dataclass 為 dict
    picks_data = []
    for p in result.get("picks", []):
        picks_data.append({
            "stock_id": getattr(p, "stock_id", ""),
            "stock_name": getattr(p, "stock_name", ""),
            "total_score": getattr(p, "total_score", 0),
            "technical_score": getattr(p, "technical_score", 0),
            "news_score": getattr(p, "news_score", 0),
            "fundamental_score": getattr(p, "fundamental_score", 0),
            "institutional_score": getattr(p, "institutional_score", 0),
            "momentum_score": getattr(p, "momentum_score", 0),
            "risk_score": getattr(p, "risk_score", 0),
            "close_price": getattr(p, "close_price", 0),
            "rating": getattr(p, "rating", ""),
            "analysis": getattr(p, "analysis", ""),
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "market_note": result.get("market_note", ""),
        "candidates_count": result.get("candidates_count", 0),
        "picks": picks_data,
    }

    with open("output/daily_picks.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info(f"✅ 完成！推薦 {len(picks_data)} 檔股票，寫入 output/daily_picks.json")
    log.info(f"   市場判斷: {result.get('market_note', 'N/A')}")

    sys.exit(0)

except Exception as e:
    log.exception(f"❌ 執行失敗: {e}")
    sys.exit(1)
