"""
⚡ 當沖推薦股排程執行器
由 Windows Task Scheduler 觸發，盤前執行
輸出結果到 output/daytrade_picks.json
"""
import sys
import os
import json
import logging
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

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
        logging.FileHandler("logs/daytrade_picks.log", encoding="utf-8"),
        SafeStreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("daytrade_picks")

log.info("=" * 50)
log.info("⚡ 當沖推薦股排程開始")

try:
    from day_trading_picks import get_day_trading_picks, get_day_trading_summary

    picks = get_day_trading_picks(top_n=8, months=3, min_score=50)
    summary = get_day_trading_summary(picks)

    picks_data = []
    for p in picks:
        picks_data.append({
            "stock_id": getattr(p, "stock_id", ""),
            "stock_name": getattr(p, "stock_name", ""),
            "total_score": getattr(p, "total_score", 0),
            "volatility_score": getattr(p, "volatility_score", 0),
            "volume_score": getattr(p, "volume_score", 0),
            "momentum_score": getattr(p, "momentum_score", 0),
            "technical_score": getattr(p, "technical_score", 0),
            "risk_score": getattr(p, "risk_score", 0),
            "close_price": getattr(p, "close_price", 0),
            "buy_price": getattr(p, "buy_price", 0),
            "sell_price": getattr(p, "sell_price", 0),
            "stop_loss": getattr(p, "stop_loss", 0),
            "reason": getattr(p, "reason", ""),
        })

    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "picks": picks_data,
    }

    with open("output/daytrade_picks.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info(f"✅ 完成！推薦 {len(picks_data)} 檔當沖標的，寫入 output/daytrade_picks.json")

    sys.exit(0)

except Exception as e:
    log.exception(f"❌ 執行失敗: {e}")
    sys.exit(1)
