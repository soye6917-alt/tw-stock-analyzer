#!/usr/bin/env python3
"""Debug scan issue and scan stocks."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\soye6\.openclaw\workspace\tw-stock-analyzer')
import warnings; warnings.filterwarnings('ignore')

# Import modules and register them in sys.modules
import data_fetcher
import indicators
import fundamentals

# Now scan
from low_price_surge import scan_low_price_surge, analyze_low_price_candidate
import time

start = time.time()
candidates = scan_low_price_surge(top_n=30, max_workers=12)
elapsed = time.time() - start
print(f'\n耗時: {elapsed:.0f}s')

print(f'\n共找到 {len(candidates)} 檔')

for i, c in enumerate(candidates[:5], 1):
    print(f'{i}. [{c.grade}] {c.stock_name}({c.stock_id}) ${c.price} 評分{c.score} 目標${c.target_short}(+{c.upside_pct}%) 盈虧比{c.risk_reward}x 信心{c.confidence} 信號:{c.signals[:3]}')
    print()
