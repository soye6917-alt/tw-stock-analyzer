#!/usr/bin/env python3
"""法人籌碼分析 for 三檔電線電纜股"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\soye6\.openclaw\workspace\tw-stock-analyzer')
import warnings; warnings.filterwarnings('ignore')
from data_fetcher import fetch_historical
from chip_analysis import run_chip_analysis, institutional_flow_analysis
from fundamentals import fetch_institutional_trading

stocks = {
    '1609': '大亞',
    '1605': '華新',
    '2415': '錩新',
    '1604': '聲寶',
}

for sid, name in stocks.items():
    print(f'========== {name}({sid}) ==========')
    df = fetch_historical(sid, months=3)
    if df.empty:
        print('  無法取得資料\n')
        continue
    
    result = run_chip_analysis(sid, df, fetch_institutional=True)
    lines = result.get('summary_lines', [])
    score = result.get('overall_score', 0)
    level = result.get('overall_level', '')
    print(f'  籌碼綜合評分: {score}')
    print(f'  等級: {level}')
    for line in lines:
        print(f'  {line}')
    print()
    
    # Also try institutional flow directly
    try:
        inst = fetch_institutional_trading(sid)
        if not inst.empty:
            recent = inst.tail(5)
            print(f'  近5日法人買賣超明細:')
            for idx, row in recent.iterrows():
                f = row.get('foreign_net', 0)
                it = row.get('investment_net', 0)
                dealer = row.get('dealer_net', 0)
                total = f + it + dealer
                print(f'    {idx}: 外資{f:+} 投信{it:+} 自營{dealer:+} 合計{total:+}')
    except Exception as e:
        print(f'  法人明細失敗: {e}')
    print()
