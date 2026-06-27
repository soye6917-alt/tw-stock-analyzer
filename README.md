# 不響老師 - 台股分析系統 📈

Taiwan stock analysis system with 12 analysis modes.

## Features

| # | Mode | Description |
|:--|:-----|:------------|
| 1 | 📈 看盤 | Real-time K-line + technical indicators |
| 2 | 📊 回測 | Backtesting (MA/RSI/MACD/Bollinger) |
| 3 | 🔍 多股掃描 | Multi-stock technical signal scan |
| 4 | 🌟 每日推薦 | Multi-factor daily picks |
| 5 | 🔬 專家分析 | Deep single-stock analysis |
| 6 | 🔥 低價飆股 | Low-price surge stock screener |
| 7 | 📊 籌碼戰情室 | Chip analysis (OBV, divergence) |
| 8 | 🌡️ 多空溫度計 | Market sentiment (PCR, fear/greed) |
| 9 | 🔬 配對交易 | Statistical arbitrage |
| 10 | 🧠 股票知識圖譜 | Knowledge graph (28 stocks, NetworkX) |
| 11 | 🌍 全球圖譜 | Global market KG (46 nodes, 126 edges) |
| 12 | 📈 LSTM 預測 | Deep learning price forecast |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
tw-stock-analyzer/
├── app.py                    # Main Streamlit app
├── stock_knowledge_graph.py  # Taiwan stock KG
├── professional/
│   ├── global_knowledge_graph.py  # Global market KG
│   ├── auto_briefing.py           # Pre-market briefing
│   ├── stat_arb.py                # Statistical arbitrage
│   ├── chip_analysis.py           # Chip analysis
│   └── ...
└── specs/
    ├── 000-spec-index.md     # 9 specs
    └── GOALS.md               # Goal-driven templates
```

## Data Sources

- TWSE API (Taiwan Stock Exchange)
- Yahoo Finance (global indices/commodities)
- FinMind (institutional trading data)
