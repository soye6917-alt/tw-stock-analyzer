# Spec-Driven Development: 不響老師 股票分析系統

> 原則：先寫規格，再實作。規格就是驗收清單。
> 靈感：AI超元域 / OpenSpec / Codex /goal

---

## Spec 001 — 知識圖譜即時更新 Agent

**狀態：** ✅ 已完成（stock_knowledge_graph.py）
**優先級：** P1

### Objective
建立股票關係知識圖譜，讓分析工具能理解股票之間的連動關係。

### Scope
只改 `professional/stock_knowledge_graph.py`，不動現有分析模組。

### Done when
1. ✅ 知識圖譜包含至少 28 檔股票的產業關係
2. ✅ 支援 impact_analysis()：給定股票代號回傳波及範圍
3. ✅ 支援 similar_stocks()：給定股票代號回傳相似股票
4. ✅ 支援跨 session 持久化（save/load JSON）
5. ✅ 使用真實價格資料建立 correlation 邊

---

## Spec 002 — 多 Agent 平行分析工作流

**狀態：** ✅ 已完成（professional_workflow.py）
**優先級：** P1

### Objective
將原本順序分析改為多 Agent 平行工作流，減少等待時間、增加分析覆蓋面。

### Scope
新增 `professional_workflow.py`，不改現有 professional_hub.py。

### Agents
1. `technical` — 技術分析師（均線/RSI/成交量）
2. `news` — 新聞情緒分析師（三源爬取）
3. `regime` — 市場判讀師（狀態分類）
4. `position` — 倉位管理師（凱利/固定風險）
5. `international` — 國際分析師（11指數+ADR+VIX）

### Done when
1. ✅ 5 個 Agent 能平行執行 (ThreadPoolExecutor)
2. ✅ Aggregator 能加權計分產出最終評級
3. ✅ app.py 第 10 模式可切換 Standard/Workflow
4. ✅ 執行時間 < 10 秒（目標 8 秒）

---

## Spec 003 — 國際市場即時監控

**狀態：** ✅ 已完成（professional_international.py）
**優先級：** P2

### Done when
1. ✅ 11 個全球指數即時行情
2. ✅ 台股 ADR 監控（TSM/UMC/ASX/WIT）
3. ✅ VIX 恐慌指數轉換為情緒標籤
4. ✅ 一鍵 `get_overnight_context()` 文字摘要

---

## Spec 004 — 每日盤前/盤後快訊

**狀態：** ✅ 已完成（professional_briefing.py）
**優先級：** P2

### Done when
1. ✅ `generate_premarket_briefing()` — 隔夜市場 + 持股狀態 + 今日關注
2. ✅ `generate_postmarket_review()` — 今日表現 + 明日規劃
3. ✅ 支援自訂持股列表

---

## Spec 005 — 不響老師 15 堂課

**狀態：** 📋 課程框架已完成，等待 Lesson 1 開始
**優先級：** P3

### Objective
建立從 K 線入門到建立交易系統的完整教學體系。

### Done when
1. ✅ 15 堂課目錄（memory/teacher/01-curriculum.md）
2. ✅ 每日市場快訊模板（02-market-intel.md）
3. ✅ 交易日記模板（03-trade-journal.md）
4. ❌ Lesson 1 實際教學（等 SOYE 開始）

---

## Spec 006 — 法人籌碼改進

**狀態：** ❌ 待開發
**優先級：** P2

### Objective
修復三大法人資料 SSL 錯誤，建立穩定的法人籌碼資料流。

### Constraints
- 資料來源：TWSE 三大法人買賣超 API
- 不新增外部依賴
- 失敗時 graceful fallback 到 OBV 分析

### Done when
1. ✅ SSL 握手錯誤不再噴出（added SESSION.verify=False + suppress InsecureRequestWarning）
2. ✅ `fetch_institutional_trading('2330')` 成功回傳外資/投信/自營商買賣超
3. ✅ `fetch_fundamentals('2330')` 成功回傳本益比/殖利率/淨值比
4. ✅ 同時修復 data_fetcher.py（與 fundaments.py 共用相同 SESSION fix）

---

## Spec 007 — LSTM 預測模組整合

**狀態：** ✅ 可正常運行
**優先級：** P3

### Objective
修復 lstm_predictor.py 的匯入錯誤，讓 LSTM/GRU/BiLSTM 深度學習預測可以正常使用。

### Done when
1. ✅ `from lstm_predictor import ...` 成功（TensorFlow/Keras 可載入）
2. ✅ 預測結果顯示在 app.py 儀表板 Tab 8
3. ✅ 支援 5/10/20 日預測區間
4. ✅ `compare_forecast_methods()` 有 LSTM vs 線性回歸對比

---

## Spec 008 — 自動排程盤前快訊

**狀態：** ✅ 已實作
**優先級：** P2

### Objective
每天早上 8:30 自動產出盤前快訊並通知 SOYE。

### Constraints
- 用 Windows Task Scheduler 而非 OpenClaw cron（因 device pairing 未設定）
- HEARTBEAT.md 作為輔助觸發

### Done when
1. ✅ `professional/auto_briefing.py` — 產生含持股+觀察+國際市場的快訊
2. ✅ Windows Task Scheduler 每天早上 8:30 執行
3. ✅ 輸出置於 `memory/briefing/`
4. ✅ HEARTBEAT.md 早上檢查排程
5. ✅ 國際市場摘要包含費半/VIX/台積電ADR

### Constraints
- OpenClaw cron 需要裝置配對
- 此之前先用 HEARTBEAT.md 觸發

### Done when
1. HEARTBEAT.md 每天早上觸發 briefing 產生
2. 產出結果 Push 到 SOYE

---

## Spec 009 — 全球市場知識圖譜延伸

**狀態：** ❌ 待開發
**優先級：** P3

### Objective
將知識圖譜從台股延伸到全球市場：
- US/JP/HK 主要指數
- 產業對應（TSMC ↔ SOX, 台塑 ↔ WTI）
- 匯率對台股影響

### Done when
1. ✅ 新增全球節點（46 nodes: 10 indices + 3 ADR + 6 commodities + 2 crypto + 26 TW stocks）
2. ✅ 建立跨市場邊（64 global→TW + 49 commodity→index = 126 edges total）
3. ✅ impact_analysis() 支援跨市場查詢（global_impact / sector_exposure / overnight_risk_assessment）
4. ✅ ADR 溢價監控（TSM/UMC/ASX，含匯率換算）
5. ✅ 整合進 Streamlit 第 12 模式「🌍 全球圖譜」（5分頁: 全球→台股影響 / ADR監控 / 隔夜風險 / 商品關係 / 圖譜概覽）
6. ✅ 圖譜持久化於 `global_graph.json`
