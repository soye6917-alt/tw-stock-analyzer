# Goal-Driven Development: 不響老師 每日任務日誌

> 五段式目標模板：
> Objective / Scope / Constraints / Done when / Stop if

---

## Active Goal (2026-06-27)

### Goal: 完整四項技術升級

**Objective:** 從 AI 超元域頻道學到的四項技術全部落地到不響老師系統。

**Scope:** 只改 `professional/` 目錄和 `app.py`。

**Constraints:**
- 所有模組必須能獨立 import（不能有循環依賴）
- 不破壞現有 9 種分析模式（只新增第 11 模式）
- 保持 cp950 編碼相容（終端輸出用英文標籤）

**Done when:**
1. ✅ 多 Agent 編排 → `professional_workflow.py`（5 Agent平行，8秒完成）
2. ✅ GitNexus 知識圖譜 → `stock_knowledge_graph.py`（28節點/190邊，支援impact_analysis）
3. ✅ Spec-Driven → `specs/000-spec-index.md`（9個spec，每個有驗收清單）
4. ✅ Goal-Driven → 本檔（active goal + task backlog）
5. ✅ 知識圖譜整合進 Streamlit → `app.py` 第 11 模式「🕸️ 關係圖譜」

**Stop if:**
- 任何模組 import 報錯
- app.py 啟動失敗
- 現有功能被破壞

**Progress:**
- 多Agent: 100% ✅
- 知識圖譜: 100% ✅（+ 整合進 Streamlit）
- Spec-Driven: 100% ✅
- Goal-Driven: 100% ✅

---

## Task Backlog

### P1 今晚衝刺 (6/27)

| ID | Task | Spec Ref | Effort | Status |
|:---|:-----|:---------|:------:|:------:|
| 001 | 法人籌碼 SSL 修復 | Spec-006 | 3h | ✅ Done |
| 002 | 知識圖譜整合 Streamlit | (延伸Spec-001) | 1h | ✅ Done |

### P2 本周目標

| ID | Task | Spec Ref | Effort | Status |
|:---|:-----|:---------|:------:|:------:|
| 003 | LSTM 預測匯入修復 | Spec-007 | 2h | ✅ 已能正常運作（僅TF載入慢）|
| 004 | 自動排程盤前快訊 | Spec-008 | 2h | ✅ Windows Task Scheduler 8:30AM + HEARTBEAT.md |
| 005 | 全球市場知識圖譜 | Spec-009 | 4h | ✅ Done (46 nodes / 126 edges / 5 UI tabs) |

### P3 有時間再弄

| ID | Task | Spec Ref | Effort | Status |
|:---|:-----|:---------|:------:|:------:|
| 006 | 15 堂課 Lesson 1 | Spec-005 | 2h | ❌ |
| 007 | 員購模組 | - | 1h | ❌ |

---

## Daily Goal Template

```
### Goal: [日期] - [一句話描述]

**Objective:** 

**Scope:**

**Constraints:**
- 
- 

**Done when:**
- [ ] 
- [ ] 

**Stop if:**
- 

**Token budget:** N/A (local dev)

**Progress:**
- [task]: %
```
