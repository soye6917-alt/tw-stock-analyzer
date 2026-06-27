"""
台灣股票分析系統 - Streamlit 主程式
功能:看盤、技術分析、回測、專家級+飆股預測+ETF分析
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import sys, os
# 確保專案目錄在 sys.path 中（處理 Streamlit 啟動目錄不一致）
_project_dir = os.path.dirname(os.path.abspath(__file__))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
if os.getcwd() != _project_dir:
    os.chdir(_project_dir)
from data_fetcher import (
    fetch_historical, fetch_realtime_quote, get_stock_name,
    search_stocks_by_name, fetch_stock_list, POPULAR_STOCKS
)
from indicators import add_all_indicators, get_indicator_signals
from backtest import (
    backtest_ma_crossover, backtest_rsi, backtest_macd, backtest_bollinger
)
from fundamentals import (
    fetch_fundamentals, fetch_institutional_trading,
    generate_analysis_summary, assess_recommendation
)
from daily_picks import get_daily_picks_with_context, SCAN_UNIVERSE
from stock_screener import filter_stocks
import pattern_recognition as pr
from html import escape as html_escape
from surge_predictor import (
    surge_score_stock_full, scan_surge_candidates, SurgeCandidate
)
from etf_analysis import (
    score_etf, get_etf_picks, get_category_stats,
    compare_etfs, generate_etf_analysis, ETFScore
)
from expert_analysis import run_expert_analysis
from day_trading_picks import (
    get_day_trading_picks, get_day_trading_summary, DAYTRADE_UNIVERSE
)
# ── 新模組載入器 ──
from _loader import *

# —— 快取：當沖推薦
@st.cache_data(ttl=1800, show_spinner="⚡ 正在掃描當沖標的（分析波動/量能/動能），約需 60-90 秒...")
def get_daytrade_cached(n, m, min_s):
    picks = get_day_trading_picks(top_n=n, months=m, min_score=min_s)
    summary = get_day_trading_summary(picks)
    return picks, summary

# —— 快取：每日推薦結果（避免每次重新掃描 50 檔導致超時）——
@st.cache_data(ttl=3600, show_spinner="🔍 正在掃描 50 檔重點股(含新聞情緒分析),約需 60-90 秒...")
def get_daily_picks_cached(top_n, months, include_news):
    return get_daily_picks_with_context(top_n=top_n, months=months, include_news=include_news)
from virtual_trading import (
    get_portfolio, buy_stock, sell_stock,
    get_holdings_with_prices, get_portfolio_summary,
    get_order_history, reset_portfolio,
)
from risk_management import (
    full_risk_report, kelly_criterion, calculate_position_size,
    calculate_var, sharpe_ratio, sortino_ratio
)
from ml_models import (
    prepare_features, train_xgboost, technical_consensus, ensemble_prediction
)
from monte_carlo import (
    monte_carlo_simulation, optimize_ma_crossover, optimize_rsi,
    strategy_comparison, monte_carlo_risk_analysis
)
from macro_data import (
    get_macro_summary, fetch_usd_twd_rate, fetch_us_interest_rate
)
from alerts import (
    add_alert, remove_alert, toggle_alert, get_alerts,
    check_price_alert, get_recent_events
)
from low_price_surge import (
    scan_low_price_surge, format_candidates_table
)
from chip_analysis import run_chip_analysis
from option_sentiment import run_option_sentiment_analysis
from market_cycle import run_market_cycle_analysis
from macro_data import get_macro_summary
from stat_arb import run_statistical_arbitrage, suggest_pair_trades, cointegration_test

st.set_page_config(
    page_title="台股分析系統",
    page_icon="📈",
    layout="wide",
)


mode = ""

# ============================================================
# 🎓 專業分析 — 不響老師 · 完整個股分析報告
# ============================================================
if mode == "🎓 專業分析":
    import sys as _pro_sys
    if _pro_sys.stdout.encoding.upper() in ("CP950", "BIG5"):
        _pro_sys.stdout.reconfigure(encoding="utf-8")
    
    st.subheader("🎓 不響老師 · 專業分析")
    st.caption("整合技術面 / 籌碼面 / 新聞情緒 / 市場狀態 / 倉位建議的完整分析報告")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        pro_sid = st.text_input("股票代號", value=stock_id, key="pro_sid")
    with col2:
        pro_capital = st.number_input("可用資金", min_value=10000, max_value=99999999, 
                                       value=100000, step=50000, format="%d", key="pro_cap")
    with col3:
        run_pro = st.button("🔬 執行專業分析", type="primary", use_container_width=True)
    
    # Toggle between sequential and parallel workflow
    use_workflow = st.toggle('使用平行工作流 (5 Agent 并行)', value=True,
                             help='开启：5 个分析 agent 平行执行，更快更全面。关闭：传统顺序分析')
    
    if run_pro or "pro_report" in st.session_state:
        if run_pro:
            st.session_state.pro_report = None
        
        if run_pro:
            if use_workflow:
                with st.spinner(f"正在以 5 Agent 工作流分析 {pro_sid} ..."):
                    from professional.professional_workflow import quick_analysis
                    result = quick_analysis(pro_sid, capital=pro_capital, parallel=True)
                    agent_data = result.get('agent_data', {})
                    tech_data = agent_data.get('technical', {})
                    news_data = agent_data.get('news', {})
                    regime_data = agent_data.get('regime', {})
                    pos_data = agent_data.get('position', {})
                    report = {
                        'sections': {
                            'realtime': {'price': pos_data.get('price', 0)} if pos_data.get('price', 0) > 0 else {},
                            'technical': tech_data,
                            'entry_analysis': {
                                'total_score': tech_data.get('entry_score', 50),
                                'grade': tech_data.get('entry_grade', 'N/A'),
                            },
                            'news_sentiment': news_data,
                            'market_regime': regime_data,
                            'position_sizing': pos_data,
                        },
                        'summary': {
                            'strengths': result.get('reasons', {}).get('bullish', []),
                            'weaknesses': result.get('reasons', {}).get('bearish', []),
                            'overall': '偏多' if '多' in result.get('final_grade', '') and '空' not in result.get('final_grade', '')
                                      else '偏空' if '空' in result.get('final_grade', '')
                                      else '中性',
                            'workflow_grade': result.get('final_grade', ''),
                            'workflow_score': result.get('final_score', 0),
                            'workflow_performance': result.get('performance', {})
                        }
                    }
                    st.session_state.pro_report = report
                    st.session_state.pro_sid = pro_sid
                    st.session_state.workflow_active = True
            else:
                with st.spinner(f"正在对 {pro_sid} 进行标准多维分析..."):
                    from professional.professional_hub import professional_stock_analysis
                    report = professional_stock_analysis(pro_sid, capital=pro_capital)
                    st.session_state.pro_report = report
                    st.session_state.pro_sid = pro_sid
                    st.session_state.workflow_active = False
        
        report = st.session_state.get("pro_report")
        if report and st.session_state.get("pro_sid") == pro_sid:
            
            is_workflow = st.session_state.get('workflow_active', False)
            
            if is_workflow and 'workflow_grade' in report.get('summary', {}):
                wf_summary = report['summary']
                perf = wf_summary.get('workflow_performance', {})
                st.success(f'[多Agent工作流] 评级: {wf_summary["workflow_grade"]} (分数: {wf_summary["workflow_score"]})')
                st.caption(f'执行时间: {perf.get("total", 0)}s | 资料 {perf.get("phase1_data_fetch", 0)}s | Agent {perf.get("phase2_agents", 0)}s | 聚合 {perf.get("phase3_aggregation", 0)}s')
            
            pro_tabs = st.tabs([
                "📊 即时报价", "📈 技术分析", "🎯 进场评分",
                "📰 新闻情绪", "🌍 市场状态", "💰 仓位建议", "📋 综合评语"
            ])
            
            sections = report.get('sections', {})
            
            # Tab 1: Realtime Quote
            with pro_tabs[0]:
                rt = sections.get('realtime', {})
                if rt:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("成交價", f"${rt.get('price', 'N/A')}", 
                                 delta=f"H:{rt.get('high', 0)}")
                    with c2:
                        st.metric("開盤", f"${rt.get('open', 'N/A')}", 
                                 delta=f"L:{rt.get('low', 0)}")
                    with c3:
                        st.metric("漲跌幅", f"{rt.get('range_pct', 0)}%",
                                 delta=f"量:{rt.get('volume', 0):,}")
                    with c4:
                        bid = rt.get('bid', 0)
                        ask = rt.get('ask', 0)
                        spread = round((ask - bid) / (rt.get('price', 1)) * 100, 3) if rt.get('price', 0) > 0 else 0
                        st.metric("買賣價差", f"{spread:.2f}%",
                                 delta=f"{bid:.2f} ~ {ask:.2f}")
                else:
                    st.info("即時報價目前無法取得（非交易時段）")
            
            # Tab 2: Technical Analysis
            with pro_tabs[1]:
                tech = sections.get('technical', {})
                if tech:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("收盤價", f"${tech.get('close', 0):.2f}")
                        st.metric("MA5", f"${tech.get('ma5', 0):.2f}")
                    with c2:
                        st.metric("MA20", f"${tech.get('ma20', 0):.2f}")
                        st.metric("MA60", f"${tech.get('ma60', 0):.2f}")
                    with c3:
                        st.metric("RSI(14)", f"{tech.get('rsi', 0)}")
                        st.metric("量比均量", f"{tech.get('volume_vs_avg', 0):.0f}%")
                    
                    # Trend
                    trend_20d = tech.get('trend_20d', 0)
                    trend_label = "📈 上漲" if trend_20d > 0 else "📉 下跌" if trend_20d < 0 else "➡️ 持平"
                    st.info(f"近20日趨勢: {trend_label} ({trend_20d:+.2f}%)")
                else:
                    st.info("技術分析資料不足")
            
            # Tab 3: Entry Score
            with pro_tabs[2]:
                entry = sections.get('entry_analysis', {})
                if entry:
                    score = entry.get('total_score', 0)
                    grade = entry.get('grade', 'N/A')
                    
                    col_m, col_d = st.columns([1, 1])
                    with col_m:
                        st.metric("進場評分", f"{score}/100", delta=grade[:3])
                        st.progress(score / 100)
                    with col_d:
                        st.write(f"**評級: {grade}**")
                        st.write(f"通過檢查: {entry.get('passed', '?')}")
                    
                    st.subheader("各項因子")
                    details = entry.get('details', {})
                    for key, detail in details.items():
                        passed = '✅' if detail.get('pass') else '❌'
                        weight = detail.get('weight', 0)
                        st.write(f"{passed} **{key}** (權重{weight})")
                else:
                    st.info("進場評分需要更多歷史資料")
            
            # Tab 4: News Sentiment
            with pro_tabs[3]:
                news = sections.get('news_sentiment', {})
                if news:
                    score = news.get('sentiment_score', 0)
                    label = news.get('sentiment_label', 'neutral')
                    emoji = '📈' if label == 'positive' else '📉' if label == 'negative' else '➡️'
                    st.metric(f"{emoji} 新聞情緒", label.upper(), delta=score)
                    st.write(f"共 {news.get('total_news', 0)} 則新聞")
                    st.write(news.get('summary', ''))
                else:
                    st.info("新聞情緒分析中")
            
            # Tab 5: Market Regime
            with pro_tabs[4]:
                regime = sections.get('market_regime', {})
                if regime:
                    name = regime.get('regime_name', '未知')
                    conf = regime.get('confidence', 0)
                    st.metric("當前市場狀態", name, delta=f"{conf}% 信心")
                    st.info(f"**建議: {regime.get('advice', '觀望')}**")
                    ind = regime.get('indicators', {})
                    if ind:
                        align = ind.get('alignment', '')
                        st.write(f"均線排列: {align}")
                else:
                    st.info("市場狀態分析需要更多資料")
            
            # Tab 6: Position Sizing
            with pro_tabs[5]:
                sizing = sections.get('position_sizing', {})
                if sizing:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("建議股數", sizing.get('shares', 0))
                        st.metric("投入金額", f"${sizing.get('position_value', 0):,.0f}")
                    with c2:
                        st.metric("風險比例", f"{sizing.get('risk_pct', 0)}%")
                        st.metric("最大損失", f"${sizing.get('potential_loss', 0):,.0f}")
                    st.write(f"停損價: ${sizing.get('stop_loss', 0):.2f}")
                    st.write(f"使用方式: {sizing.get('method', 'fixed_risk')}")
                else:
                    st.info("需即時報價才能計算倉位")
            
            # Tab 7: Summary
            with pro_tabs[6]:
                summary = report.get('summary', {})
                strengths = summary.get('strengths', [])
                weaknesses = summary.get('weaknesses', [])
                overall = summary.get('overall', '中性')
                
                emoji_o = '📈' if overall == '偏多' else '📉' if overall == '偏空' else '➡️'
                st.header(f"{emoji_o} 綜合評級: {overall}")
                
                if strengths:
                    st.subheader("✅ 優勢")
                    for s in strengths:
                        st.write(f"  ✅ {s}")
                
                if weaknesses:
                    st.subheader("⚠️ 劣勢")
                    for w in weaknesses:
                        st.write(f"  ⚠️ {w}")
                
                if not strengths and not weaknesses:
                    st.info("資料不足，無法產生綜合評語")
        else:
            st.info("請輸入股票代號並點擊「執行專業分析」")
# ============================================================
# 頁面標題 & 側邊欄
# ============================================================
st.title("📈 台股分析系統")
st.caption("資料來源:TWSE / TPEx 公開資料 | 僅供分析參考,非投資建議")

# 初始化虛擬交易 session state(雲端部署用)
if "vt_portfolio" not in st.session_state:
    from virtual_trading import new_portfolio, _load_portfolio
    # 先嘗試從 JSON 檔案還原（本機持久化），再回退到新的空組合
    try:
        from virtual_trading import _portfolio_path
        import os as _os
        if _os.path.exists(_portfolio_path()):
            st.session_state.vt_portfolio = _load_portfolio()
        else:
            st.session_state.vt_portfolio = new_portfolio()
    except Exception:
        st.session_state.vt_portfolio = new_portfolio()

with st.sidebar:
    st.header("⚙️ 設定")

    # 初始化 session state
    if "stock_id" not in st.session_state:
        st.session_state.stock_id = "2330"

    # 股票選擇(改用 session_state)
    stock_id_input = st.text_input("股票代號", value=st.session_state.stock_id, max_chars=6, key="sid_input")
    st.session_state.stock_id = stock_id_input

    stock_name = get_stock_name(st.session_state.stock_id)
    st.caption(f"📌 {stock_name}")

    # 名稱搜尋
    name_q = st.text_input("🔍 搜尋公司名稱", placeholder="輸入名稱關鍵字...")
    if name_q:
        matches = search_stocks_by_name(name_q)
        if matches:
            for code, name in matches[:10]:
                col_a, col_b = st.columns([1, 3])
                col_a.code(code)
                if col_b.button(name, key=f"pick_{code}", use_container_width=True):
                    st.session_state.stock_id = code
                    st.rerun()
        else:
            st.info("❌ 無符合的股票")

    stock_id = st.session_state.stock_id

    # 資料期間
    months = st.slider("歷史資料長度(月)", min_value=3, max_value=60, value=12)

    # 功能選擇
    # K线显示天数
    if "candle_days" not in st.session_state:
        st.session_state.candle_days = 30
    candle_days = st.session_state.candle_days
    mode = st.radio(
        "功能模式",
        ["📈 儀表板", "🔥 低價飆股", "📊 籌碼戰情室", "🌡️ 多空溫度計", "🔬 配對交易", "🔄 策略工坊", "🔍 掃描引擎", "🧠 智慧分析", "🧰 工具箱", "🕸️ 關係圖譜", "🌍 全球圖譜", "🎓 專業分析"],
        index=0,
    )

    refresh = st.button("🔄 重新整理資料")


# ============================================================
# 資料載入
# ============================================================
@st.cache_data(ttl=300)
def load_data(sid: str, m: int):
    df = fetch_historical(sid, months=m)
    if not df.empty:
        df = add_all_indicators(df)
    return df

data = load_data(stock_id, months)

if data.empty:
    st.error(f"❌ 無法取得股票 {stock_id} 的資料,請確認代號是否正確")
    st.stop()

# ============================================================
# 🔥 低價飆股 — 精準掃描即將起漲的低價潛力股
# ============================================================
if mode == "🔥 低價飆股":
    import sys as _sys
    if _sys.stdout.encoding.upper() in ("CP950", "BIG5"):
        _sys.stdout.reconfigure(encoding="utf-8")
    st.subheader("🔥 低價飆股掃描")
    st.caption("專注股價 < $100 的標的，找出爆量突破 + 技術共振 + 即將起漲的潛力股")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_n = st.number_input("顯示筆數", min_value=5, max_value=50, value=20, key="lp_top_n")
    with col2:
        min_grade = st.selectbox("最低等級", ["D", "C", "B", "A"], index=1, key="lp_grade")
    with col3:
        scan_btn = st.button("🚀 開始掃描", type="primary", use_container_width=True)

    grade_labels = {"A": "A級 強烈買入信號", "B": "B級 值得關注", "C": "C級 潛在觀察", "D": "D級 條件不足"}

    if scan_btn or "lp_results" in st.session_state:
        if scan_btn:
            with st.spinner("正在掃描 200+ 檔低價潛力股，約需 90-120 秒..."):
                candidates = scan_low_price_surge(top_n=top_n)
                st.session_state.lp_results = candidates
                st.session_state.lp_scan_time = __import__("datetime").datetime.now().strftime("%H:%M:%S")
                st.rerun()

        if "lp_results" in st.session_state and st.session_state.lp_results:
            candidates = st.session_state.lp_results
            scan_time = st.session_state.get("lp_scan_time", "")
            st.success(f"掃描完成！共篩選出 {len(candidates)} 檔潛力股（{scan_time}）")

            grade_counts = {g: sum(1 for c in candidates if c.grade == g) for g in ["A", "B", "C", "D"]}
            meta_cols = st.columns(4)
            for i, g in enumerate(["A", "B", "C", "D"]):
                meta_cols[i].metric(f"{g}級 {grade_labels[g]}", grade_counts[g])

            grade_filter = {"D": 0, "C": 60, "B": 75, "A": 90}
            min_score = grade_filter.get(min_grade, 0)
            filtered = [c for c in candidates if c.score >= min_score]

            if filtered:
                df = format_candidates_table(filtered)
                st.dataframe(df, use_container_width=True, hide_index=True,
                    column_config={
                        "短期目標": st.column_config.NumberColumn("短期目標", format="$%.2f"),
                        "中期目標": st.column_config.NumberColumn("中期目標", format="$%.2f"),
                        "波段目標": st.column_config.NumberColumn("波段目標", format="$%.2f"),
                        "漲幅%": st.column_config.NumberColumn("預估漲%", format="%.1f%%"),
                        "停損": st.column_config.NumberColumn("停損", format="$%.2f"),
                        "盈虧比": st.column_config.NumberColumn("盈虧比", format="%.1f"),
                    })

                st.subheader("詳細分析")
                for c in filtered[:5]:
                    conf_icon = "!" if c.confidence == "高" else "~" if c.confidence == "中" else "?"
                    with st.expander(f"[{c.grade}] {c.stock_name} ({c.stock_id}) — ${c.price} — 評分 {c.score} [{conf_icon} {c.confidence}]"):
                        # 價格預測
                        st.markdown("** 目標價位預測 **")
                        tcols = st.columns(5)
                        tcols[0].metric("短期(1-5d)", f"${c.target_short:.2f}", f"{c.upside_pct:+.1f}%")
                        tcols[1].metric("中期(1-2w)", f"${c.target_medium:.2f}")
                        tcols[2].metric("波段(1m)", f"${c.target_peak:.2f}")
                        tcols[3].metric("停損", f"${c.support_1:.2f}")
                        tcols[4].metric("盈虧比", f"{c.risk_reward:.1f}")

                        # 技術面
                        st.markdown("** 技術指標 **")
                        tcols2 = st.columns(4)
                        tcols2[0].metric("漲跌幅", f"{c.price_change_pct:+.1f}%")
                        tcols2[1].metric("量比", f"{c.volume_ratio:.1f}x")
                        tcols2[2].metric("RSI", f"{c.rsi_14:.0f}")
                        tcols2[3].metric("連紅", f"{c.consecutive_green}天")

                        # 阻力/支撐
                        st.markdown(f"** 阻力位: ** ${c.resistance_1:.2f} / ${c.resistance_2:.2f} / ${c.resistance_3:.2f}")

                        # 信號
                        if c.signals:
                            st.write("** 信號: **")
                            for sig in c.signals:
                                st.markdown(f"- {sig}")
            else:
                st.info("沒有符合所選等級的標的")
        else:
            st.info("點擊「🚀 開始掃描」按鈕開始分析")

    with st.expander("評分說明"):
        st.markdown(
            "評分維度 (滿分 100):\n"
            "- 量能異常 (最高 20 分): 今日成交量 > 20日均量 1.5倍為爆量\n"
            "- 價格動能 (最高 20 分): 單日漲幅越高分數越高\n"
            "- RSI 位置 (最高 15 分): 30-40 低檔即將翻多,55-70 強勢區\n"
            "- 技術突破 (最高 15 分): 均線多頭排列 + MACD 黃金交叉\n"
            "- 連續收紅 (最高 10 分): 連 N 日收紅 K\n"
            "- 籌碼面 (最高 10 分): 外資/投信買超 + 低檔盤整\n"
            "- 基本面 (最高 10 分): 營收成長 + 本益比合理\n\n"
            "等級分類:\n"
            "- A 級 (90+) 強烈買入信號\n"
            "- B 級 (75-89) 值得關注\n"
            "- C 級 (60-74) 潛在觀察\n"
            "- D 級 (<60) 條件不足"
        )





# ============================================================
# 📊 籌碼戰情室 — 三大法人 + OBV + 籌碼集中度分析
# ============================================================
if mode == "📊 籌碼戰情室":
    st.subheader("📊 籌碼戰情室")
    st.caption("法人買賣超 · OBV能量線 · 量價背離 · 融資融券 — 一頁看穿誰在買")

    # 輸入股票
    chip_sid = st.text_input("股票代號", value="2330", key="chip_sid",
                             placeholder="輸入台股代號，如 2330")
    chip_btn = st.button("🔍 分析籌碼", type="primary", key="chip_btn")

    if chip_btn or chip_sid:
        with st.spinner(f"正在分析 {chip_sid} 籌碼面..."):
            from data_fetcher import fetch_historical
            from fundamentals import fetch_institutional_trading
            import pandas as pd

            df = fetch_historical(chip_sid, months=3)

            if df.empty:
                st.error(f"無法取得 {chip_sid} 資料")
            else:
                result = run_chip_analysis(chip_sid, df, fetch_institutional=True)
                err = result.get("error")
                summary = result.get("summary_lines", [])

                # 綜合評分
                score = result.get("overall_score", 0)
                level = result.get("overall_level", "")
                st.metric("籌碼綜合評分", f"{score}/100", level[:8])

                if summary:
                    for line in summary:
                        st.markdown(line)

                with st.expander("📖 籌碼分析說明"):
                    st.markdown(""
                        "**OBV (On-Balance Volume):** 量能累積線，OBV 上升=買盤持續進場\n"
                        "**量價背離:** 股價跌但OBV在創高 = 主力偷偷吃貨\n"
                        "**法人籌碼:** 外資/投信/自營商近3日買賣超總和\n"
                        "**綜合評分:** OBV(30) + 背離(40) + 法人(30) = 0~100"
                    "")

# ============================================================
# 🌡️ 多空溫度計 — 大盤多空氣氛 + 市場週期 + 總經總覽
# ============================================================
if mode == "🌡️ 多空溫度計":
    st.subheader("🌡️ 大盤多空溫度計")
    st.caption("選擇權PC Ratio · 恐慌指數 · 市場週期 · 總體經濟 — 判斷現在該做多還做空")

    temp_tabs = st.tabs(["😱 恐慌/貪婪指數", "🔄 市場週期", "🌍 總經總覽"])

    with temp_tabs[0]:
        with st.spinner("分析選擇權數據中..."):
            opt_result = run_option_sentiment_analysis()
            opt_err = opt_result.get("error")

            if opt_err:
                st.warning(f"選擇權數據暫時無法取得: {opt_err}")

            # 恐慌貪婪指數
            fg = opt_result.get("fear_greed", {})
            fg_score = fg.get("fear_greed_index", 50) if fg else 50
            fg_label = fg.get("label", "中性") if fg else "中性"

            col1, col2, col3 = st.columns(3)
            col1.metric("恐慌/貪婪指數", f"{fg_score:.0f}", fg_label)
            col2.metric("PC Ratio", f"{opt_result.get('pc_ratio', 0):.2f}" if not opt_err else "N/A")
            col3.metric("波動率", f"{opt_result.get('volatility', 0):.1f}%" if not opt_err else "N/A")

            # 解讀
            if fg_score >= 75:
                st.warning("市場極度貪婪，注意高檔反轉風險")
            elif fg_score <= 25:
                st.success("市場極度恐慌，可能是低檔買點")
            elif fg_score >= 60:
                st.info("市場偏貪婪，謹慎追高")
            elif fg_score <= 40:
                st.info("市場偏恐慌，留意超跌機會")
            else:
                st.info("市場情緒中性")

            # 選擇權壓力/支撐
            oc = opt_result.get("option_clusters", {})
            max_call = oc.get("max_call", 0) if oc else 0
            max_put = oc.get("max_put", 0) if oc else 0
            if max_call or max_put:
                st.markdown("**選擇權最大未平倉:**")
                ccols = st.columns(2)
                if max_call:
                    ccols[0].metric("CALL 最大OI", f"${max_call}")
                if max_put:
                    ccols[1].metric("PUT 最大OI", f"${max_put}")

    with temp_tabs[1]:
        with st.spinner("分析市場週期中..."):
            from data_fetcher import fetch_historical
            twii = fetch_historical("^TWII", months=12)
            if not twii.empty:
                cycle_result = run_market_cycle_analysis(twii)
                markov = cycle_result.get("markov", {})
                kalman = cycle_result.get("kalman", {})

                col1, col2 = st.columns(2)
                if markov:
                    regime = markov.get("current_regime", "未知")
                    col1.metric("馬可夫狀態", regime)
                if kalman:
                    trend = kalman.get("current_trend", 0)
                    col2.metric("卡爾曼趨勢", f"{trend:+.2f}%")

                hessian = cycle_result.get("hessian", {})
                if hessian:
                    matrix = hessian.get("matrix_label", "")
                    st.info(f"市場矩陣: {matrix}")
                if not twii.empty:
                    from indicators import add_all_indicators
                    twii = add_all_indicators(twii)
                    rsi14 = twii["RSI_14"].iloc[-1] if "RSI_14" in twii.columns else 50
                    st.metric("大盤RSI(14)", f"{rsi14:.0f}")
            else:
                st.warning("無法取得大盤資料")

    with temp_tabs[2]:
        with st.spinner("抓取總經數據中..."):
            macro = get_macro_summary()
            if macro:
                cols = st.columns(3)
                if "美元/台幣" in macro:
                    cols[0].metric("美元/台幣", macro["美元/台幣"])
                if "美債10Y" in macro:
                    cols[1].metric("美債殖利率", macro["美債10Y"])
                if "台股期貨" in macro:
                    cols[2].metric("台指期", macro["台股期貨"])
                for k, v in macro.items():
                    if k not in ("美元/台幣", "美債10Y", "台股期貨"):
                        if isinstance(v, str):
                            cols[0 if k in macro else 0].markdown(f"**{k}:** {v}")
            else:
                st.warning("無法取得總經數據")

    with st.expander("🌡️ 溫度計使用說明"):
        st.markdown(""
            "**恐慌/貪婪指數:** 0-100，越低越恐慌（可能超跌），越高越貪婪（可能過熱）\n"
            "**PC Ratio:** Put/Call 未平倉比，越高代表避險需求越大\n"
            "**波動率:** 類似台版VIX，越高代表市場恐慌\n"
            "**馬可夫狀態:** bull/bear/recovery/overheat 四種市場狀態\n"
            "**卡爾曼趨勢:** 去噪音後的趨勢方向(%), 正=多頭, 負=空頭\n"
            "**市場矩陣:** 結合動能+波動率的 4象限定位"
        "")

# ============================================================
# 🔬 配對交易 — 統計套利 + 共整合配對掃描
# ============================================================
if mode == "🔬 配對交易":
    st.subheader("🔬 統計套利配對交易")
    st.caption("共整合檢定 · 價差回歸 · 同產業配對掃描 — 找到會回歸的股票對")
    import pandas as pd
    import numpy as np

    pair_tabs = st.tabs(["🎯 配對掃描", "📊 單配對分析", "📖 說明"])

    with pair_tabs[0]:
        st.markdown("**同產業配對掃描**")
        st.caption("輸入2~10檔同產業股票代號，找出最具共整合關係的配對")
        pair_input = st.text_area(
            "股票代號 (逗號/空格分隔)",
            value="2330, 2303, 2454, 2401, 3037",
            key="pair_list"
        )
        pair_scan_btn = st.button("🔍 掃描配對", type="primary", key="pair_scan")

        if pair_scan_btn:
            ids = [x.strip() for x in pair_input.replace(",", " ").split() if x.strip()]
            if len(ids) < 2:
                st.error("請至少輸入2檔股票代號")
            else:
                with st.spinner(f"掃描 {len(ids)} 檔股票的配對關係..."):
                    result = suggest_pair_trades(ids)
                    pairs = result if isinstance(result, list) else []

                    if pairs and "error" not in pairs[0]:
                        st.success(f"找到 {len(pairs)} 個配對")
                        data_list = []
                        for pair in pairs:
                            data_list.append({
                                "配對": f"{pair.get('stock_a', '')} - {pair.get('stock_b', '')}",
                                "共整合分數": pair.get("coint_score", 0),
                                "半衰期(天)": pair.get("half_life_hours", 0) / 24 if pair.get("half_life_hours") else "",
                                "目前價差(sigma)": f"{pair.get('zscore', 0):.2f}σ",
                                "建議": "做多A/做空B" if pair.get("zscore", 0) < -1 else "做多B/做空A" if pair.get("zscore", 0) > 1 else "觀望",
                            })
                        st.dataframe(pd.DataFrame(data_list).round(2), use_container_width=True)

                        for pair in pairs:
                            with st.expander(f"配對: {pair.get('stock_a', '')} vs {pair.get('stock_b', '')}", expanded=False):
                                st.markdown(f"**共整合分數:** {pair.get('coint_score', 0):.2f}")
                                st.markdown(f"**半衰期:** {pair.get('half_life_hours', 0)/24 if pair.get('half_life_hours') else 'N/A':.1f}天")
                                st.markdown(f"**目前價差:** {pair.get('zscore', 0):.2f}σ 離開平均值")
                                st.markdown(f"**建議策略:** {data_list[-1]['建議'] if data_list else '觀望'}")
                                if abs(pair.get('zscore', 0)) > 1.5:
                                    st.info(f"價差偏離平均 {pair.get('zscore', 0):.2f}σ，有回歸潛力")
                    else:
                        st.info("未找到具統計顯著的配對")

    with pair_tabs[1]:
        st.markdown("**單一配對共整合分析**")
        c1a, c1b = st.columns(2)
        with c1a:
            sid_a = st.text_input("股票A", value="2330", key="pair_a")
        with c1b:
            sid_b = st.text_input("股票B", value="2303", key="pair_b")

        if st.button("分析配對", key="pair_analyze"):
            with st.spinner(f"分析 {sid_a} vs {sid_b}..."):
                result = run_statistical_arbitrage(sid_a, sid_b)
                if result.get("pairs"):
                    p = result["pairs"][0]
                    st.metric("共整合分數", f"{p.get('coint_score', 0):.2f}")
                    st.metric("目前價差", f"{p.get('zscore', 0):.2f}σ")
                    hlf = p.get("half_life_hours", 0)
                    if hlf:
                        st.metric("回歸半衰期", f"{hlf/24:.1f}天")
                    beta = p.get("beta", 0)
                    if beta:
                        st.metric("避險比例 (β)", f"{beta:.2f}")

                    if abs(p.get('zscore', 0)) > 2:
                        st.success(f"價差偏離{abs(p.get('zscore', 0)):.1f}σ，強烈回歸信號")
                    elif abs(p.get('zscore', 0)) > 1:
                        st.info(f"價差偏離{abs(p.get('zscore', 0)):.1f}σ，開始關注")

                    # 信號
                    signals = p.get("signals", [])
                    if signals:
                        st.markdown("**信號:**")
                        for sig in signals:
                            st.markdown(f"- {sig}")
                else:
                    st.info("此配對無顯著共整合關係")

    with pair_tabs[2]:
        st.markdown(""
            "**什麼是統計套利配對交易?**\n"
            "\n"
            "兩檔同產業股票如果有長期的價格關係，當短期價差偏離時，"
            "押注它們會回歸均值。\n"
            "\n"
            "**關鍵指標:**\n"
            "- **共整合分數:** >0.7 表示有顯著的長期關係\n"
            "- **半衰期:** 價差偏離後平均幾天會回歸一半\n"
            "- **Z-Score:** 目前價差偏離平均幾個標準差，>2σ 或 <-2σ 為強信號\n"
            "\n"
            "**策略:**\n"
            "- Z-score > 2: 做空強勢股 + 做多弱勢股 (預期回歸)\n"
            "- Z-score < -2: 做多強勢股 + 做空空弱勢股\n"
            "- 持倉到 Z-score 回到 0 附近平倉\n"
        "")

# ============================================================
# 模式 1:看盤與技術分析
# ============================================================

# ============================================================
# 📈 儀表板
# ============================================================


# ============================================================
# 📈 儀表板
# ============================================================
if mode == "📈 儀表板":
    tab_labels = ['📊 即時看盤', '🔬 深度分析']
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.subheader("📈 即時看盤")
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.metric(f"{stock_name} ({stock_id})", f"${data['Close'].iloc[-1]:.2f}",
                     f"{(data['Close'].iloc[-1] - data['Close'].iloc[-2]):+.2f}")
            st.caption(f"最高:{data['High'].iloc[-1]:.2f} 最低:{data['Low'].iloc[-1]:.2f} 量:{data['Volume'].iloc[-1]:,.0f}")
        with col_r:
            chart_type = st.selectbox("圖表類型", ["K線+MA", "收盤價", "成交量"], label_visibility="collapsed")
        df = data.tail(candle_days * 20).copy()
        if chart_type == "K線+MA":
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                        low=df['Low'], close=df['Close'], name='K線'))
            for ma, col, w in [('MA5','orange',1),('MA10','blue',1),('MA20','green',1.5),('MA60','red',1.5)]:
                if ma in df.columns and not df[ma].isna().all():
                    fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines',
                                            name=ma, line=dict(color=col, width=w)))
            fig.update_layout(title=f"{stock_id} {stock_name} K線", height=500, template="plotly_white",
                             xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "收盤價":
            fig = go.Figure(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='收盤價'))
            for ma in ['MA5','MA10','MA20','MA60']:
                if ma in df.columns and not df[ma].isna().all():
                    fig.add_trace(go.Scatter(x=df.index, y=df[ma], mode='lines', name=ma))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = go.Figure(go.Bar(x=df.index, y=df['Volume'], name='成交量'))
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("📊 技術指標")
        tc = st.columns(4)
        if 'RSI' in data.columns: tc[0].metric("RSI", f"{data['RSI'].iloc[-1]:.1f}")
        if 'MACD' in data.columns: tc[1].metric("MACD", f"{data['MACD'].iloc[-1]:.3f}")
        if 'K' in data.columns: tc[2].metric("KDJ-K", f"{data['K'].iloc[-1]:.1f}")
        if 'BB_Upper' in data.columns:
            bb_pct = (data['Close'].iloc[-1] - data['BB_Lower'].iloc[-1]) / (data['BB_Upper'].iloc[-1] - data['BB_Lower'].iloc[-1]) * 100
            tc[3].metric("布林位置", f"{bb_pct:.1f}%")
        st.subheader("📋 近期行情")
        st.dataframe(data.tail(10)[['Open','High','Low','Close','Volume']].round(2), use_container_width=True)

    with tabs[1]:
            analysis_months = st.slider("分析期間(月)", 6, 36, 12, key="exp_months")
            col_start = st.columns([2, 1])
            with col_start[0]:
                run_btn = st.button("🔍 執行深度分析", type="primary", use_container_width=True)
            with col_start[1]:
                st.caption("分析約需 15~30 秒，耐心等候 ⏱️")

            _exp_all_data = {}

            if run_btn or "last_expert_result" in st.session_state:
                with st.spinner("🔍 正在執行各項分析..."):
                    if run_btn:
                        result = run_expert_analysis(stock_id, stock_name, months=analysis_months)
                        st.session_state["last_expert_result"] = result
                    else:
                        result = st.session_state["last_expert_result"]

                if "error" in result:
                    st.error(result["error"])
                else:
                    rec = result.get("recommendation", {})
                    if not rec:
                        from expert_analysis import expert_recommendation, comprehensive_technical_analysis, detect_abnormal_signals, analyze_news_and_market, analyze_industry_outlook
                        import pattern_recognition as pr
                        _df_exp = result.get("data", result.get("df", pd.DataFrame()))
                        tech = comprehensive_technical_analysis(_df_exp)
                        tech["patterns"] = pr.detect_all_patterns(_df_exp)
                        abnormal = detect_abnormal_signals(stock_id, _df_exp)
                        news = analyze_news_and_market(stock_id, stock_name)
                        industry = analyze_industry_outlook(stock_id, stock_name)
                        rec = expert_recommendation(stock_id, stock_name, _df_exp, tech, abnormal, news, industry)
                        result["recommendation"] = rec

                    rec = result.get("recommendation", {})
                    entry_exit = rec.get("entry_exit", {})
                    targets = entry_exit.get("targets", {})
                    stop_loss = entry_exit.get("stop_loss", 0)
                    current_price = result.get("current_price", 0) or \
                        (result.get("data", pd.DataFrame())["Close"].iloc[-1] if not result.get("data", pd.DataFrame()).empty else 0)
                    overall_score = rec.get("overall_score", 0)
                    rating = rec.get("rating", "中立")
                    rating_emoji = rec.get("rating_emoji", "⚖️")
                    rating_color = rec.get("rating_color", "#7f8c8d")
                    strategy = rec.get("strategy", "")
                    _exp_all_data = result

                    # 目標價小方塊
                    target_html_parts = []
                    for tidx in sorted(targets.keys()):
                        t = targets[tidx]
                        tp = t.get("price", 0)
                        gain = t.get("gain_pct", 0)
                        if gain >= 10:
                            tcolor = "#ff4444"
                        elif gain >= 5:
                            tcolor = "#ff8800"
                        elif gain > 0:
                            tcolor = "#44aa44"
                        else:
                            tcolor = "#888888"
                        target_html_parts.append(
                            f'<div style="text-align:center;background:{tcolor};'
                            f'border-radius:8px;padding:6px 10px;min-width:80px">'
                            f'<div style="font-size:11px;opacity:0.8;">T{tidx}</div>'
                            f'<div style="font-size:16px;font-weight:700;">{tp:.0f}</div>'
                            f'<div style="font-size:11px;">{gain:+.1f}%</div></div>'
                        )
                    target_html = "\n".join(target_html_parts)

                    industry_info = result.get("industry_analysis", {})
                    industry_name = industry_info.get("industry", "—") if isinstance(industry_info, dict) else "—"
                    industry_name_esc = html_escape(industry_name)

                    score_bar_pct = max(5, min(95, (overall_score + 100) / 2))
                    html_card = (
                        f'<div style="background:linear-gradient(135deg,{rating_color} 0%,{rating_color}dd 100%);'
                        f'border-radius:16px;padding:20px 24px;margin:10px 0 20px 0;'
                        f'box-shadow:0 4px 15px rgba(0,0,0,0.15);color:white;'
                        f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">'
                        f'<div><span style="font-size:13px;opacity:0.8;">綜合評分</span>'
                        f'<div style="font-size:44px;font-weight:800;line-height:1.1;">{overall_score:+.0f}</div></div>'
                        f'<div style="text-align:right;">'
                        f'<span style="font-size:13px;opacity:0.8;">操作建議</span>'
                        f'<div style="font-size:28px;font-weight:700;">{rating_emoji} {rating}</div>'
                        f'<div style="font-size:13px;opacity:0.9;">{html_escape(strategy)}</div></div></div>'
                        f'<div style="margin-top:10px;height:5px;background:rgba(255,255,255,0.3);border-radius:3px;">'
                        f'<div style="width:{score_bar_pct:.0f}%;height:100%;background:white;border-radius:3px;"></div></div>'
                        f'<div style="display:flex;justify-content:space-around;margin-top:14px;gap:8px;flex-wrap:wrap;">'
                        f'<div style="text-align:center;background:rgba(255,255,255,0.15);border-radius:10px;padding:8px 14px;min-width:90px;">'
                        f'<div style="font-size:11px;opacity:0.7;">現價</div>'
                        f'<div style="font-size:20px;font-weight:700;">{current_price:.2f}</div></div>'
                        + target_html +
                        (f'<div style="text-align:center;background:rgba(255,255,255,0.15);border-radius:10px;padding:8px 14px;min-width:80px;">'
                         f'<div style="font-size:11px;opacity:0.7;">停損</div>'
                         f'<div style="font-size:16px;font-weight:600;color:#ff8a80;">{stop_loss:.2f}</div></div>'
                         if stop_loss else '') +
                        f'</div>'
                        f'<div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">'
                        f'<span style="background:rgba(255,255,255,0.15);padding:3px 10px;border-radius:12px;'
                        f'font-size:12px;">🏭 {industry_name_esc}</span></div></div>'
                    )
                    st.components.v1.html(html_card, height=280)

                    # ── 分頁 ──
                    tabs = st.tabs(
                        ["📈 技術指標圖", "📋 技術評分解讀", "🏭 產業前景", "📰 新聞消息", "🔍 異常訊號", "📒 基本面速覽", "🔄 市場循環", "🧠 LSTM預測", "💰 籌碼分析", "📊 選擇權情緒", "📐 統計套利", "📚 參考資訊源"],
                        key="expert_tabs"
                    )

                    # Tab 1: K線+技術指標圖
                    with tabs[0]:
                        data_df = result.get("data", result.get("df", pd.DataFrame()))
                        if not data_df.empty:
                            st.subheader(f"{stock_id} {stock_name} K線圖 + MACD + RSI")
                            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                                vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
                            fig.add_trace(go.Candlestick(
                                x=data_df.index, open=data_df["Open"], high=data_df["High"],
                                low=data_df["Low"], close=data_df["Close"], name=stock_id), row=1, col=1)
                            for ma_col, ma_color, ma_name in [
                                ("MA20", "#9b59b6", "月線"),
                                ("MA60", "#3498db", "季線"),
                                ("MA120", "#2ecc71", "半年線"),
                            ]:
                                if ma_col in data_df.columns:
                                    fig.add_trace(go.Scatter(
                                        x=data_df.index, y=data_df[ma_col],
                                        line=dict(color=ma_color, width=1), name=ma_name), row=1, col=1)
                            if "MACD" in data_df.columns and "MACD_signal" in data_df.columns:
                                macd_colors = [
                                    "#26a69a" if v >= s else "#ef5350"
                                    for v, s in zip(data_df["MACD"], data_df["MACD_signal"])
                                ]
                                fig.add_trace(go.Bar(
                                    x=data_df.index, y=data_df["MACD_hist"],
                                    name="DIF-MACD", marker_color=macd_colors), row=2, col=1)
                                fig.add_trace(go.Scatter(
                                    x=data_df.index, y=data_df["MACD"],
                                    line=dict(color="#1565c0", width=1.5), name="DIF"), row=2, col=1)
                                fig.add_trace(go.Scatter(
                                    x=data_df.index, y=data_df["MACD_signal"],
                                    line=dict(color="#e65100", width=1.5), name="MACD"), row=2, col=1)
                            if "RSI" in data_df.columns:
                                fig.add_trace(go.Scatter(
                                    x=data_df.index, y=data_df["RSI"],
                                    line=dict(color="#7b1fa2", width=1.5), name="RSI"), row=3, col=1)
                                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
                            fig.update_layout(height=600, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified")
                            fig.update_xaxes(rangeslider_visible=False)
                            st.plotly_chart(fig, use_container_width=True)

                    # Tab 2: 技術評分解讀
                    with tabs[1]:
                        tech_info = result.get("technical", {})
                        st.subheader("📊 多週期技術分析")
                        if tech_info:
                            indicators = tech_info.get("indicators", {})
                            if indicators:
                                ncols = min(len(indicators), 4)
                                cols = st.columns(ncols)
                                for ci, (k, v) in enumerate(indicators.items()):
                                    with cols[ci % ncols]:
                                        if isinstance(v, (int, float)):
                                            st.metric(k, f"{v:.1f}")
                                        else:
                                            st.metric(k, str(v))
                            summary = tech_info.get("summary", [])
                            if summary:
                                with st.expander("📋 技術面摘要", expanded=True):
                                    for s in summary:
                                        st.markdown(s)

                            # 型態辨識結果
                            patterns_in = tech_info.get("patterns", [])
                            if isinstance(patterns_in, list) and patterns_in:
                                with st.expander("📐 型態辨識結果", expanded=False):
                                    for p in patterns_in:
                                        conf = p.get("confidence", "中")
                                        icon = {"高": "🟢", "中": "🟡", "低": "🔵"}.get(conf, "⚪")
                                        st.markdown(f"{icon} **{p.get('type', '未知')}** — 確信度: {conf}")
                        else:
                            st.info("技術分析數據尚未載入")

                    # Tab 3: 產業前景
                    with tabs[2]:
                        ind_info = result.get("industry_analysis", {})
                        if ind_info and isinstance(ind_info, dict) and not ind_info.get("error"):
                            st.info(f"🏭 **產業：{ind_info.get('industry', '—')}**")
                            outlook = ind_info.get("outlook_score", 0)
                            if outlook >= 5:
                                st.success(f"景氣展望評分：{outlook:+.0f} / +10 📈")
                            elif outlook >= 0:
                                st.info(f"景氣展望評分：{outlook:+.0f} / +10 ⚖️")
                            else:
                                st.warning(f"景氣展望評分：{outlook:+.0f} / +10 📉")
                            ind_news = ind_info.get("news", [])
                            if ind_news:
                                st.subheader("📰 產業新聞")
                                for art in ind_news[:10]:
                                    if isinstance(art, dict):
                                        st.markdown(f"- [{art.get('title', '')}]({art.get('url', '')})")
                                    else:
                                        st.markdown(f"- {art}")
                            ind_summary = ind_info.get("summary_lines", ind_info.get("summary", []))
                            if ind_summary:
                                with st.expander("📖 產業分析說明"):
                                    for l in ind_summary:
                                        st.markdown(l)
                        else:
                            st.info("產業前景分析暫無數據")

                    # Tab 4: 新聞消息
                    with tabs[3]:
                        news_info = result.get("news_analysis", {})
                        if news_info and isinstance(news_info, dict):
                            sentiment = news_info.get("sentiment_score", 0)
                            if sentiment >= 5:
                                st.success(f"📰 新聞情緒：{sentiment:+.1f}（偏多）")
                            elif sentiment <= -5:
                                st.warning(f"📰 新聞情緒：{sentiment:+.1f}（偏空）")
                            else:
                                st.info(f"📰 新聞情緒：{sentiment:+.1f}（中立）")
                            with st.expander("📰 近期相關新聞", expanded=True):
                                headlines = news_info.get("headlines", [])
                                if headlines:
                                    for h in headlines[:30]:
                                        st.markdown(f"• {h}")
                                else:
                                    # 備援：若 headline 是 dict 格式（含 title/url/source）
                                    articles = news_info.get("articles", [])
                                    for art in articles[:20]:
                                        if isinstance(art, dict):
                                            title = art.get("title", "")
                                            url = art.get("url", "")
                                            source = art.get("source", "")
                                            snt = art.get("sentiment", 0)
                                        else:
                                            title = str(art)
                                            url = ""
                                            source = ""
                                            snt = 0
                                        icon = "🟢" if snt > 0 else "🔴" if snt < 0 else "⚪"
                                        if url:
                                            st.markdown(f"{icon} [{title}]({url})  *({source})*")
                                        else:
                                            st.markdown(f"{icon} {title}  *({source})*")
                            summary = news_info.get("summary", [])
                            if summary:
                                with st.expander("📋 新聞分析摘要"):
                                    for l in summary:
                                        st.markdown(l)
                        else:
                            st.info("新聞分析尚未執行")

                    # Tab 5: 異常訊號
                    with tabs[4]:
                        ab_info = result.get("abnormal_signals", {})
                        if ab_info and isinstance(ab_info, dict):
                            risk_level = ab_info.get("risk_level", "低")
                            if risk_level == "高":
                                st.error(f"🔴 異常風險等級：{risk_level}")
                            elif risk_level == "中":
                                st.warning(f"🟡 異常風險等級：{risk_level}")
                            else:
                                st.success(f"🟢 異常風險等級：{risk_level}")
                            signals = ab_info.get("signals", [])
                            if signals:
                                st.subheader(f"🔍 偵測到 {len(signals)} 個異常訊號")
                                for sig in signals[:15]:
                                    if isinstance(sig, dict):
                                        sig_type = sig.get("type", "未知")
                                        level = sig.get("level", "")
                                        with st.expander(f"{'⚠️' if level == '高' else '📌'} {sig_type}", expanded=False):
                                            for k, v in sig.items():
                                                if k != "type":
                                                    st.text(f"{k}: {v}")
                                    else:
                                        st.markdown(f"- {sig}")
                            else:
                                st.info("✅ 未偵測到明顯異常訊號")
                        else:
                            st.info("異常訊號分析尚未執行")

                    # Tab 6: 基本面速覽
                    with tabs[5]:
                        fund_info = result.get("fundamentals", {})
                        if fund_info and isinstance(fund_info, dict) and "error" not in fund_info:
                            cols = st.columns(3)
                            cols[0].metric("本益比 (P/E)", f"{fund_info.get('pe_ratio', 'N/A')}")
                            cols[1].metric("殖利率", f"{fund_info.get('dividend_yield', 'N/A')}")
                            cols[2].metric("股價淨值比 (P/B)", f"{fund_info.get('pb_ratio', 'N/A')}")
                            st.caption(f"資料日期：{fund_info.get('report_season', 'N/A')}")
                            # MOPS 月營收
                            mops_rev = result.get("mops", {}).get("revenue", [])
                            if mops_rev:
                                with st.expander("📊 月營收趨勢"):
                                    rev_df = pd.DataFrame(mops_rev)
                                    if not rev_df.empty:
                                        st.dataframe(rev_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("基本面資料暫無法取得")

                    # Tab 7: 🔄 市場循環分析 (馬可夫 + Kalman + Hessian)
                    with tabs[6]:
                        st.subheader("🔄 市場循環狀態分析")
                        df_cycle = result.get("data", result.get("df", pd.DataFrame()))
                        if not df_cycle.empty and len(df_cycle) >= 30:
                            with st.spinner("🔄 計算市場循環狀態..."):
                                cycle_res = run_market_cycle_analysis(df_cycle)
                                if cycle_res.get("error"):
                                    st.warning(f"市場循環分析失敗: {cycle_res['error']}")
                                else:
                                    # ── Hessian 市場狀態（主卡） ──
                                    hess = cycle_res.get("hessian", {})
                                    if hess and hess.get("error") is None:
                                        st.subheader("🧩 Hessian 市場狀態矩陣")
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("市場狀態", hess.get("state_name", "?"))
                                        col2.metric("趨勢", hess.get("trend", "?"))
                                        col3.metric("波動", hess.get("volatility", "?"))
                                        
                                        with st.expander("📋 策略建議", expanded=True):
                                            st.markdown(f"**類型**: {hess.get('state_type', 'N/A')}")
                                            st.markdown(f"**策略**: {hess.get('strategy', 'N/A')}")
                                            st.markdown(f"**風險提示**: {hess.get('risk', 'N/A')}")
                                    
                                    # ── 馬可夫狀態 ──
                                    markov = cycle_res.get("markov", {})
                                    if markov:
                                        with st.expander("🔄 馬可夫狀態轉換分析", expanded=True):
                                            col1, col2 = st.columns(2)
                                            col1.metric("當前狀態", markov.get("current_label", "N/A"), f"持續 {markov.get('state_duration', 0)} 日")
                                            col2.metric("預測下一狀態", markov.get("next_label", "N/A"))
                                            
                                            st.markdown("**轉移機率:**")
                                            nprobs = markov.get("next_probs", {})
                                            for label, prob in nprobs.items():
                                                bar = "█" * int(prob / 10) + "░" * (10 - int(prob / 10))
                                                st.markdown(f"  {label}: {bar} {prob:.0f}%")
                                            
                                            st.markdown(f"**穩定性**: {markov.get('stability', 0):.1f}%")
                                            for s_label, s_stats in markov.get("state_stats", {}).items():
                                                labels = {0: "🟢 多頭", 1: "⚪ 盤整", 2: "🔴 空頭"}
                                                st.markdown(f"  {labels.get(s_label, '?')}: 佔 {s_stats.get('pct', 0):.1f}% 時間 | 平均日收益 {s_stats.get('avg_return', 0):+.4f}%")
                                    
                                    # ── Kalman Filter ──
                                    kalman = cycle_res.get("kalman", {})
                                    if kalman and kalman.get("error") is None:
                                        with st.expander("🎯 Kalman Filter 趨勢追蹤", expanded=False):
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric("趨勢方向", kalman.get("current_trend", "N/A"))
                                            col2.metric("強度", f"{kalman.get('trend_strength', 0)}/100")
                                            col3.metric("速度", f"{kalman.get('current_velocity', 0):+.6f}")
                                            if kalman.get("recent_turn"):
                                                t = kalman["recent_turn"]
                                                st.info(f"近期轉折: {t.get('direction', '?')} 在價格 {t.get('price', 0):.2f}")
                        else:
                            st.info(f"資料不足 (需≥30 筆)，市場循環分析無法執行")

                    # Tab 8: 🧠 LSTM 深度學習預測
                    with tabs[7]:
                        st.subheader("🧠 LSTM 深度學習價格預測")
                        df_lstm = result.get("data", result.get("df", pd.DataFrame()))
                        if not df_lstm.empty and len(df_lstm) >= 60:
                            with st.spinner("🔄 正在訓練 LSTM 模型（約需 10-20 秒）..."):
                                try:
                                    lstm_res = quick_lstm_forecast(df_lstm, forecast_days=10)
                                    if lstm_res.get("error"):
                                        st.warning(f"LSTM 預測失敗: {lstm_res['error']}")
                                    else:
                                        current_p = lstm_res["current_price"]
                                        target_p = lstm_res["target_price_5d"]
                                        change_pct = lstm_res["target_change_5d_pct"]
                                        direction_emoji = "📈" if change_pct > 0 else "📉"
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("當前價格", f"{current_p:.2f}")
                                        col2.metric("5日目標價", f"{target_p:.2f}", f"{change_pct:+.2f}%")
                                        col3.metric("方向準確率", lstm_res.get("direction_accuracy", "N/A"))
                                        col4.metric("MAPE 誤差", f"{lstm_res.get('mape', 0):.1f}%")

                                        # 預測走勢圖
                                        forecast_5d = lstm_res.get("forecast_5d", [])
                                        forecast_10d = lstm_res.get("forecast_10d", [])
                                        if forecast_5d:
                                            # 合併歷史尾巴 + 預測
                                            hist_tail = df_lstm['Close'].iloc[-20:].values.tolist()
                                            all_vals = hist_tail + forecast_5d
                                            fig = go.Figure()
                                            fig.add_trace(go.Scatter(
                                                y=hist_tail, mode='lines', name='歷史',
                                                line=dict(color='#3498db', width=2)))
                                            fig.add_trace(go.Scatter(
                                                y=all_vals[-6:], mode='lines+markers',
                                                name='5日預測', 
                                                line=dict(color='#e74c3c', width=2, dash='dash'),
                                                marker=dict(size=8)))
                                            if forecast_10d:
                                                all_vals_10 = hist_tail + forecast_10d
                                                fig.add_trace(go.Scatter(
                                                    y=all_vals_10[-11:], mode='lines+markers',
                                                    name='10日預測',
                                                    line=dict(color='#f39c12', width=2, dash='dot'),
                                                    marker=dict(size=6)))
                                            fig.update_layout(
                                                title="LSTM 價格預測走勢",
                                                height=350, margin=dict(l=20, r=20, t=40, b=20),
                                                hovermode="x unified",
                                                xaxis_title="交易日",
                                                yaxis_title="價格",
                                            )
                                            st.plotly_chart(fig, use_container_width=True)

                                        # 與線性模型對比
                                        with st.expander("📊 LSTM vs 線性回歸預測對比", expanded=False):
                                            comp = compare_forecast_methods(df_lstm)
                                            if comp.get("error") is None:
                                                st.json({
                                                    "LSTM 趨勢": comp.get("lstm_trend", "?"),
                                                    "線性趨勢": comp.get("linear_trend", "?"),
                                                    "平均偏差": f"{comp.get('average_deviation', 0):.2f}",
                                                    "方向一致": "✅ 是" if comp.get("direction_agreement") else ("❌ 否" if comp.get("direction_agreement") is False else "N/A"),
                                                })
                                                if comp.get("warning"):
                                                    st.info(comp["warning"])
                                            else:
                                                st.info(f"對比無法進行: {comp.get('error', '未知錯誤')}")
                                except Exception as e:
                                    st.warning(f"LSTM 預測異常: {str(e)}")
                        else:
                            st.info(f"資料不足 (需 ≥ 60 筆，目前 {len(df_lstm) if not df_lstm.empty else 0} 筆)，LSTM 預測無法執行")

                    # Tab 9: 💰 籌碼面 + OBV 分析
                    with tabs[8]:
                        st.subheader("💰 籌碼面深度分析")
                        df_chip = result.get("data", result.get("df", pd.DataFrame()))
                        if not df_chip.empty and len(df_chip) >= 30:
                            with st.spinner("🔄 分析籌碼面數據..."):
                                try:
                                    chip_res = run_chip_analysis(stock_id, df_chip)
                                    if chip_res.get("error"):
                                        st.warning(f"籌碼分析失敗: {chip_res['error']}")
                                    else:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            overall_lvl = chip_res.get("overall_level", "⚪ 待評估")
                                            overall_score = chip_res.get("overall_score", 50)
                                            st.metric("📊 籌碼面綜合評分", f"{overall_score}/100", overall_lvl[:4] if "(" in overall_lvl else overall_lvl)
                                        with col2:
                                            inst_res = chip_res.get("institutional_result", {})
                                            if inst_res and inst_res.get("error") is None:
                                                st.metric("🏢 法人籌碼評分", f"{inst_res.get('score', 50)}/100", inst_res.get('level', ''))

                                        obv_res = chip_res.get("obv_result", {})
                                        if obv_res and obv_res.get('error') is None:
                                            with st.expander("📊 OBV 能量潮分析", expanded=True):
                                                col1, col2 = st.columns(2)
                                                col1.metric("OBV 趨勢", obv_res.get("obv_trend", "N/A"))
                                                col2.metric("OBV 分數", f"{obv_res.get('score', 0):+.1f}")
                                                for sig in obv_res.get("signals", []):
                                                    st.markdown(f"- {sig}")
                                                for div in obv_res.get("divergences", []):
                                                    st.markdown(f"- {div['type']}: {div['detail']}")

                                        div_res = chip_res.get("divergence_result", {})
                                        if div_res and div_res.get('error') is None:
                                            with st.expander("📈 量價背離評分", expanded=True):
                                                st.metric("量價評分", f"{div_res.get('total_score', 50):.0f}/100", div_res.get('level', ''))
                                                for d in div_res.get("details", []):
                                                    pct = d["score"] / d["max"] * 100
                                                    bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                                                    st.markdown(f"{d['dimension']}: {bar} ({d['score']}/{d['max']})")

                                        inst_res = chip_res.get("institutional_result", {})
                                        if inst_res and inst_res.get("error") is None:
                                            with st.expander("🏢 三大法人動向", expanded=True):
                                                for sig in inst_res.get("signals", []):
                                                    st.markdown(f"- {sig}")
                                                cats = inst_res.get("categories", {})
                                                if cats:
                                                    st.dataframe(pd.DataFrame(cats).T, use_container_width=True)

                                        st.markdown("---")
                                        for line in chip_res.get("summary_lines", []):
                                            st.markdown(line)
                                except Exception as e:
                                    st.warning(f"籌碼分析異常: {str(e)}")
                        else:
                            st.info(f"資料不足 (需 ≥ 30 筆)，籌碼分析無法執行")

                    # Tab 10: 📊 選擇權情緒 + 恐慌貪婪
                    with tabs[9]:
                        st.subheader("📊 選擇權市場情緒 + 恐慌貪婪指數")
                        with st.spinner("🔄 分析選擇權市場與恐慌貪婪..."):
                            df_opt = result.get("data", result.get("df", pd.DataFrame()))
                            opt_price = float(df_opt['Close'].iloc[-1]) if not df_opt.empty else None
                            opt_res = run_option_sentiment_analysis(current_price=opt_price)
                            if opt_res.get("error"):
                                st.warning(f"選擇權分析失敗: {opt_res['error']}")
                            else:
                                # 恐慌貪婪主卡
                                fg = opt_res.get("fear_greed", {})
                                if fg:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        idx = fg.get("fear_greed_index", 50)
                                        level = fg.get("level", "⚪ 中性")
                                        st.metric("🎭 恐慌貪婪指數", f"{idx}/100", level)
                                    with col2:
                                        st.markdown(f"**💡 建議**: {fg.get('advice', 'N/A')}")
                                    
                                    with st.expander("📊 細項評分", expanded=True):
                                        for comp_name, comp_data in fg.get("components", {}).items():
                                            pct = comp_data["score"] / comp_data["max"] * 100
                                            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                                            st.markdown(f"**{comp_name}**: {bar} ({comp_data['score']}/{comp_data['max']}) {comp_data['label']}")

                                # Put/Call Ratio
                                pc = opt_res.get("pc_analysis", {})
                                if pc and pc.get("pc_ratio"):
                                    with st.expander("📊 Put/Call Ratio 選擇權未平倉", expanded=True):
                                        col1, col2 = st.columns(2)
                                        col1.metric("P/C Ratio", f"{pc.get('pc_ratio', 0):.3f}", pc.get('pc_signal', ''))
                                        col2.metric("波動率 (VIX近似)", f"{opt_res.get('vix_analysis', {}).get('vix_estimate', 0):.1f}%", opt_res.get('vix_analysis', {}).get('vix_signal', ''))
                                        
                                        st.markdown(f"**Call Max OI**: {pc.get('max_oi_call_strike', 0):.0f} (未平倉 {pc.get('max_oi_call_volume', 0):,})")
                                        st.markdown(f"**Put Max OI**: {pc.get('max_oi_put_strike', 0):.0f} (未平倉 {pc.get('max_oi_put_volume', 0):,})")
                                        oi_range = pc.get("oi_range_analysis", {})
                                        if oi_range:
                                            st.markdown(f"**🎯 最大OI集結區**: {oi_range.get('range_low', 0):.0f} ~ {oi_range.get('range_high', 0):.0f}")
                                            if opt_price:
                                                st.markdown(f"**📍 結算傾向**: {oi_range.get('settlement_bias', 'N/A')}")

                                # 選擇權支撐壓力
                                opt_levels = opt_res.get("option_levels", {})
                                if opt_levels:
                                    with st.expander("🛡️ 選擇權支撐壓力", expanded=False):
                                        if opt_levels.get("top_support"):
                                            s = opt_levels["top_support"][0]
                                            st.markdown(f"**支撐**: {s['strike']:.0f} (OI: {s['oi']:,})")
                                        if opt_levels.get("top_resistance"):
                                            r = opt_levels["top_resistance"][0]
                                            st.markdown(f"**壓力**: {r['strike']:.0f} (OI: {r['oi']:,})")

                    # Tab 11: 📐 統計套利 + 配對分析
                    with tabs[10]:
                        st.subheader("📐 統計套利與配對分析")
                        df_sa = result.get("data", result.get("df", pd.DataFrame()))
                        if not df_sa.empty:
                            with st.spinner("🔄 分析統計套利機會..."):
                                sa_res = run_statistical_arbitrage(stock_id, df_sa)
                                if sa_res.get("error"):
                                    st.warning(f"統計套利分析失敗: {sa_res['error']}")
                                else:
                                    # 與大盤相關性
                                    bm = sa_res.get("benchmark_analysis", {})
                                    if bm:
                                        col1, col2 = st.columns(2)
                                        col1.metric("📊 與大盤相關性", f"{bm.get('correlation_with_0050', 0):.3f}")
                                        col2.metric("Beta 係數", f"{bm.get('beta_to_0050', 0):.3f}")
                                        with st.expander("📋 Beta 解讀", expanded=False):
                                            beta = bm.get("beta_to_0050", 1)
                                            if beta > 1.5:
                                                st.markdown(f"β = {beta:.2f} → 高波動股，大盤漲1%該股漲{beta:.2f}%")
                                            elif beta > 0.8:
                                                st.markdown(f"β = {beta:.2f} → 與大盤同步")
                                            else:
                                                st.markdown(f"β = {beta:.2f} → 低波動防禦股")

                                    # 共整合配對
                                    pairs = sa_res.get("pairs", [])
                                    if pairs:
                                        with st.expander("🔄 共整合配對發現", expanded=True):
                                            for pair in pairs:
                                                other = pair["stock_b"] if pair["stock_a"] == stock_id else pair["stock_a"]
                                                z = pair.get("zscore", 0)
                                                coint_mark = "✅" if pair.get("is_cointegrated") else "🔶"
                                                st.markdown(f"{coint_mark} **{stock_id} ↔ {other}** | z-score: {z:.2f} | 相關: {pair.get('correlation', 0):.3f}")
                                                st.caption(pair.get('cointegration', {}).get('zscore_signal', ''))

                                    # 台積電 ADR (如果是2330)
                                    tsm = sa_res.get("tsm_adr", {})
                                    if tsm and tsm.get("premium_pct") is not None:
                                        with st.expander("🌎 台積電 ADR 溢價分析", expanded=True):
                                            col1, col2, col3 = st.columns(3)
                                            col1.metric("TSM ADR", f"${tsm.get('tsm_adr_price', 0):.2f}")
                                            col2.metric("台積電現貨", f"{tsm.get('tsm_spot_price', 0):.1f}")
                                            col3.metric("折溢價", f"{tsm.get('premium_pct', 0):+.2f}%")
                                            st.markdown(f"**隱含價格**: {tsm.get('implied_twd_price', 0):.1f}")
                                            st.markdown(f"**訊號**: {tsm.get('arbitrage_signal', 'N/A')}")
                                    
                                    for line in sa_res.get("summary_lines", []):
                                        st.markdown(line)
                        else:
                            st.info("需要歷史資料進行統計套利分析")

                    # Tab 12: 參考資訊源
                    with tabs[11]:
                        st.markdown("""
                        **📚 分析資料來源：**
                        - 📈 技術面 — TWSE/TPEx 歷史股價（均線、RSI、MACD、KDJ、布林通道）
                        - 🧠 LSTM — 基於 TensorFlow/Keras 的深度學習時間序列預測
                        - 💰 籌碼 — OBV (能量潮)、量價背離、三大法人買賣超
                        - 🔄 市場循環 — 馬可夫狀態轉換 + Kalman Filter 趨勢追蹤 + Hessian 市場狀態矩陣
                        - 📊 選擇權 — 期交所 TXO 未平倉分析 (Put/Call Ratio + 最大 OI)
                        - 📐 統計套利 — 共整合測試、價差回歸均值、ADR 溢價
                        - 📰 新聞情緒 — Yahoo奇摩股市、UDN、MoneyDJ、鉅亨網、Google News
                        - 🏭 產業前景 — 針對性產業新聞與景氣分析
                        - 🔍 異常訊號 — 爆量、跳空缺口、三大法人異常進出
                        - 📒 基本面 — TWSE 公開資訊（P/E、殖利率、P/B）
                        - 📐 型態辨識 — W底、M頭、頭肩頂/底、箱型突破等
                        """)
                        st.caption("⚠️ 本分析僅供參考，不構成投資建議")

                    # ── 詳細分析報告 ──
                    with st.expander("📋 完整分析報告", expanded=False):
                        for line in rec.get("detailed_reasoning", []):
                            st.markdown(line)


# ============================================================
# 🔄 策略工坊
# ============================================================
if mode == "🔄 策略工坊":
    tab_labels = ['📈 策略回測', '⚙️ 參數最佳化', '🎲 蒙地卡羅']
    tabs = st.tabs(tab_labels)

    with tabs[0]:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("策略參數")
                strategy = st.selectbox(
                    "選擇策略",
                    ["均線交叉", "RSI 策略", "MACD 策略", "布林通道"],
                    index=0,
                )
                initial_capital = st.number_input("初始資金", value=1_000_000, step=100_000, format="%d")

                if strategy == "均線交叉":
                    fast = st.slider("快線週期", 5, 30, 5)
                    slow = st.slider("慢線週期", 20, 120, 20)
                    run_btn = st.button("🚀 執行回測")
                elif strategy == "RSI 策略":
                    rsi_period = st.slider("RSI 週期", 5, 30, 14)
                    oversold = st.slider("超賣線", 10, 40, 30)
                    overbought = st.slider("超買線", 60, 90, 70)
                    run_btn = st.button("🚀 執行回測")
                elif strategy == "MACD 策略":
                    run_btn = st.button("🚀 執行回測")
                else:  # 布林
                    bb_period = st.slider("通道週期", 10, 50, 20)
                    bb_std = st.slider("標準差倍數", 1, 3, 2)
                    run_btn = st.button("🚀 執行回測")

            with col2:
                if "bt_result" not in st.session_state:
                    st.session_state.bt_result = None

                if run_btn or st.session_state.bt_result is not None:
                    if run_btn:
                        with st.spinner("回測計算中..."):
                            if strategy == "均線交叉":
                                result = backtest_ma_crossover(data, fast, slow, initial_capital)
                            elif strategy == "RSI 策略":
                                result = backtest_rsi(data, rsi_period, oversold, overbought, initial_capital)
                            elif strategy == "MACD 策略":
                                result = backtest_macd(data, initial_capital)
                            else:
                                result = backtest_bollinger(data, bb_period, bb_std, initial_capital)
                        st.session_state.bt_result = result
                    else:
                        result = st.session_state.bt_result

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # 績效摘要
                        st.subheader("📊 績效摘要")
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("總報酬率", f"{result['total_return_pct']:+.2f}%")
                        m2.metric("Buy & Hold", f"{result['buy_hold_return_pct']:+.2f}%")
                        m3.metric("最大回撤", f"{result['max_drawdown_pct']:.2f}%")
                        m4.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                        m5.metric("交易次數", result["num_trades"])

                        # 權益曲線
                        if not result["portfolio_values"].empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=result["portfolio_values"].index,
                                y=result["portfolio_values"].values,
                                name="策略權益",
                                line=dict(color="#2980b9", width=2),
                            ))
                            # Buy & Hold 比較
                            bh_values = initial_capital * (data["Close"] / data["Close"].iloc[0])
                            fig.add_trace(go.Scatter(
                                x=data["日期"],
                                y=bh_values,
                                name="Buy & Hold",
                                line=dict(color="#95a5a6", width=1, dash="dash"),
                            ))
                            fig.add_hline(y=initial_capital, line_dash="dot", line_color="gray")
                            fig.update_layout(
                                height=400,
                                template="plotly_white",
                                title="權益曲線 vs Buy & Hold",
                                yaxis_title="帳戶價值",
                                margin=dict(l=40, r=20, t=40, b=20),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # 交易明細
                        if not result["trades"].empty:
                            with st.expander("📋 交易明細", expanded=False):
                                trades_display = result["trades"].copy()
                                trades_display["日期"] = trades_display["日期"].dt.strftime("%Y-%m-%d")
                                trades_display["價格"] = trades_display["價格"].round(2)
                                trades_display["金額"] = trades_display["金額"].round(0).astype(int)
                                trades_display["剩餘現金"] = trades_display["剩餘現金"].round(0).astype(int)
                                st.dataframe(trades_display, use_container_width=True, hide_index=True)
                else:
                    st.info("👈 設定參數後點擊「執行回測」")




    with tabs[1]:
        st.info("內容載入中...")

    with tabs[2]:
        st.info("內容載入中...")


# ============================================================
# 🔍 掃描引擎
# ============================================================
if mode == "🔍 掃描引擎":
    tab_labels = ['🔎 多股掃描', '📡 選股篩選', '🏆 每日推薦', '🚀 飆股預測', '⚡ 當沖推薦']
    tabs = st.tabs(tab_labels)

    with tabs[0]:
            st.caption("掃描熱門股的最新技術指標狀態")

            selected_stocks = st.multiselect(
                "選擇要掃描的股票",
                options=list(POPULAR_STOCKS.keys()),
                default=list(POPULAR_STOCKS.keys())[:10],
                format_func=lambda x: f"{x} {POPULAR_STOCKS[x]}",
            )

            if st.button("🔍 開始掃描", type="primary"):
                if not selected_stocks:
                    st.warning("請至少選一支股票")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for i, sid in enumerate(selected_stocks):
                        status_text.text(f"掃描中:{sid} {POPULAR_STOCKS.get(sid, '')}")
                        df = fetch_historical(sid, months=6)
                        if not df.empty:
                            df = add_all_indicators(df)
                            signals = get_indicator_signals(df)
                            latest = df.iloc[-1]
                            price_change = latest["Close"] - df.iloc[-2]["Close"]
                            change_pct = price_change / df.iloc[-2]["Close"] * 100

                            results.append({
                                "代號": sid,
                                "名稱": POPULAR_STOCKS.get(sid, ""),
                                "收盤價": f"{latest['Close']:.2f}",
                                "漲跌幅": f"{change_pct:+.2f}%",
                                "RSI": signals.get("RSI", ""),
                                "MACD": signals.get("MACD", ""),
                                "均線": signals.get("均線", ""),
                                "布林": signals.get("布林", ""),
                            })
                        progress_bar.progress((i + 1) / len(selected_stocks))

                    status_text.empty()
                    if results:
                        result_df = pd.DataFrame(results)
                        st.dataframe(result_df, use_container_width=True, hide_index=True)

                        # 快速統計
                        st.subheader("📊 掃描摘要")
                        bullish_count = sum(1 for r in results if "🟢" in r.get("均線", ""))
                        bearish_count = sum(1 for r in results if "🔴" in r.get("均線", ""))
                        oversold_count = sum(1 for r in results if "🟢" in r.get("RSI", ""))
                        overbought_count = sum(1 for r in results if "🟡" in r.get("RSI", ""))

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("多方 (均線)", bullish_count)
                        c2.metric("空方 (均線)", bearish_count)
                        c3.metric("RSI 超賣", oversold_count)
                        c4.metric("RSI 超買", overbought_count)

                        csv = result_df.to_csv().encode("utf-8-sig")
                        st.download_button("⬇️ 下載掃描結果", csv, "stock_scan.csv", "text/csv")
                    else:
                        st.warning("掃描完成但無有效資料")





    with tabs[1]:
            st.caption("自訂條件篩選全部上市股票——價位、本益比、殖利率、成交量、技術訊號")

            with st.expander("⚙️ 篩選條件", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    price_range = st.slider("股價範圍", 0, 2000, (0, 500))
                    vol_min = st.number_input("最低成交量(張)", 0, 100000, 1000, step=500)
                with col2:
                    pe_range = st.slider("本益比範圍", 0, 200, (0, 50))
                    dy_range = st.slider("殖利率範圍(%)", 0, 20, (0, 15))
                with col3:
                    tech_filter = st.selectbox("技術訊號", ["全部", "多頭", "空頭", "中立"])
                    max_stocks = st.slider("最多掃描檔數", 50, 400, 150, step=50)

            if st.button("🔍 開始篩選", type="primary", use_container_width=True):
                with st.spinner(f"正在掃描 {max_stocks} 檔股票(約需 30-60 秒)..."):
                    df = filter_stocks(
                        price_min=price_range[0], price_max=price_range[1],
                        pe_min=pe_range[0], pe_max=pe_range[1],
                        dy_min=dy_range[0], dy_max=dy_range[1],
                        vol_min=vol_min,
                        tech_signal=tech_filter if tech_filter != "全部" else None,
                        max_stocks=max_stocks,
                    )
                if df.empty:
                    st.warning("沒有符合條件的股票，請將條件放寬")
                else:
                    st.success(f"🟢 找到 {len(df)} 檔符合條件的股票")
                    display = df[['sid', 'name', 'price', 'change_pct', 'volume',
                                'pe', 'div_yield', 'signal', 'rsi']].copy()
                    display.columns = ["代碼", "名稱", "價位", "變動(%)",
                                   "成交量", "PE", "殖利率(%)", "技術訊號", "RSI"]
                    display["價位"] = display["價位"].apply(lambda x: f"${x:,.2f}")
                    display["變動(%)"] = display["變動(%)"].apply(
                        lambda x: f"🟢 {x:+.2f}%" if x >= 0 else f"🔴 {x:+.2f}%")
                    display["成交量"] = display["成交量"].apply(
                        lambda x: f"{x:,}" if x >= 1000 else str(x))
                    display["PE"] = display["PE"].apply(
                        lambda x: str(round(x, 1)) if x and x > 0 else "-")
                    display["殖利率(%)"] = display["殖利率(%)"].apply(
                        lambda x: f"{x:.1f}%" if x and x > 0 else "-")
                    st.dataframe(display, use_container_width=True, hide_index=True)
                    csv = df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("⬇ 下載 CSV", csv, "stock_screener.csv",
                                      "text/csv", use_container_width=True)


    with tabs[2]:
            st.header("🏆 專家級每日推薦 - Top 5 精選")
            st.caption("基於技術面(35%) + 新聞情緒(20%) + 基本面(15%) + 籌碼面(15%) + 動能(10%) + 風險(5%) 多因子評分系統")

            # 參數設定
            with st.expander("⚙️ 掃描設定", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    scan_months = st.slider("分析週期(月)", 3, 12, 6, key="dp_months")
                with col2:
                    top_n = st.slider("推薦檔數", 3, 10, 5, key="dp_topn")
                with col3:
                    include_news = st.checkbox("🗞️ 新聞情緒分析", value=True,
                        help="從 Yahoo 奇摩股市抓取最新新聞進行關鍵字情緒分析")

                show_all_scores = st.checkbox("顯示所有評分細項", value=False)

            # 掃描範圍說明
            st.info(f"📋 掃描範圍:{len(SCAN_UNIVERSE)} 檔市場重點股(半導體、電子、金融、傳產、航運、ETF)│含新聞情緒分析")

            if st.button("🚀 開始掃描評分", type="primary", use_container_width=True):
                # 已移至 @st.cache_data show_spinner
                result = get_daily_picks_cached(top_n=top_n, months=scan_months, include_news=include_news)
                picks = result["picks"]
                market_note = result["market_note"]
                market_ctx = result.get("market_ctx", {})
                scan_time = result["timestamp"]

                if not picks:
                    st.error("⚠️ 掃描失敗,請稍後再試")
                else:
                    # 市場總評
                    avg = sum(p.total_score for p in picks) / len(picks)
                    candidates_count = result.get("candidates_count", 50)

                    # 大盤背景卡
                    if market_ctx:
                        trend_icon = market_ctx.get("trend_short", "⚪")
                        ma20 = market_ctx.get("ma20", 0)
                        ma60 = market_ctx.get("ma60", 0)
                        mkt_note = market_ctx.get("note", "")
                        mkt_ret = market_ctx.get("return_20d", 0)
                        ret_str = f"近月 {mkt_ret:+.1f}%" if mkt_ret else ""
                        st.markdown(
                            f"""
                            <div style="
                                background: linear-gradient(135deg, #0d2137 0%, #1a1a2e 100%);
                                border-radius: 12px;
                                padding: 14px 20px;
                                margin: 6px 0 18px 0;
                                border-left: 4px solid #e94560;
                            ">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="color: white; font-size: 16px; font-weight: 600;">
                                        🏛️ 大盤(0050){trend_icon}
                                    </span>
                                    <span style="color: rgba(255,255,255,0.5); font-size: 13px;">
                                        月線 {ma20:.0f} | 季線 {ma60:.0f} {ret_str}
                                    </span>
                                </div>
                                <div style="color: rgba(255,255,255,0.8); font-size: 14px; margin-top: 4px;">
                                    {mkt_note}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # 排行榜
                    for rank, pick in enumerate(picks, 1):
                        s = pick.total_score

                        # 獎牌
                        if rank == 1:
                            medal = "🥇"
                            border_color = "#ffd700"
                            bg_color = "rgba(255,215,0,0.08)"
                        elif rank == 2:
                            medal = "🥈"
                            border_color = "#c0c0c0"
                            bg_color = "rgba(192,192,192,0.08)"
                        elif rank == 3:
                            medal = "🥉"
                            border_color = "#cd7f32"
                            bg_color = "rgba(205,127,50,0.08)"
                        else:
                            medal = f"{rank}"
                            border_color = "#333"
                            bg_color = "transparent"

                        # 顏色
                        if s >= 70:
                            rating_color = "#00c853"
                        elif s >= 55:
                            rating_color = "#64dd17"
                        elif s >= 40:
                            rating_color = "#ffd600"
                        elif s >= 25:
                            rating_color = "#ff9100"
                        else:
                            rating_color = "#ff1744"

                        with st.container():
                            st.markdown(
                                f"""
                                <div style="
                                    border: 2px solid {border_color};
                                    border-radius: 16px;
                                    padding: 20px 24px;
                                    margin: 12px 0;
                                    background: {bg_color};
                                ">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div style="display: flex; align-items: center; gap: 12px;">
                                            <span style="font-size: 28px;">{medal}</span>
                                            <div>
                                                <span style="font-size: 22px; font-weight: 700;">{pick.stock_id}</span>
                                                <span style="font-size: 16px; color: #666; margin-left: 8px;">{pick.stock_name}</span>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 32px; font-weight: 800; color: {rating_color};">
                                                {s:.0f}
                                            </div>
                                            <div style="font-size: 13px; color: #888;">
                                                / 100 分
                                            </div>
                                        </div>
                                    </div>
                                    <div style="display: flex; gap: 20px; margin-top: 12px; flex-wrap: wrap;">
                                        <div style="flex: 1; min-width: 120px;">
                                            <div style="display: flex; align-items: center; gap: 8px;">
                                                <span style="font-size: 20px;">{pick.rating.split()[-1]}</span>
                                                <span style="font-size: 16px; font-weight: 600;">{pick.rating}</span>
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                                            <div style="text-align: center; min-width: 60px;">
                                                <div style="font-size: 11px; color: #888;">現價</div>
                                                <div style="font-size: 16px; font-weight: 700;">{pick.current_price:.1f}</div>
                                            </div>
                                            <div style="text-align: center; min-width: 60px;">
                                                <div style="font-size: 11px; color: #888;">漲跌</div>
                                                <div style="font-size: 16px; font-weight: 700; color: {'#e74c3c' if pick.change_pct >= 0 else '#2ecc71'};">{pick.change_pct:+.1f}%</div>
                                            </div>
                                            <div style="text-align: center; min-width: 70px;">
                                                <div style="font-size: 11px; color: #888;">進場</div>
                                                <div style="font-size: 14px; font-weight: 700;">{pick.entry_zone}</div>
                                            </div>
                                            <div style="text-align: center; min-width: 70px;">
                                                <div style="font-size: 11px; color: #888;">停損</div>
                                                <div style="font-size: 14px; font-weight: 700;">{pick.stop_loss:.1f}</div>
                                            </div>
                                        </div>
                                    </div>
                                    """,
                                unsafe_allow_html=True,
                            )

                            # 雷達圖 - 六項分數
                            cols = st.columns(6)
                            categories = ["技術", "新聞", "基本面", "籌碼", "動能", "風險"]
                            values = [
                                pick.tech_score, pick.news_score,
                                pick.fund_score, pick.inst_score,
                                pick.momentum_score, pick.risk_score
                            ]
                            max_vals = [35, 20, 15, 15, 10, 5]
                            pcts = [(v / m * 100) if m > 0 else 0 for v, m in zip(values, max_vals)]

                            for ci, (cat, pct, val, mval) in enumerate(zip(categories, pcts, values, max_vals)):
                                pct_clamped = max(0, min(100, pct))
                                bar_color = "#00c853" if pct_clamped >= 70 else "#ffd600" if pct_clamped >= 45 else "#ff1744"
                                cols[ci].markdown(
                                    f"""
                                    <div style="text-align: center;">
                                        <div style="font-size: 12px; color: #888; margin-bottom: 4px;">{cat}</div>
                                        <div style="
                                            width: 100%;
                                            height: 6px;
                                            background: #333;
                                            border-radius: 3px;
                                        ">
                                            <div style="
                                                width: {pct_clamped:.0f}%;
                                                height: 100%;
                                                background: {bar_color};
                                                border-radius: 3px;
                                            "></div>
                                        </div>
                                        <div style="font-size: 11px; color: #aaa; margin-top: 2px;">{val:.0f}/{mval}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            # 分析說明
                            if show_all_scores:
                                st.markdown("")
                                for line in pick.analysis:
                                    st.markdown(line)

                            st.markdown("</div>", unsafe_allow_html=True)

                        # 展開詳細訊號
                        with st.expander(f"📡 查看 {pick.stock_id} 詳細分析", expanded=(rank <= 1)):
                            tab1, tab2, tab3 = st.tabs(["📊 技術訊號", "🎯 進出場建議", "🗞️ 新聞"])

                            with tab1:
                                sig_cols = st.columns(4)
                                if pick.signals:
                                    for i, (k, v) in enumerate(pick.signals.items()):
                                        sig_cols[i % 4].metric(k, v)

                                st.markdown("**💡 分析摘要:**")
                                for line in pick.analysis:
                                    st.markdown(line)

                            with tab2:
                                er_col1, er_col2, er_col3 = st.columns(3)

                                # 進場建議
                                with er_col1:
                                    st.markdown(f"**{pick.entry_zone}**")
                                    st.markdown(pick.entry_note if pick.entry_note else "資料不足")
                                    if pick.ma20_price > 0:
                                        st.markdown(f"**月線 (MA20):** {pick.ma20_price:.1f}")
                                        st.markdown(f"**季線 (MA60):** {pick.ma60_price:.1f}")

                                # 停損停利
                                with er_col2:
                                    st.markdown("**🔴 停損價**")
                                    if pick.stop_loss > 0:
                                        pct = ((pick.stop_loss / pick.current_price) - 1) * 100
                                        st.markdown(f"**{pick.stop_loss:.1f}** ({pct:+.1f}%)")
                                        st.markdown(f"ATR: {pick.atr:.1f}")

                                    st.markdown("")
                                    st.markdown("**🎯 目標價**")
                                    if pick.target_1 > 0:
                                        tp1_pct = ((pick.target_1 / pick.current_price) - 1) * 100
                                        st.markdown(f"目標1 **{pick.target_1:.1f}** ({tp1_pct:+.1f}%)")
                                    if pick.target_2 > 0:
                                        tp2_pct = ((pick.target_2 / pick.current_price) - 1) * 100
                                        st.markdown(f"目標2 **{pick.target_2:.1f}** ({tp2_pct:+.1f}%)")

                                # 風報比 + 支撐壓力
                                with er_col3:
                                    st.markdown("**⚖️ 風報比**")
                                    if pick.risk_reward_ratio > 0:
                                        st.markdown(f"1 : **{pick.risk_reward_ratio:.1f}**")
                                        st.caption("(每虧1元賺多少)")

                                    st.markdown("")
                                    st.markdown("**📊 技術位置**")
                                    if pick.support_level > 0:
                                        sup_pct = ((pick.support_level / pick.current_price) - 1) * 100
                                        st.markdown(f"支撐: **{pick.support_level:.1f}** ({sup_pct:+.1f}%)")
                                    if pick.resistance_level > 0:
                                        res_pct = ((pick.resistance_level / pick.current_price) - 1) * 100
                                        st.markdown(f"壓力: **{pick.resistance_level:.1f}** ({res_pct:+.1f}%)")

                            with tab3:
                                if pick.news_headlines:
                                    st.markdown("**🗞️ 最新新聞:**")
                                    for h in pick.news_headlines:
                                        st.markdown(f"  • {h[:80]}")
                                else:
                                    st.info("無近期新聞資料")

                    # 掃描完成提示
                    st.success(f"✅ 掃描完成!推薦時效至今日收盤 ({scan_time}),建議在盤中/盤後評估")

                    # 免責
                    st.divider()
                    st.caption(
                        "⚠️ 本推薦系統基於多因子量化評分模型,僅供分析參考,不構成投資建議。"
                        "過往績效不代表未來表現,投資有賺有賠,請審慎評估。"
                    )
            else:
                # 顯示預覽
                st.info("👆 點擊「開始掃描評分」按鈕,系統將分析 50 檔重點股並推薦前 5 名")

                # 展示評分系統
                with st.expander("📖 評分方法說明", expanded=True):
                    st.markdown("""
                    ### 🧮 多因子評分模型(含新聞情緒)

                    | 面向 | 權重 | 評估項目 |
                    |------|------|----------|
                    | 📊 **技術面** | **35%** | 均線排列、RSI、MACD、布林通道、KDJ、成交量 |
                    | 🗞️ **新聞情緒** | **20%** | Yahoo 奇摩股市最新新聞關鍵字分析(正向/負向/熱度) |
                    | 📋 **基本面** | **15%** | 本益比、殖利率、股價淨值比 |
                    | 🏢 **籌碼面** | **15%** | 外資買賣超、投信買賣超、自營商買賣超 |
                    | 🚀 **動能力** | **10%** | 一週/一月漲跌幅、近期動能加速度 |
                    | 🛡️ **穩定度** | **5%** | 波動率、近期最大回撤 |

                    ### 🎯 評級標準

                    | 分數區間 | 評級 | 建議 |
                    |----------|------|------|
                    | 70-100 | ⭐ 強力推薦 | 多因子共振,優先考慮 |
                    | 55-69 | 📈 推薦買進 | 多項指標正面,可布局 |
                    | 40-54 | 👀 值得關注 | 部分指標轉佳,觀察時機 |
                    | 25-39 | ⚖️ 中立觀望 | 多空交錯,等待方向 |
                    | 0-24 | ❌ 暫時避開 | 指標偏弱,建議迴避 |

                    ### 🗞️ 新聞情緒分析方式

                    透過 Yahoo 奇摩股市擷取各股最新新聞標題,進行關鍵字比對:
                    - **正向關鍵字**:創高、突破、利多、成長、營收、受惠、AI、訂單...
                    - **負向關鍵字**:大跌、利空、衰退、虧損、賣壓、調降、裁員...

                    綜合正負向訊號數,標準化為 -10 ~ +10 分,納入總評分。

                    ### 📡 掃描範圍(50 檔重點股)

                    涵蓋半導體、電子代工、面板/PCB、金融、傳產龍頭、航運、ETF 等主要類股。

                    整個流程約需 30-60 秒,請耐心等候。 ⏱️
                    """)

                    st.info("💡 **專家提示:** 建議每日收盤後執行一次掃描,獲取最新推薦。多日連續上榜的股票代表多因子持續看好,可優先關注。")




    with tabs[3]:
            st.caption("六因子評分 — 爆量、突破、技術共振、新聞催化、動能加速、籌碼助攻")

            sp_tab1, sp_tab2 = st.tabs(["🔍 掃描全市場", "📈 單股分析"], key="sp_tabs")

            # ─── Tab 1: 掃描全市場 ───
            with sp_tab1:
                st.subheader("🔍 全市場飆股掃描")
                st.caption("掃描 200 檔重點股，兩階段篩選出飆股潛力候選")

                sp_top_n = st.slider("顯示前幾名", 5, 30, 10, key="sp_top_n")
                sp_months = st.slider("分析期間(月)", 3, 12, 6, key="sp_months")

                if st.button("🚀 開始掃描全市場", type="primary", use_container_width=True, key="sp_scan_btn"):
                    with st.spinner("🔍 第一階段：快速技術篩選 200 檔..."):
                        import time as _sp_t
                        _sp_start = _sp_t.time()
                        candidates = scan_surge_candidates(top_n=sp_top_n, months=sp_months)
                        _sp_elapsed = _sp_t.time() - _sp_start
                        st.session_state["sp_candidates"] = candidates
                        st.success(f"✅ 掃描完成！耗時 {_sp_elapsed:.0f} 秒，找到 {len(candidates)} 檔潛力股")

                if "sp_candidates" in st.session_state:
                    candidates = st.session_state["sp_candidates"]
                    if candidates:
                        rows = []
                        for i, c in enumerate(candidates):
                            rows.append({
                                "排名": i + 1,
                                "代號": c.stock_id,
                                "名稱": c.stock_name,
                                "總分": f"{c.surge_score:.0f}",
                                "爆量": f"{c.vol_score:.0f}/20",
                                "突破": f"{c.breakout_score:.0f}/20",
                                "技術": f"{c.tech_score:.0f}/20",
                                "新聞": f"{c.news_score:.0f}/15",
                                "動能": f"{c.momentum_score:.0f}/15",
                                "籌碼": f"{c.inst_score:.0f}/10",
                                "信心": c.conviction,
                                "現價": f"{c.current_price:.2f}" if c.current_price else "—",
                            })
                        df_sp = pd.DataFrame(rows)
                        st.dataframe(df_sp, use_container_width=True, hide_index=True)

                        st.subheader("🏆 Top 5 詳細分析")
                        for c in candidates[:5]:
                            expanded = c is candidates[0]
                            rank_icon = "🥇" if c is candidates[0] else "🥈" if c is candidates[1] else "🥉" if c is candidates[2] else "📌"
                            with st.expander(f"{rank_icon} {c.stock_name} ({c.stock_id}) — 總分 {c.surge_score:.0f}", expanded=expanded):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("現價", f"{c.current_price:.2f}" if c.current_price else "—")
                                    if hasattr(c, 'change_pct'):
                                        st.metric("漲跌幅", f"{c.change_pct:+.2f}%")
                                with col2:
                                    st.metric("信心指數", c.conviction)

                                factors = [
                                    ("爆量", c.vol_score, 20, "#ff6b6b"),
                                    ("突破", c.breakout_score, 20, "#ffa502"),
                                    ("技術", c.tech_score, 20, "#2ed573"),
                                    ("新聞", c.news_score, 15, "#1e90ff"),
                                    ("動能", c.momentum_score, 15, "#a29bfe"),
                                    ("籌碼", c.inst_score, 10, "#fd79a8"),
                                ]
                                for fname, fscore, fmax, _ in factors:
                                    pct = min(100, fscore / fmax * 100)
                                    st.markdown(f"**{fname}** {fscore:.0f}/{fmax}")
                                    st.progress(pct / 100)

                                sigs = getattr(c, 'surge_signals', [])
                                risks = getattr(c, 'risk_warnings', [])
                                col_left, col_right = st.columns(2)
                                with col_left:
                                    if sigs:
                                        st.success("✅ 正面訊號：" + "、".join(sigs[:5]))
                                with col_right:
                                    if risks:
                                        st.warning("⚠️ 風險：" + "、".join(risks[:5]))
                    else:
                        st.info("💭 未掃描到明顯的飆股候選")

            # ─── Tab 2: 單股分析 ───
            with sp_tab2:
                st.subheader("📈 單一股票飆股潛力分析")
                sp_sid = st.text_input("股票代碼", "2330", max_chars=6, key="sp_sid").strip()
                sp_sname = get_stock_name(sp_sid)
                st.caption(f"{sp_sid} {sp_sname}")
                sp_single_months = st.slider("分析期間(月)", 3, 12, 6, key="sp_single_months")

                if st.button("🔍 分析此股票", type="primary", use_container_width=True, key="sp_analyze_single"):
                    with st.spinner(f"正在分析 {sp_sid} {sp_sname}..."):
                        c = surge_score_stock_full(sp_sid, sp_sname, months=sp_single_months)
                        st.session_state["sp_single_result"] = c

                if "sp_single_result" in st.session_state:
                    c = st.session_state["sp_single_result"]
                    if c.error:
                        st.error(f"❌ {c.error}")
                    else:
                        cols = st.columns(6)
                        metrics_data = [
                            ("🏆 總分", f"{c.surge_score:.0f}/100"),
                            ("💵 現價", f"{c.current_price:.2f}" if c.current_price else "—"),
                            ("📊 漲跌", f"{c.change_pct:+.2f}%" if hasattr(c, 'change_pct') else "—"),
                            ("📈 量比", f"{c.volume_ratio:.2f}x" if hasattr(c, 'volume_ratio') else "—"),
                            ("🎯 突破", f"{c.high_break_pct:+.1f}%" if hasattr(c, 'high_break_pct') else "—"),
                            ("⚡ 信心", c.conviction),
                        ]
                        for j, (label, val) in enumerate(metrics_data):
                            with cols[j]:
                                st.metric(label, val)

                        st.subheader("📊 六因子評分")
                        factors = [
                            ("📈 爆量因子", c.vol_score, 20, "成交量異常放大"),
                            ("🚀 突破因子", c.breakout_score, 20, "價格突破型態"),
                            ("🔧 技術因子", c.tech_score, 20, "多指標技術共振"),
                            ("📰 新聞因子", c.news_score, 15, "新聞情緒催化"),
                            ("💨 動能因子", c.momentum_score, 15, "價格動能加速"),
                            ("🏦 籌碼因子", c.inst_score, 10, "法人籌碼助攻"),
                        ]
                        for fname, fscore, fmax, fdesc in factors:
                            with st.container():
                                cols_f = st.columns([2, 1])
                                with cols_f[0]:
                                    pct = min(100, fscore / fmax * 100)
                                    st.markdown(f"**{fname}** {fscore:.0f}/{fmax}")
                                    st.progress(pct / 100)
                                with cols_f[1]:
                                    st.caption(fdesc)

                        sigs = getattr(c, 'surge_signals', [])
                        risks = getattr(c, 'risk_warnings', [])
                        # ── 飆漲判斷原因（白話版）──
                        st.subheader("🔍 為什麼這支股票被注意？")
                        reasons = []

                        # 爆量判斷
                        vs = c.vol_score
                        if vs >= 15:
                            reasons.append(("🔥 爆量噴出", "成交量爆大量！這代表很多人在搶著買這支股票，市場資金正在湧入，有主力大戶在拉抬。如果之前沒什麼量、最近突然爆量，往往是起漲的訊號。"))
                        elif vs >= 10:
                            reasons.append(("📈 量能放大", "最近成交量比平常多很多，代表這支股票開始受到關注。可能是好消息要出來了，有人在提前佈局。"))
                        elif vs >= 5:
                            reasons.append(("👀 量能微增", "成交量有一點點增加，不過還不算太明顯。可以繼續觀察，看量會不會進一步放大再決定。"))

                        # 突破判斷
                        bs = c.breakout_score
                        if bs >= 15:
                            reasons.append(("🚀 突破關鍵價位", "股價突破了重要壓力區！就像水庫滿了之後洩洪一樣，一旦突破往往會有一段不錯的漲勢。現在進場有機會跟上這波行情。"))
                        elif bs >= 10:
                            reasons.append(("🎯 逼近壓力區", "股價已經漲到關鍵位置附近，正在測試能不能衝過去。如果能放量突破，就是不錯的進場點。"))
                        elif bs >= 5:
                            reasons.append(("📊 型態轉強", "技術型態有在慢慢轉好，股價從底部慢慢爬上來。不過還沒有正式突破，可以先放觀察名單。"))

                        # 技術共振判斷
                        ts = c.tech_score
                        if ts >= 15:
                            reasons.append(("⚡ 技術全面看多", "多個技術指標同時亮紅燈！RSI、MACD、均線都顯示多頭行情，這種多指標共振的情況下，上漲的勝率比較高。"))
                        elif ts >= 10:
                            reasons.append(("📐 技術偏多", "技術面看起來不錯，多數指標都偏向正面。不過還不到全面翻多的程度，可以小額試單。"))
                        elif ts >= 5:
                            reasons.append(("🔧 技術微偏多", "技術指標有一點點轉好的跡象，但還不是很明確。建議再觀望一下，等更多訊號確認。"))

                        # 新聞情緒判斷
                        ns = c.news_score
                        if ns >= 12:
                            reasons.append(("📰 新聞風向超好", "最近的新聞幾乎都是好消息！不管是公司的業績、還是產業的前景，媒體都在報好的一面。市場情緒偏向樂觀。"))
                        elif ns >= 8:
                            reasons.append(("🗞️ 新聞偏正面", "最近的新聞風向對這支股票有利，出來的消息大多是正面的。不過也要小心會不會是利多出盡。"))
                        elif ns >= 4:
                            reasons.append(("📋 新聞中性偏多", "新聞消息沒什麼大利空，偶爾有一些正面報導。雖然不是重大利多，但至少沒有壞消息來亂。"))

                        # 動能判斷
                        ms = c.momentum_score
                        if ms >= 12:
                            reasons.append(("💨 上漲加速度很強", "漲勢正在加速！就像車子油門踩到底一樣，最近幾天的漲幅一筆比一筆大，多方攻勢凌厲。"))
                        elif ms >= 8:
                            reasons.append(("🏃 動能持續增強", "股價上漲的力道在持續，不是曇花一現的那種。每天漲一點、穩穩往上的節奏，後續可能還會繼續。"))
                        elif ms >= 4:
                            reasons.append(("🚶 動能緩步向上", "雖然股價有在漲，但速度不快，屬於慢牛型。這種比較適合中長期持有，不適合短線追高。"))

                        # 籌碼判斷
                        ins = c.inst_score
                        if ins >= 8:
                            reasons.append(("🏦 法人偷偷在買", "外資和投信最近在默默吃貨！法人通常有專業的研究團隊，他們在買的股票通常有一定的依據，跟著法人走勝率較高。"))
                        elif ins >= 5:
                            reasons.append(("🏢 有法人關注", "三大法人對這支股票有興趣，買賣超偏向買方。雖然買得還不算多，但至少方向是對的。"))
                        elif ins >= 2:
                            reasons.append(("📊 籌碼略偏正面", "籌碼面有一點點好轉的跡象，不過法人的動作還不明顯，還需要再觀察。"))

                        # 綜合判斷
                        st.markdown("---")
                        st.markdown(f"**📝 總結：** {c.conviction}")
                        if c.surge_score >= 70:
                            st.success("💡 這檔股票多項指標都顯示強勁的飆漲潛力，建議列入重點觀察名單，但要嚴格設定停損。")
                        elif c.surge_score >= 55:
                            st.success("💡 這檔股票有不少正面訊號，短線有機會發動，可以小額試單並設好停損。")
                        elif c.surge_score >= 40:
                            st.info("💡 這檔股票有部分指標轉好，可以先放觀察清單，等更多確認訊號再進場。")
                        elif c.surge_score >= 25:
                            st.info("💡 這檔股票有一些初步的正面訊號，但還不夠強烈，建議再等一等。")
                        else:
                            st.warning("💡 這檔股票目前訊號混亂，不建議急著進場。")

                        # 逐項判斷原因（用卡片呈現）
                        st.markdown("---")
                        st.markdown("**📋 各項判斷原因：**")
                        if not reasons:
                            st.info("目前各項因子評分偏低，沒有明顯的飆漲訊號。")
                        for icon_title, desc in reasons:
                            icon = icon_title.split(" ")[0]
                            title = icon_title[len(icon)+1:]
                            st.markdown(
                                f"<div style='background:#1a1a2e;border:1px solid #333;border-radius:10px;"
                                f"padding:12px 16px;margin:6px 0;'>"
                                f"<b>{icon_title}</b><br>"
                                f"<span style='color:#ccc;font-size:14px;'>{desc}</span></div>",
                                unsafe_allow_html=True
                            )

                        st.markdown("---")

                        if sigs:
                            st.subheader("✅ 飆股訊號")
                            for s in sigs[:8]:
                                st.markdown(f"- ✅ {s}")
                        if risks:
                            st.subheader("⚠️ 風險警告")
                            for r in risks[:8]:
                                st.markdown(f"- ⚠️ {r}")

                        analysis_lines = getattr(c, 'analysis', [])
                        if analysis_lines:
                            with st.expander("📋 分析說明", expanded=False):
                                for l in analysis_lines:
                                    st.markdown(l)



    with tabs[4]:
        st.info("內容載入中...")


# ============================================================
# 🧠 智慧分析
# ============================================================
if mode == "🧠 智慧分析":
    tab_labels = ['📐 型態辨識', '🤖 AI 預測', '📊 風險管理', '🌍 總經分析', '🧠 專家分析']
    tabs = st.tabs(tab_labels)

    with tabs[0]:
            st.header("📐 技術型態辨識")
            st.caption("自動辨識型態——W底、M頭、頭肩頂/底、箱型突破")

            sid = st.text_input("股票代碼", "2330", max_chars=6).strip()
            sname = get_stock_name(sid)
            st.caption(f"{sid} {sname}")

            lookback_days = st.number_input(
                "分析期間(日)", 30, 365, 120, step=30)
            months = max(3, lookback_days // 30 + 1)
            if st.button("🔍 執行型態辨識", type="primary", use_container_width=True):
                with st.spinner("正在分析..."):
                    df = load_data(sid, months)
                    if df.empty:
                        st.error("無法取得資料")
                    else:
                        df = add_all_indicators(df)
                        patterns = pr.detect_all_patterns(df, lookback=lookback_days)
                        if not patterns:
                            st.info("💭 無辨識到明顯的技術型態")
                        else:
                            st.success(f"🟢 找到 {len(patterns)} 個型態")
                            for p in patterns:
                                conf = p.get("confidence", "中")
                                conf_icon = {"高": "🟢", "中": "🟡", "低": "🔵"}.get(conf, "⚪")
                                with st.expander(f"{conf_icon} {p["type"]}  (確信度:{conf})", expanded=True):
                                    info = " | ".join(
                                        f"**{k}:** {v}" for k, v in p.items()
                                        if k not in ("type", "confidence"))
                                    st.markdown(info)
                                    fig = go.Figure()
                                    fig.add_trace(go.Candlestick(
                                        x=df.index, open=df["Open"], high=df["High"],
                                        low=df["Low"], close=df["Close"], name=f"{sid} {sname}"))
                                    fig.update_layout(title=f"{sid} {sname} - {p["type"]}",
                                                      height=400, margin=dict(l=20, r=20, t=40, b=20))
                                    st.plotly_chart(fig, use_container_width=True)


    with tabs[1]:
        st.subheader("🤖 AI 多模型預測")
        st.caption("XGBoost / 隨機森林 / 邏輯回歸 多模型融合預測隔日漲跌方向")

        ai_tabs = st.tabs(["📈 XGBoost", "🧠 技術共識", "🔀 集成投票"])
        with ai_tabs[0]:
            with st.spinner("正在訓練 XGBoost 模型..."):
                features = prepare_features(data)
                xgb_result = train_xgboost(features)
            if "error" in xgb_result:
                st.error(f"❌ {xgb_result['error']}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("準確率", xgb_result["accuracy_pct"])
                c2.metric("訓練樣本", xgb_result["training_samples"])
                c3.metric("測試樣本", xgb_result["test_samples"])
                c4.metric("最新預測", xgb_result["latest_prediction"], f"機率 {xgb_result['latest_probability']:.0%}")
                st.info(f"📊 最新預測：**{xgb_result['latest_prediction']}**（信心度：{xgb_result['latest_probability']:.1%}）")
                pred_df = pd.DataFrame(xgb_result["recent_predictions"])
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

        with ai_tabs[1]:
            with st.spinner("計算技術共識中..."):
                consensus = technical_consensus(data)
            if "error" in consensus:
                st.error(f"❌ {consensus['error']}")
            else:
                st.metric("綜合分數", consensus["total_score"], delta=f"{consensus['signal_count']} 個指標投票")
                st.info(f"**結論：{consensus['conclusion']}**")
                for name, vote, weight in consensus.get("signals", []):
                    emoji = "🟢" if vote > 0 else ("🔴" if vote < 0 else "⚪")
                    st.markdown(f"{emoji} **{name}** — {'看漲' if vote>0 else '看跌' if vote<0 else '中立'}（權重:{weight}）")

        with ai_tabs[2]:
            with st.spinner("訓練多個模型投票..."):
                ensemble = ensemble_prediction(data)
            if "error" in ensemble:
                st.error(f"❌ {ensemble['error']}")
            else:
                st.subheader(f"🗳️ {ensemble['consensus']} ({ensemble['up_votes']}↑ / {ensemble['down_votes']}↓)")
                for mn, mr in ensemble.get("models", {}).items():
                    with st.expander(f"📦 {mn}（{mr.get('accuracy',0):.1%}）"):
                        k = "top_features" if "top_features" in mr else "top_coefficients"
                        if k in mr:
                            st.dataframe(pd.DataFrame(mr[k]), use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("📊 風險管理與資金控管")
        st.caption("凱利公式、VaR、夏普比率、部位計算")
        close_series = data["Close"]
        returns = close_series.pct_change().dropna().values
        equity_curve = close_series.copy()

        rm_tabs = st.tabs(["📊 風險報告", "💰 凱利公式", "📐 部位計算", "📉 VaR"])
        with rm_tabs[0]:
            with st.spinner("計算風險指標..."):
                report = full_risk_report(returns, equity_curve)
            cols = st.columns(3)
            cols[0].metric("夏普比率", f"{report['sharpe_ratio']:.2f}")
            cols[1].metric("索提諾比率", f"{report['sortino_ratio']:.2f}")
            cols[2].metric("卡瑪比率", f"{report['calmar_ratio']:.2f}")
            cols = st.columns(3)
            cols[0].metric("年化報酬率", f"{report['annual_return']:.2f}%")
            cols[1].metric("年化波動率", f"{report['annual_volatility']:.2f}%")
            cols[2].metric("勝率", f"{report['win_rate']:.1%}")
            with st.expander("📉 最大回撤"):
                dd = report['max_drawdown']
                st.metric("最大回撤", f"{dd['max_drawdown_pct']:.2f}%")
                st.metric("平均持續天數", f"{dd['avg_drawdown_duration_days']:.1f}天")
                st.metric("回撤次數", dd['drawdown_count'])

        with rm_tabs[1]:
            wr = st.number_input("勝率(%)", 0.0, 100.0, 55.0, 0.5) / 100
            aw = st.number_input("平均獲利", 0.0, 100.0, 5.0, 0.1)
            al = st.number_input("平均虧損", 0.1, 100.0, 3.0, 0.1)
            kelly = kelly_criterion(wr, aw, al)
            if "error" not in kelly:
                st.info(kelly["suggestion"])
                kc = st.columns(3)
                kc[0].metric("完整凱利", f"{kelly['full_kelly']:.1%}")
                kc[1].metric("半凱利", f"{kelly['half_kelly']:.1%}")
                kc[2].metric("¼凱利", f"{kelly['quarter_kelly']:.1%}")

        with rm_tabs[2]:
            pv = st.number_input("帳戶資金", 100000, 10000000, 1000000, 50000)
            rp = st.slider("每筆風險(%)", 0.5, 5.0, 2.0, 0.5) / 100
            ep = st.number_input("進場價", 1.0, 10000.0, float(close_series.iloc[-1]), 0.5)
            sl = st.number_input("停損價", 1.0, 10000.0, float(close_series.iloc[-1] * 0.95), 0.5)
            if st.button("📐 計算部位"):
                pos = calculate_position_size(pv, rp, ep, sl)
                if "error" not in pos:
                    pc = st.columns(4)
                    pc[0].metric("建議張數", pos['shares'])
                    pc[1].metric("投入資金", f"${pos['position_cost']:,}")
                    pc[2].metric("佔比", f"{pos['portfolio_pct']:.1f}%")
                    pc[3].metric("風險額", f"${pos['risk_amount']}")

        with rm_tabs[3]:
            conf = st.select_slider("信心水準", [0.90, 0.95, 0.99], 0.95)
            var_result = calculate_var(returns, conf, "all")
            if "error" not in var_result:
                vc = st.columns(3)
                for i, m in enumerate(["historical", "parametric", "monte_carlo"]):
                    k = f"{m}_var_pct"
                    if k in var_result:
                        vc[i].metric(f"{m.capitalize()} VaR", var_result[k])
                if "cvar_pct" in var_result:
                    st.metric("CVaR", var_result["cvar_pct"])
                st.info(f"以目前股價 ${float(close_series.iloc[-1]):.2f} 計算：")
                for m in ["historical", "parametric", "monte_carlo"]:
                    k = f"{m}_var"
                    if k in var_result:
                        lp = float(close_series.iloc[-1]) * (1 + var_result[k])
                        st.write(f"• {m.capitalize()}：極端跌至 **${lp:.2f}**")

    with tabs[3]:
        st.subheader("🌍 總體經濟概覽")
        st.caption("匯率、利率、美國指數連動")
        with st.spinner("取得總經數據..."):
            macro = get_macro_summary()
        st.subheader("💱 匯率")
        fx = macro.get("currency", {})
        if "error" not in fx:
            fxc = st.columns(3)
            if fx.get("rate"):
                fxc[0].metric("USD/TWD", f"{fx['rate']:.4f}")
            else:
                fxc[0].metric("即期買入", f"{fx.get('spot_buy',0):.4f}")
                fxc[1].metric("即期賣出", f"{fx.get('spot_sell',0):.4f}")
                fxc[2].metric("現金買入", f"{fx.get('cash_buy',0):.4f}")
        st.subheader("🏦 利率")
        rc = st.columns(2)
        fed = macro.get("us_interest_rate", {})
        if "error" not in fed:
            rc[0].metric("聯邦利率", f"{fed.get('rate','N/A')}%")
        twr = macro.get("tw_interest_rate", {})
        if "error" not in twr:
            rc[1].metric("台灣重貼現率", f"{twr.get('rate','N/A')}%")
        st.subheader("🇺🇸 美國指數")
        usm = macro.get("us_market", {})
        if "error" not in usm and usm:
            usc = st.columns(len(usm))
            for i, (n, d) in enumerate(usm.items()):
                usc[i].metric(n, f"{d['price']:,}", f"{d['change_pct']:+.2f}%")


# ============================================================
# 🧰 工具箱
# ============================================================
if mode == "🧰 工具箱":
    tab_labels = ['🎮 虛擬交易', '📊 ETF 分析', '🔔 價格警示']
    tabs = st.tabs(tab_labels)

    with tabs[0]:
            st.caption("用虛擬資金 $1,000,000 練習台股買賣,即時報價,真實手續費")

            # ── 自動存檔（瀏覽器 localStorage）──
            _vt_key = "streamlit_vt_portfolio"

            # 自動還原：首次載入時從瀏覽器讀取（必須在存檔之前執行）
            if "vt_restored" not in st.session_state:
                st.session_state.vt_restored = False
            if not st.session_state.vt_restored:
                saved = st.query_params.get("__vt_restore")
                if saved:
                    try:
                        data = json.loads(saved)
                        if data.get("orders") and len(data["orders"]) > len(st.session_state.vt_portfolio.get("orders", [])):
                            st.session_state.vt_portfolio = data
                        st.session_state.vt_restored = True
                        st.query_params.clear()
                        st.rerun()
                    except Exception:
                        st.session_state.vt_restored = True
                else:
                    st.components.v1.html(
                        f"""<script>try{{
        var d=localStorage.getItem('{_vt_key}');
        if(d&&d.length>20){{var u=new URL(window.location.href);u.searchParams.set('__vt_restore',encodeURIComponent(d));window.location.replace(u.toString());}}
        }}catch(e){{}}</script>""",
                        height=0,
                    )
                    st.session_state.vt_restored = True

            # 自動儲存：還原完成後才寫入 localStorage（避免用空白狀態覆蓋有效資料）
            if st.session_state.vt_restored:
                st.components.v1.html(
                    f"""<script>try{{
        localStorage.setItem('{_vt_key}',JSON.stringify({json.dumps(st.session_state.vt_portfolio, ensure_ascii=False)}));
        }}catch(e){{}}</script>""",
                    height=0,
                )

            # 快捷買入(從每日推薦跳轉用)
            if "quick_buy" in st.session_state and st.session_state.quick_buy:
                pass  # 由後續邏輯處理

            # 分頁
            vtab1, vtab2, vtab3, vtab4 = st.tabs(["💼 資產總覽", "💰 買入股票", "📂 庫存明細", "📜 交易紀錄"], key="vt_tabs")

            # ─── Tab 1: 資產總覽 ───
            with vtab1:
                summary = get_portfolio_summary(st.session_state.vt_portfolio)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💵 可用現金", f"${summary['cash']:,.0f}")
                col2.metric("📦 持股市值", f"${summary['holdings_value']:,.0f}")
                col3.metric("🏦 總資產", f"${summary['total_value']:,.0f}")
                pnl_color = "inverse" if summary['overall_pnl'] >= 0 else "off"
                col4.metric(
                    "📈 總損益",
                    f"${summary['overall_pnl']:+,.0f}",
                    f"{summary['overall_pnl_pct']:+.2f}%",
                    delta_color=pnl_color,
                )

                # 狀態
                st.markdown("")
                col1, col2, col3 = st.columns(3)
                col1.info(f"🟢 已實現損益:${summary['realized_pnl']:+,.0f}")
                col2.info(f"🟡 未實現損益:${summary['unrealized_pnl']:+,.0f}")
                col3.info(f"📊 持有 {summary['holdings_count']} 檔 | 總交易 {summary['order_count']} 筆")

                st.markdown("")

                # 操作提示
                st.markdown("""
                ---
                **🎯 新手操作建議:**
                1. 先去 **買入股票** 分頁模擬買股
                2. 在 **庫存明細** 查看即時損益
                3. 到 **交易紀錄** 回顧自己的操作

                💡 市場只有 $$100 萬本金,控制好每筆資金別 All-in!
                """)

                with st.expander("⚙️ 進階操作", expanded=False):
                    # 匯出 / 匯入
                    col_s, col_l, col_r = st.columns(3)
                    with col_s:
                        portfolio_json = json.dumps(
                            st.session_state.vt_portfolio, ensure_ascii=False, indent=2
                        )
                        st.download_button(
                            "📥 匯出資產",
                            data=portfolio_json,
                            file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    with col_l:
                        uploaded_file = st.file_uploader(
                            "📤 匯入資產", type=["json"], label_visibility="collapsed",
                            key="portfolio_upload",
                        )
                        if uploaded_file:
                            try:
                                data = json.load(uploaded_file)
                                # 基本驗證
                                if "cash" in data and "holdings" in data and "orders" in data:
                                    st.session_state.vt_portfolio = data
                                    st.success("✅ 資產已從檔案載入!")
                                    st.rerun()
                                else:
                                    st.error("❌ 檔案格式錯誤,缺少必要欄位")
                            except Exception as e:
                                st.error(f"❌ 載入失敗:{e}")
                    with col_r:
                        st.warning("⚠️ 重設後所有交易紀錄和庫存將被清除!")
                        if st.button("🔄 重設資產組合", type="secondary", use_container_width=True):
                            result = reset_portfolio(st.session_state.vt_portfolio)
                            if "portfolio" in result:
                                st.session_state.vt_portfolio = result["portfolio"]
                            st.success(result["message"])
                            st.rerun()

                    st.info(
                        "💡 **提示:** 資料存在瀏覽器中(Session),關閉頁面會消失。"
                        "建議定期點「匯出資產」下載備份。下次使用時點「匯入資產」還原即可。"
                    )

            # ─── Tab 2: 買入股票 ───
            with vtab2:
                st.subheader("💰 買入股票")

                # 快速選擇熱門股
                quick_stocks = ["2330", "2317", "2454", "0050", "2881", "2412", "2308", "2002"]
                st.caption("快速選擇:")
                quick_cols = st.columns(len(quick_stocks))
                for i, sid in enumerate(quick_stocks):
                    sname = get_stock_name(sid)
                    if quick_cols[i].button(f"{sname}", key=f"qs_{sid}", use_container_width=True):
                        st.session_state["buy_stock_id"] = sid

                # 自動帶入上次輸入的股號
                default_sid = st.session_state.get("buy_stock_id", "2330")

                col1, col2 = st.columns([2, 1])

                with col1:
                    buy_sid = st.text_input("股票代號", value=default_sid,
                        key="buy_input", placeholder="例如 2330").strip()

                    buy_shares = st.number_input("股數", min_value=1, max_value=100000, value=1000, step=100,
                        help="一般交易單位為 1000 股(1張),零股最小 1 股")

                    # 取得即時報價
                    if buy_sid:
                        with st.spinner("查詢報價中..."):
                            quote = fetch_realtime_quote(buy_sid)

                        if "error" not in quote:
                            price = quote["price"]
                            sname = quote.get("name", buy_sid)
                            st.session_state["buy_stock_id"] = buy_sid

                            total_cost = price * buy_shares
                            fee = max(total_cost * 0.001425, 20)
                            total_paid = total_cost + fee

                            st.info(
                                f"**{sname}** ({buy_sid})\n\n"
                                f"📊 現價:**${price:.2f}** {quote.get('change_percent', 0):+.2f}%\n"
                                f"📈 開 {quote.get('open', 0):.2f} 高 {quote.get('high', 0):.2f} 低 {quote.get('low', 0):.2f}\n\n"
                                f"**試算:**\n"
                                f"買進 {buy_shares:,} 股 × ${price:.2f} = ${total_cost:,.0f}\n"
                                f"手續費 ${fee:.0f}\n"
                                f"**需付總額:${total_paid:,.0f}**"
                            )

                            cash = st.session_state.vt_portfolio["cash"]
                            if total_paid > cash:
                                st.error(f"❌ 餘額不足!可用 ${cash:,.0f},需 ${total_paid:,.0f}")
                            else:
                                if st.button("✅ 確認買入", type="primary", use_container_width=True):
                                    result = buy_stock(buy_sid, buy_shares, portfolio=st.session_state.vt_portfolio)
                                    if result["success"]:
                                        if "portfolio" in result:
                                            st.session_state.vt_portfolio = result["portfolio"]
                                        st.success(result["message"])
                                        st.balloons()
                                        st.rerun()
                                    else:
                                        st.error(result["message"])
                        else:
                            st.warning(f"⚠️ {quote.get('error', '無法取得報價')}")

                with col2:
                    st.markdown("**💵 可用資金**")
                    cash = st.session_state.vt_portfolio["cash"]
                    st.markdown(f"<h2 style='color: #4CAF50;'>${cash:,.0f}</h2>", unsafe_allow_html=True)

                    st.markdown("")
                    st.markdown("**💡 操作提示**")
                    st.markdown("""
                    - 輸入股號後自動顯示報價
                    - 一般交易單位 = 1,000 股
                    - 低於 $20 元可買零股
                    - 手續費 0.1425%(最低 $20)
                    """)

                    st.markdown("")
                    st.markdown("**🔄 也可以賣出...**")
                    st.info("切換到「庫存明細」分頁即可賣出持股")

            # ─── Tab 3: 庫存明細 ───
            with vtab3:
                st.subheader("📂 庫存明細")

                holdings = get_holdings_with_prices(st.session_state.vt_portfolio)

                if not holdings:
                    st.info("📭 尚無庫存,快去買入第一檔股票吧!")
                else:
                    # 總計
                    total_market = sum(h["market_value"] for h in holdings)
                    total_cost_h = sum(h["total_cost"] for h in holdings)
                    total_pnl = total_market - total_cost_h
                    total_pnl_pct = (total_market / total_cost_h - 1) * 100 if total_cost_h > 0 else 0

                    c1, c2, c3 = st.columns(3)
                    c1.metric("持股市值", f"${total_market:,.0f}")
                    c2.metric("投入成本", f"${total_cost_h:,.0f}")
                    pnl_color = "normal" if total_pnl >= 0 else "inverse"
                    c3.metric("未實現損益", f"${total_pnl:+,.0f}", f"{total_pnl_pct:+.2f}%", delta_color=pnl_color)

                    st.markdown("---")

                    for h in holdings:
                        expand = st.expander(
                            f"**{h['stock_name']}** ({h['stock_id']})"
                            f" | {h['shares']:,} 股"
                            f" | 成本 ${h['avg_cost']:.1f}"
                            f" | 現價 ${h['current_price']:.1f}"
                            f" | 損益 ${h['unrealized_pnl']:+,.0f} ({h['unrealized_pnl_pct']:+.2f}%)",
                            expanded=False,
                        )

                        with expand:
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("持有股數", f"{h['shares']:,}")
                            c2.metric("平均成本", f"${h['avg_cost']:.2f}")
                            c3.metric("現價", f"${h['current_price']:.2f}", f"{h['day_change_pct']:+.2f}%")
                            c4.metric("市值", f"${h['market_value']:,.0f}")

                            st.markdown("")
                            sell_shares = st.number_input(
                                f"賣出股數(最多 {h['shares']:,})",
                                min_value=1, max_value=h["shares"], value=min(h["shares"], 1000),
                                key=f"sell_{h['stock_id']}",
                            )

                            if st.button(
                                f"🔴 賣出 {sell_shares:,} 股 {h['stock_name']}",
                                key=f"sell_btn_{h['stock_id']}",
                                use_container_width=True,
                            ):
                                result = sell_stock(h["stock_id"], sell_shares, portfolio=st.session_state.vt_portfolio)
                                if result["success"]:
                                    if "portfolio" in result:
                                        st.session_state.vt_portfolio = result["portfolio"]
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])

            # ─── Tab 4: 交易紀錄 ───
            with vtab4:
                st.subheader("📜 交易紀錄")

                orders = get_order_history(100, portfolio=st.session_state.vt_portfolio)

                if not orders:
                    st.info("📭 尚無交易紀錄")
                else:
                    st.caption(f"共 {len(orders)} 筆交易(最新在前)")

                    for o in orders:
                        ot = o["order_type"]
                        icon = "🟢 買入" if ot == "buy" else "🔴 賣出"
                        price = o.get("price", 0)
                        shares = o.get("shares", 0)
                        total_val = o.get("total", price * shares)
                        fee = o.get("fee", 0)

                        if ot == "sell":
                            tax = o.get("tax", 0)
                            net = o.get("net_received", total_val - fee - tax)
                            pnl = o.get("realized_pnl", 0)

                            st.markdown(
                                f"""<div style="
                                    padding: 10px 14px;
                                    margin: 6px 0;
                                    border-radius: 8px;
                                    background: {'rgba(76,175,80,0.08)' if ot == 'buy' else 'rgba(244,67,54,0.08)'};
                                    border-left: 3px solid {'#4CAF50' if ot == 'buy' else '#f44336'};
                                ">
                                    <strong>{icon} {o['stock_name']} ({o['stock_id']})</strong>
                                    <span style="color: #888; font-size: 13px; margin-left: 10px;">{o['timestamp']}</span><br>
                                    {shares:,} 股 @ ${price:.2f} = ${total_val:,.0f}
                                    |手續 ${fee:.0f} |稅 ${tax:.0f}
                                    |實收 ${net:,.0f}
                                    |損益 <span style="color: {'#e74c3c' if pnl >= 0 else '#2ecc71'};">${pnl:+,.0f}</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""<div style="
                                    padding: 10px 14px;
                                    margin: 6px 0;
                                    border-radius: 8px;
                                    background: rgba(76,175,80,0.08);
                                    border-left: 3px solid #4CAF50;
                                ">
                                    <strong>{icon} {o['stock_name']} ({o['stock_id']})</strong>
                                    <span style="color: #888; font-size: 13px; margin-left: 10px;">{o['timestamp']}</span><br>
                                    {shares:,} 股 @ ${price:.2f} = ${total_val:,.0f}
                                    |手續 ${fee:.0f}
                                    |餘額 ${o['cash_after']:,.0f}
                                </div>""",
                                unsafe_allow_html=True,
                            )



    with tabs[1]:
            st.caption("四維度評分 — 績效(30%)、風險(25%)、殖利率(25%)、動能(20%)")

            etf_tabs = st.tabs(["🏆 ETF 排行", "📊 類別分析", "🔍 單檔分析", "⚖️ 比較"], key="etf_tabs")

            # ─── Tab 1: ETF 排行 ───
            with etf_tabs[0]:
                st.subheader("🏆 ETF 綜合評分排行")
                etf_top_n = st.slider("顯示前幾名", 5, 30, 15, key="etf_top_n")
                etf_months = st.slider("分析期間(月)", 12, 60, 36, key="etf_months", step=12)

                if st.button("📊 更新排行", type="primary", use_container_width=True, key="etf_rank_btn"):
                    with st.spinner("正在分析 ETF..."):
                        picks = get_etf_picks(top_n=etf_top_n, months=etf_months)
                        st.session_state["etf_picks"] = picks
                        cat_stats = get_category_stats(months=etf_months)
                        st.session_state["etf_cat_stats"] = cat_stats

                if "etf_picks" in st.session_state:
                    picks = st.session_state["etf_picks"]
                    if picks:
                        rows = []
                        for i, e in enumerate(picks):
                            rows.append({
                                "排名": i + 1,
                                "代號": e.stock_id,
                                "名稱": e.name,
                                "類別": e.category,
                                "總分": f"{e.total_score:.1f}",
                                "評級": e.rating,
                                "績效": f"{e.perf_score:.0f}/30",
                                "風險": f"{e.risk_score:.0f}/25",
                                "殖利率": f"{e.yield_score:.0f}/25",
                                "動能": f"{e.momentum_score:.0f}/20",
                                "現價": f"{e.current_price:.2f}" if e.current_price else "—",
                                "1M": f"{getattr(e, 'return_1m', 0):+.2f}%",
                                "3M": f"{getattr(e, 'return_3m', 0):+.2f}%",
                                "6M": f"{getattr(e, 'return_6m', 0):+.2f}%",
                                "波動": f"{getattr(e, 'volatility', 0):.1f}%",
                                "回撤": f"{getattr(e, 'max_drawdown', 0):.1f}%",
                            })
                        df_etf = pd.DataFrame(rows)
                        st.dataframe(df_etf, use_container_width=True, hide_index=True)

                        st.subheader("🏅 Top 5 詳細")
                        for e in picks[:5]:
                            expanded = e is picks[0]
                            rank_icon = "🥇" if e is picks[0] else "🥈" if e is picks[1] else "🥉" if e is picks[2] else "📌"
                            with st.expander(
                                f"{rank_icon} {e.name} ({e.stock_id}) — 總分 {e.total_score:.1f}",
                                expanded=expanded
                            ):
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("評級", e.rating)
                                    st.metric("現價", f"{e.current_price:.2f}")
                                with cols[1]:
                                    st.metric("績效", f"{e.perf_score:.0f}/30")
                                    st.metric("風險", f"{e.risk_score:.0f}/25")
                                with cols[2]:
                                    st.metric("殖利率", f"{e.yield_score:.0f}/25")
                                    st.metric("動能", f"{e.momentum_score:.0f}/20")

                                for fname, fscore, fmax, _ in [
                                    ("績效", e.perf_score, 30, "#2ed573"),
                                    ("風險", e.risk_score, 25, "#1e90ff"),
                                    ("殖利率", e.yield_score, 25, "#ffa502"),
                                    ("動能", e.momentum_score, 20, "#a29bfe"),
                                ]:
                                    pct = min(100, fscore / fmax * 100)
                                    st.markdown(f"**{fname}** {fscore:.0f}/{fmax}")
                                    st.progress(pct / 100)

                                analysis_lines = generate_etf_analysis(e)
                                with st.expander("📋 分析摘要"):
                                    for l in analysis_lines:
                                        st.markdown(l)
                else:
                    st.info("👆 點擊「更新排行」開始分析")

            # ─── Tab 2: 類別分析 ───
            with etf_tabs[1]:
                st.subheader("📊 類別統計")
                if "etf_cat_stats" in st.session_state:
                    cat_stats = st.session_state["etf_cat_stats"]
                    for cat, data in cat_stats.items():
                        with st.expander(f"📂 {cat} ({data.get('count', 0)} 檔)", expanded=True):
                            st.metric("平均總分", f"{data.get('avg_score', 0):.1f}")
                            if data.get("avg_return_1y"):
                                st.metric("平均年報酬", f"{data['avg_return_1y']:.2f}%")
                            if data.get("avg_volatility"):
                                st.metric("平均年化波動", f"{data['avg_volatility']:.2f}%")
                            if data.get("avg_dividend"):
                                st.metric("平均預估殖利率", f"{data['avg_dividend']:.2f}%")
                else:
                    st.info("👆 先到「ETF 排行」分頁執行分析")

                st.subheader("🏆 各類別最佳 ETF")
                if "etf_picks" in st.session_state:
                    picks = st.session_state["etf_picks"]
                    best_by_cat = {}
                    for e in picks:
                        cat = e.category or "其他"
                        if cat not in best_by_cat:
                            best_by_cat[cat] = e
                    for cat, best in sorted(best_by_cat.items()):
                        st.info(f"**{cat}** 🥇 {best.name} ({best.stock_id}) — 總分 {best.total_score:.1f}")

            # ─── Tab 3: 單檔分析 ───
            with etf_tabs[2]:
                st.subheader("🔍 單檔 ETF 分析")
                from etf_analysis import ETF_UNIVERSE as _ETF_U
                etf_options = {
                    f"{sid} {info['name']} ({info['category']})": sid
                    for sid, info in _ETF_U.items()
                }
                etf_choice = st.selectbox("選擇 ETF", options=list(etf_options.keys()), key="etf_choice")
                etf_sid = etf_options[etf_choice]
                etf_info = _ETF_U.get(etf_sid, {})
                st.caption(f"{etf_info.get('desc', '')}")
                etf_single_months = st.slider("分析期間(月)", 12, 60, 36, key="etf_single_months", step=12)

                if st.button("🔍 分析此 ETF", type="primary", use_container_width=True, key="etf_analyze_btn"):
                    with st.spinner(f"正在分析 {etf_choice}..."):
                        e = score_etf(etf_sid, months=etf_single_months)
                        st.session_state["etf_single_result"] = e

                if "etf_single_result" in st.session_state:
                    e = st.session_state["etf_single_result"]
                    if e.error:
                        st.error(f"❌ {e.error}")
                    else:
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("總分", f"{e.total_score:.1f}")
                            st.metric("評級", e.rating)
                        with cols[1]:
                            st.metric("績效", f"{e.perf_score:.0f}/30")
                        with cols[2]:
                            st.metric("風險", f"{e.risk_score:.0f}/25")

                        for fname, fscore, fmax in [
                            ("績效", e.perf_score, 30),
                            ("風險", e.risk_score, 25),
                            ("殖利率", e.yield_score, 25),
                            ("動能", e.momentum_score, 20),
                        ]:
                            pct = min(100, fscore / fmax * 100)
                            st.markdown(f"**{fname}** {fscore:.0f}/{fmax}")
                            st.progress(pct / 100)

                        st.subheader("📈 各期報酬")
                        ret_cols = st.columns(5)
                        ret_data = [
                            ("1月", getattr(e, 'return_1m', 0)),
                            ("3月", getattr(e, 'return_3m', 0)),
                            ("6月", getattr(e, 'return_6m', 0)),
                            ("1年", getattr(e, 'return_1y', 0)),
                            ("3年", getattr(e, 'return_3y', 0)),
                        ]
                        for j, (label, val) in enumerate(ret_data):
                            with ret_cols[j]:
                                val_str = f"{val:+.2f}%" if isinstance(val, (int, float)) else str(val)
                                st.metric(label, val_str)

                        st.subheader("🛡️ 風險指標")
                        risk_cols = st.columns(3)
                        with risk_cols[0]:
                            st.metric("年化波動率", f"{getattr(e, 'volatility', 0):.1f}%")
                        with risk_cols[1]:
                            st.metric("最大回撤", f"{getattr(e, 'max_drawdown', 0):.1f}%")
                        with risk_cols[2]:
                            st.metric("預估殖利率", f"{getattr(e, 'est_dividend_yield', 0):.2f}%")

                        analysis_lines = generate_etf_analysis(e)
                        with st.expander("📋 分析摘要"):
                            for l in analysis_lines:
                                st.markdown(l)

            # ─── Tab 4: 比較 ───
            with etf_tabs[3]:
                st.subheader("⚖️ 多檔 ETF 比較")
                from etf_analysis import ETF_UNIVERSE as _ETF_U2
                all_etf_keys = list(_ETF_U2.keys())
                default_cmp = [s for s in ["0050", "0056", "00878"] if s in all_etf_keys]
                selected_cmp = st.multiselect(
                    "選擇要比較的 ETF",
                    options=all_etf_keys,
                    default=default_cmp,
                    format_func=lambda x: f"{x} {_ETF_U2.get(x, {}).get('name', '')}",
                    key="etf_cmp"
                )
                cmp_months = st.slider("分析期間(月)", 12, 60, 36, key="cmp_months", step=12)

                if st.button("⚖️ 開始比較", type="primary", use_container_width=True, key="cmp_btn"):
                    if len(selected_cmp) < 2:
                        st.warning("請至少選 2 檔 ETF 進行比較")
                    else:
                        with st.spinner("正在比較..."):
                            cmp_df = compare_etfs(selected_cmp, months=cmp_months)
                            st.session_state["etf_cmp_result"] = cmp_df

                if "etf_cmp_result" in st.session_state:
                    cmp_df = st.session_state["etf_cmp_result"]
                    if not cmp_df.empty:
                        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
                        score_cols = [
                            c for c in cmp_df.columns
                            if "評分" in c or "總分" in c or "風險" in c or c in ["績效", "殖利率", "動能"]
                        ]
                        if score_cols:
                            fig_comp = go.Figure()
                            for _, row in cmp_df.iterrows():
                                fig_comp.add_trace(go.Bar(
                                    name=f"{row.get('代號', '')} {row.get('名稱', '')}",
                                    x=score_cols,
                                    y=[row.get(c, 0) for c in score_cols],
                                ))
                            fig_comp.update_layout(title="ETF 比較", barmode="group", height=400)
                            st.plotly_chart(fig_comp, use_container_width=True)


    with tabs[2]:
        st.subheader("🔔 價格警示與通知")
        st.caption("自訂價格警示，股價突破/跌破時自動通知")
        
        alert_tabs = st.tabs(["➕ 新增", "📋 列表", "📜 紀錄"])
        with alert_tabs[0]:
            al_stock = st.text_input("股票代號", value=stock_id, key="alert_stock_input")
            al_name = get_stock_name(al_stock)
            st.caption(f"📌 {al_name}")
            al_type = st.selectbox("類型", [("price_above", "股價突破 ≥"), ("price_below", "股價跌破 ≤")], format_func=lambda x: x[1])
            al_target = st.number_input("目標價格", 1.0, 100000.0, float(data['Close'].iloc[-1]) * 1.05, 0.5)
            al_note = st.text_input("備註", placeholder="加碼點/停損點...")
            if st.button("➕ 新增警示", type="primary", use_container_width=True):
                add_alert(al_stock, al_type[0], al_target, al_name, al_note)
                st.success("✅ 已新增！")
                st.rerun()
        with alert_tabs[1]:
            for a in get_alerts():
                aid = a.get("id", "")
                with st.container(border=True):
                    ac = st.columns([2, 2, 1, 1])
                    am = {"price_above": "≥", "price_below": "≤"}
                    ac[0].write(f"**{a.get('stock_name','?')}({a.get('stock_id','?')})**")
                    ac[1].write(f"{am.get(a.get('alert_type',''),'')} ${a.get('target_value',0):.2f}")
                    ac[2].write("🟢 啟用" if a.get("enabled",True) else "🔴 停用")
                    if ac[3].button("🗑️", key=f"del_{aid}", use_container_width=True):
                        remove_alert(aid)
                        st.rerun()
        with alert_tabs[2]:
            for evt in get_recent_events():
                st.markdown(f"• {evt.get('message', '')}")



# ── 🕸️ 關係圖譜 — 股票知識圖譜 ──
if mode == "🕸️ 關係圖譜":
    import sys as _sys
    if _sys.stdout.encoding.upper() in ("CP950", "BIG5"):
        _sys.stdout.reconfigure(encoding="utf-8")
    
    st.subheader("🕸️ 股票關係知識圖譜")
    st.caption("查看股票之間的產業/相關性/共整合關係，分析『如果某股跌，誰會被波及？』")
    
    # 載入/建構圖譜
    kg = None
    _graph_path = os.path.join(os.path.dirname(__file__), 'stock_graph.json')
    if os.path.exists(_graph_path):
        from professional.stock_knowledge_graph import StockKnowledgeGraph
        kg = StockKnowledgeGraph.load(_graph_path)
    else:
        from professional.stock_knowledge_graph import build_default_graph
        kg = build_default_graph()
    
    kg_tabs = st.tabs(["🔍 影響範圍分析", "🏷️ 相似股票搜尋", "📊 圖譜概覽"])
    
    with kg_tabs[0]:
        st.subheader("🔍 影響範圍分析 (Impact Analysis)")
        st.caption("給定一檔股票，分析『如果它大跌，誰會被波及？』")
        
        imp_col1, _ = st.columns([3, 1])
        with imp_col1:
            imp_sid = st.text_input("股票代號", value="2330", key="kg_impact_sid")
        
        if st.button("🔍 分析影響範圍", type="primary", use_container_width=True, key="kg_impact_btn") or \
           "kg_impact_result" in st.session_state:
            if st.session_state.get("kg_impact_sid") != imp_sid or imp_sid not in (st.session_state.get("kg_impact_result") or {}).get("target", ""):
                pass  # will recompute
            elif "kg_impact_result" in st.session_state:
                pass  # cached
            
            with st.spinner(f"分析 {imp_sid} 的波及範圍..."):
                result = kg.impact_analysis(imp_sid)
                st.session_state["kg_impact_result"] = result
                kg_name = kg.graph.nodes[imp_sid].get('name', '') if hasattr(kg, 'graph') and imp_sid in kg.graph else ''
            
            result = st.session_state["kg_impact_result"]
            if result.get('error'):
                st.warning(f"⚠️ {result['error']}")
            else:
                # Summary
                st.metric("📊 潛在波及總數", f"{result['total_potential_impact']} 檔")
                
                # High Impact
                hi = result.get('high_impact', [])
                mi = result.get('medium_impact', [])
                
                if hi:
                    st.subheader("🔴 高連動 (≥ 0.7)")
                    cols = st.columns(min(len(hi), 4))
                    for i, s in enumerate(hi):
                        with cols[i % 4]:
                            sname = s.get('name', '')
                            st.markdown(f"**{s['stock']}** {sname}")
                            st.caption(f"連動: {s['impact_weight']:.2f}")
                            st.caption(f"關係: {', '.join(s.get('relation', ['未知']))[:20]}")
                
                if mi:
                    st.subheader("🟡 中連動 (0.4–0.7)")
                    for s in mi[:8]:
                        sname = s.get('name', '')
                        st.markdown(f"• **{s['stock']}** {sname} — 連動權重 {s['impact_weight']:.2f}")
                
                # Full table
                affected = result.get('affected_stocks', [])
                if affected:
                    with st.expander(f"📋 完整波及列表 ({len(affected)} 檔)"):
                        from professional.stock_knowledge_graph import _get_sector
                        df_rows = []
                        for s in affected:
                            sec = _get_sector(s['stock'])
                            df_rows.append({
                                '代號': s['stock'],
                                '名稱': s.get('name', ''),
                                '產業': sec,
                                '連動權重': f"{s['impact_weight']:.2f}",
                                '關係類型': ', '.join(s.get('relation', ['']))
                            })
                        if df_rows:
                            st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)
    
    with kg_tabs[1]:
        st.subheader("🏷️ 相似股票搜尋")
        st.caption("和指定股票最相似的標的")
        
        sim_col1, sim_col2 = st.columns([3, 1])
        with sim_col1:
            sim_sid = st.text_input("股票代號", value="2618", key="kg_sim_sid")
        with sim_col2:
            sim_n = st.number_input("數量", min_value=3, max_value=20, value=8, key="kg_sim_n")
        
        if st.button("🏷️ 搜尋相似股票", type="primary", use_container_width=True, key="kg_sim_btn"):
            with st.spinner(f"搜尋跟 {sim_sid} 最像的股票..."):
                similar = kg.similar_stocks(sim_sid, top_n=sim_n)
            
            if similar:
                from professional.stock_knowledge_graph import _get_sector
                sim_df = pd.DataFrame([{
                    '代號': s['stock'],
                    '產業': _get_sector(s['stock']),
                    '相關性': f"{s['impact_weight']:.2f}",
                } for s in similar])
                st.dataframe(sim_df, use_container_width=True, hide_index=True)
                
                # Simple bar chart
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Bar(
                    x=[f"{s['stock']}" for s in similar],
                    y=[s['impact_weight'] for s in similar],
                    marker_color=['#ff6b6b' if s['impact_weight'] >= 0.7 else '#ffd93d' if s['impact_weight'] >= 0.4 else '#6bcb77' for s in similar],
                    text=[f"{s['impact_weight']:.2f}" for s in similar],
                    textposition='outside',
                ))
                fig_sim.update_layout(
                    title=f"與 {sim_sid} 最相似的 {len(similar)} 檔股票",
                    xaxis_title="股票代號",
                    yaxis_title="連動權重",
                    height=400,
                )
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.info("找不到該股票的圖譜資料")
    
    with kg_tabs[2]:
        st.subheader("📊 圖譜概覽")
        
        meta = kg.metadata
        c1, c2, c3 = st.columns(3)
        c1.metric("📌 股票數", meta.get('stock_count', 0))
        c2.metric("🔗 關係邊", meta.get('edge_count', 0))
        c3.metric("🧩 社群數", meta.get('clusters', 0))
        
        # Sector breakdown
        from professional.stock_knowledge_graph import _get_sector
        sectors = {}
        for sid in kg.nodes:
            sec = _get_sector(sid)
            if sec not in sectors:
                sectors[sec] = []
            name = kg.graph.nodes[sid].get('name', '') if hasattr(kg, 'graph') and sid in kg.graph else ''
            sectors[sec].append(f"{sid}{' '+name if name else ''}")
        
        with st.expander("🏭 產業分佈"):
            for sec, members in sorted(sectors.items()):
                st.markdown(f"**{sec}** ({len(members)}檔)")
                st.caption(', '.join(members))
        
        # Community view
        if meta.get('communities'):
            with st.expander("🧩 社群發現 (Greedy Modularity)"):
                for i, community in enumerate(meta['communities']):
                    from collections import Counter
                    comm_sectors = Counter()
                    comm_names = []
                    for sid in community:
                        sec = _get_sector(sid)
                        comm_sectors[sec] += 1
                        name = kg.graph.nodes[sid].get('name', '') if hasattr(kg, 'graph') and sid in kg.graph else ''
                        comm_names.append(f"{sid}{' '+name if name else ''}")
                    top_sec = comm_sectors.most_common(1)[0][0] if comm_sectors else '混合'
                    st.markdown(f"**社群 {i+1}** — {len(community)} 節點 — 主要產業: {top_sec}")
                    st.caption(', '.join(comm_names))
        
        # Edge type breakdown
        edge_types = {}
        for (s1, s2), data in kg.edges.items():
            t = data['type']
            if t not in edge_types:
                edge_types[t] = 0
            edge_types[t] += 1
        
        with st.expander("🔗 關係類型分佈"):
            type_df = pd.DataFrame([
                {'關係類型': t, '數量': c}
                for t, c in sorted(edge_types.items(), key=lambda x: -x[1])
            ])
            st.dataframe(type_df, use_container_width=True, hide_index=True)
            
            fig_types = go.Figure([go.Pie(
                labels=list(edge_types.keys()),
                values=list(edge_types.values()),
                hole=0.4,
            )])
            fig_types.update_layout(title="關係類型佔比", height=350)
            st.plotly_chart(fig_types, use_container_width=True)
        
        st.caption(f"🕐 圖譜建構時間: {meta.get('built_at', 'N/A')[:19]}")


# ── 🌍 全球圖譜 — Global Stock Nexus ──
if mode == "🌍 全球圖譜":
    import sys as _gg_sys
    if _gg_sys.stdout.encoding.upper() in ("CP950", "BIG5"):
        _gg_sys.stdout.reconfigure(encoding="utf-8")
    
    st.subheader("🌍 全球市場關係圖譜 (Global Stock Nexus)")
    st.caption("全球指數 / 台股ADR / 商品 / 加密貨幣 × 台股相關性分析")
    
    # 載入/建構全球圖譜
    gkg = None
    _gkg_path = os.path.join(os.path.dirname(__file__), 'global_graph.json')
    if os.path.exists(_gkg_path):
        from professional.global_knowledge_graph import GlobalKnowledgeGraph
        try:
            gkg = GlobalKnowledgeGraph.load(_gkg_path)
        except Exception:
            gkg = None
    
    if gkg is None:
        st.error("⚠️ 全球知識圖譜尚未建構，請先執行 `python professional/global_knowledge_graph.py`")
        st.stop()
    
    gkg_tabs = st.tabs(["🔍 全球→台股影響", "🌐 ADR 溢價監控", "📊 隔夜風險", "⚙️ 商品/指數關係", "📋 圖譜概覽"])
    
    with gkg_tabs[0]:
        """🔍 全球→台股影響範圍分析"""
        st.subheader("🔍 全球指數 → 台股影響分析")
        st.caption("選擇一個全球指數，查看它對哪些台股影響最大")
        
        idx_options = {}
        for nid, meta in gkg.metadata.items():
            if meta.get('group', '').endswith('_index') or meta.get('group', '') == 'US_semiconductor':
                idx_options[f"{nid} - {meta.get('name', nid)}"] = nid
        
        if idx_options:
            gkg_sel = st.selectbox("選擇全球指數", options=list(idx_options.keys()), key="gkg_impact_select")
            gkg_idx = idx_options[gkg_sel]
            
            if st.button("🔍 分析影響", type="primary", use_container_width=True, key="gkg_impact_btn"):
                with st.spinner(f"分析 {gkg_idx} 對台股的影響..."):
                    impacts = gkg.global_impact(gkg_idx, top_n=20)
                    
                    if impacts:
                        st.metric("📊 受影響台股數", len(impacts))
                        
                        imp_df = pd.DataFrame([
                            {
                                '台股代號': s['id'],
                                '名稱': s['name'],
                                '相關性': s['correlation'],
                                '強度': '高' if abs(s['correlation']) >= 0.5 else '中',
                            }
                            for s in impacts
                        ])
                        st.dataframe(imp_df, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        fig_imp = go.Figure([go.Bar(
                            x=[s['correlation'] for s in impacts],
                            y=[f"{s['id']} {s['name']}" for s in impacts],
                            orientation='h',
                            marker_color=['red' if s['correlation'] < 0 else 'green' for s in impacts],
                        )])
                        fig_imp.update_layout(
                            title=f"{gkg_idx} → 台股相關性",
                            xaxis_title="相關性",
                            height=400,
                        )
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("該指數與台股暫無顯著相關性（threshold ≥ 0.25）")
        else:
            st.warning("圖譜中無全球指數節點")
    
    with gkg_tabs[1]:
        """🌐 ADR 溢價"""
        st.subheader("🌐 台股 ADR 溢價監控")
        st.caption("ADR 價格換算回台幣後，與台股價格的溢價/折價幅度")
        
        for adr_id in ['TSM', 'UMC', 'ASX']:
            premium_info = gkg.adr_premium(adr_id)
            if premium_info:
                pct = premium_info.get('adr_premium_pct', 'N/A')
                emoji = '🟢' if (isinstance(pct, (int, float)) and pct > 0) else ('🔴' if isinstance(pct, (int, float)) and pct < 0 else '⚪')
                st.markdown(f"{emoji} **{premium_info['name']}** ({adr_id})")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("ADR 價格(USD)", f"${premium_info.get('adr_usd', 'N/A')}")
                col2.metric("換算台股(TWD)", f"${premium_info.get('adr_converted_twd', 'N/A')}")
                col3.metric("台股收盤", f"${premium_info.get('tw_twd', 'N/A')}")
                if isinstance(pct, (int, float)):
                    col4.metric("溢價/折價", f"{pct:+.3f}%")
                else:
                    col4.metric("溢價/折價", str(pct))
                st.divider()
    
    with gkg_tabs[2]:
        """📊 隔夜風險"""
        st.subheader("📊 隔夜風險評估 (US Market → TW Stocks)")
        st.caption("美國市場波動對個股的潛在傳導風險")
        
        risk = gkg.overnight_risk_assessment()
        
        risk_levels = [
            ("🔴 高風險 (≥ 0.4)", risk['high_risk']),
            ("🟡 中風險 (0.2–0.4)", risk['medium_risk']),
            ("🟢 低風險 (< 0.2)", risk['low_risk']),
        ]
        
        for label, items in risk_levels:
            st.subheader(label)
            if items:
                risk_df = pd.DataFrame([
                    {'台股': f"{s['stock_id']} {s['name']}", '平均相關性': s['avg_correlation']}
                    for s in items
                ])
                st.dataframe(risk_df, use_container_width=True, hide_index=True, height=min(len(items) * 35 + 40, 300))
            else:
                st.caption("（無資料）")
    
    with gkg_tabs[3]:
        """⚙️ 商品/指數關係"""
        st.subheader("⚙️ 商品/指數交叉相關性")
        st.caption("原油、黃金、美元指數 vs 全球指數的相關性")
        
        if gkg.graph:
            # Build commodity-index matrix
            comm_nodes = list(set().union(
                set(gkg.metadata.get(n, {}).get('group', '') for n in gkg.graph.keys()
                    if gkg.metadata.get(n, {}).get('group') in ('commodity', 'currency', 'bond', 'crypto'))
            ))
            # Better approach: find commodity nodes
            comm_ids = [n for n in gkg.graph if gkg.metadata.get(n, {}).get('group') in ('commodity', 'currency', 'bond', 'crypto')]
            idx_ids = [n for n in gkg.graph if gkg.metadata.get(n, {}).get('group', '').endswith('_index')]
            
            if comm_ids and idx_ids:
                comm_matrix = pd.DataFrame(index=[f"{gkg.metadata.get(c, {}).get('name', c)}" for c in comm_ids],
                                         columns=[f"{gkg.metadata.get(i, {}).get('name', i)}" for i in idx_ids],
                                         dtype=float)
                for c in comm_ids:
                    for i in idx_ids:
                        val = gkg.graph.get(c, {}).get(i, {}).get('weight', 0)
                        comm_matrix.loc[gkg.metadata.get(c, {}).get('name', c),
                                       gkg.metadata.get(i, {}).get('name', i)] = val
                
                st.dataframe(comm_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
                           use_container_width=True)
                
                st.caption("🟦 正相關 | 🟥 負相關 | 顏色越深相關性越高")
            else:
                st.info("暫無商品/指數圖譜資料")
        else:
            st.info("圖譜未建構")
    
    with gkg_tabs[4]:
        """📋 圖譜概覽"""
        st.subheader("📋 全球圖譜概覽")
        
        gkg_stats = gkg.stats()
        c1, c2 = st.columns(2)
        c1.metric("🌐 節點數", gkg_stats['nodes'])
        c2.metric("🔗 關係邊", gkg_stats['edges'])
        
        with st.expander("🏷️ 節點分組"):
            for g, count in sorted(gkg_stats.get('node_groups', {}).items(), key=lambda x: -x[1]):
                st.markdown(f"**{g}**: {count} 節點")
                members = [n for n, m in gkg.metadata.items() if m.get('group') == g]
                names = [f"{m.get('name', n)}" for n, m in gkg.metadata.items() if m.get('group') == g]
                st.caption(', '.join(names))
        
        with st.expander("🔗 關係類型"):
            for rel, count in sorted(gkg_stats.get('edge_relations', {}).items(), key=lambda x: -x[1]):
                st.markdown(f"**{rel}**: {count} 邊")
        
        with st.expander("🔍 搜尋節點連線"):
            gkg_search = st.text_input("輸入節點代號 (e.g. ^GSPC, TSM, 2330, CL=F)", value="2330", key="gkg_search")
            neighbors = gkg.neighbors(gkg_search)
            if neighbors:
                st.dataframe(pd.DataFrame([
                    {'節點': n['id'], '名稱': n['name'], '分組': n['group'], '相關性': n['weight'], '關係': n['relation']}
                    for n in neighbors
                ]).sort_values('相關性', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info(f"找不到 {gkg_search} 的連線資料")


# 頁尾
# ============================================================
st.divider()
st.caption(
    "⚠️ 免責聲明:本系統僅供學習與分析參考,不構成任何投資建議。"
    "資料來源為 TWSE/TPEx 公開資訊,即時性與準確性請以官方為準。"
)
