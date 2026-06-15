"""
台灣股票分析系統 - Streamlit 主程式
功能：看盤、技術分析、回測、分析輔助
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
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
from virtual_trading import (
    get_portfolio, buy_stock, sell_stock,
    get_holdings_with_prices, get_portfolio_summary,
    get_order_history, reset_portfolio,
)

st.set_page_config(
    page_title="台股分析系統",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# 頁面標題 & 側邊欄
# ============================================================
st.title("📈 台股分析系統")
st.caption("資料來源：TWSE / TPEx 公開資料 ｜ 僅供分析參考，非投資建議")

# 初始化虛擬交易 session state（雲端部署用）
if "vt_portfolio" not in st.session_state:
    from virtual_trading import new_portfolio
    st.session_state.vt_portfolio = new_portfolio()

with st.sidebar:
    st.header("⚙️ 設定")
    
    # 初始化 session state
    if "stock_id" not in st.session_state:
        st.session_state.stock_id = "2330"
    
    # 股票選擇（改用 session_state）
    stock_id_input = st.text_input("股票代號", value=st.session_state.stock_id, max_chars=6, key="sid_input")
    st.session_state.stock_id = stock_id_input
    
    stock_name = get_stock_name(st.session_state.stock_id)
    st.caption(f"📌 {stock_name}")
    
    # 名稱搜尋
    name_q = st.text_input("🔍 搜尋公司名稱", placeholder="輸入名稱關鍵字…")
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
    months = st.slider("歷史資料長度（月）", min_value=3, max_value=60, value=12)
    
    # 功能選擇
    mode = st.radio(
        "功能模式",
        ["📊 看盤與技術分析", "🔄 策略回測", "📋 多股掃描", "🏆 每日推薦", "🎮 虛擬交易", "🧠 分析輔助"],
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
    st.error(f"❌ 無法取得股票 {stock_id} 的資料，請確認代號是否正確")
    st.stop()


# ============================================================
# 模式 1：看盤與技術分析
# ============================================================
if mode == "📊 看盤與技術分析":
    st.header(f"{stock_id} {stock_name}")
    
    # 即時報價
    with st.spinner("取得即時報價中..."):
        quote = fetch_realtime_quote(stock_id)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    if "error" not in quote:
        change_pct = quote.get("change_percent", 0)
        arrow = "🟢" if change_pct >= 0 else "🔴"
        col1.metric("成交價", f"{quote.get('price', 0):.2f}",
                    f"{quote.get('change', 0):+.2f}" if quote.get('change', 0) != 0 else "0")
        col2.metric("開盤", f"{quote.get('open', 0):.2f}")
        col3.metric("最高", f"{quote.get('high', 0):.2f}")
        col4.metric("最低", f"{quote.get('low', 0):.2f}")
        col5.metric("成交量", f"{quote.get('volume', 0):,}")
    else:
        st.info(f"💡 盤中即時報價：{quote.get('error', '無法取得')}")
        # 改用昨日收盤
        last_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[-2] if len(data) > 1 else last_close
        change = last_close - prev_close
        pct = change / prev_close * 100
        col1.metric("昨收", f"{last_close:.2f}", f"{change:+.2f}")
        col2.metric("前日收", f"{prev_close:.2f}")
    
    # 技術訊號摘要
    st.subheader("📡 技術訊號")
    signals = get_indicator_signals(data)
    if signals:
        sig_cols = st.columns(len(signals))
        for i, (key, val) in enumerate(signals.items()):
            sig_cols[i].metric(key, val)
    
    # 主圖：K線 + 均線 + 布林
    st.subheader("📉 股價走勢")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("K線與技術指標", "成交量", "RSI"),
    )
    
    # K線
    fig.add_trace(
        go.Candlestick(
            x=data["日期"], open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"],
            name="K線", showlegend=False,
            increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",
        ),
        row=1, col=1,
    )
    
    # 均線
    for ma_period, color in [(5, "#f39c12"), (10, "#3498db"),
                              (20, "#9b59b6"), (60, "#95a5a6")]:
        col_name = f"MA{ma_period}"
        if col_name in data.columns:
            fig.add_trace(
                go.Scatter(x=data["日期"], y=data[col_name],
                           name=col_name, line=dict(width=1, color=color)),
                row=1, col=1,
            )
    
    # 布林通道
    if "BB_Upper" in data.columns:
        fig.add_trace(
            go.Scatter(x=data["日期"], y=data["BB_Upper"],
                       name="布林上軌", line=dict(width=1, color="#e67e22", dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=data["日期"], y=data["BB_Lower"],
                       name="布林下軌", line=dict(width=1, color="#e67e22", dash="dash")),
            row=1, col=1,
        )
    
    # 成交量
    colors = ["#e74c3c" if c >= o else "#2ecc71"
              for c, o in zip(data["Close"], data["Open"])]
    fig.add_trace(
        go.Bar(x=data["日期"], y=data["Volume"], name="成交量",
               marker_color=colors, opacity=0.6),
        row=2, col=1,
    )
    
    # RSI
    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(x=data["日期"], y=data["RSI"],
                       name="RSI", line=dict(color="#8e44ad", width=1)),
            row=3, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        margin=dict(l=40, r=20, t=30, b=20),
    )
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # MACD 副圖
    if "MACD" in data.columns:
        st.subheader("📊 MACD")
        fig2 = make_subplots(rows=1, cols=1)
        fig2.add_trace(
            go.Scatter(x=data["日期"], y=data["MACD"],
                       name="MACD", line=dict(color="#2980b9")),
        )
        fig2.add_trace(
            go.Scatter(x=data["日期"], y=data["MACD_Signal"],
                       name="Signal", line=dict(color="#e67e22")),
        )
        # MACD Histogram
        colors_hist = ["#e74c3c" if v < 0 else "#2ecc71"
                       for v in data["MACD_Hist"]]
        fig2.add_trace(
            go.Bar(x=data["日期"], y=data["MACD_Hist"],
                   name="柱狀圖", marker_color=colors_hist, opacity=0.5),
        )
        fig2.update_layout(height=300, template="plotly_white",
                           margin=dict(l=40, r=20, t=10, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    
    # 近期數據表
    with st.expander("📋 近期數據", expanded=False):
        show_df = data[["日期", "Open", "High", "Low", "Close", "Volume",
                        "MA5", "MA20", "RSI", "MACD"]].tail(30).copy()
        show_df["日期"] = show_df["日期"].dt.strftime("%Y-%m-%d")
        show_df["成交量(張)"] = (show_df["Volume"] / 1000).round(0).astype(int)
        show_df = show_df.drop(columns=["Volume"])
        show_df = show_df.rename(columns={
            "Open": "開盤", "High": "最高", "Low": "最低",
            "Close": "收盤", "MA5": "MA5", "MA20": "MA20",
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)
        csv = show_df.to_csv().encode("utf-8-sig")
        st.download_button("⬇️ 下載 CSV", csv, f"{stock_id}_data.csv", "text/csv")


# ============================================================
# 模式 2：策略回測
# ============================================================
elif mode == "🔄 策略回測":
    st.header(f"🔄 策略回測 — {stock_id} {stock_name}")
    
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


# ============================================================
# 模式 4：🏆 每日推薦（專家級 Top 5）
# ============================================================
elif mode == "🏆 每日推薦":
    st.header("🏆 專家級每日推薦 — Top 5 精選")
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
    st.info(f"📋 掃描範圍：{len(SCAN_UNIVERSE)} 檔市場重點股（半導體、電子、金融、傳產、航運、ETF）│含新聞情緒分析")
    
    if st.button("🚀 開始掃描評分", type="primary", use_container_width=True):
        with st.spinner("🔍 正在掃描 50 檔重點股（含新聞情緒分析），約需 30-60 秒..."):
            result = get_daily_picks_with_context(top_n=top_n, months=scan_months, include_news=include_news)
        picks = result["picks"]
        market_note = result["market_note"]
        market_ctx = result.get("market_ctx", {})
        scan_time = result["timestamp"]
        
        if not picks:
            st.error("⚠️ 掃描失敗，請稍後再試")
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
                                🏛️ 大盤（0050）{trend_icon}
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
                        
                        st.markdown("**💡 分析摘要：**")
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
                                st.markdown(f"目標① **{pick.target_1:.1f}** ({tp1_pct:+.1f}%)")
                            if pick.target_2 > 0:
                                tp2_pct = ((pick.target_2 / pick.current_price) - 1) * 100
                                st.markdown(f"目標② **{pick.target_2:.1f}** ({tp2_pct:+.1f}%)")
                        
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
                            st.markdown("**🗞️ 最新新聞：**")
                            for h in pick.news_headlines:
                                st.markdown(f"  • {h[:80]}")
                        else:
                            st.info("無近期新聞資料")
            
            # 掃描完成提示
            st.success(f"✅ 掃描完成！推薦時效至今日收盤 ({scan_time})，建議在盤中/盤後評估")
            
            # 免責
            st.divider()
            st.caption(
                "⚠️ 本推薦系統基於多因子量化評分模型，僅供分析參考，不構成投資建議。"
                "過往績效不代表未來表現，投資有賺有賠，請審慎評估。"
            )
    else:
        # 顯示預覽
        st.info("👆 點擊「開始掃描評分」按鈕，系統將分析 50 檔重點股並推薦前 5 名")
        
        # 展示評分系統
        with st.expander("📖 評分方法說明", expanded=True):
            st.markdown("""
            ### 🧮 多因子評分模型（含新聞情緒）
            
            | 面向 | 權重 | 評估項目 |
            |------|------|----------|
            | 📊 **技術面** | **35%** | 均線排列、RSI、MACD、布林通道、KDJ、成交量 |
            | 🗞️ **新聞情緒** | **20%** | Yahoo 奇摩股市最新新聞關鍵字分析（正向/負向/熱度） |
            | 📋 **基本面** | **15%** | 本益比、殖利率、股價淨值比 |
            | 🏢 **籌碼面** | **15%** | 外資買賣超、投信買賣超、自營商買賣超 |
            | 🚀 **動能力** | **10%** | 一週/一月漲跌幅、近期動能加速度 |
            | 🛡️ **穩定度** | **5%** | 波動率、近期最大回撤 |
            
            ### 🎯 評級標準
            
            | 分數區間 | 評級 | 建議 |
            |----------|------|------|
            | 70-100 | ⭐ 強力推薦 | 多因子共振，優先考慮 |
            | 55-69 | 📈 推薦買進 | 多項指標正面，可布局 |
            | 40-54 | 👀 值得關注 | 部分指標轉佳，觀察時機 |
            | 25-39 | ⚖️ 中立觀望 | 多空交錯，等待方向 |
            | 0-24 | ❌ 暫時避開 | 指標偏弱，建議迴避 |
            
            ### 🗞️ 新聞情緒分析方式
            
            透過 Yahoo 奇摩股市擷取各股最新新聞標題，進行關鍵字比對：
            - **正向關鍵字**：創高、突破、利多、成長、營收、受惠、AI、訂單...
            - **負向關鍵字**：大跌、利空、衰退、虧損、賣壓、調降、裁員...
            
            綜合正負向訊號數，標準化為 -10 ~ +10 分，納入總評分。
            
            ### 📡 掃描範圍（50 檔重點股）
            
            涵蓋半導體、電子代工、面板/PCB、金融、傳產龍頭、航運、ETF 等主要類股。
            
            整個流程約需 30-60 秒，請耐心等候。 ⏱️
            """)
            
            st.info("💡 **專家提示：** 建議每日收盤後執行一次掃描，獲取最新推薦。多日連續上榜的股票代表多因子持續看好，可優先關注。")


# ============================================================
# 模式 5：🎮 虛擬交易
# ============================================================
elif mode == "🎮 虛擬交易":
    st.header("🎮 虛擬交易市場")
    st.caption("用虛擬資金 $1,000,000 練習台股買賣，即時報價，真實手續費")
    
    # 快捷買入（從每日推薦跳轉用）
    if "quick_buy" in st.session_state and st.session_state.quick_buy:
        pass  # 由後續邏輯處理
    
    # 分頁
    vtab1, vtab2, vtab3, vtab4 = st.tabs(["💼 資產總覽", "💰 買入股票", "📂 庫存明細", "📜 交易紀錄"])
    
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
        col1.info(f"🟢 已實現損益：${summary['realized_pnl']:+,.0f}")
        col2.info(f"🟡 未實現損益：${summary['unrealized_pnl']:+,.0f}")
        col3.info(f"📊 持有 {summary['holdings_count']} 檔 | 總交易 {summary['order_count']} 筆")
        
        st.markdown("")
        
        # 操作提示
        st.markdown("""
        ---
        **🎯 新手操作建議：**
        1. 先去 **買入股票** 分頁模擬買股
        2. 在 **庫存明細** 查看即時損益
        3. 到 **交易紀錄** 回顧自己的操作
        
        💡 市場只有 $$100 萬本金，控制好每筆資金別 All-in！
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
                            st.success("✅ 資產已從檔案載入！")
                            st.rerun()
                        else:
                            st.error("❌ 檔案格式錯誤，缺少必要欄位")
                    except Exception as e:
                        st.error(f"❌ 載入失敗：{e}")
            with col_r:
                st.warning("⚠️ 重設後所有交易紀錄和庫存將被清除！")
                if st.button("🔄 重設資產組合", type="secondary", use_container_width=True):
                    result = reset_portfolio(st.session_state.vt_portfolio)
                    if "portfolio" in result:
                        st.session_state.vt_portfolio = result["portfolio"]
                    st.success(result["message"])
                    st.rerun()
            
            st.info(
                "💡 **提示：** 資料存在瀏覽器中（Session），關閉頁面會消失。"
                "建議定期點「匯出資產」下載備份。下次使用時點「匯入資產」還原即可。"
            )
    
    # ─── Tab 2: 買入股票 ───
    with vtab2:
        st.subheader("💰 買入股票")
        
        # 快速選擇熱門股
        quick_stocks = ["2330", "2317", "2454", "0050", "2881", "2412", "2308", "2002"]
        st.caption("快速選擇：")
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
                help="一般交易單位為 1000 股（1張），零股最小 1 股")
            
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
                        f"📊 現價：**${price:.2f}** {quote.get('change_percent', 0):+.2f}%\n"
                        f"📈 開 {quote.get('open', 0):.2f} 高 {quote.get('high', 0):.2f} 低 {quote.get('low', 0):.2f}\n\n"
                        f"**試算：**\n"
                        f"買進 {buy_shares:,} 股 × ${price:.2f} = ${total_cost:,.0f}\n"
                        f"手續費 ${fee:.0f}\n"
                        f"**需付總額：${total_paid:,.0f}**"
                    )
                    
                    cash = st.session_state.vt_portfolio["cash"]
                    if total_paid > cash:
                        st.error(f"❌ 餘額不足！可用 ${cash:,.0f}，需 ${total_paid:,.0f}")
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
            - 手續費 0.1425%（最低 $20）
            """)
            
            st.markdown("")
            st.markdown("**🔄 也可以賣出...**")
            st.info("切換到「庫存明細」分頁即可賣出持股")
    
    # ─── Tab 3: 庫存明細 ───
    with vtab3:
        st.subheader("📂 庫存明細")
        
        holdings = get_holdings_with_prices(st.session_state.vt_portfolio)
        
        if not holdings:
            st.info("📭 尚無庫存，快去買入第一檔股票吧！")
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
                    f" ｜ {h['shares']:,} 股"
                    f" ｜ 成本 ${h['avg_cost']:.1f}"
                    f" ｜ 現價 ${h['current_price']:.1f}"
                    f" ｜ 損益 ${h['unrealized_pnl']:+,.0f} ({h['unrealized_pnl_pct']:+.2f}%)",
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
                        f"賣出股數（最多 {h['shares']:,}）",
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
                        else:
                            st.error(result["message"])
    
    # ─── Tab 4: 交易紀錄 ───
    with vtab4:
        st.subheader("📜 交易紀錄")
        
        orders = get_order_history(100, portfolio=st.session_state.vt_portfolio)
        
        if not orders:
            st.info("📭 尚無交易紀錄")
        else:
            st.caption(f"共 {len(orders)} 筆交易（最新在前）")
            
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
                            ｜手續 ${fee:.0f} ｜稅 ${tax:.0f}
                            ｜實收 ${net:,.0f}
                            ｜損益 <span style="color: {'#e74c3c' if pnl >= 0 else '#2ecc71'};">${pnl:+,.0f}</span>
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
                            ｜手續 ${fee:.0f}
                            ｜餘額 ${o['cash_after']:,.0f}
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ============================================================
# 模式 6：分析輔助
# ============================================================
elif mode == "🧠 分析輔助":
    st.header(f"🧠 分析輔助 — {stock_id} {stock_name}")
    
    tabs = st.tabs(["📊 綜合分析摘要", "📋 基本面", "🏢 三大法人", "📈 多週期技術"])
    
    # Tab 1: 綜合分析摘要
    with tabs[0]:
        # 買賣建議大卡（最上方顯眼位置）
        with st.spinner("評分中..."):
            rec = assess_recommendation(stock_id, stock_name, data)
        
        score = rec["score"]
        rating = rec["rating"]
        pt = rec.get("price_targets", {})
        
        # 根據評級顯示顏色
        if "強烈買進" in rating:
            card_color = "#00a86b"
            emoji = "🚀"
        elif "買進" in rating:
            card_color = "#27ae60"
            emoji = "📈"
        elif "賣出" in rating and "強烈" in rating:
            card_color = "#e74c3c"
            emoji = "☠️"
        elif "賣出" in rating:
            card_color = "#c0392b"
            emoji = "📉"
        else:
            card_color = "#7f8c8d"
            emoji = "⏳"
        
        # 價格區間字串
        buy_str = ""
        sell_str = ""
        stop_str = ""
        if pt:
            if pt.get("buy_zones"):
                bz = pt["buy_zones"]
                buy_low = min(b[0] for b in bz)
                buy_high = max(b[1] for b in bz)
                buy_str = f"買進 {buy_low:.2f} ~ {buy_high:.2f}"
            if pt.get("sell_zones"):
                sz = pt["sell_zones"]
                sell_low = min(s[0] for s in sz)
                sell_high = max(s[1] for s in sz)
                sell_str = f"賣出 {sell_low:.2f} ~ {sell_high:.2f}"
            if pt.get("stop_loss"):
                stop_str = f"停損 {pt['stop_loss']:.2f}"
        
        current_price = pt.get("current_price", 0)
        
        html_card = f"""
        <div style="
            background: linear-gradient(135deg, {card_color} 0%, {card_color}dd 100%);
            border-radius: 16px;
            padding: 24px 30px;
            margin: 10px 0 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="font-size: 14px; opacity: 0.8;">綜合評分</span>
                    <div style="font-size: 48px; font-weight: 800; line-height: 1.1;">
                        {score:+.0f}
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 14px; opacity: 0.8;">操作建議</span>
                    <div style="font-size: 32px; font-weight: 700;">
                        {emoji} {rating}
                    </div>
                </div>
            </div>
            <div style="margin-top: 12px; height: 6px; background: rgba(255,255,255,0.3); border-radius: 3px;">
                <div style="width: {(score + 100) / 2:.0f}%; height: 100%; background: white; border-radius: 3px;"></div>
            </div>
            <div style="display: flex; justify-content: space-around; margin-top: 16px; gap: 10px; flex-wrap: wrap;">
                <div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 10px; padding: 10px 18px; min-width: 120px;">
                    <div style="font-size: 12px; opacity: 0.7;">現價</div>
                    <div style="font-size: 22px; font-weight: 700;">{current_price:.2f}</div>
                </div>
                {('<div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 10px; padding: 10px 18px; min-width: 120px;"><div style="font-size: 12px; opacity: 0.7;">買進區間</div><div style="font-size: 18px; font-weight: 600; color: #a8e6cf;">' + buy_str + '</div></div>') if buy_str else ''}
                {('<div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 10px; padding: 10px 18px; min-width: 120px;"><div style="font-size: 12px; opacity: 0.7;">賣出區間</div><div style="font-size: 18px; font-weight: 600; color: #ffb3b3;">' + sell_str + '</div></div>') if sell_str else ''}
                {('<div style="text-align: center; background: rgba(255,255,255,0.15); border-radius: 10px; padding: 10px 18px; min-width: 120px;"><div style="font-size: 12px; opacity: 0.7;">停損</div><div style="font-size: 18px; font-weight: 600; color: #ff8a80;">' + stop_str + '</div></div>') if stop_str else ''}
            </div>
        </div>
    """
    st.components.v1.html(html_card, height=220)
        
        # 詳細說明
        st.subheader("📋 評分明細")
        for line in rec["details"]:
            st.markdown(line)
        
        # 強弱勢對比
        st.subheader("📊 大盤對比")
        try:
            col1, col2 = st.columns(2)
            with col1:
                twii = fetch_historical("t00", months=6)
                if not twii.empty and len(data) > 1:
                    stock_return = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100
                    # 大盤用加權指數替代：抓 0050 近似大盤
                    t50 = fetch_historical("0050", months=6)
                    if not t50.empty:
                        market_return = (t50["Close"].iloc[-1] - t50["Close"].iloc[0]) / t50["Close"].iloc[0] * 100
                        col1.metric(f"{stock_id} 期間報酬", f"{stock_return:+.2f}%")
                        col2.metric("0050 同期報酬", f"{market_return:+.2f}%")
                        diff = stock_return - market_return
                        if diff > 0:
                            st.success(f"✅ {stock_id} 表現優於大盤 {diff:+.2f}%")
                        else:
                            st.warning(f"⚠️ {stock_id} 表現落後大盤 {diff:+.2f}%")
        except Exception:
            pass
    
    # Tab 2: 基本面
    with tabs[1]:
        st.subheader("📋 基本面速覽")
        with st.spinner("查詢中..."):
            fund = fetch_fundamentals(stock_id)
        if "error" not in fund:
            cols = st.columns(3)
            cols[0].metric("本益比 (P/E)", f"{fund['pe_ratio']:.2f}" if fund['pe_ratio'] else "N/A")
            cols[1].metric("殖利率", f"{fund['dividend_yield']:.2f}%" if fund['dividend_yield'] else "N/A")
            cols[2].metric("股價淨值比 (P/B)", f"{fund['pb_ratio']:.2f}" if fund['pb_ratio'] else "N/A")
            
            # PE 評估
            pe = fund.get("pe_ratio")
            if pe:
                if pe < 10:
                    st.info(f"💡 本益比 {pe:.1f} 偏低，可能價值低估")
                elif pe < 18:
                    st.info(f"✅ 本益比 {pe:.1f} 在合理偏低區間")
                elif pe < 25:
                    st.info(f"⚖️ 本益比 {pe:.1f} 在合理區間")
                elif pe < 40:
                    st.info(f"⚠️ 本益比 {pe:.1f} 偏高，留意成長能否支撐")
                else:
                    st.info(f"🔴 本益比 {pe:.1f} 顯著偏高，可能存在泡沫風險")
            
            # 殖利率評估
            dy = fund.get("dividend_yield")
            if dy:
                if dy > 8:
                    st.info(f"💡 殖利率 {dy:.1f}% 非常高，留意是否為一次性配息")
                elif dy > 5:
                    st.info(f"✅ 殖利率 {dy:.1f}% 不錯，高於定存")
                elif dy > 3:
                    st.info(f"✅ 殖利率 {dy:.1f}% 尚可")
                else:
                    st.info(f"⚪ 殖利率 {dy:.1f}% 偏低")
            
            st.caption(f"資料日期：{fund.get('report_season', 'N/A')}")
        else:
            st.warning(f"無法取得基本面資料：{fund['error']}")
    
    # Tab 3: 三大法人
    with tabs[2]:
        st.subheader("🏢 三大法人買賣超")
        with st.spinner("查詢中..."):
            inst_df = fetch_institutional_trading(stock_id)
        if not inst_df.empty:
            display_cols = ["類別", "買進(張)", "賣出(張)", "買賣超(張)"]
            inst_display = inst_df[["類別", "買進(張)", "賣出(張)", "買賣超(張)"]].copy()
            
            # 顏色標記
            def color_net(val):
                if isinstance(val, (int, float)):
                    return "🟢" if val > 0 else "🔴" if val < 0 else "⚪"
                return ""
            
            inst_display["方向"] = inst_display["買賣超(張)"].apply(
                lambda x: color_net(x) + f" {x:+,d}" if isinstance(x, (int, float)) else str(x)
            )
            
            st.dataframe(inst_display, use_container_width=True, hide_index=True)
        else:
            st.warning("今日三大法人資料尚未更新，或非交易日")
        
        st.caption("📌 外資>0:買超 / <0:賣超；投信>0:買超 / <0:賣超；自營>0:買超 / <0:賣超")
    
    # Tab 4: 多週期技術分析
    with tabs[3]:
        st.subheader("📈 多週期技術面")
        
        period_tabs = st.tabs(["月線(20日)", "季線(60日)", "半年線(120日)"])
        
        for pt_idx, (p_name, p_days, p_color) in enumerate([
            ("月線", 20, "#9b59b6"),
            ("季線", 60, "#3498db"),
            ("半年線", 120, "#2ecc71"),
        ]):
            with period_tabs[pt_idx]:
                if f"MA{p_days}" in data.columns:
                    latest_val = data[f"MA{p_days}"].iloc[-1]
                    prev_val = data[f"MA{p_days}"].iloc[-2] if len(data) > 1 else latest_val
                    change = latest_val - prev_val
                    
                    ma_dir = "📈 上揚" if change > 0 else "📉 下跌"
                    st.metric(f"{p_name} ({latest_val:.1f})", 
                              f"{latest_val:.1f}",
                              f"{change:+.2f} ({ma_dir})")
                    
                    # 與股價的距離
                    close = data["Close"].iloc[-1]
                    dist = (close - latest_val) / latest_val * 100
                    if abs(dist) < 1:
                        st.info(f"📊 股價與{p_name}幾乎貼齊 ({dist:+.2f}%)")
                    elif dist > 0:
                        st.info(f"📈 股價在{p_name}之上 {dist:+.2f}%（偏多）")
                    else:
                        st.info(f"📉 股價在{p_name}之下 {abs(dist):.2f}%（偏空）")
                    
                    # 均線斜率趨勢
                    slope_days = 10
                    if len(data) > slope_days:
                        recent_ma = data[f"MA{p_days}"].iloc[-slope_days:].values
                        slope = (recent_ma[-1] - recent_ma[0]) / recent_ma[0] * 100
                        if slope > 2:
                            st.success(f"✅ {p_name}近期斜率 +{slope:.1f}%，上升趨勢明確")
                        elif slope > 0:
                            st.info(f"📊 {p_name}近期斜率 +{slope:.1f}%，緩步上揚")
                        elif slope > -2:
                            st.warning(f"⚠️ {p_name}近期斜率 {slope:.1f}%，平緩偏弱")
                        else:
                            st.error(f"🔴 {p_name}近期斜率 {slope:.1f}%，明顯下彎")
elif mode == "📋 多股掃描":
    st.header("📋 多股掃描")
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
                status_text.text(f"掃描中：{sid} {POPULAR_STOCKS.get(sid, '')}")
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


# ============================================================
# 頁尾
# ============================================================
st.divider()
st.caption(
    "⚠️ 免責聲明：本系統僅供學習與分析參考，不構成任何投資建議。"
    "資料來源為 TWSE/TPEx 公開資訊，即時性與準確性請以官方為準。"
)
