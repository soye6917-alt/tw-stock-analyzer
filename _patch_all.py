import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
lines = f.readlines()
f.close()

# ---- 1. Add radio option ----
for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith('"\U0001f4e1') and s.endswith("輔助\"]"):
        # Already added, skip
        break
    if '"\U0001f4e1 \u9078\u80a1\u7be9\u9078\u5668"' in s or '"📡 選股篩選器"' in s:
        break
else:
    # Add to radio
    for i, line in enumerate(lines):
        if '"📊 看盤與技術分析", "🔄 策略回測", "📋 多股掃描", "🏆 每日推薦", "🎮 虛擬交易", "🧠 分析輔助"' in line:
            lines[i] = line.replace(
                '"📊 看盤與技術分析", "🔄 策略回測", "📋 多股掃描", "🏆 每日推薦", "🎮 虛擬交易", "🧠 分析輔助"',
                '"📊 看盤與技術分析", "🔄 策略回測", "📋 多股掃描", "📡 選股篩選器", "🏆 每日推薦", "📐 型態辨識", "🎮 虛擬交易", "🧠 分析輔助"'
            )
            break

# ---- 2. Add imports ----
for i, line in enumerate(lines):
    if "from daily_picks" in line:
        lines.insert(i+1, 'from stock_screener import filter_stocks\n')
        lines.insert(i+2, 'import pattern_recognition as pr\n')
        break

# ---- 3. Add Screener mode block (before 每日推薦) ----
SCREENER_BLOCK = [
    "\n",
    "# ============================================================\n",
    "# 模式 4.5: 📡 選股篩選器\n",
    "# ============================================================\n",
    'elif mode == "📡 選股篩選器":\n',
    '    st.header("📡 選股篩選器")\n',
    '    st.caption("自訂條件篩選全部上市股票——價位、本益比、殖利率、成交量、技術訊號")\n',
    "\n",
    '    with st.expander("⚙️ 篩選條件", expanded=True):\n',
    '        col1, col2, col3 = st.columns(3)\n',
    "        with col1:\n",
    '            price_range = st.slider("股價範圍", 0, 2000, (0, 500))\n',
    '            vol_min = st.number_input("最低成交量(張)", 0, 100000, 1000, step=500)\n',
    "        with col2:\n",
    '            pe_range = st.slider("本益比範圍", 0, 200, (0, 50))\n',
    '            dy_range = st.slider("殖利率範圍(%)", 0, 20, (0, 15))\n',
    "        with col3:\n",
    '            tech_filter = st.selectbox("技術訊號", ["全部", "多頭", "空頭", "中立"])\n',
    '            max_stocks = st.slider("最多掃描檔數", 50, 400, 150, step=50)\n',
    "\n",
    '    if st.button("🔍 開始篩選", type="primary", use_container_width=True):\n',
    '        with st.spinner(f"正在掃描 {max_stocks} 檔股票(約需 30-60 秒)..."):\n',
    "            df = filter_stocks(\n",
    "                price_min=price_range[0], price_max=price_range[1],\n",
    "                pe_min=pe_range[0], pe_max=pe_range[1],\n",
    "                dy_min=dy_range[0], dy_max=dy_range[1],\n",
    "                vol_min=vol_min,\n",
    '                tech_signal=tech_filter if tech_filter != "全部" else None,\n',
    "                max_stocks=max_stocks,\n",
    "            )\n",
    "        if df.empty:\n",
    '            st.warning("沒有符合條件的股票，請將條件放寬")\n',
    "        else:\n",
    '            st.success(f"🟢 找到 {len(df)} 檔符合條件的股票")\n',
    "            display = df[['sid', 'name', 'price', 'change_pct', 'volume',\n",
    "                        'pe', 'div_yield', 'signal', 'rsi']].copy()\n",
    '            display.columns = ["代碼", "名稱", "價位", "變動(%)",\n',
    '                           "成交量", "PE", "殖利率(%)", "技術訊號", "RSI"]\n',
    '            display["價位"] = display["價位"].apply(lambda x: f"${x:,.2f}")\n',
    '            display["變動(%)"] = display["變動(%)"].apply(\n',
    '                lambda x: f"🟢 {x:+.2f}%" if x >= 0 else f"🔴 {x:+.2f}%")\n',
    '            display["成交量"] = display["成交量"].apply(\n',
    '                lambda x: f"{x:,}" if x >= 1000 else str(x))\n',
    '            display["PE"] = display["PE"].apply(\n',
    '                lambda x: str(round(x, 1)) if x and x > 0 else "-")\n',
    '            display["殖利率(%)"] = display["殖利率(%)"].apply(\n',
    '                lambda x: f"{x:.1f}%" if x and x > 0 else "-")\n',
    "            st.dataframe(display, use_container_width=True, hide_index=True)\n",
    '            csv = df.to_csv(index=False).encode("utf-8-sig")\n',
    '            st.download_button("⬇ 下載 CSV", csv, "stock_screener.csv",\n',
    '                              "text/csv", use_container_width=True)\n',
    "\n",
]

# Insert screener block before daily picks
for i, line in enumerate(lines):
    if "# 模式 4:\U0001f3c6" in line:
        for nl in reversed(SCREENER_BLOCK):
            lines.insert(i, nl)
        break

# ---- 4. Add Pattern Recognition mode block (before 虛擬交易) ----
PATTERN_BLOCK = [
    "\n",
    "# ============================================================\n",
    "# 模式 5.5: 📐 技術型態辨識\n",
    "# ============================================================\n",
    'elif mode == "📐 型態辨識":\n',
    '    st.header("📐 技術型態辨識")\n',
    '    st.caption("自動辨識型態——W底、M頭、頭肩頂/底、箱型突破")\n',
    "\n",
    '    sid = st.text_input("股票代碼", "2330", max_chars=6).strip()\n',
    '    sname = get_stock_name(sid)\n',
    '    st.caption(f"{sid} {sname}")\n',
    "\n",
    "    lookback_days = st.number_input(\n",
    '        "分析期間(日)", 30, 365, 120, step=30)\n',
    "    months = max(3, lookback_days // 30 + 1)\n",
    '    if st.button("🔍 執行型態辨識", type="primary", use_container_width=True):\n',
    '        with st.spinner("正在分析..."):\n',
    "            df = load_data(sid, months)\n",
    '            if df.empty:\n',
    '                st.error("無法取得資料")\n',
    "            else:\n",
    "                df = add_all_indicators(df)\n",
    "                patterns = pr.detect_all_patterns(df, lookback=lookback_days)\n",
    "                if not patterns:\n",
    '                    st.info("💭 無辨識到明顯的技術型態")\n',
    "                else:\n",
    '                    st.success(f"🟢 找到 {len(patterns)} 個型態")\n',
    "                    for p in patterns:\n",
    '                        conf = p.get("confidence", "中")\n',
    '                        conf_icon = {"高": "🟢", "中": "🟡", "低": "🔵"}.get(conf, "⚪")\n',
    '                        with st.expander(f"{conf_icon} {p[\"type\"]}  (確信度:{conf})", expanded=True):\n',
    "                            info = \" | \".join(\n",
    '                                f"**{k}:** {v}" for k, v in p.items()\n',
    '                                if k not in ("type", "confidence"))\n',
    "                            st.markdown(info)\n",
    "                            fig = go.Figure()\n",
    "                            fig.add_trace(go.Candlestick(\n",
    '                                x=df.index, open=df["Open"], high=df["High"],\n',
    '                                low=df["Low"], close=df["Close"], name=f"{sid} {sname}"))\n',
    '                            fig.update_layout(title=f"{sid} {sname} - {p[\"type\"]}",\n',
    "                                              height=400, margin=dict(l=20, r=20, t=40, b=20))\n",
    "                            st.plotly_chart(fig, use_container_width=True)\n",
    "\n",
]

# Insert pattern block before virtual trading
for i, line in enumerate(lines):
    if "# 模式 5:\U0001f3ae" in line:
        for nl in reversed(PATTERN_BLOCK):
            lines.insert(i, nl)
        break

f = open(FPATH, "w", encoding="utf-8")
f.writelines(lines)
f.close()

import py_compile
try:
    py_compile.compile(FPATH, doraise=True)
    print("OK - syntax check passed")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
