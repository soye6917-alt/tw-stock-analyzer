import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
content = f.read()
f.close()

OLD = '''elif mode == "🎮 虛擬交易":
    st.header("🎮 虛擬交易市場")
    st.caption("用虛擬資金 $1,000,000 練習台股買賣，即時報價，真實手續費")

    # 快捷買入(從每日推薦跳轉用)
    if "quick_buy" in st.session_state and st.session_state.quick_buy:
        pass  # 由後續邏輯處理'''

NEW = '''elif mode == "🎮 虛擬交易":
    st.header("🎮 虛擬交易市場")
    st.caption("用虛擬資金 $1,000,000 練習台股買賣，即時報價，真實手續費")

    # ── 自動存檔：每次操作後存到瀏覽器 localStorage ──
    _vt_key = "streamlit_vt_portfolio"
    # 自動儲存（每次頁面渲染時寫入）
    st.components.v1.html(
        f"""<script>
try {{ localStorage.setItem('{_vt_key}', JSON.stringify({json.dumps(st.session_state.vt_portfolio, ensure_ascii=False)})); }}
catch(e) {{}}
</script>""",
        height=0,
    )
    # 自動還原（僅首次載入時執行一次）
    if "vt_restored" not in st.session_state:
        st.session_state.vt_restored = False
    if not st.session_state.vt_restored:
        _saved_raw = st.query_params.get("__vt_restore")
        if _saved_raw:
            try:
                _data = json.loads(_saved_raw)
                if _data.get("orders") and len(_data["orders"]) > len(st.session_state.vt_portfolio.get("orders", [])):
                    st.session_state.vt_portfolio = _data
                st.session_state.vt_restored = True
                st.query_params.clear()
                st.rerun()
            except Exception:
                st.session_state.vt_restored = True
        else:
            st.components.v1.html(
                f"""<script>
try {{
    var d = localStorage.getItem('{_vt_key}');
    if (d && d.length > 20) {{
        var u = new URL(window.location.href);
        u.searchParams.set('__vt_restore', encodeURIComponent(d));
        window.location.replace(u.toString());
    }}
}} catch(e) {{}}
</script>""",
                height=0,
            )
            st.session_state.vt_restored = True

    # 快捷買入(從每日推薦跳轉用)
    if "quick_buy" in st.session_state and st.session_state.quick_buy:
        pass  # 由後續邏輯處理'''

if OLD in content:
    content = content.replace(OLD, NEW, 1)
    f = open(FPATH, "w", encoding="utf-8")
    f.write(content)
    f.close()
    
    # Syntax check
    import py_compile
    try:
        py_compile.compile(FPATH, doraise=True)
        print("OK - syntax check passed")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
else:
    print("ERROR: Could not find the old text to replace!")
    # Debug: find similar text
    idx = content.find('elif mode == "')
    if idx >= 0:
        print(f"Found 'elif mode' at position {idx}")
        print(content[idx:idx+300])
