import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
c = f.read()
f.close()

# Replace the virtual trading header with auto-save version
OLD_HEADER = '''elif mode == "\U0001f3ae \u865b\u64ec\u4ea4\u6613":
    st.header("\U0001f3ae \u865b\u64ec\u4ea4\u6613\u5e02\u5834")
    st.caption("\u7528\u865b\u64ec\u8cc7\u91d1 $1,000,000 \u7df4\u7fd2\u53f0\u80a1\u8cb7\u8ce3\uff0c\u5373\u6642\u5831\u50f9\uff0c\u771f\u5be6\u624b\u7e8c\u8cbb")

    # \u5feb\u6377\u8cb7\u5165(\u5f9e\u6bcf\u65e5\u63a8\u85a6\u8df3\u8f49\u7528)'''

# Find the exact position
pos = c.find(OLD_HEADER)
if pos < 0:
    print("Old header not found with unicode escapes. Trying direct match...")
    # Try with regular commas
    OLD_HEADER2 = 'elif mode == "\U0001f3ae \u865b\u64ec\u4ea4\u6613":\n    st.header("\U0001f3ae \u865b\u64ec\u4ea4\u6613\u5e02\u5834")\n    st.caption("\u7528\u865b\u64ec\u8cc7\u91d1 $1,000,000 \u7df4\u7fd2\u53f0\u80a1\u8cb7\u8ce3,\u5373\u6642\u5831\u50f9,\u771f\u5be6\u624b\u7e8c\u8cbb")\n\n    # \u5feb\u6377\u8cb7\u5165(\u5f9e\u6bcf\u65e5\u63a8\u85a6\u8df3\u8f49\u7528)'
    pos = c.find(OLD_HEADER2)
    if pos >= 0:
        print(f"Found at position {pos} (comma variant)")
        OLD_HEADER = OLD_HEADER2

if pos >= 0:
    print(f"Found at position {pos}")
    # Build replacement
    # Need dump of current session state for auto-save, but batch mode uses empty dict
    dummy_json = '{"cash": 1000000, "holdings": {}, "orders": []}'
    
    NEW_HEADER = '''elif mode == "\U0001f3ae \u865b\u64ec\u4ea4\u6613":
    st.header("\U0001f3ae \u865b\u64ec\u4ea4\u6613\u5e02\u5834")
    st.caption("\u7528\u865b\u64ec\u8cc7\u91d1 $1,000,000 \u7df4\u7fd2\u53f0\u80a1\u8cb7\u8ce3\uff0c\u5373\u6642\u5831\u50f9\uff0c\u771f\u5be6\u624b\u7e8c\u8cbb")

    # \u81ea\u52d5\u5b58\u6a94\uff1a\u6bcf\u6b21\u64cd\u4f5c\u5f8c\u5b58\u5230\u700f\u89bd\u5668 localStorage
    _vt_key = "streamlit_vt_portfolio"
    st.components.v1.html(
        f"""<script>
try {{ localStorage.setItem('{_vt_key}', JSON.stringify({dummy_json})); }}
catch(e) {{}}
</script>""",
        height=0,
    )
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

    # \u5feb\u6377\u8cb7\u5165(\u5f9e\u6bcf\u65e5\u63a8\u85a6\u8df3\u8f49\u7528)'''
    
    c = c.replace(OLD_HEADER, NEW_HEADER, 1)
    f = open(FPATH, "w", encoding="utf-8")
    f.write(c)
    f.close()
    
    import py_compile
    try:
        py_compile.compile(FPATH, doraise=True)
        print("OK - syntax check passed")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
else:
    print("Could not find the old text!")
    # Show what's around position 28905
    print(f"Text at 28905: {repr(c[28905:29050])}")
