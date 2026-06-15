import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
c = f.read()
f.close()

# Find exact text at position 28905
old = c[28905:29180]
print("OLD TEXT:")
print(repr(old))
print()

# Build the replacement header
# Note: {{ -> { in outer f-string, so runtime Python sees single braces
new_header = (
    'elif mode == "\U0001f3ae \u865b\u64ec\u4ea4\u6613":\n'
    '    st.header("\U0001f3ae \u865b\u64ec\u4ea4\u6613\u5e02\u5834")\n'
    '    st.caption("\u7528\u865b\u64ec\u8cc7\u91d1 $1,000,000 \u7df4\u7fd2\u53f0\u80a1\u8cb7\u8ce3\uff0c\u5373\u6642\u5831\u50f9\uff0c\u771f\u5be6\u624b\u7e8c\u8cbb")\n'
    '\n'
    '    # \u81ea\u52d5\u5b58\u6a94\uff1a\u6bcf\u6b21\u64cd\u4f5c\u5f8c\u5b58\u5230\u700f\u89bd\u5668 localStorage\n'
    '    _vt_key = "streamlit_vt_portfolio"\n'
    '    st.components.v1.html(\n'
    '        f"""<script>\n'
    'try {{ localStorage.setItem(\'{_vt_key}\', JSON.stringify({json.dumps(st.session_state.vt_portfolio, ensure_ascii=False)})); }}\n'
    'catch(e) {{}}\n'
    '</script>""",\n'
    '        height=0,\n'
    '    )\n'
    '    if "vt_restored" not in st.session_state:\n'
    '        st.session_state.vt_restored = False\n'
    '    if not st.session_state.vt_restored:\n'
    '        _saved_raw = st.query_params.get("__vt_restore")\n'
    '        if _saved_raw:\n'
    '            try:\n'
    '                _data = json.loads(_saved_raw)\n'
    '                if _data.get("orders") and len(_data["orders"]) > len(st.session_state.vt_portfolio.get("orders", [])):\n'
    '                    st.session_state.vt_portfolio = _data\n'
    '                st.session_state.vt_restored = True\n'
    '                st.query_params.clear()\n'
    '                st.rerun()\n'
    '            except Exception:\n'
    '                st.session_state.vt_restored = True\n'
    '        else:\n'
    '            st.components.v1.html(\n'
    '                f"""<script>\n'
    'try {{\n'
    '    var d = localStorage.getItem(\'{_vt_key}\');\n'
    '    if (d && d.length > 20) {{\n'
    '        var u = new URL(window.location.href);\n'
    '        u.searchParams.set(\'__vt_restore\', encodeURIComponent(d));\n'
    '        window.location.replace(u.toString());\n'
    '    }}\n'
    '}} catch(e) {{}}\n'
    '</script>""",\n'
    '                height=0,\n'
    '            )\n'
    '            st.session_state.vt_restored = True\n'
    '\n'
    '    # \u5feb\u6377\u8cb7\u5165(\u5f9e\u6bcf\u65e5\u63a8\u85a6\u8df3\u8f49\u7528)\n'
)

c = c.replace(old, new_header, 1)

# Verify
if new_header in c:
    print("Replacement succeeded!")
else:
    print("WARNING: Replacement text not found in output!")
    # Show diff around insertion point
    idx2 = c.find('localStorage')
    if idx2 > 0:
        print(f"\nOutput around localStorage at {idx2}:")
        print(repr(c[idx2-50:idx2+200]))

f = open(FPATH, "w", encoding="utf-8")
f.write(c)
f.close()

import py_compile
try:
    py_compile.compile(FPATH, doraise=True)
    print("OK - syntax check passed")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
