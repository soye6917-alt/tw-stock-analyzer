import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

f = open(r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py", "r", encoding="utf-8")
lines = f.readlines()
f.close()

changes = []
for i, l in enumerate(lines):
    s = l.strip()
    # Fix virtual trading tabs
    if s.startswith("vtab1, vtab2, vtab3, vtab4 = st.tabs(") and "key=" not in s:
        lines[i] = l.rstrip() + ', key="vt_tabs")\n'
        changes.append(f"  Line {i+1}: vt_tabs")
    # Fix analysis tabs
    if s.startswith("tabs = st.tabs(") and '分析' in s and "key=" not in s:
        lines[i] = l.rstrip() + ', key="analysis_tabs")\n'
        changes.append(f"  Line {i+1}: analysis_tabs")

for c in changes:
    print(c)

if changes:
    f2 = open(r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py", "w", encoding="utf-8")
    f2.writelines(lines)
    f2.close()
    print("Done - file updated")
else:
    print("No changes needed")
