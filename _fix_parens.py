import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

f = open(r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py", "r", encoding="utf-8")
lines = f.readlines()
f.close()

for i, l in enumerate(lines):
    if i == 708:  # vt_tabs line (0-indexed)
        old = l
        lines[i] = l.replace(']), key="vt_tabs")', '], key="vt_tabs")')
        if old != lines[i]:
            print(f"Fixed line {i+1}: vt_tabs")

    if i == 1001:  # analysis_tabs line
        old = l
        lines[i] = l.replace(']), key="analysis_tabs")', '], key="analysis_tabs")')
        if old != lines[i]:
            print(f"Fixed line {i+1}: analysis_tabs")

f2 = open(r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py", "w", encoding="utf-8")
f2.writelines(lines)
f2.close()
print("Done")
