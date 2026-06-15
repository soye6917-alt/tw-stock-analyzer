import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
c = f.read()
f.close()

# Add imports
old = "from daily_picks import get_daily_picks_with_context, SCAN_UNIVERSE\r\nfrom virtual_trading import ("
new = "from daily_picks import get_daily_picks_with_context, SCAN_UNIVERSE\r\nfrom stock_screener import filter_stocks\r\nimport pattern_recognition as pr\r\nfrom virtual_trading import ("

if old in c:
    c = c.replace(old, new, 1)
    print("Added stock_screener + pattern_recognition imports")
else:
    print("ERROR: Could not find import line!")
    idx = c.find("from daily_picks")
    if idx >= 0:
        print(f"Found at {idx}: {repr(c[idx:idx+80])}")

f = open(FPATH, "w", encoding="utf-8")
f.write(c)
f.close()

import py_compile
try:
    py_compile.compile(FPATH, doraise=True)
    print("OK - syntax check passed")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
