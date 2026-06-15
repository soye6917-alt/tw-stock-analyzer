import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
lines = f.readlines()
f.close()

# Find the virtual trading section
vt_emoji = "\U0001f3ae"  # 🎮
for i, line in enumerate(lines):
    if vt_emoji in line and "elif mode" in line:
        print(f"Virtual trading section at line {i+1}: {line.rstrip()}")
        # Show next few lines
        for j in range(i, min(i+10, len(lines))):
            print(f"  {j+1}: {lines[j].rstrip()}")
        break
