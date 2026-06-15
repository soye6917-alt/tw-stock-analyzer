import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
c = f.read()
f.close()

idx = c.find("elif mode == ")
count = 0
for i, ch in enumerate(c):
    if c[i:i+13] == "elif mode == ":
        count += 1
        end = c.index("\n", i)
        print(f"  #{count}: line starting at pos {i}: {c[i:end+1].strip()[:80]}")

print(f"\nTotal elif mode blocks: {count}")
