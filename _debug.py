import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPATH = r"F:\Portable_V1.2.2\data\UserProfile\.openclaw\workspace\tw-stock-analyzer\app.py"
f = open(FPATH, "r", encoding="utf-8")
c = f.read()
f.close()

# Find the virtual trading mode section
# Pattern: "elif mode == " followed by virtual trading emoji
target = 'elif mode == "\U0001f3ae'
pos = c.find(target)
print(f"Virtual trading starts at: {pos}")

# Show 200 chars around the broken area
for start in range(pos, min(pos + 1000, len(c)), 200):
    end = min(start + 200, len(c))
    chunk = c[start:end]
    # Print with visible line breaks
    escape_bracket = chunk.replace("{", "{{").replace("}", "}}")
    print(f"\n--- pos {start} ---")
    print(repr(chunk[:200]))
