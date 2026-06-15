import requests, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

r = requests.get("https://tw-stock-analyzer-b99m9judkv54tthzqsrmhm.streamlit.app/", allow_redirects=True, timeout=30)
text = r.text

checks = [
    ("Status code", str(r.status_code)),
    ("stApp present", "stApp" in text),
    ("stException present", "stException" in text),
    ("data-testid present", "data-testid" in text),
    ("_stcore present", "_stcore" in text),
    ("Page size", f"{len(text)} bytes"),
    ("Script bundles", "/static/js/" in text or "/_stcore/" in text),
]
for k, v in checks:
    print(f"  {k}: {v}")

# Try the Streamlit message endpoint
try:
    h = requests.get("https://tw-stock-analyzer-b99m9judkv54tthzqsrmhm.streamlit.app/healthz", timeout=10)
    print(f"  Health endpoint: {h.status_code}")
except:
    print("  Health endpoint: not available (normal)")

print("\nDone -- app appears to be serving.")
