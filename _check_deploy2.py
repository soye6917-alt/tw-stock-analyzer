import requests, sys, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE = "https://tw-stock-analyzer-b99m9judkv54tthzqsrmhm.streamlit.app"

# Try various Streamlit internal endpoints
endpoints = [
    "/healthz",
    "/healthz-lb",
    "/_stcore/health",
    "/_stcore/host-config",
    "/_stcore/upload_file/",
    "/_stcore/stream",
]

for ep in endpoints:
    try:
        r = requests.get(f"{BASE}{ep}", timeout=10, allow_redirects=False)
        print(f"  {ep}: {r.status_code}")
        if r.text and len(r.text) < 500:
            print(f"    Body: {r.text}")
    except Exception as e:
        print(f"  {ep}: ERROR - {e}")

# Also check main page content
print("\n--- Main page content ---")
r = requests.get(BASE, allow_redirects=True, timeout=30)
# Extract text content
text = re.sub(r'<script[^>]*>.*?</script>', '', r.text, flags=re.DOTALL)
text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
text = re.sub(r'<[^>]+>', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()
print(text[:1000])
