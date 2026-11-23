"""Time long-text requests and probe /health while one runs."""
import json
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8765"
WORDS = "the quick brown fox jumps over the lazy dog near the river bank today "


def post(body):
    req = urllib.request.Request(BASE + "/predict", json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        code = f"{e.code} {e.read()[:200]!r}"
    return code, time.perf_counter() - t


for n in (500, 2000, 8000):
    text = WORDS * (n // 14)  # ~1 token per word
    done = threading.Event()
    health = []

    def probe():
        while not done.is_set():
            t = time.perf_counter()
            urllib.request.urlopen(BASE + "/health", timeout=60).read()
            health.append(time.perf_counter() - t)
            time.sleep(0.2)

    th = threading.Thread(target=probe)
    th.start()
    code, dt = post({"text": [text]})
    done.set()
    th.join()
    print(f"~{n} tokens: {code} in {dt:.2f}s; /health worst {max(health) * 1000:.0f} ms")

print("over-long:", post({"text": [WORDS * 2400]}))
print("batch 32 x ~500 tokens:", post({"text": [WORDS * 36] * 32}))
