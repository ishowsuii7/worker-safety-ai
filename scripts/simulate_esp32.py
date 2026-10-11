"""Pretends to be the ESP32 wearable, so the backend can be tested without hardware.

Usage:  python -m scripts.simulate_esp32 [worker] [seconds] [url]
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request


def reading_at(t):
    """One simulated reading, t = seconds since the start.
    0-15 s normal | 15-25 gas rises | 25-35 gas leak | 35-45 gas falls | 52-55 fall flag."""
    if t < 15:
        gas = 900
    elif t < 25:
        gas = 900 + (t - 15) / 10 * 2300
    elif t < 35:
        gas = 3200
    elif t < 45:
        gas = 3200 - (t - 35) / 10 * 2300
    else:
        gas = 900
    gas += random.uniform(-40, 40)
    temp = 31 + t * 0.15 + random.uniform(-0.3, 0.3)        # slow warm-up, crosses 38 C at ~47 s
    fall = 52 <= t < 55
    return round(max(0, min(4095, gas))), round(temp, 1), fall


def main(worker=7, seconds=60, url="http://127.0.0.1:8000/api/sensors"):
    print(f"Simulating wearable #{worker} for {seconds} s -> {url}")
    start = time.time()
    t = 0
    while t < seconds:
        gas, temp, fall = reading_at(t)
        body = json.dumps({"worker": worker, "gas": gas, "temp": temp, "fall": fall}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                answer = json.loads(resp.read())
            buzzer = "BUZZER ON" if answer.get("buzzer") else ""
            print(f"t={t:3d}s gas={gas:4d} temp={temp:4.1f} fall={int(fall)} -> "
                  f"{answer['status']:8s} {buzzer} {', '.join(answer['reasons'])}")
        except (urllib.error.URLError, OSError) as e:
            print(f"Could not reach the server ({e}). Is uvicorn running?")
            return
        t += 1
        time.sleep(max(0, start + t - time.time()))
    print("Done.")


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]) if len(a) > 1 else 7,
         int(a[2]) if len(a) > 2 else 60,
         a[3] if len(a) > 3 else "http://127.0.0.1:8000/api/sensors")