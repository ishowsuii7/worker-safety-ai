"""Wearable sensor readings: storage and threshold logic (no web code here)."""
import sqlite3
from datetime import datetime

import yaml

DB_PATH = "data/sensors.db"
CONFIG_PATH = "configs/sensors.yaml"
ONLINE_WITHIN_S = 15

DEFAULTS = {
    "gas": {"warning": 1800, "critical": 2800},
    "temperature_c": {"warning": 38, "critical": 45},
}
ORDER = {"ok": 0, "warning": 1, "critical": 2}


def load_thresholds(path=CONFIG_PATH):
    """Read the thresholds file on every call, so edits apply without a restart."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    return {k: {**d, **(cfg.get(k) or {})} for k, d in DEFAULTS.items()}


def evaluate(gas, temp_c, fall, thr):
    """Returns (status, reasons). status is 'ok', 'warning' or 'critical'."""
    status, reasons = "ok", []

    def bump(level, why):
        nonlocal status
        reasons.append(why)
        if ORDER[level] > ORDER[status]:
            status = level

    if gas >= thr["gas"]["critical"]:
        bump("critical", "gas level critical")
    elif gas >= thr["gas"]["warning"]:
        bump("warning", "gas level high")

    if temp_c >= thr["temperature_c"]["critical"]:
        bump("critical", "temperature critical")
    elif temp_c >= thr["temperature_c"]["warning"]:
        bump("warning", "temperature high")

    if fall:
        bump("critical", "fall detected by the wearable")
    return status, reasons


def _connect(db_path=DB_PATH):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, worker_id INTEGER, gas REAL, temp_c REAL,
            fall INTEGER, status TEXT, reasons TEXT
        )""")
    return db


def save_reading(worker, gas, temp_c, fall, status, reasons, db_path=DB_PATH):
    db = _connect(db_path)
    try:
        db.execute(
            "INSERT INTO sensor_readings (ts, worker_id, gas, temp_c, fall, status, reasons) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), worker, gas, temp_c,
             int(fall), status, "; ".join(reasons)))
        db.commit()
    finally:
        db.close()


def latest_per_worker(db_path=DB_PATH):
    db = _connect(db_path)
    try:
        rows = db.execute(
            "SELECT * FROM sensor_readings WHERE id IN "
            "(SELECT MAX(id) FROM sensor_readings GROUP BY worker_id) ORDER BY worker_id"
        ).fetchall()
    finally:
        db.close()
    out = []
    now = datetime.now()
    for r in rows:
        d = dict(r)
        age = (now - datetime.fromisoformat(d["ts"])).total_seconds()
        d["age_s"] = round(age)
        d["online"] = age < ONLINE_WITHIN_S
        d["fall"] = bool(d["fall"])
        out.append(d)
    return out


def history(worker, limit=100, db_path=DB_PATH):
    db = _connect(db_path)
    try:
        rows = db.execute(
            "SELECT * FROM sensor_readings WHERE worker_id = ? ORDER BY id DESC LIMIT ?",
            (worker, limit)).fetchall()
    finally:
        db.close()
    return [dict(r) for r in rows]