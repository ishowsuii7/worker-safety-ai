import sqlite3
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import sensors

DB_PATH = "data/events.db"
SNAP_DIR = Path("data/snapshots")
SNAP_DIR.mkdir(parents=True, exist_ok=True)
Path("data").mkdir(exist_ok=True)

app = FastAPI(title="Worker Safety AI")
app.mount("/snapshots", StaticFiles(directory=str(SNAP_DIR)), name="snapshots")


def query(sql, params=()):
    """Run a read-only query. Returns [] if the database or table does not exist yet."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in db.execute(sql, params).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        db.close()


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- camera events
@app.get("/api/events")
def list_events(limit: int = Query(50, le=500),
                type_: str | None = Query(None, alias="type"),
                severity: str | None = None):
    where, params = [], []
    if type_:
        where.append("type = ?")
        params.append(type_)
    if severity:
        where.append("severity = ?")
        params.append(severity)

    sql = "SELECT * FROM events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    rows = query(sql, params)
    for r in rows:
        name = Path(str(r["snapshot"]).replace("\\", "/")).name
        r["snapshot_url"] = f"/snapshots/{name}"
    return rows


@app.get("/api/stats")
def stats():
    by_type = {r["type"]: r["n"] for r in
               query("SELECT type, COUNT(*) AS n FROM events GROUP BY type")}
    by_sev = {r["severity"]: r["n"] for r in
              query("SELECT severity, COUNT(*) AS n FROM events GROUP BY severity")}
    workers = query("SELECT COUNT(DISTINCT worker_id) AS n FROM events")
    return {
        "total": sum(by_type.values()),
        "critical": by_sev.get("critical", 0),
        "warning": by_sev.get("warning", 0),
        "workers_involved": workers[0]["n"] if workers else 0,
        "by_type": by_type,
    }


# ------------------------------------------------------------ wearable (ESP32)
class SensorReading(BaseModel):
    worker: int = Field(ge=0, le=9999)
    gas: float = Field(ge=0, le=4095)        # raw ESP32 ADC value
    temp: float = Field(ge=-40, le=125)      # degrees Celsius
    fall: bool = False                       # optional: wearable's own fall flag


@app.post("/api/sensors")
def post_sensors(r: SensorReading):
    """The ESP32 posts one reading every second or so.
    The answer tells it whether to sound its buzzer."""
    status, reasons = sensors.evaluate(r.gas, r.temp, r.fall, sensors.load_thresholds())
    sensors.save_reading(r.worker, r.gas, r.temp, r.fall, status, reasons)
    return {"status": status, "reasons": reasons, "buzzer": status == "critical"}


@app.get("/api/sensors/latest")
def sensors_latest():
    return sensors.latest_per_worker()


@app.get("/api/sensors/history")
def sensors_history(worker: int, limit: int = Query(100, le=1000)):
    return sensors.history(worker, limit)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return Path("dashboard/index.html").read_text(encoding="utf-8")