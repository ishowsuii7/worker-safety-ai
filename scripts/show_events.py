import sqlite3

db = sqlite3.connect("data/events.db")
rows = db.execute("SELECT id, video_time, type, severity, worker_id, frame_no, snapshot "
                  "FROM events ORDER BY id").fetchall()
for r in rows:
    print(r)
print(len(rows), "events")
db.close()