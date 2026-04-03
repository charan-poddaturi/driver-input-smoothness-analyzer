"""
DISA — SQLite Database Layer
==============================
Stores analysis sessions, optimizer runs, and anomaly events persistently.
Uses plain sqlite3 (stdlib) — no ORM required.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DB_PATH = Path(__file__).parent.parent / "data" / "disa.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """Create tables if they don't exist."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            filename    TEXT,
            fs          REAL DEFAULT 25.0,
            n_samples   INTEGER,
            duration_s  REAL,
            overall_score REAL,
            grade       TEXT,
            sub_scores  TEXT,    -- JSON
            features    TEXT,    -- JSON
            weights     TEXT     -- JSON
        );

        CREATE TABLE IF NOT EXISTS anomaly_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            start_time  REAL,
            end_time    REAL,
            severity    REAL,
            event_type  TEXT,
            channels    TEXT,    -- JSON list
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            method      TEXT,
            csv_files   TEXT,    -- JSON list
            weights     TEXT,    -- JSON
            mse         REAL,
            mae         REAL,
            default_mse REAL,
            improvement_pct REAL,
            sensitivity TEXT     -- JSON
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def save_session(
    session_id: str,
    filename: str,
    fs: float,
    n_samples: int,
    duration_s: float,
    score_result: dict,
    anomalies: List[dict],
    db_path: Path = DB_PATH,
):
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    cur.execute(
        """
        INSERT OR REPLACE INTO sessions
          (id, created_at, filename, fs, n_samples, duration_s,
           overall_score, grade, sub_scores, features, weights)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            now,
            filename,
            fs,
            n_samples,
            duration_s,
            score_result.get("overall_score"),
            score_result.get("grade"),
            json.dumps(score_result.get("sub_scores", {})),
            json.dumps(score_result.get("normalized_features", {})),
            json.dumps(score_result.get("weights", [])),
        ),
    )

    # Save anomaly events
    for ev in anomalies:
        cur.execute(
            """
            INSERT INTO anomaly_events
              (session_id, start_time, end_time, severity, event_type, channels)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                ev.get("start_time"),
                ev.get("end_time"),
                ev.get("severity"),
                ev.get("event_type"),
                json.dumps(ev.get("affected_channels", [])),
            ),
        )

    conn.commit()
    conn.close()


def get_session(session_id: str, db_path: Path = DB_PATH) -> Optional[dict]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    for key in ("sub_scores", "features", "weights"):
        if d.get(key):
            d[key] = json.loads(d[key])
    return d


def list_sessions(db_path: Path = DB_PATH) -> List[dict]:
    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, filename, overall_score, grade FROM sessions ORDER BY created_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_optimizer_run(run_data: dict, db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO optimizer_runs
          (created_at, method, csv_files, weights, mse, mae, default_mse, improvement_pct, sensitivity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            run_data.get("method"),
            json.dumps(run_data.get("csv_files", [])),
            json.dumps(run_data.get("optimized_weights")),
            run_data.get("mse"),
            run_data.get("mae"),
            run_data.get("default_mse"),
            run_data.get("mse_improvement_pct"),
            json.dumps(run_data.get("sensitivity", {})),
        ),
    )
    conn.commit()
    conn.close()
