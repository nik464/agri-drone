"""
feedback_store.py — SQLite-backed storage for agronomist corrections (E2).

Schema:
  feedback        — individual correction records
  feedback_images — optional original image bytes (for retrain)

Thread-safe via one connection per call + WAL mode.
"""

import json
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from loguru import logger


_DB_DIR = Path(__file__).resolve().parent.parent.parent.parent / "outputs" / "feedback"
_DB_PATH = _DB_DIR / "feedback.db"


# ── Data models ──────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    """A single agronomist correction."""
    id: Optional[int] = None
    detection_id: Optional[int] = None          # links to detection_history id
    image_hash: str = ""                        # MD5 from _image_hash()
    filename: str = ""

    # What the system predicted
    predicted_disease: str = ""
    predicted_confidence: float = 0.0

    # What the agronomist says
    correct_disease: str = ""                   # ground-truth label
    severity_rating: Optional[int] = None       # 1-5 (agronomist's assessment)
    notes: str = ""

    # Which models were wrong
    classifier_prediction: str = ""
    rule_engine_prediction: str = ""
    llm_prediction: str = ""

    crop_type: str = ""
    created_at: str = ""


# ── Database management ──────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    """Open (and optionally create) the feedback database."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS feedback (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id        INTEGER,
                image_hash          TEXT NOT NULL,
                filename            TEXT DEFAULT '',

                predicted_disease   TEXT NOT NULL,
                predicted_confidence REAL DEFAULT 0.0,

                correct_disease     TEXT NOT NULL,
                severity_rating     INTEGER,
                notes               TEXT DEFAULT '',

                classifier_prediction TEXT DEFAULT '',
                rule_engine_prediction TEXT DEFAULT '',
                llm_prediction       TEXT DEFAULT '',

                crop_type           TEXT DEFAULT '',
                created_at          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_image_hash
                ON feedback(image_hash);
            CREATE INDEX IF NOT EXISTS idx_feedback_predicted
                ON feedback(predicted_disease);
            CREATE INDEX IF NOT EXISTS idx_feedback_correct
                ON feedback(correct_disease);
            CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON feedback(created_at);

            CREATE TABLE IF NOT EXISTS feedback_images (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id  INTEGER NOT NULL REFERENCES feedback(id) ON DELETE CASCADE,
                image_data   BLOB NOT NULL,
                format       TEXT DEFAULT 'jpeg',
                created_at   TEXT NOT NULL
            );
        """)
        conn.commit()
        logger.info(f"Feedback database initialized at {_DB_PATH}")
    finally:
        conn.close()


# ── CRUD ─────────────────────────────────────────────────────

def save_feedback(record: FeedbackRecord, image_bytes: bytes | None = None) -> int:
    """Insert a feedback record. Returns the new row id."""
    if not record.created_at:
        record.created_at = datetime.now().isoformat()

    conn = _get_conn()
    try:
        cur = conn.execute("""
            INSERT INTO feedback (
                detection_id, image_hash, filename,
                predicted_disease, predicted_confidence,
                correct_disease, severity_rating, notes,
                classifier_prediction, rule_engine_prediction, llm_prediction,
                crop_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.detection_id, record.image_hash, record.filename,
            record.predicted_disease, record.predicted_confidence,
            record.correct_disease, record.severity_rating, record.notes,
            record.classifier_prediction, record.rule_engine_prediction,
            record.llm_prediction,
            record.crop_type, record.created_at,
        ))
        fb_id = cur.lastrowid

        if image_bytes:
            conn.execute("""
                INSERT INTO feedback_images (feedback_id, image_data, format, created_at)
                VALUES (?, ?, 'jpeg', ?)
            """, (fb_id, image_bytes, record.created_at))

        conn.commit()
        logger.info(f"Feedback #{fb_id} saved: predicted={record.predicted_disease}, "
                     f"correct={record.correct_disease}")
        return fb_id
    finally:
        conn.close()


def get_all_feedback(
    limit: int = 500,
    offset: int = 0,
    disease_filter: str | None = None,
) -> list[dict]:
    """Retrieve feedback records, newest first."""
    conn = _get_conn()
    try:
        query = "SELECT * FROM feedback"
        params: list = []
        if disease_filter:
            query += " WHERE predicted_disease = ? OR correct_disease = ?"
            params.extend([disease_filter, disease_filter])
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_feedback_by_id(fb_id: int) -> dict | None:
    """Get a single feedback record."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM feedback WHERE id = ?", (fb_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_feedback_count() -> int:
    """Total number of feedback records."""
    conn = _get_conn()
    try:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    finally:
        conn.close()


def get_feedback_since(since_iso: str) -> list[dict]:
    """Get feedback records created after a timestamp."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE created_at > ? ORDER BY created_at",
            (since_iso,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_misclassified_pairs() -> list[dict]:
    """Get (predicted, correct) pairs where predicted != correct, with counts."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT predicted_disease, correct_disease, COUNT(*) as count
            FROM feedback
            WHERE predicted_disease != correct_disease
            GROUP BY predicted_disease, correct_disease
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_feedback_images_for_disease(disease_key: str) -> list[tuple[int, bytes]]:
    """Get stored images for a specific corrected disease (for retraining)."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT fi.feedback_id, fi.image_data
            FROM feedback_images fi
            JOIN feedback f ON fi.feedback_id = f.id
            WHERE f.correct_disease = ?
        """, (disease_key,)).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()


def delete_feedback(fb_id: int) -> bool:
    """Delete a feedback record (cascades to images)."""
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM feedback WHERE id = ?", (fb_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
