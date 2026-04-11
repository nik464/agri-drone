"""
temporal_tracker.py — GPS-zone-keyed disease progression tracker.

Persists every diagnosis to a local SQLite database keyed by GPS zone
(lat/lng rounded to 4 decimal places ≈ 11 m grid cells) and provides
trend analysis across readings for the same zone.
"""

import hashlib
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# ── Defaults ──
_DEFAULT_DB_PATH = Path("data/field_history.db")
_ZONE_DECIMALS = 4  # ~11 m grid


# ── Data classes ──

@dataclass
class Reading:
    """A single recorded diagnosis for a zone."""
    timestamp: str
    disease: str
    confidence: float
    severity: float
    image_hash: str


@dataclass
class ProgressionResult:
    """Output of analyze_progression()."""
    zone_id: str
    spread_rate: float           # % confidence change per day
    trend: str                   # accelerating | stable | recovering
    days_since_first_detection: int
    urgency_override: Optional[str]
    history: list[Reading]       # last 7 readings


# ── Schema ──

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id     TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    disease     TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    severity    REAL    NOT NULL DEFAULT 0.0,
    image_hash  TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_readings_zone
    ON readings (zone_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_readings_zone_disease
    ON readings (zone_id, disease);
"""


class FieldZoneTracker:
    """
    Stores diagnosis history keyed by GPS zone and computes disease
    progression trends.

    Thread-safe: each public method acquires a dedicated connection
    via a per-thread local, and schema init is guarded by a lock.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._schema_ready = False
        self._ensure_schema()

    # ── Connection management ──

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._init_lock:
            if self._schema_ready:
                return
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.commit()
            self._schema_ready = True
            logger.info(f"FieldZoneTracker: database ready at {self._db_path}")

    # ── Public helpers ──

    @staticmethod
    def gps_to_zone(lat: float, lng: float) -> str:
        """Round GPS coordinates to a grid cell and return a zone id."""
        rlat = round(lat, _ZONE_DECIMALS)
        rlng = round(lng, _ZONE_DECIMALS)
        return f"{rlat},{rlng}"

    @staticmethod
    def image_hash(data: bytes) -> str:
        """SHA-256 hex digest (first 16 chars) of raw image bytes."""
        return hashlib.sha256(data).hexdigest()[:16]

    # ── Record a reading ──

    def record(
        self,
        zone_id: str,
        disease: str,
        confidence: float,
        severity: float = 0.0,
        image_hash: str = "",
        timestamp: Optional[str] = None,
    ) -> None:
        """Persist a single diagnosis reading for *zone_id*."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO readings (zone_id, timestamp, disease, confidence, severity, image_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (zone_id, ts, disease, confidence, severity, image_hash),
        )
        conn.commit()
        logger.debug(f"Recorded: zone={zone_id} disease={disease} conf={confidence:.2f}")

    # ── Analyse progression ──

    def analyze_progression(self, zone_id: str) -> ProgressionResult:
        """
        Compute trend for *zone_id*.

        Returns a ProgressionResult with:
        - spread_rate  (% confidence change per day)
        - trend        ("accelerating" | "stable" | "recovering")
        - urgency_override  (None, or a short advisory string)
        - history      (last 7 readings, newest-first)
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT timestamp, disease, confidence, severity, image_hash "
            "FROM readings WHERE zone_id = ? ORDER BY timestamp DESC LIMIT 30",
            (zone_id,),
        ).fetchall()

        history = [
            Reading(
                timestamp=r[0],
                disease=r[1],
                confidence=r[2],
                severity=r[3],
                image_hash=r[4],
            )
            for r in rows
        ]

        last_7 = history[:7]

        if len(history) < 2:
            return ProgressionResult(
                zone_id=zone_id,
                spread_rate=0.0,
                trend="stable",
                days_since_first_detection=0,
                urgency_override=None,
                history=last_7,
            )

        # Days since first detection of any disease (exclude healthy)
        diseased = [h for h in history if h.disease.lower() not in ("healthy", "healthy_wheat", "healthy_rice")]
        if diseased:
            first_ts = _parse_ts(diseased[-1].timestamp)
            days_since = max(0, (datetime.now(timezone.utc) - first_ts).days)
        else:
            days_since = 0

        # ── Compute spread rate ──
        # Use only diseased readings with the same top disease
        top_disease = last_7[0].disease
        same_disease = [h for h in history if h.disease == top_disease]

        if len(same_disease) >= 2:
            newest = same_disease[0]
            oldest = same_disease[-1]
            dt_days = max(
                (_parse_ts(newest.timestamp) - _parse_ts(oldest.timestamp)).total_seconds() / 86400,
                0.01,  # avoid division by zero
            )
            conf_delta = newest.confidence - oldest.confidence
            spread_rate = round((conf_delta / dt_days) * 100, 2)  # %/day
        else:
            spread_rate = 0.0

        # ── Trend classification ──
        # Compare first half vs second half of same-disease readings
        if len(same_disease) >= 4:
            mid = len(same_disease) // 2
            recent_avg = sum(r.confidence for r in same_disease[:mid]) / mid
            older_avg = sum(r.confidence for r in same_disease[mid:]) / (len(same_disease) - mid)
            delta = recent_avg - older_avg
            if delta > 0.05:
                trend = "accelerating"
            elif delta < -0.05:
                trend = "recovering"
            else:
                trend = "stable"
        elif spread_rate > 5:
            trend = "accelerating"
        elif spread_rate < -5:
            trend = "recovering"
        else:
            trend = "stable"

        # ── Urgency override ──
        urgency_override = None
        if trend == "accelerating" and spread_rate > 10:
            urgency_override = f"SPRAY NOW — spread rate {spread_rate:.0f}%/day"
        elif trend == "accelerating" and spread_rate > 5:
            urgency_override = f"Treat within 24h — spread rate {spread_rate:.0f}%/day"

        return ProgressionResult(
            zone_id=zone_id,
            spread_rate=spread_rate,
            trend=trend,
            days_since_first_detection=days_since,
            urgency_override=urgency_override,
            history=last_7,
        )

    # ── Full zone history (for frontend charts) ──

    def get_zone_history(self, zone_id: str, limit: int = 100) -> list[Reading]:
        """Return all readings for *zone_id*, newest-first."""
        limit = min(limit, 500)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT timestamp, disease, confidence, severity, image_hash "
            "FROM readings WHERE zone_id = ? ORDER BY timestamp DESC LIMIT ?",
            (zone_id, limit),
        ).fetchall()
        return [
            Reading(timestamp=r[0], disease=r[1], confidence=r[2], severity=r[3], image_hash=r[4])
            for r in rows
        ]

    def list_zones(self) -> list[dict]:
        """Return all zones with their latest reading count."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT zone_id, COUNT(*) as cnt, MAX(timestamp) as last_ts "
            "FROM readings GROUP BY zone_id ORDER BY last_ts DESC"
        ).fetchall()
        return [{"zone_id": r[0], "readings": r[1], "last_seen": r[2]} for r in rows]


# ── Module-level singleton ──

_tracker: Optional[FieldZoneTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> FieldZoneTracker:
    """Return (or create) the module-level FieldZoneTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = FieldZoneTracker()
    return _tracker


# ── Helpers ──

def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO-format timestamp, tolerating multiple formats."""
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        # Fallback: try stripping trailing 'Z'
        return datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
