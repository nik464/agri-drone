"""
field.py — Field history and progression endpoints.

GET /api/field/history?zone=<zone_id>       → full reading history
GET /api/field/progression?zone=<zone_id>   → trend analysis
GET /api/field/zones                        → list all tracked zones
"""

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from ...core.temporal_tracker import get_tracker

router = APIRouter(prefix="/field", tags=["field"])


@router.get("/history")
async def zone_history(
    zone: str = Query(..., min_length=3, description="Zone ID (e.g. '28.6139,77.2090')"),
    limit: int = Query(100, ge=1, le=500, description="Max readings to return"),
):
    """Return the full diagnosis history for a GPS zone (newest-first)."""
    tracker = get_tracker()
    readings = tracker.get_zone_history(zone, limit=limit)
    if not readings:
        raise HTTPException(status_code=404, detail=f"No history for zone '{zone}'")
    return {
        "zone_id": zone,
        "count": len(readings),
        "readings": [asdict(r) for r in readings],
    }


@router.get("/progression")
async def zone_progression(
    zone: str = Query(..., min_length=3, description="Zone ID"),
):
    """Return the disease progression analysis for a GPS zone."""
    tracker = get_tracker()
    result = tracker.analyze_progression(zone)
    return asdict(result)


@router.get("/zones")
async def list_zones():
    """List all GPS zones that have recorded readings."""
    tracker = get_tracker()
    return {"zones": tracker.list_zones()}
