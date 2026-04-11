"""
reports.py - PDF field report generation and listing endpoints.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from ...services.report_service import generate_field_report

router = APIRouter(prefix="/reports", tags=["reports"])

_REPORTS_DIR = Path("outputs/reports")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class IssueInput(BaseModel):
    issue: str
    affected_area_pct: float = 0.0
    severity: float = 0.0
    zone: str = ""


class RecommendationInput(BaseModel):
    action: str
    priority: str = "medium"
    input: str = ""
    rate: str = ""
    zone: str = ""


class ReportRequest(BaseModel):
    """Payload accepted by POST /api/reports/generate."""
    field_id: str
    crop_type: str
    scan_date: Optional[str] = None
    gps_coordinates: Optional[str] = None
    images_analyzed: int = 0
    total_detections: int = 0
    overall_field_health: int = Field(default=0, ge=0, le=100)
    risk_level: str = "unknown"
    primary_issues: List[IssueInput] = Field(default_factory=list)
    recommendations: List[RecommendationInput] = Field(default_factory=list)
    research_notes: str = ""
    follow_up_scan_days: int = 7


class ReportMeta(BaseModel):
    filename: str
    size_bytes: int
    created: str
    download_url: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate")
async def generate_report(body: ReportRequest):
    """Generate a PDF field report and return it as a file download."""
    try:
        data = body.model_dump()
        # Convert nested Pydantic models to plain dicts
        data["primary_issues"] = [i.model_dump() for i in body.primary_issues]
        data["recommendations"] = [r.model_dump() for r in body.recommendations]

        filepath = generate_field_report(data)

        if not os.path.isfile(filepath):
            raise HTTPException(status_code=500, detail="Report file was not created")

        return FileResponse(
            path=filepath,
            media_type="application/pdf",
            filename=os.path.basename(filepath),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@router.get("/list", response_model=List[ReportMeta])
async def list_reports():
    """List all previously generated PDF reports."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reports: list[ReportMeta] = []
    for f in sorted(_REPORTS_DIR.glob("*.pdf"), reverse=True):
        stat = f.stat()
        reports.append(
            ReportMeta(
                filename=f.name,
                size_bytes=stat.st_size,
                created=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                download_url=f"/api/reports/download/{f.name}",
            )
        )
    return reports


@router.get("/download/{filename}")
async def download_report(filename: str):
    """Download a specific report by filename."""
    # Prevent path traversal
    safe_name = os.path.basename(filename)
    filepath = _REPORTS_DIR / safe_name

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        path=str(filepath),
        media_type="application/pdf",
        filename=safe_name,
    )
