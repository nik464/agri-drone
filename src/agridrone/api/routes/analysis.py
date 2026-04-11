"""
analysis.py - LLM-powered field analysis API endpoint.
"""
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from ...services.llm_service import LLMService

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalysisDetectionItem(BaseModel):
    """Single detection item in the analysis request."""
    class_name: str
    confidence: float
    severity_score: Optional[float] = None
    category: Optional[str] = None
    area_pct: Optional[float] = None
    bbox: Optional[dict] = None


class AnalysisRequest(BaseModel):
    """Request body for the analysis endpoint."""
    detections: List[AnalysisDetectionItem] = Field(
        ..., description="Detection results to analyse"
    )
    crop_type: str = Field(
        ..., description="Crop type (e.g. soybean, wheat, corn)"
    )
    field_id: str = Field(
        ..., description="Unique field identifier"
    )


class IssueSchema(BaseModel):
    issue: str
    affected_area_pct: float = 0.0
    severity: float = 0.0
    zone: str = ""


class RecommendationSchema(BaseModel):
    action: str
    priority: str = "medium"
    input: str = ""
    rate: str = ""
    zone: str = ""


class AnalysisResponse(BaseModel):
    """Response returned by the analysis endpoint."""
    status: str = "success"
    backend: str = Field(
        ..., description="LLM backend used: 'claude' or 'ollama'"
    )
    overall_field_health: int = Field(
        ..., ge=0, le=100, description="Field health score 0-100"
    )
    risk_level: str = Field(
        ..., description="low | medium | high | critical"
    )
    primary_issues: List[IssueSchema] = Field(default_factory=list)
    recommendations: List[RecommendationSchema] = Field(default_factory=list)
    research_notes: str = ""
    follow_up_scan_days: int = 7


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/recommend", response_model=AnalysisResponse)
async def recommend(body: AnalysisRequest) -> AnalysisResponse:
    """
    Analyse detections using an LLM and return agronomic recommendations.

    Uses **Claude** when ``ANTHROPIC_API_KEY`` is set, otherwise falls back
    to a local **Ollama** instance (``llama3.2``).
    """
    try:
        service = LLMService()

        det_dicts = [d.model_dump(exclude_none=True) for d in body.detections]

        result = await service.analyze_detections(
            detections=det_dicts,
            crop_type=body.crop_type,
            field_id=body.field_id,
        )

        return AnalysisResponse(
            status="success",
            backend=service.backend,
            overall_field_health=int(result.get("overall_field_health", 0)),
            risk_level=str(result.get("risk_level", "unknown")),
            primary_issues=[
                IssueSchema(**i) for i in result.get("primary_issues", [])
            ],
            recommendations=[
                RecommendationSchema(**r) for r in result.get("recommendations", [])
            ],
            research_notes=str(result.get("research_notes", "")),
            follow_up_scan_days=int(result.get("follow_up_scan_days", 7)),
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
