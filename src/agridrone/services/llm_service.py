"""
llm_service.py - LLM-powered detection analysis.

Uses Claude API when ANTHROPIC_API_KEY is set, otherwise falls back to
a local Ollama instance (llama3.2) for offline field use.
"""
import json
import os
from typing import Any

import httpx
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_CLAUDE_MODEL = "claude-opus-4-5"
_CLAUDE_MAX_TOKENS = 4096

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "llama3.2"

_REQUEST_TIMEOUT = 120.0  # seconds


def _build_prompt(detections: list[dict], crop_type: str, field_id: str) -> str:
    """Build the analysis prompt sent to the LLM."""
    det_json = json.dumps(detections, indent=2, default=str)
    return (
        "You are a precision agriculture research scientist advising farmers "
        "in North India (Punjab / Haryana / western UP). "
        "The field is in the Indo-Gangetic wheat-rice belt. "
        "Analyze the following drone-based crop detection results and return "
        "ONLY valid JSON (no markdown fences, no commentary).\n\n"
        f"Field ID: {field_id}\n"
        f"Crop type: {crop_type}\n"
        f"Detection results:\n{det_json}\n\n"
        "When recommending inputs, use ONLY products available in India:\n"
        "  - Urea (46-0-0) for nitrogen top-dressing\n"
        "  - DAP (18-46-0) for phosphorus\n"
        "  - Propiconazole 25% EC for wheat leaf rust / yellow rust\n"
        "  - Tricyclazole 75% WP for rice blast\n"
        "  - Imidacloprid 17.8% SL for rice brown planthopper\n"
        "  - Pendimethalin 30% EC as pre-emergent herbicide\n"
        "All application rates MUST be in kg/hectare or ml/hectare.\n"
        "Reference ICAR (Indian Council of Agricultural Research) crop advisories "
        "and state agricultural university recommendations where applicable.\n\n"
        "Return JSON with exactly these keys:\n"
        "{\n"
        '  "overall_field_health": <integer 0-100>,\n'
        '  "risk_level": "<low|medium|high|critical>",\n'
        '  "estimated_yield_loss_pct": <float — predicted yield loss percentage>,\n'
        '  "primary_issues": [\n'
        '    {"issue": "<string>", "affected_area_pct": <float>, '
        '"severity": <float 0-1>, "zone": "<string>"}\n'
        "  ],\n"
        '  "recommendations": [\n'
        '    {"action": "<string>", "priority": "<low|medium|high|urgent>", '
        '"input": "<Indian product name>", '
        '"rate": "<amount in kg/ha or ml/ha>", '
        '"zone": "<string>"}\n'
        "  ],\n"
        '  "icar_advisory_reference": "<relevant ICAR advisory or bulletin name>",\n'
        '  "research_notes": "<paragraph citing ICAR / state agri-university findings>",\n'
        '  "follow_up_scan_days": <integer>\n'
        "}\n"
    )


def _get_backend() -> str:
    """Return 'claude' if ANTHROPIC_API_KEY is set, else 'ollama'."""
    return "claude" if os.environ.get("ANTHROPIC_API_KEY") else "ollama"


class LLMService:
    """
    Unified async interface for LLM-powered field analysis.

    Automatically selects Claude (cloud) or Ollama (local) based on the
    presence of the ``ANTHROPIC_API_KEY`` environment variable.
    """

    def __init__(self) -> None:
        self.backend = _get_backend()
        logger.info(f"LLMService initialised — backend={self.backend}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_detections(
        self,
        detections: list[dict[str, Any]],
        crop_type: str,
        field_id: str,
    ) -> dict[str, Any]:
        """
        Ask the LLM to interpret detection results and return agronomic
        recommendations.

        Args:
            detections: List of detection dicts (class_name, confidence, …).
            crop_type: E.g. "soybean", "wheat", "corn".
            field_id: Unique field identifier.

        Returns:
            Dict with overall_field_health, risk_level, primary_issues,
            recommendations, research_notes, follow_up_scan_days.
        """
        prompt = _build_prompt(detections, crop_type, field_id)

        if self.backend == "claude":
            return await self._call_claude(prompt)
        return await self._call_ollama(prompt)

    # ------------------------------------------------------------------
    # Private – Claude
    # ------------------------------------------------------------------

    async def _call_claude(self, prompt: str) -> dict[str, Any]:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": _CLAUDE_MODEL,
            "max_tokens": _CLAUDE_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            logger.debug("Sending request to Claude API …")
            resp = await client.post(_CLAUDE_URL, headers=headers, json=payload)
            resp.raise_for_status()

        body = resp.json()
        raw_text = body["content"][0]["text"]
        return _parse_llm_json(raw_text)

    # ------------------------------------------------------------------
    # Private – Ollama
    # ------------------------------------------------------------------

    async def _call_ollama(self, prompt: str) -> dict[str, Any]:
        payload = {
            "model": _OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            logger.debug("Sending request to Ollama …")
            resp = await client.post(_OLLAMA_URL, json=payload)
            resp.raise_for_status()

        body = resp.json()
        raw_text = body.get("response", "{}")
        return _parse_llm_json(raw_text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict[str, Any]:
    """
    Best-effort extraction of JSON from an LLM response.

    Strips optional markdown fences and trailing text before parsing.
    """
    text = raw.strip()
    # Remove ```json … ``` wrappers some models add
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON — returning raw text")
        return {
            "overall_field_health": 0,
            "risk_level": "unknown",
            "estimated_yield_loss_pct": 0.0,
            "primary_issues": [],
            "recommendations": [],
            "icar_advisory_reference": "",
            "research_notes": text,
            "follow_up_scan_days": 7,
        }
