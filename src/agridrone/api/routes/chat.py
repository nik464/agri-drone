"""
chat.py — Conversational agricultural advisor endpoint.

Accepts a farmer's follow-up question together with the current diagnosis
context and streams a response from the local Ollama LLM.
"""

import json
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from ...knowledge import kb_loader

router = APIRouter(prefix="/chat", tags=["chat"])

# ── Ollama configuration (same instance as the rest of the app) ──
_OLLAMA_URL = "http://localhost:11434"
_CHAT_MODEL = "llava"          # use the same model already pulled for validation
_TIMEOUT = 120.0               # seconds


# ── Request / response schemas ──

class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    disease_context: dict = Field(default_factory=dict)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    crop_type: str = Field(default="wheat")


class ChatResponseSync(BaseModel):
    answer: str
    model: str
    elapsed_ms: float


# ── Helpers ──

def _build_system_prompt(ctx: dict, crop_type: str) -> str:
    """Build a tightly-scoped system prompt from the diagnosis context."""

    disease_key = ctx.get("disease_key", "")
    disease_name = ctx.get("disease", ctx.get("disease_name", "Unknown"))
    confidence = ctx.get("confidence", 0)
    health_score = ctx.get("health_score", "N/A")
    risk_level = ctx.get("risk_level", "unknown")
    yield_loss = ctx.get("yield_loss", "N/A")
    urgency = ctx.get("urgency", "N/A")

    # Pull rich profile from KB if available
    profile = None
    if disease_key:
        profile = kb_loader.get_profile(disease_key)
    if profile is None and disease_name:
        # Fuzzy-match by display name
        for k, p in (kb_loader.get_all_profiles() or {}).items():
            if disease_name.lower() in p.display_name.lower():
                profile = p
                disease_key = k
                break

    # Treatment list
    treatments = ctx.get("treatment", [])
    if not treatments and profile:
        treatments = profile.treatment
    treatment_block = "\n".join(f"  • {t}" for t in treatments) if treatments else "  (none available)"

    # Symptoms
    symptoms = []
    if profile:
        symptoms = profile.symptoms
    symptoms_block = "\n".join(f"  • {s}" for s in symptoms[:5]) if symptoms else "  (none listed)"

    # Seasonal stage
    seasonal_block = ""
    try:
        risks = kb_loader.get_seasonal_risk(crop_type)
        if risks:
            current = risks[0]
            seasonal_block = (
                f"\nSeasonal stage: {current.stage} "
                f"(months: {', '.join(current.months)}, "
                f"temp: {current.temperature_range_c[0]}–{current.temperature_range_c[1]}°C)\n"
                f"High-risk diseases now: {', '.join(current.high_risk)}\n"
            )
    except Exception:
        pass

    # Favourable conditions
    conditions = ""
    if profile and profile.favorable_conditions:
        conditions = f"\nFavourable conditions for this disease: {profile.favorable_conditions}"

    return f"""You are an expert agricultural advisor specialising in Indian crop diseases.
You are speaking with a farmer about a SPECIFIC diagnosis on their field.

═══ CURRENT DIAGNOSIS ═══
Disease: {disease_name}
Confidence: {confidence if isinstance(confidence, str) else f"{confidence:.0%}"}
Health score: {health_score}/100
Risk level: {risk_level}
Yield loss estimate: {yield_loss}
Urgency: {urgency}
{conditions}
{seasonal_block}
Known symptoms:
{symptoms_block}

Recommended treatments:
{treatment_block}

═══ RULES ═══
1. ONLY answer questions about this disease, this crop ({crop_type}), and this field.
2. If the farmer asks about an unrelated topic, politely redirect.
3. Use practical, actionable language a small-holder Indian farmer can follow.
4. Mention specific Indian-available products (brand names, dosages in ml/L or g/L).
5. If you don't know, say so — never fabricate product names or dosages.
6. Keep answers concise (3–6 sentences) unless the farmer asks for detail.
7. Reference the diagnosis data above when relevant."""


def _build_messages(system: str, history: list[ChatMessage], question: str) -> list[dict]:
    """Assemble the Ollama chat message list."""
    msgs: list[dict] = [{"role": "system", "content": system}]
    # Include last 10 turns max to keep context window manageable
    for msg in history[-10:]:
        msgs.append({"role": msg.role, "content": msg.content})
    msgs.append({"role": "user", "content": question})
    return msgs


# ── Streaming endpoint ──

@router.post("")
async def chat_stream(req: ChatRequest):
    """
    Stream a conversational response from the local LLM.

    If Ollama is unreachable, falls back to a rule-based answer
    built from the knowledge base.
    """
    # Ensure KB is loaded
    if not kb_loader.get_all_profiles():
        kb_loader.load()

    system_prompt = _build_system_prompt(req.disease_context, req.crop_type)
    messages = _build_messages(system_prompt, req.conversation_history, req.question)

    async def _stream_ollama():
        """Yield text chunks from Ollama's streaming chat API."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    f"{_OLLAMA_URL}/api/chat",
                    json={
                        "model": _CHAT_MODEL,
                        "messages": messages,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            yield _fallback_answer(req)
        except Exception as exc:
            logger.warning(f"Chat LLM stream error: {exc}")
            yield _fallback_answer(req)

    return StreamingResponse(
        _stream_ollama(),
        media_type="text/plain",
        headers={"X-Chat-Model": _CHAT_MODEL},
    )


# ── Non-streaming fallback (when Ollama is down) ──

@router.post("/sync")
async def chat_sync(req: ChatRequest):
    """Non-streaming fallback — returns the full answer at once."""
    if not kb_loader.get_all_profiles():
        kb_loader.load()

    system_prompt = _build_system_prompt(req.disease_context, req.crop_type)
    messages = _build_messages(system_prompt, req.conversation_history, req.question)

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{_OLLAMA_URL}/api/chat",
                json={
                    "model": _CHAT_MODEL,
                    "messages": messages,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("message", {}).get("content", "")
    except Exception as exc:
        logger.warning(f"Chat sync LLM error: {exc}")
        answer = _fallback_answer(req)

    return ChatResponseSync(
        answer=answer,
        model=_CHAT_MODEL,
        elapsed_ms=round((time.time() - t0) * 1000, 1),
    )


# ── KB-based fallback when LLM unavailable ──

def _fallback_answer(req: ChatRequest) -> str:
    """Generate a basic answer from the knowledge base when Ollama is offline."""
    ctx = req.disease_context
    disease_name = ctx.get("disease", ctx.get("disease_name", "the detected disease"))
    treatments = ctx.get("treatment", [])
    urgency = ctx.get("urgency", "within_7_days")
    yield_loss = ctx.get("yield_loss", "unknown")

    q = req.question.lower()

    if "delay" in q or "wait" in q:
        if urgency == "immediate":
            return (
                f"Delaying treatment for {disease_name} is NOT recommended. "
                f"This disease requires immediate action — estimated yield loss is {yield_loss}. "
                "Apply the recommended fungicide today if possible."
            )
        return (
            f"A short delay (1–2 days) is generally acceptable for {disease_name}, "
            f"but don't exceed 3 days. Urgency level: {urgency}."
        )

    if "rain" in q:
        return (
            f"Rain can wash off contact fungicides. For {disease_name}, "
            "apply a systemic fungicide (e.g. Propiconazole, Tebuconazole) "
            "which is rain-fast within 2 hours of application. "
            "Avoid spraying if rain is expected within 1 hour."
        )

    if "cheap" in q or "alternative" in q or "cost" in q:
        if treatments:
            return (
                f"Current recommendation: {treatments[0]}. "
                "A more economical option for many fungal diseases is "
                "Mancozeb 75 WP @ 2.5 g/L (₹150–200/kg). "
                "However, systemic fungicides are more effective for established infections."
            )
        return "Please consult your local Krishi Vigyan Kendra for cost-effective alternatives."

    if "spread" in q or "nearby" in q:
        return (
            f"{disease_name} can spread to adjacent plants, especially in humid conditions "
            "with wind carrying spores. Inspect a 5-metre radius around infected plants. "
            "Remove severely infected material and apply protective spray to surrounding area."
        )

    if "yield" in q or "loss" in q:
        return (
            f"Estimated yield loss from {disease_name}: {yield_loss}. "
            "Early treatment can reduce this significantly. "
            "Without treatment, losses tend to increase 5–10% per week."
        )

    return (
        f"I can answer questions about {disease_name} — its treatment, timing, "
        "spreading risk, and yield impact. The LLM is currently unavailable, "
        "so responses are drawn from the knowledge base."
    )
