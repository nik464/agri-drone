"""
voice.py — Offline voice interface for farmer interaction.

Endpoints:
  POST /voice/transcribe — Audio blob → text (whisper.cpp)
  POST /voice/speak      — Text → WAV audio  (piper-tts)

Both engines run 100% offline via subprocess — no cloud API keys needed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field


router = APIRouter(prefix="/voice", tags=["voice"])

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════

# Root dir for models — sibling to src/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MODELS_DIR = _PROJECT_ROOT / "models" / "voice"

# whisper.cpp binary + model
_WHISPER_BIN = _MODELS_DIR / "whisper-cpp" / ("main.exe" if os.name == "nt" else "main")
_WHISPER_MODEL = _MODELS_DIR / "whisper-cpp" / "models" / "ggml-base.bin"

# piper binary
_PIPER_BIN = _MODELS_DIR / "piper" / ("piper.exe" if os.name == "nt" else "piper")

# Language → piper voice ONNX model mapping
PIPER_VOICES: dict[str, str] = {
    "en": "en_US-lessac-medium.onnx",
    "hi": "hi_IN-swara-medium.onnx",
    "te": "te_IN-wikipedia-medium.onnx",
    "ta": "ta_IN-wikipedia-medium.onnx",
    "pa": "pa_IN-wikipedia-medium.onnx",
    "fi": "fi_FI-harri-medium.onnx",
    "nl": "nl_NL-mls-medium.onnx",
    "de": "de_DE-thorsten-medium.onnx",
}

# Language code → flag emoji for frontend
LANG_FLAGS: dict[str, str] = {
    "en": "🇬🇧", "hi": "🇮🇳", "te": "🇮🇳", "ta": "🇮🇳", "pa": "🇮🇳",
    "fi": "🇫🇮", "nl": "🇳🇱", "de": "🇩🇪",
}

# whisper language tokens (subset) — used for confidence parsing
_WHISPER_LANGS = {"en", "hi", "te", "ta", "pa", "fi", "nl", "de"}


# ════════════════════════════════════════════════════════════════
# Schemas
# ════════════════════════════════════════════════════════════════

class TranscribeResponse(BaseModel):
    text: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    flag: str = ""


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str = Field(default="en", pattern=r"^(en|hi|te|ta|pa|fi|nl|de)$")


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def _check_whisper() -> None:
    """Verify whisper.cpp binary and model exist."""
    if not _WHISPER_BIN.exists():
        raise HTTPException(
            status_code=503,
            detail=f"whisper.cpp binary not found at {_WHISPER_BIN}. Run setup_voice.py first.",
        )
    if not _WHISPER_MODEL.exists():
        raise HTTPException(
            status_code=503,
            detail=f"whisper model not found at {_WHISPER_MODEL}. Run setup_voice.py first.",
        )


def _check_piper(language: str) -> Path:
    """Verify piper binary and voice model exist, return voice path."""
    if not _PIPER_BIN.exists():
        raise HTTPException(
            status_code=503,
            detail=f"piper binary not found at {_PIPER_BIN}. Run setup_voice.py first.",
        )
    voice_name = PIPER_VOICES.get(language)
    if not voice_name:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    voice_path = _MODELS_DIR / "piper" / "voices" / voice_name
    if not voice_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Piper voice model for '{language}' not found at {voice_path}. Run setup_voice.py first.",
        )
    return voice_path


def _parse_whisper_output(stdout: str) -> tuple[str, str, float]:
    """Parse whisper.cpp stdout into (text, language, confidence).

    whisper.cpp outputs lines like:
      [00:00:00.000 --> 00:00:03.200]   The plant has yellow spots.
    And with --print-special it may include language token info.
    """
    text_lines = []
    for line in stdout.splitlines():
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Timestamp-bracketed transcript lines
        if line.startswith("[") and "-->" in line:
            # Extract text after the closing bracket
            bracket_end = line.find("]")
            if bracket_end != -1:
                segment = line[bracket_end + 1:].strip()
                if segment:
                    text_lines.append(segment)
        elif not line.startswith("whisper_") and not line.startswith("main:"):
            # Plain text output (some builds)
            text_lines.append(line)

    text = " ".join(text_lines).strip()

    # Language detection: whisper outputs "auto-detected language: xx" on stderr
    # We parse it from the combined output
    language = "en"
    confidence = 0.85  # default when whisper doesn't report
    for line in stdout.splitlines():
        if "auto-detected language:" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                lang_part = parts[-1].strip().lower()
                # Extract two-letter code
                for code in _WHISPER_LANGS:
                    if code in lang_part:
                        language = code
                        break
        if "probability" in line.lower() or "prob" in line.lower():
            # Try to extract probability value
            import re
            match = re.search(r"(\d+\.?\d*)\s*%", line)
            if match:
                confidence = min(float(match.group(1)) / 100.0, 1.0)

    return text, language, confidence


async def _run_subprocess(cmd: list[str], timeout: float = 30.0) -> tuple[str, str]:
    """Run a subprocess asynchronously with timeout."""
    logger.debug(f"Running: {' '.join(str(c) for c in cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail="Voice processing timed out")

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        logger.error(f"Subprocess failed (rc={proc.returncode}): {stderr[:500]}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice engine error: {stderr[:200]}",
        )

    return stdout, stderr


def _convert_webm_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert webm/ogg audio to 16kHz mono WAV using ffmpeg if available."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        import subprocess
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1",
             "-f", "wav", str(output_path)],
            capture_output=True, timeout=15,
        )
        if result.returncode == 0:
            return
        logger.warning(f"ffmpeg conversion failed: {result.stderr[:200]}")

    # Fallback: just rename — whisper.cpp handles some formats directly
    if input_path != output_path:
        shutil.copy2(input_path, output_path)


# ════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe uploaded audio to text using whisper.cpp (offline).

    Accepts audio/webm, audio/wav, audio/ogg, audio/mp4.
    Returns detected text, language, and confidence.
    """
    _check_whisper()

    # Validate content type
    ct = (audio.content_type or "").lower()
    allowed = {"audio/webm", "audio/wav", "audio/wave", "audio/ogg",
               "audio/mp4", "audio/mpeg", "audio/x-wav", "application/octet-stream"}
    if ct and ct not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type: {ct}")

    # Save uploaded audio to temp file
    suffix = ".webm" if "webm" in ct else ".wav" if "wav" in ct else ".ogg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        content = await audio.read()
        if len(content) > 25 * 1024 * 1024:  # 25MB limit
            raise HTTPException(status_code=413, detail="Audio file too large (max 25MB)")
        if len(content) < 100:
            raise HTTPException(status_code=400, detail="Audio file is empty or too small")
        tmp_in.write(content)
        tmp_in_path = Path(tmp_in.name)

    wav_path = tmp_in_path.with_suffix(".wav")
    try:
        # Convert to WAV if needed (whisper.cpp expects 16kHz mono WAV)
        if suffix != ".wav":
            _convert_webm_to_wav(tmp_in_path, wav_path)
        else:
            wav_path = tmp_in_path

        # Run whisper.cpp
        cmd = [
            str(_WHISPER_BIN),
            "-m", str(_WHISPER_MODEL),
            "-f", str(wav_path),
            "-l", "auto",         # auto-detect language
            "--no-timestamps",     # cleaner output
            "-t", "4",            # threads
        ]

        stdout, stderr = await _run_subprocess(cmd, timeout=30.0)
        combined = stdout + "\n" + stderr
        text, language, confidence = _parse_whisper_output(combined)

        if not text:
            text = "(no speech detected)"
            confidence = 0.0

        logger.info(f"Transcribed: lang={language}, conf={confidence:.2f}, text={text[:80]}...")

        return TranscribeResponse(
            text=text,
            language=language,
            confidence=round(confidence, 3),
            flag=LANG_FLAGS.get(language, "🌐"),
        )
    finally:
        # Cleanup temp files
        for p in (tmp_in_path, wav_path):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


@router.post("/speak")
async def speak_text(req: SpeakRequest):
    """Convert text to speech using piper-tts (offline).

    Returns audio/wav stream for direct playback.
    """
    voice_path = _check_piper(req.language)

    # Sanitize input text — strip control characters
    clean_text = req.text.replace("\x00", "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Empty text after sanitization")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tmp_out_path = Path(tmp_out.name)

    try:
        # piper reads from stdin, writes WAV to --output_file
        cmd = [
            str(_PIPER_BIN),
            "--model", str(voice_path),
            "--output_file", str(tmp_out_path),
        ]

        logger.debug(f"TTS: lang={req.language}, text={clean_text[:60]}...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=clean_text.encode("utf-8")),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(status_code=504, detail="TTS timed out")

        if proc.returncode != 0:
            err = stderr_bytes.decode("utf-8", errors="replace")[:200]
            logger.error(f"Piper failed: {err}")
            raise HTTPException(status_code=500, detail=f"TTS error: {err}")

        if not tmp_out_path.exists() or tmp_out_path.stat().st_size < 44:
            raise HTTPException(status_code=500, detail="TTS produced empty output")

        # Stream the WAV file back
        wav_bytes = tmp_out_path.read_bytes()
        logger.info(f"TTS complete: {len(wav_bytes)} bytes, lang={req.language}")

        return StreamingResponse(
            iter([wav_bytes]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Content-Length": str(len(wav_bytes)),
            },
        )
    finally:
        try:
            if tmp_out_path.exists():
                tmp_out_path.unlink()
        except OSError:
            pass


# ════════════════════════════════════════════════════════════════
# Info endpoint
# ════════════════════════════════════════════════════════════════

@router.get("/status")
async def voice_status():
    """Check which voice engines are available."""
    whisper_ok = _WHISPER_BIN.exists() and _WHISPER_MODEL.exists()
    piper_ok = _PIPER_BIN.exists()

    available_voices = {}
    for lang, voice_file in PIPER_VOICES.items():
        voice_path = _MODELS_DIR / "piper" / "voices" / voice_file
        available_voices[lang] = {
            "available": voice_path.exists(),
            "flag": LANG_FLAGS.get(lang, "🌐"),
            "model": voice_file,
        }

    return {
        "whisper": {"available": whisper_ok, "model": "ggml-base"},
        "piper": {"available": piper_ok},
        "voices": available_voices,
        "supported_languages": list(PIPER_VOICES.keys()),
    }
