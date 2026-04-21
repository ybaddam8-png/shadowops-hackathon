"""
main.py — ShadowOps FastAPI WebSocket Server
=============================================
WebSocket: ws://localhost:8000/ws
REST:      GET /health | /state | /forensics | /reports | /health-scores

Changes:
  - _mock_supervisor: 4-action output (ALLOW/BLOCK/FORK/QUARANTINE)
  - _infer_llama: parses QUARANTINE token
  - process_live_action response now includes ambiguity_score and
    quarantine_hold fields from the updated env
"""

import json
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from shadowops_env import (
    UniversalShadowEnv,
    extract_features,
    build_llama_prompt,
    compute_ambiguity,
    QUARANTINE_HOLD_STEPS,
)
from api.models import InboundMessage, OutboundMessage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shadowops")

# ── Config ────────────────────────────────────────────────────
USE_REAL_MODEL   = False
LLAMA_MODEL_PATH = "./training/shadowops_model"

# ── Llama loader ──────────────────────────────────────────────
_llama_model = _llama_tokenizer = None


def _load_llama():
    global _llama_model, _llama_tokenizer
    if _llama_model:
        return
    from unsloth import FastLanguageModel
    _llama_model, _llama_tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLAMA_MODEL_PATH, max_seq_length=512,
        dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(_llama_model)


def _infer_llama(prompt: str) -> str:
    import torch
    inputs = _llama_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _llama_model.generate(
            **inputs, max_new_tokens=5,
            temperature=0.01, do_sample=False,
        )
    decoded = _llama_tokenizer.decode(out[0], skip_special_tokens=True)
    last    = decoded.strip().split()[-1].upper()
    # Check exact match first (most common path)
    if last in ("ALLOW", "BLOCK", "FORK", "QUARANTINE"):
        return last
    # Fallback: scan full decoded text in priority order
    for w in ("QUARANTINE", "FORK", "BLOCK", "ALLOW"):
        if w in decoded.upper():
            return w
    return "BLOCK"


def _mock_supervisor(risk_vector: list, ambiguity: float) -> str:
    """
    4-action heuristic supervisor used when USE_REAL_MODEL = False.

    Decision logic:
      - High risk (>0.65)  → FORK   (deceive, don't alert attacker)
      - Ambiguous (>0.40)  → QUARANTINE (hold 3 steps, gather context)
      - Medium risk (>0.35)→ BLOCK
      - Low risk           → ALLOW
    """
    risk = (
        risk_vector[0] * 0.35 +
        risk_vector[1] * 0.25 +
        risk_vector[3] * 0.20 +
        risk_vector[6] * 0.20
    )
    if risk > 0.65:
        return "FORK"
    if ambiguity > 0.40:
        return "QUARANTINE"
    if risk > 0.35:
        return "BLOCK"
    return "ALLOW"


def _decide(prompt: str, risk_vector: list) -> str:
    ambiguity = compute_ambiguity(risk_vector)
    if USE_REAL_MODEL:
        _load_llama()
        return _infer_llama(prompt)
    return _mock_supervisor(risk_vector, ambiguity)


# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="ShadowOps API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = UniversalShadowEnv(mode="live", seed=99)

# ── REST ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama" if USE_REAL_MODEL else "mock",
            "version": "3.0.0", "action_space": ["ALLOW", "BLOCK", "FORK", "QUARANTINE"]}


@app.get("/state")
def get_state():
    return JSONResponse(env.get_production_snapshot())


@app.get("/forensics")
def get_forensics():
    return JSONResponse(env.get_forensic_log())


@app.get("/reports")
def get_reports():
    return JSONResponse(env.get_incident_reports())


@app.get("/health-scores")
def get_health():
    return JSONResponse(env.get_health_scores())


# ── WebSocket ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("Frontend connected")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = InboundMessage(**json.loads(raw))
            except Exception as e:
                await websocket.send_json({"error": f"Bad payload: {e}"})
                continue

            domain      = payload.domain
            intent      = payload.action.intent
            raw_payload = payload.action.raw_payload
            log.info("Inbound: %s | %s", domain, intent)

            risk_vector = extract_features(domain, intent, raw_payload)
            prompt      = build_llama_prompt(domain, intent, raw_payload, risk_vector)
            decision    = _decide(prompt, risk_vector)
            log.info("Decision: %s", decision)

            result   = env.process_live_action(domain, intent, raw_payload, decision)
            response = OutboundMessage(**result)

            await asyncio.sleep(0.4)
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        log.info("Frontend disconnected")
    except Exception as e:
        log.error("WS error: %s", e)
        await websocket.close()