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

try:
    from training.shadowops_training_common import build_q_aware_decision, q_aware_demo_policy_action
    _Q_AWARE_POLICY_AVAILABLE = True
except Exception as exc:
    build_q_aware_decision = None
    q_aware_demo_policy_action = None
    _Q_AWARE_POLICY_AVAILABLE = False
    _Q_AWARE_POLICY_IMPORT_ERROR = str(exc)

try:
    from agent_memory import add_record, summarize_memory_context
except Exception:
    add_record = None
    summarize_memory_context = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shadowops")

# ── Config ────────────────────────────────────────────────────
USE_REAL_MODEL   = False
LLAMA_MODEL_PATH = "./training/shadowops_model"
DEMO_POLICY_NAME = "q_aware_demo_policy"
VALID_DECISIONS = {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}

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
    4-action threshold fallback used only if the Q-aware demo policy fails.

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


def _fallback_decision_details(domain: str, risk_vector: list, ambiguity: float, decision: str) -> dict:
    risk_score = (
        risk_vector[0] * 0.35 +
        risk_vector[1] * 0.25 +
        risk_vector[3] * 0.20 +
        risk_vector[6] * 0.20
    )
    return {
        "decision": decision,
        "confidence": round(max(0.0, min(1.0, 1.0 - ambiguity * 0.5)), 3),
        "uncertainty": round(max(0.0, min(1.0, ambiguity)), 3),
        "risk_score": round(float(risk_score), 3),
        "cumulative_risk_score": round(float(risk_score), 3),
        "cumulative_risk_reason": "Threshold fallback does not compute cumulative memory risk.",
        "missing_evidence": [],
        "required_evidence": [],
        "explanation": "Threshold fallback decision; Q-aware policy details unavailable.",
        "safe_outcome": "Hold action until a valid supervisor decision is available.",
        "evidence_plan": [],
        "structured_safe_outcome": {
            "outcome": "Hold action until a valid supervisor decision is available.",
            "remediation_steps": ["Request a valid supervisor decision before execution."],
            "rollback_required": False,
            "human_review_required": True,
            "monitoring_required": False,
            "explanation": "Fallback policy has no evidence planner context.",
            "evidence_needed": [],
            "allowed_next_actions": ["request valid supervisor decision"],
        },
        "decision_trace": {
            "domain": domain or "unknown",
            "risk_signals": [],
            "safe_signals": [],
            "cumulative_risk_score": round(float(risk_score), 3),
            "memory_signals": [],
            "missing_evidence": [],
            "evidence_steps": [],
            "final_decision": decision,
            "safety_rationale": "Fallback policy returned a conservative supervisor decision.",
        },
        "memory_context": {},
        "policy_name": "threshold_fallback_policy",
        "domain": domain,
        "mitre_tactic": "Unknown",
        "mitre_technique": "Unknown",
        "risk_indicators": [],
        "safe_indicators": [],
    }


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_decision_details(
    details: dict,
    *,
    domain: str,
    risk_vector: list,
    ambiguity: float,
) -> dict:
    if not isinstance(details, dict):
        details = _fallback_decision_details(domain, risk_vector, ambiguity, "QUARANTINE")

    decision = str(details.get("decision") or details.get("action_taken") or "QUARANTINE").upper()
    if decision not in VALID_DECISIONS:
        decision = "QUARANTINE"

    risk_score = _as_float(details.get("risk_score"), _as_float(details.get("cumulative_risk_score"), 0.0))
    cumulative_risk = _as_float(details.get("cumulative_risk_score"), risk_score)
    missing_evidence = [str(item) for item in _as_list(details.get("missing_evidence"))]
    required_evidence = [str(item) for item in _as_list(details.get("required_evidence"))]
    risk_indicators = [str(item) for item in _as_list(details.get("risk_indicators"))]
    safe_indicators = [str(item) for item in _as_list(details.get("safe_indicators"))]
    evidence_plan = [
        item for item in _as_list(details.get("evidence_plan"))
        if isinstance(item, dict)
    ]
    safe_outcome = str(details.get("safe_outcome") or "Quarantine action until missing evidence is provided.")
    structured_safe_outcome = _as_dict(details.get("structured_safe_outcome"))
    structured_safe_outcome.setdefault("outcome", safe_outcome)
    structured_safe_outcome.setdefault("remediation_steps", ["Collect required evidence before execution."])
    structured_safe_outcome.setdefault("rollback_required", False)
    structured_safe_outcome.setdefault("human_review_required", decision in {"FORK", "QUARANTINE", "BLOCK"})
    structured_safe_outcome.setdefault("monitoring_required", decision in {"ALLOW", "FORK", "QUARANTINE"})
    structured_safe_outcome.setdefault("explanation", "Safe defaults applied to supervisor response.")
    structured_safe_outcome.setdefault("evidence_needed", [item.get("ask", "") for item in evidence_plan] or missing_evidence)
    structured_safe_outcome.setdefault("allowed_next_actions", ["hold action", "collect missing evidence"])
    if not isinstance(structured_safe_outcome.get("remediation_steps"), list):
        structured_safe_outcome["remediation_steps"] = ["Collect required evidence before execution."]
    if not isinstance(structured_safe_outcome.get("evidence_needed"), list):
        structured_safe_outcome["evidence_needed"] = missing_evidence
    if not isinstance(structured_safe_outcome.get("allowed_next_actions"), list):
        structured_safe_outcome["allowed_next_actions"] = ["hold action", "collect missing evidence"]
    decision_trace = _as_dict(details.get("decision_trace"))
    decision_trace.setdefault("domain", str(details.get("domain") or domain or "unknown"))
    decision_trace.setdefault("risk_signals", risk_indicators)
    decision_trace.setdefault("safe_signals", safe_indicators)
    decision_trace.setdefault("cumulative_risk_score", round(max(0.0, min(1.0, cumulative_risk)), 3))
    memory_context = _as_dict(details.get("memory_context"))
    memory_signals = list(memory_context.get("risky_chains", []))
    decision_trace.setdefault("memory_signals", memory_signals)
    decision_trace.setdefault("missing_evidence", missing_evidence)
    decision_trace.setdefault(
        "evidence_steps",
        [
            {
                "step": item.get("step"),
                "priority": item.get("priority"),
                "ask": item.get("ask"),
                "blocks_decision": item.get("blocks_decision", False),
            }
            for item in evidence_plan
        ],
    )
    decision_trace.setdefault("final_decision", decision)
    decision_trace.setdefault("safety_rationale", f"Safe outcome: {safe_outcome}")
    if not isinstance(decision_trace.get("risk_signals"), list):
        decision_trace["risk_signals"] = risk_indicators
    if not isinstance(decision_trace.get("safe_signals"), list):
        decision_trace["safe_signals"] = safe_indicators
    if not isinstance(decision_trace.get("memory_signals"), list):
        decision_trace["memory_signals"] = memory_signals
    if not isinstance(decision_trace.get("missing_evidence"), list):
        decision_trace["missing_evidence"] = missing_evidence
    if not isinstance(decision_trace.get("evidence_steps"), list):
        decision_trace["evidence_steps"] = []

    return {
        **details,
        "decision": decision,
        "confidence": round(max(0.0, min(1.0, _as_float(details.get("confidence"), 0.5))), 3),
        "uncertainty": round(max(0.0, min(1.0, _as_float(details.get("uncertainty"), ambiguity))), 3),
        "risk_score": round(max(0.0, min(1.0, risk_score)), 3),
        "cumulative_risk_score": round(max(0.0, min(1.0, cumulative_risk)), 3),
        "cumulative_risk_reason": str(details.get("cumulative_risk_reason") or "No cumulative risk reason provided."),
        "missing_evidence": missing_evidence,
        "required_evidence": required_evidence,
        "explanation": str(details.get("explanation") or "Supervisor returned safe default decision details."),
        "safe_outcome": safe_outcome,
        "evidence_plan": evidence_plan,
        "structured_safe_outcome": structured_safe_outcome,
        "decision_trace": decision_trace,
        "memory_context": memory_context,
        "policy_name": str(details.get("policy_name") or DEMO_POLICY_NAME),
        "domain": str(details.get("domain") or domain or "unknown"),
        "mitre_tactic": str(details.get("mitre_tactic") or "Unknown"),
        "mitre_technique": str(details.get("mitre_technique") or "Unknown"),
        "risk_indicators": risk_indicators,
        "safe_indicators": safe_indicators,
    }


def _decide(
    domain: str,
    intent: str,
    raw_payload: str,
    prompt: str,
    risk_vector: list,
    *,
    actor: str = "unknown",
    session_id: str = "default",
    service: str = "",
    environment: str = "production",
    provided_evidence: list | None = None,
) -> dict:
    ambiguity = compute_ambiguity(risk_vector)
    if USE_REAL_MODEL:
        _load_llama()
        model_decision = _infer_llama(prompt)
        details = _fallback_decision_details(domain, risk_vector, ambiguity, model_decision)
        details["policy_name"] = "llama_model"
        return _safe_decision_details(details, domain=domain, risk_vector=risk_vector, ambiguity=ambiguity)
    if _Q_AWARE_POLICY_AVAILABLE and build_q_aware_decision is not None:
        try:
            memory_context = (
                summarize_memory_context(session_id)
                if summarize_memory_context is not None
                else None
            )
            details = build_q_aware_decision(
                domain,
                intent,
                raw_payload,
                risk_vector,
                actor=actor,
                session_id=session_id,
                service=service or domain,
                environment=environment,
                provided_evidence=provided_evidence or [],
                memory_context=memory_context,
            )
            return _safe_decision_details(details, domain=domain, risk_vector=risk_vector, ambiguity=ambiguity)
        except Exception as exc:
            log.warning("Q-aware demo policy failed; using threshold fallback: %s", exc)
    details = _fallback_decision_details(domain, risk_vector, ambiguity, _mock_supervisor(risk_vector, ambiguity))
    return _safe_decision_details(details, domain=domain, risk_vector=risk_vector, ambiguity=ambiguity)


def _process_inbound(payload: InboundMessage) -> dict:
    domain      = payload.domain
    intent      = payload.action.intent
    raw_payload = payload.action.raw_payload
    risk_vector = extract_features(domain, intent, raw_payload)
    prompt      = build_llama_prompt(domain, intent, raw_payload, risk_vector)
    decision_details = _decide(
        domain,
        intent,
        raw_payload,
        prompt,
        risk_vector,
        actor=payload.actor,
        session_id=payload.session_id,
        service=payload.service or domain,
        environment=payload.environment,
        provided_evidence=payload.provided_evidence,
    )
    decision = decision_details["decision"]
    result = env.process_live_action(domain, intent, raw_payload, decision)
    result["supervisor_decision"].update(decision_details)
    result["supervisor_decision"]["action_taken"] = decision
    result["supervisor_decision"]["decision"] = decision
    if add_record is not None:
        add_record({
            "actor": payload.actor,
            "session_id": payload.session_id,
            "service": payload.service or domain,
            "domain": domain,
            "environment": payload.environment,
            "timestamp": decision_details.get("timestamp", 0),
            "decision": decision,
            "risk_score": decision_details.get("risk_score", 0.0),
            "action_summary": raw_payload,
            "indicators": decision_details.get("risk_indicators", []),
        })
    return result


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
            "policy": "llama_model" if USE_REAL_MODEL else DEMO_POLICY_NAME,
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


@app.post("/decision")
def post_decision(payload: InboundMessage):
    return JSONResponse(_process_inbound(payload))


@app.post("/simulate")
def post_simulate(payload: InboundMessage):
    return JSONResponse(_process_inbound(payload))


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

            result   = _process_inbound(payload)
            log.info("Decision: %s", result["supervisor_decision"]["decision"])
            response = OutboundMessage(**result)

            await asyncio.sleep(0.4)
            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        log.info("Frontend disconnected")
    except Exception as e:
        log.error("WS error: %s", e)
        await websocket.close()
