"""
main.py — ShadowOps FastAPI WebSocket Server.

Stability-first runtime:
  - default decision path uses pickle classifier
  - optional LoRA explainer is isolated behind toggle
"""

import asyncio
import datetime
import json
import logging
import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.models import InboundMessage, OutboundMessage
from shadowops_env import UniversalShadowEnv, build_llama_prompt, compute_ambiguity, extract_features

try:
    from agent_memory import add_record
except Exception:
    add_record = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("shadowops")

# ── Stable toggles ─────────────────────────────────────────────
USE_PICKLE_CLASSIFIER = True
USE_LORA_EXPLAINER = False

MODEL_PATH_CANDIDATES = [
    "exports/model_fixed.pkl",
    "../export/model_fixed.pkl",
    "../export/model.pkl",
]
LORA_ADAPTER_PATH = "./shadowops_qwen3_1p7b_model"
VALID_DECISIONS = {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}
FORENSICS_JSONL = Path("forensics.jsonl")

app = FastAPI(title="ShadowOps API", version="3.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
env = UniversalShadowEnv(mode="live", seed=99)

runtime_state: dict[str, Any] = {
    "alive": True,
    "ready": False,
    "mode": "threshold_fallback",
    "model_loaded": False,
    "classifier_loaded": False,
    "adapter_loaded": False,
    "device": "cpu",
    "startup_error": None,
}

_clf = None
_labels = None
_qwen_model = None
_qwen_tokenizer = None
_lora_model = None
_lora_tokenizer = None
_model_path_in_use = None


def _append_forensics(event: dict[str, Any]) -> None:
    try:
        with FORENSICS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("forensics append failed: %s", exc)


def _risk_score(risk_vector: list[float]) -> float:
    return float(
        risk_vector[0] * 0.35
        + risk_vector[1] * 0.25
        + risk_vector[3] * 0.20
        + risk_vector[6] * 0.20
    )


def _threshold_decision(risk_vector: list[float], ambiguity: float) -> str:
    risk = _risk_score(risk_vector)
    if risk > 0.65:
        return "FORK"
    if ambiguity > 0.40:
        return "QUARANTINE"
    if risk > 0.35:
        return "BLOCK"
    return "ALLOW"


def _fallback_decision_details(domain: str, risk_vector: list[float], ambiguity: float, decision: str) -> dict[str, Any]:
    score = round(_risk_score(risk_vector), 3)
    decision = str(decision or "QUARANTINE").upper()
    if decision not in VALID_DECISIONS:
        decision = "QUARANTINE"
    return {
        "decision": decision,
        "confidence": round(max(0.0, min(1.0, 1.0 - ambiguity * 0.5)), 3),
        "uncertainty": round(max(0.0, min(1.0, ambiguity)), 3),
        "risk_score": score,
        "cumulative_risk_score": score,
        "missing_evidence": [],
        "required_evidence": [],
        "explanation": "Stable fallback decision.",
        "safe_outcome": "Decision generated with conservative runtime safeguards.",
        "structured_safe_outcome": {"remediation_steps": "Manual review required."},
        "evidence_plan": [],
        "decision_trace": {"final_decision": decision},
        "policy_name": runtime_state["mode"],
        "domain": domain,
        "mitre_tactic": "Unknown",
        "mitre_technique": "Unknown",
    }


def _load_pickle_classifier() -> None:
    global _clf, _labels, _qwen_model, _qwen_tokenizer, _model_path_in_use
    if _clf is not None:
        return
    selected = None
    for candidate in MODEL_PATH_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            selected = p
            break
    if selected is None:
        raise FileNotFoundError(f"missing classifier artifact candidates: {MODEL_PATH_CANDIDATES}")
    _model_path_in_use = str(selected)
    with open(selected, "rb") as f:
        data = pickle.load(f)
    _clf = data["clf"]
    _labels = data["labels"]

    try:
        from unsloth import FastLanguageModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        runtime_state["device"] = device
        _qwen_model, _qwen_tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-1.7B",
            max_seq_length=256,
            load_in_4bit=(device == "cuda"),
        )
        if getattr(_qwen_tokenizer, "pad_token_id", None) is None:
            _qwen_tokenizer.pad_token = _qwen_tokenizer.eos_token
        FastLanguageModel.for_inference(_qwen_model)
    except Exception as exc:
        log.warning("Optional Qwen model import failed, running in fallback mode: %s", exc)
        runtime_state["startup_error"] = str(exc)


def _load_lora_explainer() -> None:
    global _lora_model, _lora_tokenizer
    if _lora_model is not None:
        return
    required = [
        Path(LORA_ADAPTER_PATH) / "adapter_model.safetensors",
        Path(LORA_ADAPTER_PATH) / "adapter_config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing adapter files: {missing}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runtime_state["device"] = device
    base = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-1.7B")
    _lora_tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")
    _lora_model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    _lora_model.to(device)
    if getattr(_lora_tokenizer, "pad_token_id", None) is None:
        _lora_tokenizer.pad_token = _lora_tokenizer.eos_token


def _infer_classifier(text: str) -> str:
    import torch

    inputs = _qwen_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    if runtime_state["device"] == "cuda":
        inputs = inputs.to("cuda")
    with torch.no_grad():
        h = _qwen_model(**inputs, output_hidden_states=True).hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    probs = _clf.predict_proba(emb)[0]
    return str(_labels[int(probs.argmax())]).upper()


def _warmup_runtime() -> None:
    runtime_state["ready"] = False
    runtime_state["startup_error"] = None
    runtime_state["mode"] = "threshold_fallback"
    try:
        if USE_PICKLE_CLASSIFIER:
            _load_pickle_classifier()
            runtime_state["classifier_loaded"] = True
            runtime_state["model_loaded"] = True
            runtime_state["mode"] = "pickle_classifier"
            log.info("classifier artifact loaded from %s", _model_path_in_use)
        if USE_LORA_EXPLAINER:
            _load_lora_explainer()
            runtime_state["adapter_loaded"] = True
            runtime_state["model_loaded"] = True
            runtime_state["mode"] = "pickle_classifier+lora_explainer" if USE_PICKLE_CLASSIFIER else "lora_explainer"
        runtime_state["ready"] = True
        log.info("startup warmup success mode=%s device=%s", runtime_state["mode"], runtime_state["device"])
    except Exception as exc:
        runtime_state["startup_error"] = str(exc)
        runtime_state["ready"] = False
        log.exception("startup warmup failed; using threshold fallback: %s", exc)


@app.on_event("startup")
def _startup() -> None:
    _warmup_runtime()


def _decide(domain: str, intent: str, raw_payload: str, risk_vector: list[float]) -> dict[str, Any]:
    ambiguity = compute_ambiguity(risk_vector)
    text = f"{intent} {raw_payload}"
    try:
        if runtime_state["classifier_loaded"] and _clf is not None and _qwen_model is not None:
            decision = _infer_classifier(text)
            details = _fallback_decision_details(domain, risk_vector, ambiguity, decision)
            details["policy_name"] = "pickle_classifier"
            details["explanation"] = "Decision from stable classifier path."
            return details
    except Exception as exc:
        log.warning("classifier inference failed, falling back: %s", exc)
    decision = _threshold_decision(risk_vector, ambiguity)
    details = _fallback_decision_details(domain, risk_vector, ambiguity, decision)
    details["policy_name"] = "threshold_fallback"
    return details


def _process_inbound(payload: InboundMessage) -> dict[str, Any]:
    domain = payload.domain
    intent = payload.action.intent
    raw_payload = payload.action.raw_payload
    risk_vector = extract_features(domain, intent, raw_payload)
    _ = build_llama_prompt(domain, intent, raw_payload, risk_vector)

    decision_details = _decide(domain, intent, raw_payload, risk_vector)
    decision = decision_details["decision"]
    result = env.process_live_action(domain, intent, raw_payload, decision)
    result["supervisor_decision"].update(decision_details)
    result["supervisor_decision"]["action_taken"] = decision
    result["supervisor_decision"]["decision"] = decision

    event = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "domain": domain,
        "intent": intent,
        "decision": decision,
        "policy_name": decision_details.get("policy_name"),
        "risk_score": decision_details.get("risk_score", 0.0),
        "session_id": payload.session_id,
        "actor": payload.actor,
    }
    _append_forensics(event)
    if add_record is not None:
        add_record(event)
    return result


@app.get("/health")
def health():
    return {"status": "alive", "service": "shadowops-api", "version": "3.2.0"}


@app.get("/ready")
def ready():
    model_ready = runtime_state["model_loaded"] and _qwen_model is not None
    return {
        "status": "ready" if runtime_state["ready"] else "not_ready",
        "q_aware_ready": model_ready,
        "model_ready": model_ready,
        "fallback_ready": runtime_state["ready"],
        "mode": runtime_state["mode"],
        "classifier_loaded": runtime_state["classifier_loaded"],
        "adapter_loaded": runtime_state["adapter_loaded"],
        "device": runtime_state["device"],
        "startup_errors": runtime_state.get("startup_error"),
    }


@app.get("/state")
def get_state():
    return JSONResponse(env.state())


@app.get("/forensics")
def get_forensics():
    return JSONResponse(env.get_forensic_log())


@app.get("/reports")
def get_reports():
    return JSONResponse(env.get_incident_reports())


@app.get("/health-scores")
def get_health_scores():
    return JSONResponse(env.get_health_scores())


@app.post("/decision")
def post_decision(payload: InboundMessage):
    try:
        return JSONResponse(_process_inbound(payload))
    except Exception as exc:
        log.exception("decision route failure: %s", exc)
        fallback = _fallback_decision_details(payload.domain, [0.0] * 16, 1.0, "QUARANTINE")
        return JSONResponse(
            {
                "domain": payload.domain,
                "worker_action": {
                    "intent": payload.action.intent,
                    "raw_payload": payload.action.raw_payload,
                    "is_malicious": False,
                },
                "supervisor_decision": {
                    "action_taken": fallback["decision"],
                    "risk_vector": [0.0] * 16,
                    "ambiguity_score": 1.0,
                    "quarantine_steps_remaining": 0,
                    **fallback,
                },
                "environment_state": {"is_shadow_active": False, "domain_data": {}},
                "health_scores": {},
                "quarantine_status": {},
                "quarantine_hold": None,
                "forensic_log": [],
                "incident_report": None,
            }
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log.info("Frontend connected")
    try:
        while True:
            raw = await websocket.receive_text()
            payload = InboundMessage(**json.loads(raw))
            response = OutboundMessage(**_process_inbound(payload))
            await asyncio.sleep(0.2)
            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        log.info("Frontend disconnected")