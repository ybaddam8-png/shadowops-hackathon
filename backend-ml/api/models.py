"""api/models.py — Pydantic contracts matching schema_contract.json v3"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ── Inbound ───────────────────────────────────────────────────

class WorkerActionIn(BaseModel):
    intent:      str
    raw_payload: str


class InboundMessage(BaseModel):
    domain: str
    action: WorkerActionIn
    actor: str = "unknown"
    session_id: str = "default"
    service: str = ""
    environment: str = "production"
    provided_evidence: List[str] = Field(default_factory=list)


# ── Outbound sub-models ───────────────────────────────────────

class WorkerActionOut(BaseModel):
    intent:       str
    raw_payload:  str
    is_malicious: bool


class SupervisorDecision(BaseModel):
    action_taken:               str        # ALLOW | BLOCK | FORK | QUARANTINE
    risk_vector:                List[float] = Field(..., min_items=16, max_items=16)  # 16-dim risk feature vector
    ambiguity_score:            float  # [0,1] — how close to 0.5 risk midpoint
    quarantine_steps_remaining: int    # 0 if no active hold
    decision:                   Optional[str] = None
    confidence:                 Optional[float] = None
    uncertainty:                Optional[float] = None
    risk_score:                 Optional[float] = None
    cumulative_risk_score:      Optional[float] = None
    missing_evidence:           List[str] = Field(default_factory=list)
    required_evidence:          List[str] = Field(default_factory=list)
    explanation:                Optional[str] = None
    safe_outcome:               Optional[str] = None
    policy_name:                Optional[str] = None
    domain:                     Optional[str] = None
    mitre_tactic:               Optional[str] = None
    mitre_technique:            Optional[str] = None


class EnvironmentState(BaseModel):
    is_shadow_active: bool
    domain_data:      Dict[str, Any]


class MitreInfo(BaseModel):
    tactic:         str
    technique_id:   str
    technique_name: str
    confidence:     float


class IncidentReport(BaseModel):
    report_id:       str
    timestamp:       str
    domain:          str
    intent:          str
    severity:        str
    confidence:      float
    mitre:           MitreInfo
    blast_radius:    Dict[str, Any]
    cloudtrail:      List[Dict[str, Any]]
    recommendation:  str
    payload_snippet: str


class ForensicEvent(BaseModel):
    domain:    str
    intent:    str
    payload:   str
    step:      int
    timestamp: str


class QuarantineHoldStatus(BaseModel):
    """Per-domain quarantine hold observable state — sent to frontend every step."""
    active:             bool
    steps_remaining:    int          # 0-3
    context_signals:    List[Dict[str, Any]]  # signals emitted during hold
    latest_signal:      Optional[Dict[str, Any]] = None


class DomainQuarantineStatus(BaseModel):
    """Combines domain-level auto-quarantine + action-level hold."""
    domain_quarantined:    bool   # auto-quarantine after 3 consecutive hits
    hold_active:           bool   # QUARANTINE action hold in progress
    hold_steps_remaining:  int    # 0 if no hold


class OutboundMessage(BaseModel):
    domain:              str
    worker_action:       WorkerActionOut
    supervisor_decision: SupervisorDecision
    environment_state:   EnvironmentState
    health_scores:       Dict[str, float]
    # quarantine_status: one entry per domain with full observable state
    quarantine_status:   Dict[str, Any]
    # quarantine_hold: only populated when decision == QUARANTINE
    quarantine_hold:     Optional[QuarantineHoldStatus] = None
    forensic_log:        List[Dict[str, Any]] = Field(default_factory=list)
    incident_report:     Optional[IncidentReport] = None
