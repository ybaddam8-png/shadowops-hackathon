"""In-memory deterministic session memory for ShadowOps decisions."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_PATH = BACKEND_DIR / "data" / "session_memory.json"


def _parse_timestamp(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if text.isdigit():
        return float(text)
    with_z = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(with_z)
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _norm(value: str) -> str:
    return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split())


@dataclass(frozen=True)
class ActionMemoryRecord:
    actor: str
    session_id: str
    service: str
    domain: str
    environment: str
    timestamp: Any
    decision: str
    risk_score: float
    action_summary: str
    indicators: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ActionMemoryRecord":
        return cls(
            actor=str(payload.get("actor") or "unknown"),
            session_id=str(payload.get("session_id") or "default"),
            service=str(payload.get("service") or payload.get("domain") or "unknown"),
            domain=str(payload.get("domain") or "unknown"),
            environment=str(payload.get("environment") or "production"),
            timestamp=payload.get("timestamp", 0),
            decision=str(payload.get("decision") or payload.get("supervisor_decision") or "UNKNOWN"),
            risk_score=float(payload.get("risk_score", 0.0)),
            action_summary=str(payload.get("action_summary") or payload.get("raw_payload") or ""),
            indicators=list(payload.get("indicators") or []),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "actor": self.actor,
            "session_id": self.session_id,
            "service": self.service,
            "domain": self.domain,
            "environment": self.environment,
            "timestamp": self.timestamp,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "action_summary": self.action_summary,
            "indicators": list(self.indicators),
        }


class SessionMemory:
    def __init__(
        self,
        max_actions_per_session: int = 20,
        decay_window_seconds: float = 3600.0,
        *,
        persistence_enabled: bool = True,
        storage_path: Path | str = DEFAULT_MEMORY_PATH,
    ):
        self.max_actions_per_session = max_actions_per_session
        self.decay_window_seconds = decay_window_seconds
        self.persistence_enabled = persistence_enabled
        self.storage_path = Path(storage_path)
        self._by_session: dict[str, deque[ActionMemoryRecord]] = defaultdict(
            lambda: deque(maxlen=self.max_actions_per_session)
        )
        if self.persistence_enabled:
            self.load()

    def load(self) -> None:
        if not self.persistence_enabled or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
            for session_id, records in sessions.items():
                queue = self._by_session[str(session_id)]
                for item in records[-self.max_actions_per_session:]:
                    if isinstance(item, dict):
                        queue.append(ActionMemoryRecord.from_mapping(item))
        except Exception:
            self._by_session.clear()

    def save(self) -> None:
        if not self.persistence_enabled:
            return
        payload = {
            "version": 1,
            "max_actions_per_session": self.max_actions_per_session,
            "sessions": {
                session_id: [record.to_mapping() for record in records]
                for session_id, records in self._by_session.items()
            },
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_record(self, record: ActionMemoryRecord | dict[str, Any]) -> ActionMemoryRecord:
        if isinstance(record, dict):
            record = ActionMemoryRecord.from_mapping(record)
        self._by_session[record.session_id].append(record)
        self.save()
        return record

    def get_recent_actions(self, session_id: str, limit: int = 10) -> list[ActionMemoryRecord]:
        records = list(self._by_session.get(str(session_id), ()))
        return records[-limit:]

    def _all_records(self) -> list[ActionMemoryRecord]:
        records: list[ActionMemoryRecord] = []
        for session_records in self._by_session.values():
            records.extend(session_records)
        return records

    def _reference_time(self, records: Iterable[ActionMemoryRecord]) -> float:
        values = [_parse_timestamp(record.timestamp) for record in records]
        return max(values) if values else 0.0

    def _decayed_score(self, records: Iterable[ActionMemoryRecord]) -> float:
        rows = list(records)
        if not rows:
            return 0.0
        reference = self._reference_time(rows)
        weighted = []
        for record in rows:
            age = max(0.0, reference - _parse_timestamp(record.timestamp))
            decay = max(0.0, 1.0 - age / max(self.decay_window_seconds, 1.0))
            indicator_boost = min(0.25, 0.04 * len(record.indicators))
            decision_boost = 0.10 if record.decision.upper() in {"BLOCK", "FORK", "QUARANTINE"} else 0.0
            weighted.append(min(1.0, record.risk_score + indicator_boost + decision_boost) * decay)
        return max(0.0, min(1.0, sum(weighted) / max(len(weighted), 1) + 0.08 * max(0, len(rows) - 1)))

    def compute_actor_risk(self, actor: str) -> float:
        actor = _norm(actor or "unknown")
        return self._decayed_score(record for record in self._all_records() if _norm(record.actor) == actor)

    def compute_session_risk(self, session_id: str) -> float:
        return self._decayed_score(self._by_session.get(str(session_id), ()))

    def compute_service_risk(self, service: str) -> float:
        service = _norm(service or "unknown")
        return self._decayed_score(record for record in self._all_records() if _norm(record.service) == service)

    def _record_tokens(self, record: ActionMemoryRecord) -> set[str]:
        text = _norm(record.action_summary + " " + " ".join(record.indicators))
        tokens = set(record.indicators)
        if "firewall" in text or "security group" in text or "open port" in text:
            tokens.add("firewall_open")
        if "admin" in text or "administratoraccess" in text or "privilege" in text:
            tokens.add("iam_admin")
        if "export" in text or "exfil" in text or "transfer" in text:
            tokens.add("data_export")
        if "secret" in text and ("ci" in text or "workflow" in text):
            tokens.add("ci_secret_access")
        if "workflow" in text or "pipeline" in text:
            tokens.add("workflow_modification")
        if "deploy" in text or "production" in text:
            tokens.add("production_deploy")
        if "public" in text and ("bucket" in text or "s3" in text):
            tokens.add("public_bucket")
        if "external transfer" in text or "external" in text and "transfer" in text:
            tokens.add("external_transfer")
        if "failed auth" in text or "failed login" in text:
            tokens.add("failed_auth")
        if "production change" in text:
            tokens.add("production_change")
        return {_norm(token).replace(" ", "_") for token in tokens}

    def detect_risky_chains(self, session_id: str) -> list[str]:
        sequence = [self._record_tokens(record) for record in self._by_session.get(str(session_id), ())]
        chain_specs = [
            ("firewall open -> IAM admin creation -> data export", ["firewall_open", "iam_admin", "data_export"]),
            ("CI secret access -> workflow modification -> production deploy", ["ci_secret_access", "workflow_modification", "production_deploy"]),
            ("public bucket -> external transfer -> permission escalation", ["public_bucket", "external_transfer", "iam_admin"]),
            ("failed auth -> privilege escalation -> production change", ["failed_auth", "iam_admin", "production_change"]),
        ]
        matches = []
        for name, required in chain_specs:
            cursor = 0
            for tokens in sequence:
                if required[cursor] in tokens:
                    cursor += 1
                    if cursor == len(required):
                        matches.append(name)
                        break
        return matches

    def summarize_memory_context(self, session_id: str) -> dict[str, Any]:
        recent = self.get_recent_actions(session_id, limit=self.max_actions_per_session)
        actor = recent[-1].actor if recent else "unknown"
        service = recent[-1].service if recent else "unknown"
        chains = self.detect_risky_chains(session_id)
        return {
            "session_id": str(session_id),
            "recent_action_count": len(recent),
            "actor": actor,
            "actor_risk": self.compute_actor_risk(actor),
            "session_risk": self.compute_session_risk(session_id),
            "service": service,
            "service_risk": self.compute_service_risk(service),
            "risky_chains": chains,
            "recent_decisions": [record.decision for record in recent[-5:]],
            "recent_indicators": sorted({indicator for record in recent for indicator in record.indicators}),
        }


DEFAULT_MEMORY = SessionMemory()


def add_record(record: ActionMemoryRecord | dict[str, Any]) -> ActionMemoryRecord:
    return DEFAULT_MEMORY.add_record(record)


def get_recent_actions(session_id: str, limit: int = 10) -> list[ActionMemoryRecord]:
    return DEFAULT_MEMORY.get_recent_actions(session_id, limit)


def compute_actor_risk(actor: str) -> float:
    return DEFAULT_MEMORY.compute_actor_risk(actor)


def compute_session_risk(session_id: str) -> float:
    return DEFAULT_MEMORY.compute_session_risk(session_id)


def compute_service_risk(service: str) -> float:
    return DEFAULT_MEMORY.compute_service_risk(service)


def detect_risky_chains(session_id: str) -> list[str]:
    return DEFAULT_MEMORY.detect_risky_chains(session_id)


def summarize_memory_context(session_id: str) -> dict[str, Any]:
    return DEFAULT_MEMORY.summarize_memory_context(session_id)
