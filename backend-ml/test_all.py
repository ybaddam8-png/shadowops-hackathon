import json
import os
from urllib import request


BASE = os.getenv("SHADOWOPS_BASE_URL", "http://127.0.0.1:8000")
REQUIRED_TOP_LEVEL = {
    "domain",
    "worker_action",
    "supervisor_decision",
    "environment_state",
    "health_scores",
    "quarantine_status",
    "forensic_log",
}
REQUIRED_SUPERVISOR = {"action_taken", "decision", "risk_vector", "ambiguity_score"}


def _fetch_json(url: str):
    with request.urlopen(url, timeout=20) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _assert(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    status, health = _fetch_json(f"{BASE}/health")
    _assert(status == 200, "health endpoint failed")
    _assert(health.get("status") == "alive", "health status should be alive")

    status, ready = _fetch_json(f"{BASE}/ready")
    _assert(status == 200, "ready endpoint failed")
    _assert("mode" in ready, "ready payload missing mode")

    payload = {
        "domain": "SOC",
        "action": {
            "intent": "UPDATE_FIREWALL",
            "raw_payload": "Routine test traffic from CI with User-Agent: pytest/8.2",
        },
        "actor": "test_all",
        "session_id": "smoke-all",
        "service": "soc",
        "environment": "production",
        "provided_evidence": [],
    }
    status, decision = _post_json(f"{BASE}/decision", payload)
    _assert(status == 200, "decision endpoint failed")
    _assert(REQUIRED_TOP_LEVEL.issubset(decision.keys()), "decision payload missing top-level fields")
    supervisor = decision["supervisor_decision"]
    _assert(REQUIRED_SUPERVISOR.issubset(supervisor.keys()), "supervisor payload missing required fields")
    _assert(supervisor["decision"] in {"ALLOW", "BLOCK", "FORK", "QUARANTINE"}, "invalid action returned")
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
