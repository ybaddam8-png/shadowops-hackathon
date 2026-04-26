import json
import os
from urllib import request


BASE = os.getenv("SHADOWOPS_BASE_URL", "http://127.0.0.1:8000")


def _get(path: str):
    with request.urlopen(f"{BASE}{path}", timeout=20) as r:
        body = r.read().decode("utf-8")
        return r.status, json.loads(body)


def _post(path: str, payload: dict):
    req = request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")
        return r.status, json.loads(body)


def main():
    status, health = _get("/health")
    print("GET /health", status, json.dumps(health, indent=2))

    status, ready = _get("/ready")
    print("GET /ready", status, json.dumps(ready, indent=2))

    payload = {
        "domain": "GITHUB",
        "action": {
            "intent": "MERGE_PR",
            "raw_payload": "Developer running integration tests with User-Agent: pytest/8.2",
        },
        "actor": "test_client",
        "session_id": "smoke-api",
        "service": "github",
        "environment": "production",
        "provided_evidence": [],
    }
    status, decision = _post("/decision", payload)
    print("POST /decision", status)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
