"""Compatibility entrypoint for the ShadowOps OpenEnv environment.

This module intentionally wraps ``openenv_shadowops`` instead of duplicating
environment logic. Hackathon packaging can point at this file while existing
backend and test code can continue using ``openenv_shadowops.py``.
"""

from __future__ import annotations

from typing import Any

from openenv_shadowops import ShadowOpsOpenEnv


class ShadowOpsOpenEnvV1(ShadowOpsOpenEnv):
    """OpenEnv-compatible ShadowOps environment with stable metadata."""

    metadata = {
        **ShadowOpsOpenEnv.metadata,
        "openenv_entrypoint": "backend-ml/openenv_shadowops_env.py:ShadowOpsOpenEnvV1",
        "schema_contract": "backend-ml/schema_contract.json",
        "episode_metadata": {
            "deterministic_seed": True,
            "multi_step_incident_trajectory": True,
            "model_free_by_default": True,
        },
    }


def make_env(**kwargs: Any) -> ShadowOpsOpenEnvV1:
    """Factory used by OpenEnv runners and simple smoke tests."""

    return ShadowOpsOpenEnvV1(**kwargs)
