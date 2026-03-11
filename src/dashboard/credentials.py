"""Credential validation for each provider."""
from __future__ import annotations

import os
from pathlib import Path


CREDENTIAL_REQUIREMENTS: dict[str, list[dict]] = {
    "aws": {
        "display": "AWS (Rekognition)",
        "pipelines": ["rekognition", "rek_claude_haiku"],
        "checks": [
            {"env": "AWS_ACCESS_KEY_ID", "label": "AWS_ACCESS_KEY_ID"},
            {"env": "AWS_SECRET_ACCESS_KEY", "label": "AWS_SECRET_ACCESS_KEY"},
        ],
    },
    "google_vision": {
        "display": "Google Cloud Vision",
        "pipelines": ["google_vision", "gv_gemini_flash", "gv_claude_sonnet"],
        "checks": [
            {"env": "GOOGLE_APPLICATION_CREDENTIALS", "label": "GOOGLE_APPLICATION_CREDENTIALS", "is_file": True},
        ],
    },
    "openrouter": {
        "display": "OpenRouter (LLMs)",
        "pipelines": ["claude_haiku", "gemini_flash", "gpt41_mini", "gv_gemini_flash", "rek_claude_haiku", "gv_claude_haiku"],
        "checks": [
            {"env": "OPENROUTER_API_KEY", "label": "OPENROUTER_API_KEY"},
        ],
    },
}


def check_credential(check: dict) -> tuple[bool, str]:
    env_var = check["env"]
    value = os.environ.get(env_var, "")

    if not value:
        return False, f"{check['label']} is not set"

    if check.get("is_file"):
        if not Path(value).exists():
            return False, f"{check['label']} file not found: {value}"

    masked = value[:8] + "..." + value[-4:] if len(value) > 16 else value[:4] + "..."
    return True, masked


def check_all_credentials() -> list[dict]:
    results = []
    for key, cred_info in CREDENTIAL_REQUIREMENTS.items():
        checks_ok = True
        details = []
        for check in cred_info["checks"]:
            ok, msg = check_credential(check)
            details.append({"label": check["label"], "ok": ok, "message": msg})
            if not ok:
                checks_ok = False

        results.append({
            "key": key,
            "display": cred_info["display"],
            "pipelines": cred_info["pipelines"],
            "ok": checks_ok,
            "details": details,
        })
    return results


def get_available_pipelines() -> set[str]:
    available = set()
    cred_results = check_all_credentials()

    pipeline_requirements: dict[str, list[str]] = {}
    for cred in CREDENTIAL_REQUIREMENTS.values():
        for pipeline_key in cred["pipelines"]:
            pipeline_requirements.setdefault(pipeline_key, [])

    from src.pipelines.registry import PIPELINE_REGISTRY
    for pipeline_key in PIPELINE_REGISTRY:
        required_creds = []
        for cred_key, cred_info in CREDENTIAL_REQUIREMENTS.items():
            if pipeline_key in cred_info["pipelines"]:
                required_creds.append(cred_key)

        all_ok = True
        for cred_key in required_creds:
            cred_result = next((c for c in cred_results if c["key"] == cred_key), None)
            if cred_result and not cred_result["ok"]:
                all_ok = False
                break

        if all_ok:
            available.add(pipeline_key)

    return available
