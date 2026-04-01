"""Environment audit trail: git, package versions, prompt template id."""

from __future__ import annotations

import subprocess
from importlib import metadata
from typing import Any

# Bump when hiring system / final JSON instructions change materially.
PROMPT_TEMPLATE_ID = "hiring_system_final_json_v1"


def git_commit_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=None,
        )
        if out.returncode == 0:
            return out.stdout.strip()[:40]
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def package_versions(packages: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def audit_metadata() -> dict[str, Any]:
    try:
        import gshp

        gshp_ver = getattr(gshp, "__version__", "unknown")
    except ImportError:
        gshp_ver = "unknown"
    pkgs = package_versions(["networkx", "pydantic", "openai", "httpx"])
    pkgs["gshp"] = gshp_ver
    return {
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "git_commit": git_commit_short(),
        "python_packages": pkgs,
    }
