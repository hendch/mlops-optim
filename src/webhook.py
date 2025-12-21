"""Webhook server to trigger the CI pipeline remotely."""

import subprocess  # nosec B404  # Bandit: subprocess is used intentionally and safely

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="CI Webhook")

CI_COMMAND: list[str] = ["make", "ci"]


@app.post("/trigger")
def trigger_ci():
    """Trigger the local CI pipeline (make ci)."""
    try:
        # Fire-and-forget CI process.
        subprocess.Popen(CI_COMMAND)  # nosec  # fixed command, no shell, no user input
    except OSError as exc:
        # If `make` is missing or something fails at OS level, return 500
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start CI process: {exc}",
        ) from exc

    return JSONResponse({"status": "started", "pipeline": "ci"})
