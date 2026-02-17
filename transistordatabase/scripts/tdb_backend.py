"""Entry point for the TDB headless backend (FastAPI REST API)."""
from __future__ import annotations


def main() -> None:
    """Start the FastAPI server as a headless REST API on port 8000."""
    import uvicorn
    uvicorn.run(
        "transistordatabase.gui_web.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
