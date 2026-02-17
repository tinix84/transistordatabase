"""Entry point for the TDB web GUI (FastAPI + Vue 3 frontend)."""
from __future__ import annotations

import threading
import webbrowser


def _open_browser() -> None:
    """Open the default browser to the local web GUI."""
    webbrowser.open("http://localhost:8000")


def main() -> None:
    """Start FastAPI with the bundled Vue 3 frontend and open the browser."""
    import uvicorn
    threading.Timer(1.5, _open_browser).start()
    uvicorn.run(
        "transistordatabase.gui_web.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
