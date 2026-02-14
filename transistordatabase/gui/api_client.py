"""REST API client for the Transistor Database FastAPI backend.

Provides a Python interface for the PyQt5 GUI to communicate with the
backend. All methods return plain dicts/lists and handle HTTP errors by
raising ``ApiError``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


class ApiError(Exception):
    """Raised when an API call fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class TransistorApiClient:
    """HTTP client wrapping the FastAPI backend endpoints.

    :param base_url: Base URL of the running backend (e.g.
        ``http://localhost:8001``).
    :param timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _handle(self, resp: requests.Response) -> Any:
        """Check response status and return JSON payload."""
        if resp.ok:
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return resp.content
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise ApiError(resp.status_code, detail)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_transistors(self) -> list[dict[str, Any]]:
        """Get all transistors from the database."""
        resp = self._session.get(self._url("/api/transistors"), timeout=self.timeout)
        return self._handle(resp)

    def get_transistor(self, name: str) -> dict[str, Any]:
        """Get a single transistor by name."""
        resp = self._session.get(
            self._url(f"/api/transistors/{name}"), timeout=self.timeout,
        )
        return self._handle(resp)

    def create_transistor(self, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new transistor from a dict."""
        resp = self._session.post(
            self._url("/api/transistors"), json=data, timeout=self.timeout,
        )
        return self._handle(resp)

    def update_transistor(self, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing transistor."""
        resp = self._session.put(
            self._url(f"/api/transistors/{name}"), json=data, timeout=self.timeout,
        )
        return self._handle(resp)

    def delete_transistor(self, name: str) -> dict[str, Any]:
        """Delete a transistor by name."""
        resp = self._session.delete(
            self._url(f"/api/transistors/{name}"), timeout=self.timeout,
        )
        return self._handle(resp)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_json(self, file_path: str | Path) -> dict[str, Any]:
        """Upload a transistor JSON file."""
        p = Path(file_path)
        with open(p, "rb") as f:
            resp = self._session.post(
                self._url("/api/transistors/upload"),
                files={"file": (p.name, f, "application/json")},
                timeout=self.timeout,
            )
        return self._handle(resp)

    # ------------------------------------------------------------------
    # Validation & Comparison
    # ------------------------------------------------------------------

    def validate(self, name: str) -> dict[str, Any]:
        """Validate a transistor."""
        resp = self._session.post(
            self._url(f"/api/transistors/{name}/validate"), timeout=self.timeout,
        )
        return self._handle(resp)

    def compare(self, names: list[str]) -> dict[str, Any]:
        """Compare multiple transistors."""
        resp = self._session.post(
            self._url("/api/transistors/compare"),
            json=names,
            timeout=self.timeout,
        )
        return self._handle(resp)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        name: str,
        fmt: str,
        save_path: str | Path | None = None,
    ) -> bytes | Path:
        """Export a transistor in the given format.

        :param name: Transistor name.
        :param fmt: One of ``json``, ``csv``, ``spice``, ``plecs``,
            ``matlab``, ``gecko``, ``ltspice``.
        :param save_path: If provided, write the response body to this
            file and return the Path.  Otherwise return raw bytes.
        """
        resp = self._session.post(
            self._url(f"/api/transistors/{name}/export/{fmt}"),
            timeout=self.timeout,
        )
        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise ApiError(resp.status_code, detail)

        if save_path is not None:
            p = Path(save_path)
            p.write_bytes(resp.content)
            return p
        return resp.content

    # ------------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------------

    def plot_channel(
        self, name: str, component: str = "switch",
    ) -> dict[str, Any]:
        """Get channel characteristic plot data."""
        resp = self._session.get(
            self._url(f"/api/plots/channel/{name}"),
            params={"component": component},
            timeout=self.timeout,
        )
        return self._handle(resp)

    def plot_switching(
        self,
        name: str,
        loss_type: str = "e_on",
        plot_type: str = "i_e",
    ) -> dict[str, Any]:
        """Get switching loss plot data."""
        resp = self._session.get(
            self._url(f"/api/plots/switching/{name}"),
            params={"loss_type": loss_type, "plot_type": plot_type},
            timeout=self.timeout,
        )
        return self._handle(resp)

    def plot_soa(self, name: str) -> dict[str, Any]:
        """Get safe operating area plot data."""
        resp = self._session.get(
            self._url(f"/api/plots/soa/{name}"), timeout=self.timeout,
        )
        return self._handle(resp)

    def plot_thermal(self, name: str) -> dict[str, Any]:
        """Get thermal impedance plot data."""
        resp = self._session.get(
            self._url(f"/api/plots/thermal/{name}"), timeout=self.timeout,
        )
        return self._handle(resp)

    def plot_gate_charge(self, name: str) -> dict[str, Any]:
        """Get gate charge plot data."""
        resp = self._session.get(
            self._url(f"/api/plots/gate_charge/{name}"), timeout=self.timeout,
        )
        return self._handle(resp)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if the backend is reachable."""
        try:
            resp = self._session.get(self._url("/"), timeout=5.0)
            return resp.ok
        except requests.ConnectionError:
            return False
