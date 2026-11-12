"""Preserve legacy gate transport and validate original optional-phase numbers."""
import json

from fastapi import APIRouter
from fastapi.routing import APIRoute

from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute


class GateChargeRoute(FiniteJsonRoute):
    """Apply finite original-token validation only to the optional settling mode."""

    max_body_bytes = None

    def get_route_handler(self):
        """Keep the previous no-target body policy and handler behavior."""
        legacy = APIRoute.get_route_handler(self)
        validated = super().get_route_handler()

        async def handler(request):
            try:
                policy = json.loads(await request.body())
            except (ValueError, UnicodeDecodeError, RecursionError):
                return await legacy(request)
            if not isinstance(policy, dict) or policy.get('settling_target_v') is None:
                return await legacy(request)
            return await validated(request)

        return handler


router = APIRouter(route_class=GateChargeRoute)
