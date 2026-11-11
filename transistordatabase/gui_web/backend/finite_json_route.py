"""Bound JSON request bytes and preserve finite numeric literals."""
import json
import math
from decimal import Decimal

from fastapi.responses import JSONResponse

from transistordatabase.gui_web.backend.parameter_routes import ParameterRoute


class FiniteJsonRoute(ParameterRoute):
    """Bound actual request bytes before JSON parsing or validation."""

    max_body_bytes = 4 * 1024 * 1024

    def get_route_handler(self):
        """Limit streamed request bodies independently of Content-Length."""
        original = super().get_route_handler()

        async def handler(request):
            chunks = []
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if self.max_body_bytes is not None and size > self.max_body_bytes:
                    return JSONResponse(status_code=413, content={
                        "detail": f"The request exceeds the {self.max_body_bytes} byte limit"})
                chunks.append(chunk)
            request._body = b"".join(chunks)
            # Inspect numeric literals before the standard JSON parser can turn
            # a nonzero decimal such as 1e-400 into a fabricated zero.
            def strict_float(literal):
                number = float(literal)
                mantissa = literal.lower().split("e", 1)[0]
                if (not math.isfinite(number)
                        or (number == 0 and any(c in "123456789" for c in mantissa))):
                    raise ValueError("A JSON number exceeds the finite numerical range")
                # Preserve exact membership checks before typed float conversion.
                # Exact zero avoids Decimal exponent limits and retains its sign.
                return number if number == 0 else Decimal(literal)

            try:
                request._json = json.loads(request._body, parse_float=strict_float)
                pending = [request._json]
                while pending:
                    value = pending.pop()
                    if isinstance(value, str):
                        value.encode("utf-8")
                    elif isinstance(value, dict):
                        pending.extend(value.keys())
                        pending.extend(value.values())
                    elif isinstance(value, list):
                        pending.extend(value)
            except json.JSONDecodeError:
                # Preserve the standard typed error for malformed JSON.
                pass
            except (UnicodeDecodeError, RecursionError):
                return JSONResponse(status_code=422, content={
                    "detail": "Supply a valid JSON object within the parser limits"})
            except UnicodeEncodeError:
                return JSONResponse(status_code=422, content={
                    "detail": "Supply UTF-8 scalar text without isolated Unicode surrogates"})
            except ValueError as exc:
                return JSONResponse(status_code=422, content={"detail": str(exc)})
            return await original(request)

        return handler

