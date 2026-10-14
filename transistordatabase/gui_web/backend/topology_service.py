"""Transport-facing orchestration for headless topology calculations."""
from __future__ import annotations

from typing import Any, Dict, Optional

from transistordatabase.core.models import Transistor
from transistordatabase.topologies.operating_point import (
    SUPPORTED_TOPOLOGIES,
    calculate_topology_operating_point,
    calculate_topology_sweep,
)


class TopologyCalculatorService:
    """Delegate topology work to reusable core calculations."""

    SUPPORTED_TOPOLOGIES = list(SUPPORTED_TOPOLOGIES)

    @staticmethod
    def calculate(
        topology: str,
        parameters: Dict[str, Any],
        transistor: Optional[Transistor] = None,
        *,
        switch_transistor: Transistor | None = None,
        diode_transistor: Transistor | None = None,
    ) -> Dict[str, Any]:
        """Calculate one point and an optional Python-generated sweep.

        ``transistor`` remains as a compatibility argument for callers that use
        one TDB device for both switch and diode roles.
        """
        if transistor is not None:
            switch_transistor = switch_transistor or transistor
            diode_transistor = diode_transistor or transistor
        values = dict(parameters)
        sweep = values.pop("sweep", None)
        result = calculate_topology_operating_point(
            topology, values, switch_transistor, diode_transistor
        )
        if sweep is not None:
            result["sweep"] = calculate_topology_sweep(
                topology, values, sweep, switch_transistor, diode_transistor
            )
        return result

    @staticmethod
    def compare_topologies(
        parameters: Dict[str, Any],
        transistor: Optional[Transistor] = None,
        *,
        switch_transistor: Transistor | None = None,
        diode_transistor: Transistor | None = None,
    ) -> Dict[str, Any]:
        """Compare supported topologies and retain per-topology domain errors."""
        if (
            parameters.get("mode") == "loss-balanced"
            and parameters.get("conduction_regime", "CCM") == "DCM"
        ):
            raise ValueError(
                "Topology comparison does not support loss-balanced DCM; "
                "scalar Buck, Boost, Buck-Boost, and Flyback DCM operating "
                "points are supported individually"
            )
        results: Dict[str, Any] = {}
        for topology in TopologyCalculatorService.SUPPORTED_TOPOLOGIES:
            try:
                topology_parameters = dict(parameters)
                if topology != "flyback":
                    topology_parameters.pop("turns_ratio_nps", None)
                results[topology] = TopologyCalculatorService.calculate(
                    topology, topology_parameters, transistor,
                    switch_transistor=switch_transistor,
                    diode_transistor=diode_transistor,
                )
            except ValueError as exc:
                results[topology] = {"error": str(exc)}

        efficiencies = {
            topology: result["performance"]["efficiency_percent"]
            for topology, result in results.items()
            if "error" not in result and result["performance"].get("efficiency_percent") is not None
        }
        results["comparison"] = {
            "efficiencies_percent": efficiencies,
            "best_topology": max(efficiencies, key=efficiencies.get) if efficiencies else None,
            "ranking_available": bool(efficiencies),
            "reason": (
                None if efficiencies else
                (
                    "Estimate mode does not calculate efficiency."
                    if parameters.get("mode") == "estimate" else
                    "No topology produced a comparable efficiency; check the required device curves and operating conditions."
                )
            ),
            "loss_scope": (
                "channel_conduction_only"
                if parameters.get("mode") in ("conduction-only", "loss-balanced") else
                "channel_conduction_and_switch_turn_on_turn_off"
                if parameters.get("mode") == "detailed" else None
            ),
        }
        return results
