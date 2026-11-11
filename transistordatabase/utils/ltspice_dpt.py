"""LTspice Double Pulse Test (DPT) generator and analyzer.

Generates LTspice netlists for double pulse testing and analyzes simulation
results to extract switching loss parameters. PyLTSpice is an optional
dependency - basic netlist generation works without it, but simulation
execution requires it.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

from transistordatabase.core.models import MeasurementEvidence, SwitchingLossData, Transistor

logger = logging.getLogger(__name__)

# Soft import of PyLTSpice
try:
    from PyLTSpice import SimRunner, SpiceEditor  # noqa: F401
    HAS_PYLTSPICE = True
except ImportError:
    HAS_PYLTSPICE = False


@dataclass
class DPTConfig:
    """Configuration for double pulse test.

    :param v_dc: DC bus voltage in V.
    :param i_target: Target inductor current at turn-off in A.
    :param l_load: Load inductance in H.
    :param r_g_on: Gate resistance for turn-on in Ohm.
    :param r_g_off: Gate resistance for turn-off in Ohm.
    :param v_g_on: Gate drive voltage for turn-on in V.
    :param v_g_off: Gate drive voltage for turn-off in V.
    :param t_dead: Dead time between pulses in s.
    :param t_pulse2: Duration of second pulse in s.
    """

    v_dc: float = 400.0
    i_target: float = 20.0
    l_load: float = 100e-6  # H
    r_g_on: float = 10.0    # Ohm
    r_g_off: float = 10.0   # Ohm
    v_g_on: float = 15.0    # V
    v_g_off: float = -5.0   # V
    t_dead: float = 5e-6    # s
    t_pulse2: float = 2e-6  # s

    model_kind: str = "generic_nmos"

    def validate(self) -> None:
        """Validate physical and representable timing domains for the template."""
        for key in ("v_dc", "i_target", "l_load", "r_g_on", "r_g_off",
                    "v_g_on", "v_g_off", "t_dead", "t_pulse2"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        for key in ("v_dc", "i_target", "l_load", "r_g_on", "r_g_off",
                    "t_dead", "t_pulse2"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        # The uncalibrated level-1 NMOS below has VTO=3 V. Steering diodes
        # reduce the applied rail by approximately a diode drop.
        if not self.v_g_off + 1.0 < 3.0 < self.v_g_on - 1.0:
            raise ValueError("Generic NMOS requires v_g_off + 1 V < 3 V < v_g_on - 1 V")
        if self.model_kind != "generic_nmos":
            raise ValueError("Only the explicitly selected generic_nmos template is supported")
        pulse = self.i_target * (self.l_load / self.v_dc)
        if not math.isfinite(pulse) or pulse <= 0:
            raise ValueError("First pulse duration must be finite and positive")

    @property
    def t_pulse1(self) -> float:
        """Calculate first pulse duration to reach target current.

        Uses the inductor equation: I = V * t / L => t = I * L / V.

        :return: Duration of first pulse in seconds.
        :raises ValueError: If DC bus voltage is not positive.
        """
        self.validate()
        return self.i_target * (self.l_load / self.v_dc)


@dataclass
class DPTResult:
    """Results from double pulse test analysis.

    :param e_on: Turn-on switching energy in J.
    :param e_off: Turn-off switching energy in J.
    :param t_r: Current rise time in s.
    :param t_f: Current fall time in s.
    :param i_peak: Peak current during turn-on in A.
    :param v_overshoot: Peak drain voltage in the bounded off-event region, V.
        Legacy field name; this is an absolute peak, not excess above the bus.
    :param di_dt_on: di/dt during turn-on in A/s.
    :param dv_dt_off: dv/dt during turn-off in V/s.
    """

    e_on: float | None = None
    e_off: float | None = None
    t_r: float = 0.0
    t_f: float = 0.0
    i_peak: float = 0.0
    v_overshoot: float = 0.0
    di_dt_on: float = 0.0
    dv_dt_off: float = 0.0
    i_on: float | None = None
    i_off: float | None = None
    on_window: tuple[float, float] | None = None
    off_window: tuple[float, float] | None = None
    energy_window_convention: str = "unspecified"
    event_evidence: dict = field(default_factory=dict)


class LTspiceDPT:
    """LTspice Double Pulse Test generator and analyzer.

    Provides methods to generate LTspice netlists for double pulse testing,
    run simulations (requires PyLTSpice), and analyze waveforms to extract
    switching loss parameters.
    """

    def __init__(self, config: DPTConfig) -> None:
        """Initialize LTspice DPT with the given configuration.

        :param config: Double pulse test configuration.
        """
        self.config = config

    MODEL_DISCLOSURE = (
        "Generic uncalibrated NMOS template: VTO=3 V, KP=50 A/V², RD=10 mΩ, "
        "Cgs=2 nF, Cgd=100 pF and Cds=200 pF; idealized freewheel/steering "
        "diodes. The selected transistor name is a label only. This is not a "
        "vendor device model or a prediction of its switching energy. First "
        "pulse duration uses ideal I·L/V and excludes losses and gate delay. "
        "Pulse widths and dead time denote flat drive plateaus, with fixed "
        "10 ns edges and a 1 µs initial delay."
    )

    @staticmethod
    def netlist_filename(transistor: Optional[Transistor] = None) -> str:
        """Return a portable filename without interpreting the label as a path."""
        name = transistor.metadata.name if transistor else "DUT"
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:100] or "DUT"
        return f"dpt_{label}.cir"

    def pulse_schedule(self) -> list[tuple[float, float]]:
        """Two finite-edged pulses; widths/dead time refer to flat plateaus."""
        cfg = self.config
        cfg.validate()
        edge = 10e-9
        a = 1e-6
        b = a + edge
        c = b + cfg.t_pulse1
        d = c + edge
        e = d + cfg.t_dead
        f = e + edge
        g = f + cfg.t_pulse2
        h = g + edge
        stop = h + max(cfg.t_dead, 5e-6)
        times = [0.0, a, b, c, d, e, f, g, h, stop]
        if not all(math.isfinite(t) for t in times) or any(
            right <= left for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError("Pulse schedule must have finite, distinct increasing times")
        levels = [cfg.v_g_off, cfg.v_g_off, cfg.v_g_on, cfg.v_g_on,
                  cfg.v_g_off, cfg.v_g_off, cfg.v_g_on, cfg.v_g_on,
                  cfg.v_g_off, cfg.v_g_off]
        return list(zip(times, levels, strict=True))

    def render_netlist(self, transistor: Optional[Transistor] = None) -> str:
        """Render a generic DPT deck without filesystem or simulator access.

        No vendor parameters are inferred from the selected transistor. The
        external capacitors and level-1 MOSFET are illustrative constants.
        """
        cfg = self.config
        schedule = self.pulse_schedule()
        name = " ".join((transistor.metadata.name if transistor else "DUT").split())
        pwl = " ".join(f"{time:.17g} {voltage:.17g}" for time, voltage in schedule)
        lines = [
            f"* Generic Double Pulse Test; selected label: {name}",
            "* UNCALIBRATED GENERIC TEMPLATE - selected name is metadata only",
            "* VTO=3 V KP=50 RD=0.01; Cgs=2n Cgd=100p Cds=200p",
            "* Idealized diodes; no vendor model, recovery, package parasitics or thermal model",
            f"* Ideal first pulse I*L/V = {cfg.t_pulse1:.17g} s; actual current differs",
            f"Vbus bus 0 {cfg.v_dc!r}",
            f"Lload bus sw {cfg.l_load!r}",
            "M_dut sw gate 0 0 GENERIC_DPT_NMOS",
            "D_freewheel sw bus GENERIC_FREEWHEEL",
            f"R_on drive on_path {cfg.r_g_on!r}",
            "D_on on_path gate GATE_STEERING",
            "D_off gate off_path GATE_STEERING",
            f"R_off off_path drive {cfg.r_g_off!r}",
            f"Vdrive drive 0 PWL({pwl})",
            "Cgs gate 0 2n",
            "Cgd gate sw 100p",
            "Cds sw 0 200p",
            ".model GENERIC_DPT_NMOS NMOS(VTO=3 KP=50 RD=0.01)",
            ".model GENERIC_FREEWHEEL D(IS=1e-12 RS=0.01 N=1)",
            ".model GATE_STEERING D(IS=1e-12 RS=0.01 N=1)",
            f".tran 0 {schedule[-1][0]:.17g} 0 1n",
            ".options plotwinsize=0",
            ".save V(gate) V(drive) V(sw) I(Lload) Id(M_dut) I(D_freewheel)",
            ".end",
        ]
        return "\n".join(lines) + "\n"

    def generate_netlist(
        self, output_path: Path, transistor: Optional[Transistor] = None,
    ) -> Path:
        """Write the pure renderer's text; explicit file paths are preserved.

        Directory destinations use .cir because this is a textual netlist,
        not an LTspice .asc schematic. Existing files are never overwritten.
        """
        text = self.render_netlist(transistor)
        output_path = Path(output_path)
        path = (output_path / self.netlist_filename(transistor)
                if output_path.is_dir() else output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return path

    def run_simulation(self, netlist_path: Path) -> Optional[Path]:
        """Run LTspice simulation on the given netlist.

        Requires PyLTSpice to be installed.

        :param netlist_path: Path to the LTspice netlist.
        :return: Path to raw output file, or None if simulation failed.
        :raises ImportError: If PyLTSpice is not installed.
        """
        if not HAS_PYLTSPICE:
            raise ImportError(
                "PyLTSpice is required for simulation execution. "
                "Install with: pip install PyLTSpice"
            )

        runner = SimRunner(output_folder=str(netlist_path.parent))
        # run_now waits for the owned task and returns its actual output paths;
        # an asynchronous run followed by exists() races simulation completion.
        raw_path, _log_path = runner.run_now(str(netlist_path))
        return Path(raw_path) if raw_path is not None and Path(raw_path).exists() else None

    def analyze_waveforms(
        self,
        time: npt.NDArray[np.float64],
        v_ds: npt.NDArray[np.float64],
        i_d: npt.NDArray[np.float64],
        v_gs: Optional[npt.NDArray[np.float64]] = None,
    ) -> DPTResult:
        """Measure first turn-off and next turn-on using the measured gate.

        A gate trace is required: missing/unresolved events raise ValueError.
        The trace may start during the first high pulse. Gate midpoint edges
        bound each event, drain 1/99% crossings locate the local commutation,
        and local current levels set 10/90% times. Energies integrate measured
        Vds*Id over the union of local 1/99% drain/current transitions (no
        conduction subtraction). This convention is not a vendor IEC window.
        Later second-pulse current growth cannot set first-event thresholds.
        """
        if (not math.isfinite(self.config.v_dc) or self.config.v_dc <= 0 or
                not math.isfinite(self.config.v_g_on) or
                not math.isfinite(self.config.v_g_off) or
                self.config.v_g_on <= self.config.v_g_off):
            raise ValueError("Declare a positive bus and finite increasing gate rails")
        if v_gs is None:
            raise ValueError("Measured v_gs is required to identify DPT events")
        arrays = [np.asarray(a, dtype=float) for a in (time, v_ds, i_d, v_gs)]
        time, v_ds, i_d, v_gs = arrays
        if any(a.ndim != 1 or len(a) != len(time) for a in arrays) or len(time) < 8:
            raise ValueError("Waveforms must be aligned one-dimensional arrays (at least 8 samples)")
        if any(not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("Waveforms must contain finite values")
        if np.any(np.diff(time) <= 0):
            raise ValueError("Time must be strictly increasing")
        gate_high = v_gs >= (self.config.v_g_on + self.config.v_g_off) / 2
        falls = np.flatnonzero(gate_high[:-1] & ~gate_high[1:]) + 1
        rises = np.flatnonzero(~gate_high[:-1] & gate_high[1:]) + 1
        if not len(falls):
            raise ValueError("No first falling gate edge was resolved")
        off_edge = falls[0]
        next_rises = rises[rises > off_edge]
        if not len(next_rises):
            raise ValueError("No second rising gate edge was resolved")
        on_edge = next_rises[0]
        prior_rises = rises[rises < off_edge]
        first_start = int(prior_rises[-1]) if len(prior_rises) else 0
        later_falls = falls[falls > on_edge]
        second_end = int(later_falls[0]) if len(later_falls) else len(time) - 1
        def midpoint_index(first, last):
            return int(np.searchsorted(time, (time[first] + time[last]) / 2))

        middle = midpoint_index(off_edge, on_edge)
        off_left = midpoint_index(first_start, off_edge)
        on_right = midpoint_index(on_edge, second_end)

        def crossing(signal, level, start, end, rising):
            x, y = signal[start:end], signal[start + 1:end + 1]
            mask = ((x <= level) & (y > level) if rising
                    else (x >= level) & (y < level))
            indices = np.flatnonzero(mask)
            if not len(indices):
                raise ValueError(f"Unresolved {'rising' if rising else 'falling'} switching threshold")
            index = start + int(indices[0])
            return float(time[index] + (level - signal[index]) /
                         (signal[index + 1] - signal[index]) *
                         (time[index + 1] - time[index]))

        def event(left, edge, right, turn_on):
            if right - left < 6:
                raise ValueError("Insufficient samples to resolve the switching event")
            n = max(2, (right - left) // 10)
            va = float(np.median(v_ds[left:left + n]))
            vb = float(np.median(v_ds[right - n:right]))
            low, high = min(va, vb), max(va, vb)
            if high - low < .5 * self.config.v_dc or (va > vb) != turn_on:
                raise ValueError("Gate event does not contain a resolved drain commutation")
            vstart = crossing(v_ds, low + (high-low) * (.99 if turn_on else .01),
                              left, right, not turn_on)
            vend = crossing(v_ds, low + (high-low) * (.01 if turn_on else .99),
                            left, right, not turn_on)
            # The first off-pulse plateau or the end of the second on-transition
            # determines the current level, never the full-record maximum.
            if turn_on:
                current_high = float(np.interp(vend, time, i_d))
                current_low = float(np.median(i_d[left:left+n]))
            else:
                before = int(np.searchsorted(time, vstart))
                current_high = float(np.max(i_d[left:before + 1]))
                current_low = float(np.median(i_d[right-n:right]))
            amplitude = current_high - current_low
            if not math.isfinite(amplitude) or amplitude <= 0:
                raise ValueError("Switching event must have a positive resolved current step")
            fractions = (.01, .1, .9, .99) if turn_on else (.99, .9, .1, .01)
            thresholds = [crossing(i_d, current_low + amplitude * fraction,
                                   left, right, turn_on) for fraction in fractions]
            start, stop = min(vstart, thresholds[0]), max(vend, thresholds[-1])
            if stop <= start or thresholds[2] <= thresholds[1]:
                raise ValueError("Switching thresholds must resolve increasing event times")
            selected = (time > start) & (time < stop)
            t = np.r_[start, time[selected], stop]
            v = np.interp(t, time, v_ds)
            current = np.interp(t, time, i_d)
            energy = float(np.trapezoid(v * current, t))
            result = (energy, thresholds[2] - thresholds[1],
                      float(np.max(current)), float(np.max(v_ds[left:right + 1])),
                      .8 * amplitude / (thresholds[2] - thresholds[1]),
                      .8 * (high-low) / (crossing(v_ds, low+.9*(high-low), left, right,
                                                 not turn_on) -
                                             crossing(v_ds, low+.1*(high-low), left, right,
                                                      not turn_on)),
                      current_high, (start, stop))
            if not all(math.isfinite(value) for value in result[:7]):
                raise ValueError("Derived switching results must be finite")
            evidence = {
                "gate_edge_s": float(time[edge]),
                "local_current_low_a": current_low,
                "local_current_high_a": current_high,
                "local_vds_low_v": low, "local_vds_high_v": high,
                "current_threshold_times_s": thresholds,
                "voltage_1_99_times_s": [vstart, vend],
                "integration_window_s": [start, stop],
            }
            return (*result, evidence)

        off = event(off_left, off_edge, middle, False)
        on = event(middle, on_edge, on_right, True)
        return DPTResult(e_on=on[0], e_off=off[0], t_r=on[1], t_f=off[1],
                         i_peak=on[2], v_overshoot=off[3], di_dt_on=on[4],
                         dv_dt_off=off[5], i_on=on[6], i_off=off[6],
                         on_window=on[7], off_window=off[7],
                         energy_window_convention=(
                             "Integral of measured Vds*Id over the union of local "
                             "1%-99% drain/current transitions; no conduction "
                             "subtraction; not a vendor/IEC full-energy window"
                         ),
                         event_evidence={"turn_on": on[8], "turn_off": off[8]})

    def extract_switching_loss_data(
        self, result: DPTResult, t_j: float = 25.0,
        evidence: MeasurementEvidence | None = None,
    ) -> tuple[SwitchingLossData, SwitchingLossData]:
        """Convert DPT results to SwitchingLossData objects.

        Creates single-point switching loss data entries suitable for
        storing in a Transistor's switch component.

        :param result: DPT analysis result.
        :param t_j: Junction temperature in deg C.
        :return: Tuple of (e_on_data, e_off_data) SwitchingLossData.
        """
        cfg = self.config
        def finite(value):
            return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)

        if not finite(result.e_on) or not finite(result.e_off):
            raise ValueError("Supply explicit finite turn-on and turn-off energies")
        def valid_event(event, current, window):
            data = result.event_evidence.get(event, {})
            required = {"gate_edge_s", "local_current_low_a", "local_current_high_a",
                        "local_vds_low_v", "local_vds_high_v", "current_threshold_times_s",
                        "voltage_1_99_times_s", "integration_window_s"}
            if (not finite(current) or current <= 0 or not isinstance(window, (list, tuple)) or
                    len(window) != 2 or not all(finite(value) for value in window) or
                    window[1] <= window[0] or not isinstance(data, dict) or
                    not required.issubset(data)):
                return False
            if data["integration_window_s"] != list(window):
                return False
            for key in required:
                values = data[key] if isinstance(data[key], (list, tuple)) else [data[key]]
                if not values or not all(finite(value) for value in values):
                    return False
            return (len(data["current_threshold_times_s"]) == 4 and
                    len(data["voltage_1_99_times_s"]) == 2)

        resolved = (result.i_on is not None and result.i_off is not None and
                    result.on_window is not None and result.off_window is not None and
                    set(result.event_evidence) == {"turn_on", "turn_off"} and
                    valid_event("turn_on", result.i_on, result.on_window) and
                    valid_event("turn_off", result.i_off, result.off_window) and
                    bool(result.energy_window_convention.strip()) and
                    result.energy_window_convention != "unspecified")
        if evidence is not None and not resolved:
            raise ValueError("Measured extraction requires resolved DPT event evidence")
        source = evidence or MeasurementEvidence(
            source_kind="unknown" if resolved else "manual",
            method="waveform-local-edge" if resolved else "manual-point",
            manual_override=None if resolved else (
                "Manually supplied energies. Missing currents use the requested target, not a measured current."
            ))
        def event_evidence(event, current, window):
            if not resolved:
                return replace(source, event=event, local_current_a=current,
                               bus_voltage_v=cfg.v_dc, t_j=t_j,
                               v_g_on=cfg.v_g_on, v_g_off=cfg.v_g_off,
                               r_g_on=cfg.r_g_on, r_g_off=cfg.r_g_off)
            return replace(source, event=event, local_current_a=current, window_s=window,
                           method="waveform-local-edge", bus_voltage_v=cfg.v_dc, t_j=t_j,
                           v_g_on=cfg.v_g_on, v_g_off=cfg.v_g_off,
                           r_g_on=cfg.r_g_on, r_g_off=cfg.r_g_off,
                           convention=result.energy_window_convention,
                           thresholds=result.event_evidence[event])

        e_on_data = SwitchingLossData(
            dataset_type="single",
            t_j=t_j,
            v_supply=cfg.v_dc,
            v_g=cfg.v_g_on,
            v_g_off=cfg.v_g_off,
            e_x=result.e_on,
            i_x=result.i_on if result.i_on is not None else cfg.i_target,
            r_g=cfg.r_g_on,
            evidence=event_evidence("turn_on", result.i_on, result.on_window),
        )

        e_off_data = SwitchingLossData(
            dataset_type="single",
            t_j=t_j,
            v_supply=cfg.v_dc,
            v_g=cfg.v_g_on,
            v_g_off=cfg.v_g_off,
            e_x=result.e_off,
            i_x=result.i_off if result.i_off is not None else cfg.i_target,
            r_g=cfg.r_g_off,
            evidence=event_evidence("turn_off", result.i_off, result.off_window),
        )

        return e_on_data, e_off_data
