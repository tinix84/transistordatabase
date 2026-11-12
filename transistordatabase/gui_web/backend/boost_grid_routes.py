"""Stream-bound optional topology grids without changing legacy transport."""
from decimal import Decimal

from fastapi import APIRouter

from transistordatabase.gui_web.backend.finite_json_route import FiniteJsonRoute, binary64_number


def normalize_grid_numbers(value):
    """Normalize original numeric tokens before optional grid DTO conversion."""
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return binary64_number(value)
    if isinstance(value, dict):
        return {key: normalize_grid_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_grid_numbers(item) for item in value]
    return value


def validate_grid_numeric_positions(value, kind):
    """Require JSON numbers at declared numeric positions before DTO coercion."""
    def check(mapping, names):
        if isinstance(mapping, dict):
            for name in names:
                if name in mapping and mapping[name] is not None:
                    binary64_number(mapping[name])

    if kind == 'line':
        check(value, ('v_in', 'v_out', 'p_out', 'f_sw', 'l_value', 'c_out',
                      'parallel_switches', 'parallel_diodes', 'turns_ratio_nps',
                      't_j', 't_case', 'v_g', 'diode_v_g', 'v_g_off', 'r_g_on', 'r_g_off'))
        check(value.get('sweep'), ('x_min', 'x_max', 'x_points', 'y_min', 'y_max', 'y_points'))
    elif kind == 'pareto':
        check(value.get('boundary'), ('v_in', 'v_out', 'p_out', 'f_sw', 'l_value', 'c_out',
                                      't_j', 't_case', 'voltage_residual_tolerance_v'))
        candidates = value.get('candidates')
        if isinstance(candidates, list):
            for candidate in candidates:
                check(candidate, ('parallel_switches', 'parallel_diodes', 'v_g', 'diode_v_g',
                                  'v_g_off', 'r_g_on', 'r_g_off'))
        frequencies = value.get('frequencies_hz')
        if isinstance(frequencies, list):
            for frequency in frequencies:
                binary64_number(frequency)
    else:
        raise ValueError('Unknown bounded topology request kind')
    return value

router = APIRouter(route_class=FiniteJsonRoute)
