"""Contains important data classes like SOA, SwitchEnergyData, GateChargeCurve, ..."""
# Python standard libraries
from __future__ import annotations
from matplotlib import pyplot as plt
from typing import Dict, Union, List, Optional
from dataclasses import dataclass, field

from datetime import datetime
import numpy as np
import numpy.typing as npt

# Local libraries
from transistordatabase.checker_functions import check_float
from transistordatabase.helper_functions import isvalid_dict, get_img_raw_data


def _obj_to_dict(obj) -> dict:
    """Convert an object into dict datatype, converting numpy arrays to lists."""
    d = dict(vars(obj))
    for att_key in d:
        if isinstance(d[att_key], np.ndarray):
            d[att_key] = d[att_key].tolist()
    return d


@dataclass
class GateChargeCurve:
    """A class to hold gate charge characteristics of switch which is added as a optional attribute inside switch class."""

    v_supply: float  #: same as drain-to-source (v_ds)/ collector-emitter (v_ce) voltages
    t_j: float  #: junction temperature
    i_channel: float  #: channel current at which the graph is recorded
    i_g: Optional[float]  #: gate to source/emitter current
    graph_q_v: npt.NDArray[np.float64]  #: a 2D numpy array to store gate charge dependant on gate to source voltage

    def __init__(self, args):
        """
        Initialize a GateChargeCurve object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Switch class and thus does not need to be
        # checked again here.
        self.i_channel = args.get('i_channel')
        self.v_supply = args.get('v_supply')
        self.t_j = args.get('t_j')
        self.i_g = args.get('i_g')
        self.graph_q_v = args.get('graph_q_v')

    def convert_to_dict(self) -> dict:
        """
        Convert a GateChargeCurve object into dict datatype.

        :return: GateChargeCurve object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def get_plots(self, ax=None):
        """
        Plot the gate charge vs. gate source/ gate emitter voltage of switch type mosfet and igbt respectively.

        :param ax: figure axes to append the curves

        :return: Respective plots are displayed if available else None is returned
        """
        if ax:
            label_plot = "$V_{{supply}}$ = {0} V".format(self.v_supply)
            return ax.plot(self.graph_q_v[0], self.graph_q_v[1], label=label_plot)
        else:
            plt.figure()  # needs rework because of this class being a list of transistor class members
            label_plot = " $V_{{supply}}$ = {0} V".format(self.v_supply)
            plt.plot(self.graph_q_v[0], self.graph_q_v[1], label=label_plot)
            plt.legend(fontsize=8)
            plt.xlabel('Gate Charge, $Q_{G} [nC]$')
            plt.ylabel('Gate source Voltage, $V_{gs} [V]$')
            plt.grid()
            plt.show()

@dataclass
class SOA:
    """Class to hold safe operating area characteristics of transistor type."""

    t_c: Optional[float] = None  #: case temperature
    time_pulse: Optional[float] = None  #: pulse duration
    graph_i_v: np.ndarray = field(default_factory=lambda: np.array([]))  #: a 2D numpy array to store SOA characteristics curves

    def __init__(self, args: dict):
        """
        Initialize method for SOA object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Transistor class and thus does not need to be
        # checked again here.
        self.time_pulse = args.get('time_pulse')
        self.t_c = args.get('t_c')
        self.graph_i_v = args.get('graph_i_v')

    def convert_to_dict(self) -> dict:
        """
        Convert SOA object into dict datatype.

        :return: SOA object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def get_plots(self, ax=None):
        """
        Plot drain current/reverse diode current vs drain-to-source voltage/diode applied reverse voltage of switch type mosfet/igbt.

        :param ax: figure axes to append the curves

        :return: Respective plots are displayed if available else None is returned
        """
        if ax:
            label_plot = "$t_{{pulse}}$ = {0} s".format(self.time_pulse)
            return ax.loglog(self.graph_i_v[0], self.graph_i_v[1], label=label_plot)
        else:
            plt.figure()  # needs rework because of this class being a list of transistor class members
            label_plot = " $t_{{pulse}}$ = {0} V".format(self.time_pulse)
            plt.loglog(self.graph_i_v[0], self.graph_i_v[1], label=label_plot)
            plt.legend(fontsize=8)
            plt.xlabel('Drain-to-source ($V_{ds}$)/ reverse ($V_{ce}$) voltage')
            plt.ylabel('Drain $(I_d)$/ reverse $(I_c)$ current')
            plt.grid()
            plt.show()


@dataclass
class TemperatureDependResistance:
    """Store temperature dependant resistance curve."""

    i_channel: float  #: channel current at which the graph is recorded
    v_g: float  #: gate voltage
    dataset_type: str  #: curve datatype, can be either 't_r' or 't_factor'. 't_factor' is used to denote normalized gate curves
    graph_t_r: npt.NDArray[np.float64]  #: a 2D numpy array to store the temperature related channel on resistance
    r_channel_nominal: Optional[float]  #: a mandatory field if the dataset_type is 't_factor'

    def __init__(self, args):
        """
        Initialize method for TemperatureDependResistance object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Switch class and thus does not need to be
        # checked again here.
        self.i_channel = args.get('i_channel')
        self.v_g = args.get('v_g')
        self.dataset_type = args.get('dataset_type')
        self.r_channel_nominal = args.get('r_channel_nominal')
        self.graph_t_r = args.get('graph_t_r')

    def convert_to_dict(self) -> dict:
        """
        Convert a TemperatureDependResistance object into dict datatype.

        :return: TemperatureDependResistance object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def get_plots(self, ax=None):
        """
        Plot the on-resistance vs. Junction temperature.

        :param ax: figure axes to append the curves

        :return: Respective plots are displayed if available else None is returned
        """
        if ax:
            label_plot = "$V_{{G}}$ = {0} V".format(self.v_g)
            return ax.plot(self.graph_t_r[0], self.graph_t_r[1], label=label_plot)
        else:
            plt.figure()  # needs rework because of this class being a list of transistor class members
            label_plot = " $V_{{G}}$ = {0} V".format(self.v_g)
            plt.plot(self.graph_t_r[0], self.graph_t_r[1], label=label_plot)
            plt.legend(fontsize=8)
            plt.xlabel('Junction Temperature [C°]')
            y_label = 'On Resistance [Ohm]' if self.dataset_type == 't_factor' else 'On Resistance'
            plt.ylabel(y_label)
            plt.grid()
            plt.show()

@dataclass
class EffectiveOutputCapacitance:
    """Record energy related or time related output capacitance of the switch."""

    c_o: float  #: Value of the fixed output capacitance. Units in F
    v_gs: float  #: Gate to source voltage of the switch. Units in V
    v_ds: float  #: Drain to source voltage of the switch ex: V_DS = (0-400V) i.e v_ds=400 (max value, min assumed a 0). Units in V

    def __init__(self, args):
        """
        Initialize the EffectiveOutputCapacitance object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Diode/Switch class and thus does not need to be
        # checked again here.
        self.c_o = args.get('c_o')
        self.v_gs = args.get('v_gs')
        self.v_ds = args.get('v_ds')

    def convert_to_dict(self) -> dict:
        """
        Convert a EffectiveOutputCapacitance object into dict datatype.

        :return: EffectiveOutputCapacitance object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    # ToDO: To be implemented for future boundary conditions in virtual datasheet
    def collect_data(self):
        """Get the effective output capacitance from the data."""
        c_oss_related = {}
        skipIds = []
        for attr in dir(self):
            if attr not in skipIds and not callable(getattr(self, attr)) and not attr.startswith("__") and not isinstance(getattr(self, attr), (list, dict)) \
                    and (getattr(self, attr) is not None):
                c_oss_related[attr.capitalize()] = getattr(self, attr)
        return c_oss_related


class SwitchEnergyData:
    """
    Data storage for switching losses (on/off/rr).

    - Contains switching energy data for either switch or diode. The type of Energy (E_on, E_off or E_rr) is already implicitly
    specified by how the respective objects of this class are used in a Diode- or Switch-object.
    - For each set (e.g. every curve in the datasheet) of switching energy data a separate object should be created.
    - This also includes the reference values in a datasheet given without a graph. (Those are considered as data sets with just a single data point.)
    - Data sets with more than one point are given as graph_i_e with an r_g parameter or as graph_r_e with an i_x parameter.
    - Unused parameters or datasets should be left empty.
    - Which of these cases (single point, E vs I dataset, E vs R dataset) is valid for the current object also needs to be specified by the
    dataset_type-property.
    """

    # Type of the dataset:
    # single: e_x, r_g, i_x are scalars. Given e.g. by a table in the datasheet.
    # graph_r_e: r_e is a 2-dim numpy array with two rows. i_x is a scalar. Given e.g. by an E vs R graph.
    # graph_i_e: i_e is a 2-dim numpy array with two rows. r_g is a scalar. Given e.g. by an E vs I graph.
    dataset_type: str  #: Single, graph_r_e, graph_i_e (Mandatory key)
    # Additional measurement information.
    comment: str | None  #: Comment for additional information e.g. on who made these measurements
    measurement_date: datetime | None  #: Specifies the date and time at which the measurement was done.
    measurement_testbench: str | None  #: Specifies the testbench used for the measurement.
    commutation_device: str | None  #: Second device used in half-bridge test condition
    # Test conditions. These must be given as scalars. Create additional objects for e.g. different temperatures.
    t_j: float  #: Junction temperature. Units in °C (Mandatory key)
    v_supply: float  #: Supply voltage. Units in V (Mandatory key)
    v_g: float  #: Gate voltage. Units in V (Mandatory key)
    v_g_off: float | None  #: Gate voltage for turn off. Units in V
    load_inductance: float | None  #: Load inductance. Units in H
    commutation_inductance: float | None  #: Commutation inductance. Units in H
    # Scalar dataset-parameters. Some of these can be 'None' depending on the dataset_type.
    e_x: float | None  #: Scalar dataset-parameter - switching energy. Units in J
    r_g: float | None  #: Scalar dataset-parameter - gate resistance. Units in Ohm
    i_x: float | None  #: Scalar dataset-parameter - current rating. Units in A
    # Dataset. Only one of these is allowed. The other should be 'None'.
    graph_i_e: npt.NDArray[np.float64] | None  #: Units for Row 1 = A; Row 2 = J
    graph_r_e: npt.NDArray[np.float64] | None  #: Units for Row 1 = Ohm; Row 2 = J

    # ToDo: Add MOSFET capacitance. Discuss with Philipp.
    # ToDo: Add additional class for linearized switching loss model with capacities. (See infineon application
    #  note.)
    # ToDo: Option 1: Look up table like it's currently implemented.
    # ToDo: Option 2: https://application-notes.digchip.com/070/70-41484.pdf
    # ToDO: Option 3: K_i, K_v, G_i. Add as empty class with pass.

    def __init__(self, args):
        # Validity of args is checked in the constructor of Diode/Switch class and thus does not need to be
        # checked again here.
        """
        Initialize the VoltageDependentCapacitance object.

        :param args: arguments to be passed for initialization

        .. todo:: Add warning if data is ignored because of dataset_type?
        """
        # ToDo: Add warning if data is ignored because of dataset_type?
        self.dataset_type = args.get('dataset_type')
        self.v_supply = args.get('v_supply')
        self.v_g = args.get('v_g')
        self.v_g_off = args.get('v_g_off')
        self.t_j = args.get('t_j')
        self.load_inductance = args.get('load_inductance')
        self.measurement_date = args.get('measurement_date')
        self.measurement_testbench = args.get('measurement_testbench')
        self.commutation_inductance = args.get('commutation_inductance')
        self.commutation_device = args.get('commutation_device')
        self.comment = args.get('comment')
        if self.dataset_type == 'single':
            self.e_x = args.get('e_x')
            self.r_g = args.get('r_g')
            self.i_x = args.get('i_x')
            self.t_j = args.get('t_j')
            self.graph_i_e = None
            self.graph_r_e = None
            self.graph_t_e = None
        elif self.dataset_type == 'graph_i_e':
            self.e_x = None
            self.r_g = args.get('r_g')
            self.i_x = None
            self.t_j = args.get('t_j')
            self.graph_r_e = None
            self.graph_i_e = args.get('graph_i_e')            
            self.graph_t_e = None
        elif self.dataset_type == 'graph_r_e':
            self.e_x = None
            self.r_g = None
            self.i_x = args.get('i_x')
            self.t_j = args.get('t_j')
            self.graph_r_e = args.get('graph_r_e')
            self.graph_i_e = None
            self.graph_t_e = None            
        elif self.dataset_type == 'graph_t_e':
            self.e_x = None
            self.r_g = args.get('r_g')
            self.i_x = args.get('i_x')
            self.t_j = None
            self.graph_r_e = None
            self.graph_i_e = None
            self.graph_t_e = args.get('graph_t_e')

    def convert_to_dict(self) -> dict:
        """
        Convert a SwitchEnergyData object into dict datatype.

        :return: SwitchEnergyData object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def copy(self):
        """
        Copy the existing SwitchEnergyData object and create a new object of same type.

        Created to allow deep copy of object when using gecko exporter

        :return: SwitchEnergyData object
        :rtype: SwitchEnergyData
        """
        args = {
            'dataset_type': 'graph_i_e',
            'v_supply': self.v_supply,
            'graph_i_e': self.graph_i_e,
            'graph_r_e': self.graph_r_e,
            'r_g': self.r_g,
            'i_x': self.i_x,
            'e_x': self.e_x,
            't_j': self.t_j,
            'v_g': self.v_g,
        }
        # check dictionary
        isvalid_dict(args, 'SwitchEnergyData')
        return SwitchEnergyData(args)

class ChannelData:
    """
    V-I data for either switch or diode. Data is given for only one junction temperature t_j.

    For different temperatures: Create additional ChannelData-objects and store them as a list in the respective
    Diode- or Switch-object.
    This data can be used to linearize the transistor at a specific operating point
    """

    # # Test condition: Must be given as scalar. Create additional objects for different temperatures.
    t_j: float  #: Junction temperature of switch\diode. (Mandatory key)
    v_g: float  #: Switch: Mandatory key, Diode: optional (standard diode useless, for GaN 'diode' necessary
    # Dataset: Represented as a 2xm Matrix where row 1 is the voltage and row 2 the current.
    graph_v_i: npt.NDArray[np.float64]  #: Represented as a numpy 2D array where row 1 is the voltage and row 2 the current.
    # Units of Row 1 = V; Row 2 = A (Mandatory key)

    def __init__(self, args):
        """
        Initialize a ChannelData object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Diode/Switch class and thus does not need to be
        # checked again here.
        self.t_j = args.get('t_j')
        self.graph_v_i = args.get('graph_v_i')
        self.v_g = args.get('v_g')

    def convert_to_dict(self) -> dict:
        """
        Convert a ChannelData object into dict datatype.

        :return: ChannelData object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

class LinearizedModel:
    """
    Data for a linearized Switch/Diode depending on given operating point.

    Operating point specified by t_j, i_channel and (not for all diode types) v_g.
    """

    t_j: float  #: Junction temperature of diode\switch. Units in K  (Mandatory key)
    v_g: float | None  #: Gate voltage of switch or diode. Units in V (Mandatory for Switch, Optional for some Diode types)
    i_channel: float  #: Channel current of diode\switch. Units in A (Mandatory key)
    r_channel: float  #: Channel resistance of diode\switch. Units in Ohm (Mandatory key)
    v0_channel: float  #: Channel voltage of diode\switch. Unis in V (Mandatory key)

    def __init__(self, args):
        """
        Initialize a linearizedmodel object.

        :param args: arguments to passed for initialization
        """
        self.t_j = args.get('t_j')
        self.v_g = args.get('v_g')
        self.i_channel = args.get('i_channel')
        self.r_channel = args.get('r_channel')
        self.v0_channel = args.get('v0_channel')

    def convert_to_dict(self) -> dict:
        """
        Convert LinearizedModel object into dict datatype.

        :return: LinearizedModel object of dict type
        :rtype: dict
        """
        d = dict(vars(self))
        return d

class VoltageDependentCapacitance:
    """
    Graph_v_c data for transistor class. Data is given for only one junction temperature t_j.

    For different temperatures: Create additional VoltageDependentCapacitance-objects and store them as a list in the transistor-object.
    """

    # # Test condition: Must be given as scalar. Create additional objects for different temperatures.
    t_j: float  #: Junction temperature (Mandatory key)
    # Dataset: Represented as a 2xm Matrix where row 1 is the voltage and row 2 the capacitance.
    graph_v_c: npt.NDArray[np.float64]  #: Represented as a 2D numpy array where row 1 is the voltage and row 2 the capacitance.
    # Units of Row 1 = V; Row 2 = A  (Mandatory key)

    def __init__(self, args):
        """
        Initialize the VoltageDependentCapacitance object.

        :param args: arguments to be passed for initialization
        """
        # Validity of args is checked in the constructor of Diode/Switch class and thus does not need to be
        # checked again here.
        self.t_j = args.get('t_j')
        self.graph_v_c = args.get('graph_v_c')

    def convert_to_dict(self) -> dict:
        """
        Convert a VoltageDependentCapacitance object into dict datatype.

        :return: VoltageDependentCapacitance object of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def get_plots(self, ax=None, label=None):
        """
        Plot the voltage dependant capacitance graph_v_c of the VoltageDependentCapacitance object.

        Also attaches the plot to figure axes for the purpose virtual datasheet if ax argument is specified

        :param ax: figure axes for making the graph_v_c plot in virtual datasheet
        :param label: label of the plot for virtual datasheet plot

        :return: Respective plots are displayed
        """
        if ax:
            label_plot = label + ", $T_{{J}}$ = {0} °C".format(self.t_j)
            return ax.semilogy(self.graph_v_c[0], self.graph_v_c[1], label=label_plot)
        else:
            plt.figure()  # needs rework because of this class being a list of transistor class members
            label_plot = "$T_{{J}}$ = {0}".format(self.t_j)
            plt.semilogy(self.graph_v_c[0], self.graph_v_c[1], label=label_plot)
            plt.legend(fontsize=8)
            plt.xlabel('Voltage in V')
            plt.ylabel('Capacitance in F')
            plt.grid()
            plt.show()

class FosterThermalModel:
    """
    Data to specify parameters of the Foster thermal_foster model.

    This model describes the transient
    temperature behavior as a thermal_foster RC-network. The necessary parameters can be estimated by curve-fitting
    transient temperature data supplied in graph_t_rthjc or by manually specifying the individual 2 out of 3 of the
    parameters R, C, and tau.

    .. todo::
        - Add function to estimate parameters from transient data.
        - Add function to automatically calculate missing parameters from given ones.
        - Do these need to be numpy array or should they be lists instead?
    """

    # Thermal resistances of RC-network (array).
    r_th_vector: list[float] | None  #: Thermal resistances of RC-network (array). Units in K/W (Optional key)
    # Sum of thermal_foster resistances of n-pole RC-network (scalar).
    r_th_total: float | None  #: Sum of thermal_foster resistances of n-pole RC-network (scalar). Units in K/W  (Optional key)
    # Thermal capacities of n-pole RC-network (array).
    c_th_vector: list[float] | None  #: Thermal capacities of n-pole RC-network (array). Units in J/K (Optional key)
    # Sum of thermal_foster capacities of n-pole low pass as (scalar).
    c_th_total: float | None  #: Sum of thermal_foster capacities of n-pole low pass as (scalar). Units in J/K  (Optional key)
    # Thermal time constants of n-pole RC-network (array).
    tau_vector: list[float] | None  #: Thermal time constants of n-pole RC-network (array). Units in s  (Optional key)
    # Sum of thermal_foster time constants of n-pole RC-network (scalar).
    tau_total: float | None  #: Sum of thermal_foster time constants of n-pole RC-network (scalar). Units in s (Optional key)
    # Transient data for extraction of the thermal_foster parameters specified above.
    # Represented as a 2xm Matrix where row 1 is the time and row 2 the temperature.
    graph_t_rthjc: npt.NDArray[np.float64] | None  #: Transient data for extraction of the thermal_foster parameters specified above.
    # Units of Row 1 in s; Row 2 in K/W  (Optional key)

    def __init__(self, args):
        """
        Initialize a FosterThermalModel object.

        :param args: argument to be passed for initialization
        :type args: dict

        .. note:: Can be constructed from empty or 'None' argument dictionary since no attributes are mandatory.
        """
        if isvalid_dict(args, 'FosterThermalModel'):
            self.r_th_total = args.get('r_th_total')
            self.r_th_vector = args.get('r_th_vector')
            self.c_th_total = args.get('c_th_total')
            self.c_th_vector = args.get('c_th_vector')
            self.tau_total = args.get('tau_total')
            self.tau_vector = args.get('tau_vector')
            self.graph_t_rthjc = args.get('graph_t_rthjc')
        else:  # Can be constructed from empty or 'None' argument dictionary since no attributes are mandatory.
            self.r_th_total = None
            self.r_th_vector = None
            self.c_th_total = None
            self.c_th_vector = None
            self.tau_total = None
            self.tau_vector = None
            self.graph_t_rthjc = None

    def convert_to_dict(self) -> dict:
        """
        Convert a FosterThermalModel object into dict datatype.

        :return: FosterThermalModel of dict type
        :rtype: dict
        """
        return _obj_to_dict(self)

    def get_plots(self, buffer_req: bool = False):
        """
        Plot tau vs rthjc.

        :param buffer_req: Internally required for generating virtual datasheets
        :type buffer_req: bool

        :return: Respective plots are displayed if available else None is returned
        """
        if self.graph_t_rthjc is None:
            print('No Foster impedance information exists!')
            return None
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.loglog(self.graph_t_rthjc[0], self.graph_t_rthjc[1])
        ax.set_xlabel('Time : $t$ [sec]')
        ax.set_ylabel('Thermal impedance: $Z_{th(j-c)}$ [K/W ]')
        ax.grid()
        # self.r_th_vector and self.tau_vector are optional.
        if self.r_th_vector is not None and self.tau_vector is not None:
            r_tau_vector = '\n'.join([
                '$R_{th}$ :' + " ".join(str("{:4.3f}".format(x)) for x in self.r_th_vector),
                'tau :' + " ".join(str("{:4.3f}".format(x)) for x in self.tau_vector)
            ])
            props = dict(fill=False, edgecolor='black', linewidth=2)
            ax.text(0.9, 0.2, r_tau_vector, fontsize='small', transform=ax.transAxes, bbox=props, ha='right')
        if buffer_req:
            return get_img_raw_data(plt)
        else:
            plt.show()

    def collect_data(self) -> dict:
        """
        Collect foster data in form of dictionary for generating virtual datasheet.

        :return: foster data in form of dictionary
        :rtype: dict
        """
        foster_data = {}
        foster_data['foster_plot'] = {'imp_plot': self.get_plots(True)}
        skipIds = ['graph_t_rthjc']
        for attr in dir(self):
            if attr not in skipIds and not callable(getattr(self, attr)) and not attr.startswith("__") and not isinstance(getattr(self, attr), (list, dict)) \
                    and (getattr(self, attr) is not None):
                foster_data[attr.capitalize()] = getattr(self, attr)
        return foster_data

class RawMeasurementData:
    """RAW measurement data. e.g. for voltage and current graphs from a double pulse test."""

    # Type of the dataset:
    # dpt_u_i: U/t I/t graph from double pulse measurements
    dataset_type: str  #: e.g. dpt_u_i (Mandatory key)
    dpt_on_vds: List[npt.NDArray[np.float64]] | None  #: measured Vds data at turn on event. Units in V and s
    dpt_on_id: List[npt.NDArray[np.float64]] | None  #: measured Id data at turn on event. Units in A and s
    dpt_off_vds: List[npt.NDArray[np.float64]] | None  #: measured Vds data at turn off event. Units in V and s
    dpt_off_id: List[npt.NDArray[np.float64]] | None  #: measured Vds data at turn off event. Units in A and s
    measurement_date: datetime | None  #: Specifies the measurements date and time
    measurement_testbench: str | None  #: Specifies the testbench used for the measurement.
    commutation_device: str | None  #: Second device used in half-bridge test condition
    comment: str | None  #: Comment for additional information e.g. on who made these measurements
    # Test conditions. These must be given as scalars. Create additional objects for e.g. different temperatures.
    t_j: float | None  #: Junction temperature. Units in °C
    v_supply: float | None  #: Supply voltage. Units in V
    v_g: float | None  #: Gate voltage. Units in V
    v_g_off: float | None  #: Gate voltage for turn off. Units in V
    r_g: List[npt.NDArray[np.float64]] | None  #: gate resistance. Units in Ohm
    r_g_off: List[npt.NDArray[np.float64]] | None  #: gate resistance. Units in Ohm
    load_inductance: float | None  #: Load inductance. Units in µH
    commutation_inductance: float | None  #: Commutation inductance. Units in µH

    e_off_meas = Union[Dict, None]  # Union[] is used here because | somehow didn't work
    e_on_meas = Union[Dict, None]

    def __init__(self, args):
        """
        Initialize a RawMeasurementData object.

        :param args: arguments to be passed for initialization
        """
        self.dataset_type = args.get('dataset_type')
        self.comment = args.get('dataset_type')
        if self.dataset_type == 'dpt_u_i' or self.dataset_type == 'dpt_u_i_r':
            self.dpt_on_vds = args.get('dpt_on_vds')
            self.dpt_on_id = args.get('dpt_on_id')
            self.dpt_off_vds = args.get('dpt_off_vds')
            self.dpt_off_id = args.get('dpt_off_id')
            self.v_supply = args.get('v_supply')
            self.v_g = args.get('v_g')
            self.v_g_off = args.get('v_g_off')
            self.t_j = args.get('t_j')
            self.load_inductance = args.get('load_inductance')
            self.commutation_inductance = args.get('commutation_inductance')
            self.commutation_device = args.get('commutation_device')
            self.r_g = args.get('r_g')
            self.r_g_off = args.get('r_g_off')
            self.measurement_date = args.get('measurement_date')
            self.measurement_testbench = args.get('measurement_testbench')
        else:
            self.dpt_on_vds = []
            self.dpt_on_id = []
            self.dpt_off_vds = []
            self.dpt_off_id = []

    def convert_to_dict(self) -> dict:
        """
        Convert RawMeasurementData object into dict datatype.

        :return: Switch object of dict type
        :rtype: dict
        """
        d = dict(vars(self))
        d['dpt_on_vds'] = [c.tolist() for c in self.dpt_on_vds]
        d['dpt_on_id'] = [c.tolist() for c in self.dpt_on_id]
        d['dpt_off_vds'] = [c.tolist() for c in self.dpt_off_vds]
        d['dpt_off_id'] = [c.tolist() for c in self.dpt_off_id]
        return d
