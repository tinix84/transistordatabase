"""Topology calculator mixin for MainWindow."""
from math import floor
import os

import numpy as np

from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from matplotlib.widgets import Cursor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5 import QtCore, QtGui

from transistordatabase.gui import buck_converter_functions
from transistordatabase.gui import boost_converter_functions
from transistordatabase.gui import buck_boost_converter_functions
from transistordatabase.gui._widgets import PopOutPlotWindow, MatplotlibWidget


class TopologyCalculatorMixin:
    """Mixin providing topology calculator functionality for MainWindow."""

    def clear_topology_calculator(self):
        """
        Clear numeric user inputs in Topology Calculator.

        :return:
        """
        self.lineEdit_topology_number_parallel_transistor1.clear()
        self.lineEdit_topology_number_parallel_transistor2.clear()
        self.lineEdit_topology_output_power.clear()
        self.lineEdit_topology_v_in.clear()
        self.lineEdit_topology_v_out.clear()
        self.lineEdit_topology_frequency.clear()
        self.lineEdit_topology_zeta.clear()
        self.lineEdit_topology_temperature_heatsink.clear()
        self.lineEdit_topology_thermal_resistance_heatsink.clear()
        self.lineEdit_topology_output_power_min.clear()
        self.lineEdit_topology_v_in_min.clear()
        self.lineEdit_topology_v_out_min.clear()
        self.lineEdit_topology_frequency_min.clear()
        self.lineEdit_topology_zeta_min.clear()
        self.lineEdit_topology_output_power_max.clear()
        self.lineEdit_topology_v_in_max.clear()
        self.lineEdit_topology_v_out_max.clear()
        self.lineEdit_topology_frequency_max.clear()
        self.lineEdit_topology_zeta_max.clear()
        self.label_topology_slider_r_g_on_value_transistor1.setText(str(0.0))
        self.label_topology_slider_r_g_off_value_transistor1.setText(str(0.0))
        self.slider_topology_r_g_on_transistor1.setValue(0)
        self.slider_topology_r_g_off_transistor1.setValue(0)

    def comboBox_topology_topology_changed(self):
        """
        Set the picture which shows the circuit diagram of selected topology and resizes it so that will be shown correctly.

        :return: None
        """
        if self.comboBox_topology_topology.currentText() == "Buck-Converter":
            self.label_picture_topology.setMaximumSize(QtCore.QSize(450, 115))
            self.label_picture_topology.setPixmap(QtGui.QPixmap(os.path.join(os.path.dirname(__file__), "..", "..", "buck_converter_schematic.png")))
        elif self.comboBox_topology_topology.currentText() == "Boost-Converter":
            self.label_picture_topology.setMaximumSize(QtCore.QSize(450, 110))
            self.label_picture_topology.setPixmap(QtGui.QPixmap(os.path.join(os.path.dirname(__file__), "..", "..", "boost_converter_schematic.png")))
        elif self.comboBox_topology_topology.currentText() == "Buck-Boost-Converter":
            self.label_picture_topology.setMaximumSize(QtCore.QSize(450, 115))
            self.label_picture_topology.setPixmap(QtGui.QPixmap(os.path.join(os.path.dirname(__file__), "..", "..", "buck_boost_converter_schematic.png")))

    def comboBox_topology_plot_line_contour_changed(self, comboBox_topology_plot_y_axis, comboBox_topology_plot_z_axis,
                                                    comboBox_topology_plot_line_contour):
        """
        Fill the ComboBoxes for y-axis and z-axis with variables depending on whether line or contour is selected.

        :param comboBox_topology_plot_y_axis: ComboBox to choose variable for x-axis
        :param comboBox_topology_plot_z_axis: ComboBox to choose variable for y-axis
        :param comboBox_topology_plot_line_contour: ComboBox to choose between line-plot and contour-plot
        :return: None
        """
        current_y_axis = comboBox_topology_plot_y_axis.currentText()
        current_z_axis = comboBox_topology_plot_z_axis.currentText()
        self.comboBox_topology_plots_x_axis_changed()
        if comboBox_topology_plot_line_contour.currentText() == "Line":
            comboBox_topology_plot_y_axis.setCurrentText(current_z_axis)
        elif comboBox_topology_plot_line_contour.currentText() == "Contour":
            comboBox_topology_plot_z_axis.setCurrentText(current_y_axis)

    def comboBox_topology_plots_line_contour_changed(self):
        """
        Run the function "comboBox_topology_plot_line_contour_changed" with the the ComboBoxes of all plots.

        :return: None
        """
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot1_y_axis,
                                                         self.comboBox_topology_plot1_z_axis,
                                                         self.comboBox_topology_plot1_line_contour)
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot2_y_axis,
                                                         self.comboBox_topology_plot2_z_axis,
                                                         self.comboBox_topology_plot2_line_contour)
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot3_y_axis,
                                                         self.comboBox_topology_plot3_z_axis,
                                                         self.comboBox_topology_plot3_line_contour)
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot4_y_axis,
                                                         self.comboBox_topology_plot4_z_axis,
                                                         self.comboBox_topology_plot4_line_contour)
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot5_y_axis,
                                                         self.comboBox_topology_plot5_z_axis,
                                                         self.comboBox_topology_plot5_line_contour)
        self.comboBox_topology_plot_line_contour_changed(self.comboBox_topology_plot6_y_axis,
                                                         self.comboBox_topology_plot6_z_axis,
                                                         self.comboBox_topology_plot6_line_contour)

    def comboBox_topology_plot_x_axis_changed(self, comboBox_topology_plot_x_axis, comboBox_topology_plot_y_axis,
                                              comboBox_topology_plot_z_axis, comboBox_topology_plot_line_contour):
        """
        Fill the ComboBoxes to choose the variables for y-axis and z-axis.

        This is based on the selection of the variable for the x-axis and whether line or contour is selected.

        :param comboBox_topology_plot_x_axis: ComboBox to choose variable for x-axis
        :param comboBox_topology_plot_y_axis: ComboBox to choose variable for y-axis
        :param comboBox_topology_plot_z_axis: ComboBox to choose variable for z-axis
        :param comboBox_topology_plot_line_contour: ComboBox to choose between line-plot and contour-plot
        :return: None
        """
        items_comboBox_y_z_axis = ["RMS Current Transistor1 [A]",
                                   "RMS Current Diode Transistor2 [A]",
                                   "Mean Current Transistor1 [A]",
                                   "Mean Current Diode Transistor2 [A]",
                                   "RMS Inductor Current [A]",
                                   "Mean Inductor Current [A]",
                                   "Peak Current [A]",
                                   "Conduction Losses Transistor1 [W]",
                                   "Conduction Losses Diode Transistor2 [W]",
                                   "Total Conduction Losses [W]",
                                   "Turn-on Switching Losses Transistor1 [W]",
                                   "Turn-off Switching Losses Transistor1 [W]",
                                   "Reverse Recovery Losses Diode Transistor2 [W]",
                                   "Total Switching Losses Transistor1 [W]",
                                   "Total Power Losses Transistor1 [W]",
                                   "Total Switching Losses [W]",
                                   "Temperature Switch Transistor1 [°C]",
                                   "Temperature Diode Transistor2 [°C]"]

        current_y_axis = comboBox_topology_plot_y_axis.currentText()
        comboBox_topology_plot_y_axis.clear()
        current_z_axis = comboBox_topology_plot_z_axis.currentText()
        comboBox_topology_plot_z_axis.clear()
        if comboBox_topology_plot_line_contour.currentText() == "Line":
            comboBox_topology_plot_y_axis.addItems(items_comboBox_y_z_axis)
            comboBox_topology_plot_y_axis.setCurrentText(current_y_axis)
        elif comboBox_topology_plot_line_contour.currentText() == "Contour":
            items_comboBox_y_axis = ["Vin [V]", "Vout [V]", "Output Power [W]", "Frequency [kHz]", "Zeta = f*L"]
            items_comboBox_y_axis.remove(comboBox_topology_plot_x_axis.currentText())
            comboBox_topology_plot_y_axis.addItems(items_comboBox_y_axis)
            comboBox_topology_plot_y_axis.setCurrentText(current_y_axis)
            comboBox_topology_plot_z_axis.addItems(items_comboBox_y_z_axis)
            comboBox_topology_plot_z_axis.setCurrentText(current_z_axis)

    def comboBox_topology_plots_x_axis_changed(self):
        """
        Run the function "comboBox_topology_plot_x_axis_changed" with the ComboBoxes of all plots.

        :return: None
        """
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot1_x_axis,
                                                   self.comboBox_topology_plot1_y_axis,
                                                   self.comboBox_topology_plot1_z_axis,
                                                   self.comboBox_topology_plot1_line_contour)
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot2_x_axis,
                                                   self.comboBox_topology_plot2_y_axis,
                                                   self.comboBox_topology_plot2_z_axis,
                                                   self.comboBox_topology_plot2_line_contour)
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot3_x_axis,
                                                   self.comboBox_topology_plot3_y_axis,
                                                   self.comboBox_topology_plot3_z_axis,
                                                   self.comboBox_topology_plot3_line_contour)
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot4_x_axis,
                                                   self.comboBox_topology_plot4_y_axis,
                                                   self.comboBox_topology_plot4_z_axis,
                                                   self.comboBox_topology_plot4_line_contour)
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot5_x_axis,
                                                   self.comboBox_topology_plot5_y_axis,
                                                   self.comboBox_topology_plot5_z_axis,
                                                   self.comboBox_topology_plot5_line_contour)
        self.comboBox_topology_plot_x_axis_changed(self.comboBox_topology_plot6_x_axis,
                                                   self.comboBox_topology_plot6_y_axis,
                                                   self.comboBox_topology_plot6_z_axis,
                                                   self.comboBox_topology_plot6_line_contour)

    def comboBox_topology_transistor_changed(self, comboBox_topology_transistor, comboBox_topology_v_g_on_transistor,
                                             slider_topology_r_g_on_transistor,
                                             label_topology_slider_r_g_on_value_transistor,
                                             slider_topology_r_g_off_transistor,
                                             label_topology_slider_r_g_off_value_transistor):
        """
        Fill the ComboBox to choose the gate voltage for the transistor and sets minimum and maximum value for the sliders.

        For choosing the gate resistors for transistor based on the available data stored in the transistordatabase.

        :return: None
        """
        transistor = self.tdb.load_transistor(comboBox_topology_transistor.currentText())
        comboBox_topology_v_g_on_transistor.clear()
        available_v_g_on_transistor = [str(channel.v_g) for channel in transistor.switch.channel]
        available_v_g_on_transistor_cleared = [available_v_g_on_transistor[i] for i in range(len(available_v_g_on_transistor))
                                               if i == available_v_g_on_transistor.index(available_v_g_on_transistor[i])]
        comboBox_topology_v_g_on_transistor.addItems(available_v_g_on_transistor_cleared)
        comboBox_topology_v_g_on_transistor.setCurrentText(
            str(max(channel.v_g for channel in transistor.switch.channel)))

        try:
            r_e_object_on = transistor.get_object_r_e_simplified(
                e_on_off_rr="e_on",
                t_j=max([i for i in [e_on.t_j for e_on in transistor.switch.e_on] if i is not None]),
                v_g=max([i for i in [e_on.v_g for e_on in transistor.switch.e_on] if i is not None]),
                v_supply=max([i for i in [e_on.v_supply for e_on in transistor.switch.e_on] if i is not None]),
                normalize_t_to_v=10)
            r_g_on_max = floor(10 * (np.amax(r_e_object_on.graph_r_e[0]))) / 10
            slider_topology_r_g_on_transistor.setMinimum(0)
            slider_topology_r_g_on_transistor.setMaximum(int(r_g_on_max * 100))
            slider_topology_r_g_on_transistor.setValue(int(r_g_on_max * 100))
            label_topology_slider_r_g_on_value_transistor.setText(str(round(r_g_on_max, 1)))

        except:
            try:
                r_g_on_max = max([i for i in [e_on.r_g for e_on in transistor.switch.e_on] if i is not None])
                slider_topology_r_g_on_transistor.setMinimum(int(r_g_on_max * 100))
                slider_topology_r_g_on_transistor.setMaximum(int(r_g_on_max * 100))
                if not self.start:
                    self.show_popup_message(
                        f"No energy data for different turn on gate resistor for <b>{transistor.name}</b> available!")
            except:
                if not self.start:
                    self.show_popup_message(f"No turn-on energy data for <b>{transistor.name}</b> available!")
                slider_topology_r_g_on_transistor.setMinimum(0)
                slider_topology_r_g_on_transistor.setMaximum(0)

        try:
            r_e_object_off = transistor.get_object_r_e_simplified(
                e_on_off_rr="e_off",
                t_j=max([i for i in [e_off.t_j for e_off in transistor.switch.e_off] if i is not None]),
                v_g=min([i for i in [e_off.v_g for e_off in transistor.switch.e_off] if i is not None]),
                v_supply=max([i for i in [e_off.v_supply for e_off in transistor.switch.e_off] if i is not None]),
                normalize_t_to_v=10)
            r_g_off_max = floor(10 * np.amax(r_e_object_off.graph_r_e[0])) / 10
            slider_topology_r_g_off_transistor.setMinimum(0)

            if transistor.type == "IGBT":
                r_e_object_rr = transistor.get_object_r_e_simplified(
                    e_on_off_rr="e_rr",
                    t_j=max([i for i in [e_rr.t_j for e_rr in transistor.diode.e_rr] if i is not None]),
                    v_g=min([i for i in [e_rr.v_g for e_rr in transistor.diode.e_rr] if i is not None]),
                    v_supply=max([i for i in [e_rr.v_supply for e_rr in transistor.diode.e_rr] if i is not None]),
                    normalize_t_to_v=10)
                r_g_rr_max = floor(10 * (np.amax(r_e_object_rr.graph_r_e[0]))) / 10
                r_g_max_off_rr1 = min(r_g_off_max, r_g_rr_max)
                slider_topology_r_g_off_transistor.setMaximum(int(r_g_max_off_rr1 * 100))
                slider_topology_r_g_off_transistor.setValue(int(r_g_max_off_rr1 * 100))
                label_topology_slider_r_g_off_value_transistor.setText(str(round(r_g_max_off_rr1, 1)))
            else:
                slider_topology_r_g_off_transistor.setMaximum(int(r_g_off_max * 100))
                slider_topology_r_g_off_transistor.setValue(int(r_g_off_max * 100))
                label_topology_slider_r_g_off_value_transistor.setText(str(round(r_g_off_max, 1)))
        except:
            try:
                r_g_off_max = max([i for i in [e_off.r_g for e_off in transistor.switch.e_off] if i is not None])
                slider_topology_r_g_off_transistor.setMinimum(int(r_g_off_max * 100))
                slider_topology_r_g_off_transistor.setMaximum(int(r_g_off_max * 100))
                if not self.start:
                    self.show_popup_message(
                        f"No energy data for different turn off gate resistor for <b>{transistor.name}</b> available!")
            except:
                slider_topology_r_g_off_transistor.setMinimum(0)
                slider_topology_r_g_off_transistor.setMaximum(0)
                if not self.start:
                    self.show_popup_message(f"No turn-off energy data for <b>{transistor.name}</b> available!")
        self.start = None

    def comboBox_topology_transistor1_changed(self):
        """
        Run the function to fill the comboBoxes and configurate the sliders for the topology calculator tab for transistor1.

        :return: None
        """
        print("we are before the comboBox_topology_transistor_changed ")
        self.comboBox_topology_transistor_changed(self.comboBox_topology_transistor1,
                                                  self.comboBox_topology_v_g_on_transistor1,
                                                  self.slider_topology_r_g_on_transistor1,
                                                  self.label_topology_slider_r_g_on_value_transistor1,
                                                  self.slider_topology_r_g_off_transistor1,
                                                  self.label_topology_slider_r_g_off_value_transistor1)

    def slider_topology_r_g_value_changed(self):
        """
        Set the labels below the sliders to choose gate resistors to show currently selected values.

        :return:
        """
        self.label_topology_slider_r_g_on_value_transistor1.setText(
            str(round(self.slider_topology_r_g_on_transistor1.value() / 100, 1)))
        self.label_topology_slider_r_g_off_value_transistor1.setText(
            str(round(self.slider_topology_r_g_off_transistor1.value() / 100, 1)))

    def new_annotation(self, axis):
        """
        Create an annotation and adds it to a matplotlibwidget.

        :param axis: matplotlibwidget axis to put the annotation on
        :return: annotation object
        """
        annotation = axis.annotate("", xy=(0, 0), xytext=(-110, 30), textcoords="offset pixels",
                                   bbox=dict(boxstyle="square", fc="linen", ec="k", lw="1"),
                                   arrowprops=dict(arrowstyle="-|>"))
        return annotation

    def click_event(self, button, xdata, ydata, matplotlibwidget, annotations_list):
        """
        Create an annotation for an embedded matplotlibwidget graph on left click and removes last created annotation on right click.

        :param button: clicked button
        :param xdata: x-data of matplotlib graph
        :param ydata: y-data of matplotlib graph
        :param matplotlibwidget: matplotlibwidget object
        :param annotations_list: list to store added annotations
        :return: None
        """
        if str(button) == "MouseButton.LEFT":
            click_annotation = self.new_annotation(matplotlibwidget.axis)
            annotations_list.append(click_annotation)
            click_annotation.xy = (xdata, ydata)
            text = f"({round(xdata, 2)}, {round(ydata, 2)})"
            click_annotation.set_text(text)
            click_annotation.set_visible(True)
        elif str(button) == "MouseButton.RIGHT":
            annotations_list[-1].remove()
            annotations_list.pop()
        matplotlibwidget.figure.canvas.draw_idle()

    def topology_create_plot(self, widget_topology_plot, matplotlibwidget, comboBox_topology_plot_x_axis,
                             comboBox_topology_plot_y_axis, comboBox_topology_plot_z_axis,
                             comboBox_topology_line_contour, converter):
        """
        Add a Matplotlib figure to a QWidget and create a plot based on all the possible inputs and selections.

        Uses the calculation-functions from buck_converter_functions.py, boost_converter_functions.py and buck_boost_converter_functions.py.

        :param widget_topology_plot: widget for the Matplotlib figure
        :param matplotlibwidget: Matplotlib figure
        :param comboBox_topology_plot_x_axis: ComboBox for selection of the variable for the x-axis
        :param comboBox_topology_plot_y_axis: ComboBox for selection of the variable for the y-axis
        :param comboBox_topology_plot_z_axis: ComboBox for selection of the variable for the z-axis
        :param comboBox_topology_line_contour: ComboBox for selection between line-plot and contour-plot
        :param converter: "buck_converter", "boost_converter" or "buck_boost_converter"
        :return: None
        """
        annotations_list = []

        def clicked(event):
            if event.dblclick:
                self.click_event(event.button, event.xdata, event.ydata, matplotlibwidget, annotations_list)

        self.layout = QVBoxLayout(widget_topology_plot)
        self.layout.addWidget(matplotlibwidget)
        matplotlibwidget.axis.clear()
        try:
            matplotlibwidget.axis_cm.remove()
        except:
            pass

        transistor1 = self.tdb.load_transistor(self.comboBox_topology_transistor1.currentText())
        transistor2 = self.tdb.load_transistor(self.comboBox_topology_transistor2.currentText())
        v_g_on1 = float(self.comboBox_topology_v_g_on_transistor1.currentText())
        r_g_on1 = float(self.label_topology_slider_r_g_on_value_transistor1.text())
        r_g_off1 = float(self.label_topology_slider_r_g_off_value_transistor1.text())

        try:
            t_heatsink = float(self.lineEdit_topology_temperature_heatsink.text())
            r_th_heatsink = float(self.lineEdit_topology_thermal_resistance_heatsink.text())
        except:
            pass

        try:
            transistor1 = self.tdb.parallel_transistors(transistor1, int(self.lineEdit_topology_number_parallel_transistor1.text()))
            transistor2 = self.tdb.parallel_transistors(transistor2, int(self.lineEdit_topology_number_parallel_transistor2.text()))

            if comboBox_topology_line_contour.currentText() == "Contour":
                if comboBox_topology_plot_x_axis.currentText() == "Zeta = f*L":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_zeta_min.text()),
                                             float(self.lineEdit_topology_zeta_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Vin [V]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_v_in_min.text()),
                                             float(self.lineEdit_topology_v_in_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Vout [V]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_v_out_min.text()),
                                             float(self.lineEdit_topology_v_out_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Output Power [W]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_output_power_min.text()),
                                             float(self.lineEdit_topology_output_power_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Frequency [kHz]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_frequency_min.text()),
                                             float(self.lineEdit_topology_frequency_max.text()), 100)

                if comboBox_topology_plot_y_axis.currentText() == "Zeta = f*L":
                    vec_y_axis = np.linspace(float(self.lineEdit_topology_zeta_min.text()),
                                             float(self.lineEdit_topology_zeta_max.text()), 100)
                if comboBox_topology_plot_y_axis.currentText() == "Vin [V]":
                    vec_y_axis = np.linspace(float(self.lineEdit_topology_v_in_min.text()),
                                             float(self.lineEdit_topology_v_in_max.text()), 100)
                if comboBox_topology_plot_y_axis.currentText() == "Vout [V]":
                    vec_y_axis = np.linspace(float(self.lineEdit_topology_v_out_min.text()),
                                             float(self.lineEdit_topology_v_out_max.text()), 100)
                if comboBox_topology_plot_y_axis.currentText() == "Output Power [W]":
                    vec_y_axis = np.linspace(float(self.lineEdit_topology_output_power_min.text()),
                                             float(self.lineEdit_topology_output_power_max.text()), 100)
                if comboBox_topology_plot_y_axis.currentText() == "Frequency [kHz]":
                    vec_y_axis = np.linspace(float(self.lineEdit_topology_frequency_min.text()),
                                             float(self.lineEdit_topology_frequency_max.text()), 100)

                m_x, m_y = np.meshgrid(vec_x_axis, vec_y_axis)

                if self.lineEdit_topology_zeta.text() != "":
                    m_zeta = np.full_like(m_x, float(self.lineEdit_topology_zeta.text()))
                if self.lineEdit_topology_v_in.text() != "":
                    m_v_in = np.full_like(m_x, float(self.lineEdit_topology_v_in.text()))
                if self.lineEdit_topology_v_out.text() != "":
                    m_v_out = np.full_like(m_x, float(self.lineEdit_topology_v_out.text()))
                if self.lineEdit_topology_output_power.text() != "":
                    m_output_power = np.full_like(m_x, float(self.lineEdit_topology_output_power.text()))
                if self.lineEdit_topology_frequency.text() != "":
                    m_frequency = np.full_like(m_x, float(self.lineEdit_topology_frequency.text()))

                if comboBox_topology_plot_x_axis.currentText() == "Zeta = f*L":
                    m_zeta = m_x
                if comboBox_topology_plot_x_axis.currentText() == "Vin [V]":
                    m_v_in = m_x
                if comboBox_topology_plot_x_axis.currentText() == "Vout [V]":
                    m_v_out = m_x
                if comboBox_topology_plot_x_axis.currentText() == "Output Power [W]":
                    m_output_power = m_x
                if comboBox_topology_plot_x_axis.currentText() == "Frequency [kHz]":
                    m_frequency = m_x

                if comboBox_topology_plot_y_axis.currentText() == "Zeta = f*L":
                    m_zeta = m_y
                if comboBox_topology_plot_y_axis.currentText() == "Vin [V]":
                    m_v_in = m_y
                if comboBox_topology_plot_y_axis.currentText() == "Vout [V]":
                    m_v_out = m_y
                if comboBox_topology_plot_y_axis.currentText() == "Output Power [W]":
                    m_output_power = m_y
                if comboBox_topology_plot_y_axis.currentText() == "Frequency [kHz]":
                    m_frequency = m_y

                if comboBox_topology_plot_z_axis.currentText() == "RMS Current Transistor1 [A]":
                    m_z = converter.f_m_i1_rms(zeta=m_zeta,
                                               v_in=m_v_in,
                                               v_out=m_v_out,
                                               p_out=m_output_power,
                                               v_g_on1=v_g_on1,
                                               transistor1=transistor1,
                                               transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "RMS Current Diode Transistor2 [A]":
                    m_z = converter.f_m_i2_rms(zeta=m_zeta,
                                               v_in=m_v_in,
                                               v_out=m_v_out,
                                               p_out=m_output_power,
                                               v_g_on1=v_g_on1,
                                               transistor1=transistor1,
                                               transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Mean Current Transistor1 [A]":
                    m_z = converter.f_m_i1_mean(zeta=m_zeta,
                                                v_in=m_v_in,
                                                v_out=m_v_out,
                                                p_out=m_output_power,
                                                v_g_on1=v_g_on1,
                                                transistor1=transistor1,
                                                transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Mean Current Diode Transistor2 [A]":
                    m_z = converter.f_m_i2_mean(zeta=m_zeta,
                                                v_in=m_v_in,
                                                v_out=m_v_out,
                                                p_out=m_output_power,
                                                v_g_on1=v_g_on1,
                                                transistor1=transistor1,
                                                transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "RMS Inductor Current [A]":
                    m_z = converter.f_m_i_l_rms(zeta=m_zeta,
                                                v_in=m_v_in,
                                                v_out=m_v_out,
                                                p_out=m_output_power,
                                                v_g_on1=v_g_on1,
                                                transistor1=transistor1,
                                                transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Mean Inductor Current [A]":
                    m_z = converter.f_m_i_l_mean(zeta=m_zeta,
                                                 v_in=m_v_in,
                                                 v_out=m_v_out,
                                                 p_out=m_output_power,
                                                 v_g_on1=v_g_on1,
                                                 transistor1=transistor1,
                                                 transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Peak Current [A]":
                    m_z = converter.f_m_i_peak(zeta=m_zeta,
                                               v_in=m_v_in,
                                               v_out=m_v_out,
                                               p_out=m_output_power,
                                               v_g_on1=v_g_on1,
                                               transistor1=transistor1,
                                               transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Conduction Losses Transistor1 [W]":
                    m_z = converter.f_m_conduction_losses1(zeta=m_zeta,
                                                           v_in=m_v_in,
                                                           v_out=m_v_out,
                                                           p_out=m_output_power,
                                                           v_g_on1=v_g_on1,
                                                           transistor1=transistor1,
                                                           transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Conduction Losses Diode Transistor2 [W]":
                    m_z = converter.f_m_conduction_losses2(zeta=m_zeta,
                                                           v_in=m_v_in,
                                                           v_out=m_v_out,
                                                           p_out=m_output_power,
                                                           v_g_on1=v_g_on1,
                                                           transistor1=transistor1,
                                                           transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Total Conduction Losses [W]":
                    m_z = converter.f_m_conduction_losses(zeta=m_zeta,
                                                          v_in=m_v_in,
                                                          v_out=m_v_out,
                                                          p_out=m_output_power,
                                                          v_g_on1=v_g_on1,
                                                          transistor1=transistor1,
                                                          transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Turn-on Switching Losses Transistor1 [W]":
                    m_z = converter.f_m_p_on1(zeta=m_zeta,
                                              v_in=m_v_in,
                                              v_out=m_v_out,
                                              p_out=m_output_power,
                                              v_g_on1=v_g_on1,
                                              transistor1=transistor1,
                                              transistor2=transistor2,
                                              r_g_on1=r_g_on1,
                                              frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Turn-off Switching Losses Transistor1 [W]":
                    m_z = converter.f_m_p_off1(zeta=m_zeta,
                                               v_in=m_v_in,
                                               v_out=m_v_out,
                                               p_out=m_output_power,
                                               v_g_on1=v_g_on1,
                                               transistor1=transistor1,
                                               transistor2=transistor2,
                                               r_g_off1=r_g_off1,
                                               frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Reverse Recovery Losses Diode Transistor2 [W]":
                    m_z = converter.f_m_p_rr2(zeta=m_zeta,
                                              v_in=m_v_in,
                                              v_out=m_v_out,
                                              p_out=m_output_power,
                                              v_g_on1=v_g_on1,
                                              transistor1=transistor1,
                                              transistor2=transistor2,
                                              frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Total Switching Losses Transistor1 [W]":
                    m_z = converter.f_m_p_on_off1(zeta=m_zeta,
                                                  v_in=m_v_in,
                                                  v_out=m_v_out,
                                                  p_out=m_output_power,
                                                  v_g_on1=v_g_on1,
                                                  transistor1=transistor1,
                                                  transistor2=transistor2,
                                                  r_g_on1=r_g_on1,
                                                  r_g_off1=r_g_off1,
                                                  frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Total Switching Losses [W]":
                    m_z = converter.f_m_p_on_off_rr_1_2(zeta=m_zeta,
                                                        v_in=m_v_in,
                                                        v_out=m_v_out,
                                                        p_out=m_output_power,
                                                        v_g_on1=v_g_on1,
                                                        transistor1=transistor1,
                                                        transistor2=transistor2,
                                                        r_g_on1=r_g_on1,
                                                        r_g_off1=r_g_off1,
                                                        frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Total Power Losses Transistor1 [W]":
                    m_z = converter.f_m_p1(zeta=m_zeta,
                                           v_in=m_v_in,
                                           v_out=m_v_out,
                                           p_out=m_output_power,
                                           v_g_on1=v_g_on1,
                                           transistor1=transistor1,
                                           transistor2=transistor2,
                                           r_g_on1=r_g_on1,
                                           r_g_off1=r_g_off1,
                                           frequency=m_frequency)
                if comboBox_topology_plot_z_axis.currentText() == "Temperature Switch Transistor1 [°C]":
                    m_z = converter.f_m_t_switch1(zeta=m_zeta,
                                                  v_in=m_v_in,
                                                  v_out=m_v_out,
                                                  p_out=m_output_power,
                                                  v_g_on1=v_g_on1,
                                                  r_g_on1=r_g_on1,
                                                  r_g_off1=r_g_off1,
                                                  t_heatsink=t_heatsink,
                                                  r_th_heatsink=r_th_heatsink,
                                                  frequency=m_frequency,
                                                  transistor1=transistor1,
                                                  transistor2=transistor2)
                if comboBox_topology_plot_z_axis.currentText() == "Temperature Diode Transistor2 [°C]":
                    m_z = converter.f_m_t_diode2(zeta=m_zeta,
                                                 v_in=m_v_in,
                                                 v_out=m_v_out,
                                                 p_out=m_output_power,
                                                 v_g_on1=v_g_on1,
                                                 t_heatsink=t_heatsink,
                                                 r_th_heatsink=r_th_heatsink,
                                                 frequency=m_frequency,
                                                 transistor1=transistor1,
                                                 transistor2=transistor2)

                plot = matplotlibwidget.axis.contourf(m_x, m_y, m_z, 100, cmap=cm.inferno)
                matplotlibwidget.divider = make_axes_locatable(matplotlibwidget.axis)
                matplotlibwidget.axis_cm = matplotlibwidget.divider.append_axes("right", size="3%", pad=0.03)
                matplotlibwidget.figure.colorbar(plot, cax=matplotlibwidget.axis_cm, format='%.2f')
                matplotlibwidget.axis.set_position([0.175, 0.15, 0.7, 0.75])
                matplotlibwidget.axis.ticklabel_format(useOffset=False)
                matplotlibwidget.axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                matplotlibwidget.axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                matplotlibwidget.axis.set(xlabel=comboBox_topology_plot_x_axis.currentText(),
                                          ylabel=comboBox_topology_plot_y_axis.currentText(),
                                          title=comboBox_topology_plot_z_axis.currentText())
                matplotlibwidget.figure.canvas.draw_idle()

                matplotlibwidget.cursor = Cursor(matplotlibwidget.axis, horizOn=True, vertOn=True, useblit=True,
                                                 color="Green",
                                                 linewidth=1)
                matplotlibwidget.figure.canvas.mpl_connect("button_press_event", clicked)

        except:
            matplotlibwidget.axis.clear()
            matplotlibwidget.figure.canvas.draw_idle()
            self.show_popup_message(
                "Error: " + comboBox_topology_plot_z_axis.currentText() + " could not be plotted due to missing inputs or data!")

        try:
            if comboBox_topology_line_contour.currentText() == "Line":
                if comboBox_topology_plot_x_axis.currentText() == "Zeta = f*L":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_zeta_min.text()),
                                             float(self.lineEdit_topology_zeta_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Vin [V]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_v_in_min.text()),
                                             float(self.lineEdit_topology_v_in_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Vout [V]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_v_out_min.text()),
                                             float(self.lineEdit_topology_v_out_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Output Power [W]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_output_power_min.text()),
                                             float(self.lineEdit_topology_output_power_max.text()), 100)
                if comboBox_topology_plot_x_axis.currentText() == "Frequency [kHz]":
                    vec_x_axis = np.linspace(float(self.lineEdit_topology_frequency_min.text()),
                                             float(self.lineEdit_topology_frequency_max.text()), 100)

                if self.lineEdit_topology_zeta.text() != "":
                    vec_zeta = np.full_like(vec_x_axis, float(self.lineEdit_topology_zeta.text()))
                if self.lineEdit_topology_v_in.text() != "":
                    vec_v_in = np.full_like(vec_x_axis, float(self.lineEdit_topology_v_in.text()))
                if self.lineEdit_topology_v_out.text() != "":
                    vec_v_out = np.full_like(vec_x_axis, float(self.lineEdit_topology_v_out.text()))
                if self.lineEdit_topology_output_power.text() != "":
                    vec_output_power = np.full_like(vec_x_axis, float(self.lineEdit_topology_output_power.text()))
                if self.lineEdit_topology_frequency.text() != "":
                    vec_frequency = np.full_like(vec_x_axis, float(self.lineEdit_topology_frequency.text()))

                if comboBox_topology_plot_x_axis.currentText() == "Zeta = f*L":
                    vec_zeta = vec_x_axis
                if comboBox_topology_plot_x_axis.currentText() == "Vin [V]":
                    vec_v_in = vec_x_axis
                if comboBox_topology_plot_x_axis.currentText() == "Vout [V]":
                    vec_v_out = vec_x_axis
                if comboBox_topology_plot_x_axis.currentText() == "Output Power [W]":
                    vec_output_power = vec_x_axis
                if comboBox_topology_plot_x_axis.currentText() == "Frequency [kHz]":
                    vec_frequency = vec_x_axis

                if comboBox_topology_plot_y_axis.currentText() == "RMS Current Transistor1 [A]":
                    vec_y_axis = converter.f_vec_i1_rms(zeta=vec_zeta,
                                                        v_in=vec_v_in,
                                                        v_out=vec_v_out,
                                                        p_out=vec_output_power,
                                                        v_g_on1=v_g_on1,
                                                        transistor1=transistor1,
                                                        transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "RMS Current Diode Transistor2 [A]":
                    vec_y_axis = converter.f_vec_i2_rms(zeta=vec_zeta,
                                                        v_in=vec_v_in,
                                                        v_out=vec_v_out,
                                                        p_out=vec_output_power,
                                                        v_g_on1=v_g_on1,
                                                        transistor1=transistor1,
                                                        transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Mean Current Transistor1 [A]":
                    vec_y_axis = converter.f_vec_i1_mean(zeta=vec_zeta,
                                                         v_in=vec_v_in,
                                                         v_out=vec_v_out,
                                                         p_out=vec_output_power,
                                                         v_g_on1=v_g_on1,
                                                         transistor1=transistor1,
                                                         transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Mean Current Diode Transistor2 [A]":
                    vec_y_axis = converter.f_vec_i2_mean(zeta=vec_zeta,
                                                         v_in=vec_v_in,
                                                         v_out=vec_v_out,
                                                         p_out=vec_output_power,
                                                         v_g_on1=v_g_on1,
                                                         transistor1=transistor1,
                                                         transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "RMS Inductor Current [A]":
                    vec_y_axis = converter.f_vec_i_l_rms(zeta=vec_zeta,
                                                         v_in=vec_v_in,
                                                         v_out=vec_v_out,
                                                         p_out=vec_output_power,
                                                         v_g_on1=v_g_on1,
                                                         transistor1=transistor1,
                                                         transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Mean Inductor Current [A]":
                    vec_y_axis = converter.f_vec_i_l_mean(zeta=vec_zeta,
                                                          v_in=vec_v_in,
                                                          v_out=vec_v_out,
                                                          p_out=vec_output_power,
                                                          v_g_on1=v_g_on1,
                                                          transistor1=transistor1,
                                                          transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Peak Current [A]":
                    vec_y_axis = converter.f_vec_i_peak(zeta=vec_zeta,
                                                        v_in=vec_v_in,
                                                        v_out=vec_v_out,
                                                        p_out=vec_output_power,
                                                        v_g_on1=v_g_on1,
                                                        transistor1=transistor1,
                                                        transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Conduction Losses Transistor1 [W]":
                    vec_y_axis = converter.f_vec_conduction_losses1(zeta=vec_zeta,
                                                                    v_in=vec_v_in,
                                                                    v_out=vec_v_out,
                                                                    p_out=vec_output_power,
                                                                    v_g_on1=v_g_on1,
                                                                    transistor1=transistor1,
                                                                    transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Conduction Losses Diode Transistor2 [W]":
                    vec_y_axis = converter.f_vec_conduction_losses2(zeta=vec_zeta,
                                                                    v_in=vec_v_in,
                                                                    v_out=vec_v_out,
                                                                    p_out=vec_output_power,
                                                                    v_g_on1=v_g_on1,
                                                                    transistor1=transistor1,
                                                                    transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Total Conduction Losses [W]":
                    vec_y_axis = converter.f_vec_conduction_losses(zeta=vec_zeta,
                                                                   v_in=vec_v_in,
                                                                   v_out=vec_v_out,
                                                                   p_out=vec_output_power,
                                                                   v_g_on1=v_g_on1,
                                                                   transistor1=transistor1,
                                                                   transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Turn-on Switching Losses Transistor1 [W]":
                    vec_y_axis = converter.f_vec_p_on1(zeta=vec_zeta,
                                                       v_in=vec_v_in,
                                                       v_out=vec_v_out,
                                                       p_out=vec_output_power,
                                                       v_g_on1=v_g_on1,
                                                       transistor1=transistor1,
                                                       transistor2=transistor2,
                                                       r_g_on1=r_g_on1,
                                                       frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Turn-off Switching Losses Transistor1 [W]":
                    vec_y_axis = converter.f_vec_p_off1(zeta=vec_zeta,
                                                        v_in=vec_v_in,
                                                        v_out=vec_v_out,
                                                        p_out=vec_output_power,
                                                        v_g_on1=v_g_on1,
                                                        transistor1=transistor1,
                                                        transistor2=transistor2,
                                                        r_g_off1=r_g_off1,
                                                        frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Reverse Recovery Losses Diode Transistor2 [W]":
                    vec_y_axis = converter.f_vec_p_rr2(zeta=vec_zeta,
                                                       v_in=vec_v_in,
                                                       v_out=vec_v_out,
                                                       p_out=vec_output_power,
                                                       v_g_on1=v_g_on1,
                                                       transistor1=transistor1,
                                                       transistor2=transistor2,
                                                       frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Total Switching Losses Transistor1 [W]":
                    vec_y_axis = converter.f_vec_p_on_off1(zeta=vec_zeta,
                                                           v_in=vec_v_in,
                                                           v_out=vec_v_out,
                                                           p_out=vec_output_power,
                                                           v_g_on1=v_g_on1,
                                                           transistor1=transistor1,
                                                           transistor2=transistor2,
                                                           r_g_on1=r_g_on1,
                                                           r_g_off1=r_g_off1,
                                                           frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Total Switching Losses [W]":
                    vec_y_axis = converter.f_vec_p_on_off_rr_1_2(zeta=vec_zeta,
                                                                 v_in=vec_v_in,
                                                                 v_out=vec_v_out,
                                                                 p_out=vec_output_power,
                                                                 v_g_on1=v_g_on1,
                                                                 transistor1=transistor1,
                                                                 transistor2=transistor2,
                                                                 r_g_on1=r_g_on1,
                                                                 r_g_off1=r_g_off1,
                                                                 frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Total Power Losses Transistor1 [W]":
                    vec_y_axis = converter.f_vec_p1(zeta=vec_zeta,
                                                    v_in=vec_v_in,
                                                    v_out=vec_v_out,
                                                    p_out=vec_output_power,
                                                    v_g_on1=v_g_on1,
                                                    transistor1=transistor1,
                                                    transistor2=transistor2,
                                                    r_g_on1=r_g_on1,
                                                    r_g_off1=r_g_off1,
                                                    frequency=vec_frequency)
                if comboBox_topology_plot_y_axis.currentText() == "Temperature Switch Transistor1 [°C]":
                    vec_y_axis = converter.f_vec_t_switch1(zeta=vec_zeta,
                                                           v_in=vec_v_in,
                                                           v_out=vec_v_out,
                                                           p_out=vec_output_power,
                                                           v_g_on1=v_g_on1,
                                                           r_g_on1=r_g_on1,
                                                           r_g_off1=r_g_off1,
                                                           t_heatsink=t_heatsink,
                                                           r_th_heatsink=r_th_heatsink,
                                                           frequency=vec_frequency,
                                                           transistor1=transistor1,
                                                           transistor2=transistor2)
                if comboBox_topology_plot_y_axis.currentText() == "Temperature Diode Transistor2 [°C]":
                    vec_y_axis = converter.f_vec_t_diode2(zeta=vec_zeta,
                                                          v_in=vec_v_in,
                                                          v_out=vec_v_out,
                                                          p_out=vec_output_power,
                                                          v_g_on1=v_g_on1,
                                                          t_heatsink=t_heatsink,
                                                          r_th_heatsink=r_th_heatsink,
                                                          frequency=vec_frequency,
                                                          transistor1=transistor1,
                                                          transistor2=transistor2)

                matplotlibwidget.axis.plot(vec_x_axis, vec_y_axis)
                matplotlibwidget.axis.ticklabel_format(useOffset=False)
                matplotlibwidget.axis.grid()
                matplotlibwidget.axis.set_position([0.15, 0.15, 0.8, 0.8])
                matplotlibwidget.axis.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                matplotlibwidget.axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                matplotlibwidget.axis.set(xlabel=comboBox_topology_plot_x_axis.currentText(),
                                          ylabel=comboBox_topology_plot_y_axis.currentText(), )
                matplotlibwidget.figure.canvas.draw_idle()

                matplotlibwidget.cursor = Cursor(matplotlibwidget.axis, horizOn=True, vertOn=True, useblit=True,
                                                 color="Green", linewidth=1)
                matplotlibwidget.figure.canvas.mpl_connect("button_press_event", clicked)

        except:
            matplotlibwidget.axis.clear()
            matplotlibwidget.figure.canvas.draw_idle()
            self.show_popup_message(
                "Error: " + comboBox_topology_plot_y_axis.currentText() + " could not be plotted due to missing inputs or data!")

    def topology_update_plots(self):
        """
        Set the variable to determine the topology and runs the function "topology_create_plot" for all QWidgets, Matplotlib figures and ComboBoxes.

        :return: None
        """
        if self.comboBox_topology_topology.currentText() == "Buck-Converter":
            converter = buck_converter_functions
        elif self.comboBox_topology_topology.currentText() == "Boost-Converter":
            converter = boost_converter_functions
        elif self.comboBox_topology_topology.currentText() == "Buck-Boost-Converter":
            converter = buck_boost_converter_functions

        self.topology_create_plot(self.widget_topology_plot1,
                                  self.matplotlibwidget_topology1,
                                  self.comboBox_topology_plot1_x_axis,
                                  self.comboBox_topology_plot1_y_axis,
                                  self.comboBox_topology_plot1_z_axis,
                                  self.comboBox_topology_plot1_line_contour,
                                  converter)
        self.topology_create_plot(self.widget_topology_plot2,
                                  self.matplotlibwidget_topology2,
                                  self.comboBox_topology_plot2_x_axis,
                                  self.comboBox_topology_plot2_y_axis,
                                  self.comboBox_topology_plot2_z_axis,
                                  self.comboBox_topology_plot2_line_contour,
                                  converter)
        self.topology_create_plot(self.widget_topology_plot3,
                                  self.matplotlibwidget_topology3,
                                  self.comboBox_topology_plot3_x_axis,
                                  self.comboBox_topology_plot3_y_axis,
                                  self.comboBox_topology_plot3_z_axis,
                                  self.comboBox_topology_plot3_line_contour,
                                  converter)
        self.topology_create_plot(self.widget_topology_plot4,
                                  self.matplotlibwidget_topology4,
                                  self.comboBox_topology_plot4_x_axis,
                                  self.comboBox_topology_plot4_y_axis,
                                  self.comboBox_topology_plot4_z_axis,
                                  self.comboBox_topology_plot4_line_contour,
                                  converter)
        self.topology_create_plot(self.widget_topology_plot5,
                                  self.matplotlibwidget_topology5,
                                  self.comboBox_topology_plot5_x_axis,
                                  self.comboBox_topology_plot5_y_axis,
                                  self.comboBox_topology_plot5_z_axis,
                                  self.comboBox_topology_plot5_line_contour,
                                  converter)
        self.topology_create_plot(self.widget_topology_plot6,
                                  self.matplotlibwidget_topology6,
                                  self.comboBox_topology_plot6_x_axis,
                                  self.comboBox_topology_plot6_y_axis,
                                  self.comboBox_topology_plot6_z_axis,
                                  self.comboBox_topology_plot6_line_contour,
                                  converter)

    def get_converter(self):
        """
        Return a converter_functions python file depending on which converter is currently selected.

        :return: converter_functions python file
        """
        if self.comboBox_topology_topology.currentText() == "Buck-Converter":
            converter = buck_converter_functions
        elif self.comboBox_topology_topology.currentText() == "Boost-Converter":
            converter = boost_converter_functions
        elif self.comboBox_topology_topology.currentText() == "Buck-Boost-Converter":
            converter = buck_boost_converter_functions

        return converter

    def topology_pop_out_plot1(self):
        """
        Open a new window with plot1 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow1 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out1 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow1.widget_plot,
                                  self.matplotlibwidget_pop_out1,
                                  self.comboBox_topology_plot1_x_axis,
                                  self.comboBox_topology_plot1_y_axis,
                                  self.comboBox_topology_plot1_z_axis,
                                  self.comboBox_topology_plot1_line_contour,
                                  converter)
        self.PopOutPlotWindow1.show()

    def topology_pop_out_plot2(self):
        """
        Open a new window with plot2 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow2 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out2 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow2.widget_plot,
                                  self.matplotlibwidget_pop_out2,
                                  self.comboBox_topology_plot2_x_axis,
                                  self.comboBox_topology_plot2_y_axis,
                                  self.comboBox_topology_plot2_z_axis,
                                  self.comboBox_topology_plot2_line_contour,
                                  converter)
        self.PopOutPlotWindow2.show()

    def topology_pop_out_plot3(self):
        """
        Open a new window with plot3 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow3 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out3 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow3.widget_plot,
                                  self.matplotlibwidget_pop_out3,
                                  self.comboBox_topology_plot3_x_axis,
                                  self.comboBox_topology_plot3_y_axis,
                                  self.comboBox_topology_plot3_z_axis,
                                  self.comboBox_topology_plot3_line_contour,
                                  converter)
        self.PopOutPlotWindow3.show()

    def topology_pop_out_plot4(self):
        """
        Open a new window with plot4 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow4 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out4 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow4.widget_plot,
                                  self.matplotlibwidget_pop_out4,
                                  self.comboBox_topology_plot4_x_axis,
                                  self.comboBox_topology_plot4_y_axis,
                                  self.comboBox_topology_plot4_z_axis,
                                  self.comboBox_topology_plot4_line_contour,
                                  converter)
        self.PopOutPlotWindow4.show()

    def topology_pop_out_plot5(self):
        """
        Open a new window with plot5 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow5 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out5 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow5.widget_plot,
                                  self.matplotlibwidget_pop_out5,
                                  self.comboBox_topology_plot5_x_axis,
                                  self.comboBox_topology_plot5_y_axis,
                                  self.comboBox_topology_plot5_z_axis,
                                  self.comboBox_topology_plot5_line_contour,
                                  converter)
        self.PopOutPlotWindow5.show()

    def topology_pop_out_plot6(self):
        """
        Open new window with plot6 from Topology Calculator.

        :return:
        """
        converter = self.get_converter()

        self.PopOutPlotWindow6 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out6 = MatplotlibWidget()

        self.topology_create_plot(self.PopOutPlotWindow6.widget_plot,
                                  self.matplotlibwidget_pop_out6,
                                  self.comboBox_topology_plot6_x_axis,
                                  self.comboBox_topology_plot6_y_axis,
                                  self.comboBox_topology_plot6_z_axis,
                                  self.comboBox_topology_plot6_line_contour,
                                  converter)
        self.PopOutPlotWindow6.show()
