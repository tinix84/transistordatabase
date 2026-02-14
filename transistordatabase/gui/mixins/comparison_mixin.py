"""Comparison tools mixin for MainWindow."""
from math import floor

import numpy as np

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from transistordatabase.gui import comparison_tools_functions
from transistordatabase.gui._widgets import PopOutPlotWindow, MatplotlibWidget


class ComparisonToolsMixin:
    """Mixin providing comparison tools functionality for MainWindow."""

    def clear_comparison_tools(self):
        """
        Clear numeric user inputs in Comparison Tools.

        :return: None
        """
        self.comboBox_compare_v_g_on_transistor1.setCurrentIndex(0)
        self.comboBox_compare_v_g_off_transistor1.setCurrentIndex(0)
        self.comboBox_compare_v_g_on_transistor2.setCurrentIndex(0)
        self.comboBox_compare_v_g_off_transistor2.setCurrentIndex(0)
        self.comboBox_compare_v_g_on_transistor3.setCurrentIndex(0)
        self.comboBox_compare_v_g_off_transistor3.setCurrentIndex(0)
        self.lineEdit_compare_t_j_transistor1.clear()
        self.lineEdit_compare_t_j_transistor2.clear()
        self.lineEdit_compare_t_j_transistor3.clear()
        self.lineEdit_compare_v_supply_transistor1.clear()
        self.lineEdit_compare_number_parallel_transistor1.clear()
        self.lineEdit_compare_v_supply_transistor2.clear()
        self.lineEdit_compare_number_parallel_transistor2.clear()
        self.lineEdit_compare_v_supply_transistor3.clear()
        self.lineEdit_compare_number_parallel_transistor3.clear()
        self.label_compare_r_g_on_value_transistor1.setText(str(0.0))
        self.label_compare_r_g_off_value_transistor1.setText(str(0.0))
        self.label_compare_r_g_on_value_transistor2.setText(str(0.0))
        self.label_compare_r_g_off_value_transistor2.setText(str(0.0))
        self.label_compare_r_g_on_value_transistor3.setText(str(0.0))
        self.label_compare_r_g_off_value_transistor3.setText(str(0.0))
        self.slider_compare_r_g_on_transistor1.setValue(0)
        self.slider_compare_r_g_off_transistor1.setValue(0)
        self.slider_compare_r_g_on_transistor2.setValue(0)
        self.slider_compare_r_g_off_transistor2.setValue(0)
        self.slider_compare_r_g_on_transistor3.setValue(0)
        self.slider_compare_r_g_off_transistor3.setValue(0)

    def compare_create_plot(self, widget_plot: QWidget, matplotlibwidget, comboBox_plot):
        """
        Add a Matplotlib figure to a QWidget and creates a plot based on all the possible inputs and selections.

        :param widget_compare_plot: widget for the matplotlib figure
        :type widget_compare_plot: QWidget
        :param matplotlibwidget_compare: matplotlib figure
        :param comboBox_compare_plot: comboBox to choose plot
        :return: None
        """
        matplotlibwidget.axis.clear()
        self.layout = QVBoxLayout(widget_plot)
        self.layout.addWidget(matplotlibwidget)

        try:
            matplotlibwidget.axis_cm.remove()
        except:
            pass

        try:
            transistor1 = self.tdb.load_transistor(self.comboBox_compare_transistor1.currentText())
            transistor2 = self.tdb.load_transistor(self.comboBox_compare_transistor2.currentText())
            transistor3 = self.tdb.load_transistor(self.comboBox_compare_transistor3.currentText())

            if self.lineEdit_compare_number_parallel_transistor1.text() != "1":
                transistor1 = self.tdb.parallel_transistors(transistor1, int(self.lineEdit_compare_number_parallel_transistor1.text()))
            if self.lineEdit_compare_number_parallel_transistor2.text() != "1":
                transistor2 = self.tdb.parallel_transistors(transistor2, int(self.lineEdit_compare_number_parallel_transistor2.text()))
            if self.lineEdit_compare_number_parallel_transistor3.text() != "1":
                transistor3 = self.tdb.parallel_transistors(transistor3, int(self.lineEdit_compare_number_parallel_transistor3.text()))

            r_g_on1 = float(self.label_compare_r_g_on_value_transistor1.text())
            r_g_on2 = float(self.label_compare_r_g_on_value_transistor2.text())
            r_g_on3 = float(self.label_compare_r_g_on_value_transistor3.text())
            r_g_off1 = float(self.label_compare_r_g_off_value_transistor1.text())
            r_g_off2 = float(self.label_compare_r_g_off_value_transistor2.text())
            r_g_off3 = float(self.label_compare_r_g_off_value_transistor3.text())
            v_supply1 = float(self.lineEdit_compare_v_supply_transistor1.text())
            v_supply2 = float(self.lineEdit_compare_v_supply_transistor2.text())
            v_supply3 = float(self.lineEdit_compare_v_supply_transistor3.text())
            t_j1 = float(self.lineEdit_compare_t_j_transistor1.text())
            t_j2 = float(self.lineEdit_compare_t_j_transistor2.text())
            t_j3 = float(self.lineEdit_compare_t_j_transistor3.text())
            v_g_on1 = float(self.comboBox_compare_v_g_on_transistor1.currentText())
            v_g_on2 = float(self.comboBox_compare_v_g_on_transistor2.currentText())
            v_g_on3 = float(self.comboBox_compare_v_g_on_transistor3.currentText())

            if self.comboBox_compare_v_g_off_transistor1.count() >= 1:
                v_g_off1 = float(self.comboBox_compare_v_g_off_transistor1.currentText())
            else:
                v_g_off1 = None

            if self.comboBox_compare_v_g_off_transistor2.count() >= 1:
                v_g_off2 = float(self.comboBox_compare_v_g_off_transistor2.currentText())
            else:
                v_g_off2 = None

            if self.comboBox_compare_v_g_off_transistor3.count() >= 1:
                v_g_off3 = float(self.comboBox_compare_v_g_off_transistor3.currentText())
            else:
                v_g_off3 = None

            if comboBox_plot.currentText() == "Switch Energy Data vs. Channel Current Transistor1":
                comparison_tools_functions.plot_all_energy_data(transistor1, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Energy Data vs. Channel Current Transistor2":
                comparison_tools_functions.plot_all_energy_data(transistor2, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Energy Data vs. Channel Current Transistor3":
                comparison_tools_functions.plot_all_energy_data(transistor3, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Channel Current Transistor1":
                comparison_tools_functions.plot_all_energy_data(transistor1, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Channel Current Transistor2":
                comparison_tools_functions.plot_all_energy_data(transistor2, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Channel Current Transistor3":
                comparison_tools_functions.plot_all_energy_data(transistor3, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Switch Energy Data vs. Gate Resistor Transistor1":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor1, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Energy Data vs. Gate Resistor Transistor2":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor2, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Energy Data vs. Gate Resistor Transistor3":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor3, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Gate Resistor Transistor1":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor1, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Gate Resistor Transistor2":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor2, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Energy Data vs. Gate Resistor Transistor3":
                comparison_tools_functions.plot_all_energy_data_r_g(transistor3, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Switch Channel Data Transistor1":
                comparison_tools_functions.plot_all_channel_data(transistor1, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Channel Data Transistor2":
                comparison_tools_functions.plot_all_channel_data(transistor2, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Switch Channel Data Transistor3":
                comparison_tools_functions.plot_all_channel_data(transistor3, matplotlibwidget, "switch")
            if comboBox_plot.currentText() == "Diode Channel Data Transistor1":
                comparison_tools_functions.plot_all_channel_data(transistor1, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Channel Data Transistor2":
                comparison_tools_functions.plot_all_channel_data(transistor2, matplotlibwidget, "diode")
            if comboBox_plot.currentText() == "Diode Channel Data Transistor3":
                comparison_tools_functions.plot_all_channel_data(transistor3, matplotlibwidget, "diode")

            if comboBox_plot.currentText() == "Switch Turn-on Losses":
                comparison_tools_functions.plot_e_on(transistor1=transistor1,
                                                     transistor2=transistor2,
                                                     transistor3=transistor3,
                                                     matplotlibwidget=matplotlibwidget,
                                                     t_j1=t_j1,
                                                     t_j2=t_j2,
                                                     t_j3=t_j3,
                                                     r_g_on1=r_g_on1,
                                                     r_g_on2=r_g_on2,
                                                     r_g_on3=r_g_on3,
                                                     v_supply1=v_supply1,
                                                     v_supply2=v_supply2,
                                                     v_supply3=v_supply3)

            if comboBox_plot.currentText() == "Switch Turn-off Losses":
                comparison_tools_functions.plot_e_off(transistor1=transistor1,
                                                      transistor2=transistor2,
                                                      transistor3=transistor3,
                                                      matplotlibwidget=matplotlibwidget,
                                                      t_j1=t_j1,
                                                      t_j2=t_j2,
                                                      t_j3=t_j3,
                                                      r_g_off1=r_g_off1,
                                                      r_g_off2=r_g_off2,
                                                      r_g_off3=r_g_off3,
                                                      v_supply1=v_supply1,
                                                      v_supply2=v_supply2,
                                                      v_supply3=v_supply3)

            if comboBox_plot.currentText() == "Diode Reverse Recovery Losses":
                comparison_tools_functions.plot_e_rr(transistor1=transistor1,
                                                     transistor2=transistor2,
                                                     transistor3=transistor3,
                                                     matplotlibwidget=matplotlibwidget,
                                                     t_j1=t_j1,
                                                     t_j2=t_j2,
                                                     t_j3=t_j3,
                                                     r_g_off1=r_g_off1,
                                                     r_g_off2=r_g_off2,
                                                     r_g_off3=r_g_off3,
                                                     v_supply1=v_supply1,
                                                     v_supply2=v_supply2,
                                                     v_supply3=v_supply3)

            if comboBox_plot.currentText() == "Switch Channel Data":
                comparison_tools_functions.plot_channel(transistor1=transistor1,
                                                        transistor2=transistor2,
                                                        transistor3=transistor3,
                                                        matplotlibwidget=matplotlibwidget,
                                                        t_j1=t_j1,
                                                        t_j2=t_j2,
                                                        t_j3=t_j3,
                                                        v_g_on1=v_g_on1,
                                                        v_g_on2=v_g_on2,
                                                        v_g_on3=v_g_on3,
                                                        v_g_off1=v_g_off1,
                                                        v_g_off2=v_g_off2,
                                                        v_g_off3=v_g_off3,
                                                        switch_diode="switch")

            if comboBox_plot.currentText() == "Diode Channel Data":
                comparison_tools_functions.plot_channel(transistor1=transistor1,
                                                        transistor2=transistor2,
                                                        transistor3=transistor3,
                                                        matplotlibwidget=matplotlibwidget,
                                                        t_j1=t_j1,
                                                        t_j2=t_j2,
                                                        t_j3=t_j3,
                                                        v_g_on1=v_g_on1,
                                                        v_g_on2=v_g_on2,
                                                        v_g_on3=v_g_on3,
                                                        v_g_off1=v_g_off1,
                                                        v_g_off2=v_g_off2,
                                                        v_g_off3=v_g_off3,
                                                        switch_diode="diode")

            if comboBox_plot.currentText() == "Output Capacitance Charge vs. Channel Voltage":
                comparison_tools_functions.plot_v_qoss(transistor1=transistor1,
                                                       transistor2=transistor2,
                                                       transistor3=transistor3,
                                                       matplotlibwidget=matplotlibwidget)

            if comboBox_plot.currentText() == "Output Capacitance Energy vs. Channel Voltage":
                comparison_tools_functions.plot_v_eoss(transistor1=transistor1,
                                                       transistor2=transistor2,
                                                       transistor3=transistor3,
                                                       matplotlibwidget=matplotlibwidget)
        except:
            self.show_popup_message("Error: Inputs are missing or not numeric!")

    def compare_update_plots(self):
        """
        Run the function "topology_create_plot" for all QWidgets, Matplotlib figures and ComboBoxes.

        :return: None
        """
        self.compare_create_plot(self.widget_compare_plot1,
                                 self.matplotlibwidget_compare1,
                                 self.comboBox_compare_plot1)
        self.compare_create_plot(self.widget_compare_plot2,
                                 self.matplotlibwidget_compare2,
                                 self.comboBox_compare_plot2)
        self.compare_create_plot(self.widget_compare_plot3,
                                 self.matplotlibwidget_compare3,
                                 self.comboBox_compare_plot3)
        self.compare_create_plot(self.widget_compare_plot4,
                                 self.matplotlibwidget_compare4,
                                 self.comboBox_compare_plot4)
        self.compare_create_plot(self.widget_compare_plot5,
                                 self.matplotlibwidget_compare5,
                                 self.comboBox_compare_plot5)
        self.compare_create_plot(self.widget_compare_plot6,
                                 self.matplotlibwidget_compare6,
                                 self.comboBox_compare_plot6)
        self.compare_create_plot(self.widget_compare_plot7,
                                 self.matplotlibwidget_compare7,
                                 self.comboBox_compare_plot7)
        self.compare_create_plot(self.widget_compare_plot8,
                                 self.matplotlibwidget_compare8,
                                 self.comboBox_compare_plot8)
        self.compare_create_plot(self.widget_compare_plot9,
                                 self.matplotlibwidget_compare9,
                                 self.comboBox_compare_plot9)

    def compare_pop_out_plot1(self):
        """
        Pops out plot1 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow1 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare1 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow1.widget_plot,
                                 self.matplotlibwidget_pop_out_compare1,
                                 self.comboBox_compare_plot1)

        self.PopOutPlotWindow1.show()

    def compare_pop_out_plot2(self):
        """
        Pops out plot2 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow2 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare2 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow2.widget_plot,
                                 self.matplotlibwidget_pop_out_compare2,
                                 self.comboBox_compare_plot2)

        self.PopOutPlotWindow2.show()

    def compare_pop_out_plot3(self):
        """
        Pops out plot3 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow3 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare3 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow3.widget_plot,
                                 self.matplotlibwidget_pop_out_compare3,
                                 self.comboBox_compare_plot3)

        self.PopOutPlotWindow3.show()

    def compare_pop_out_plot4(self):
        """
        Pops out plot4 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow4 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare4 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow4.widget_plot,
                                 self.matplotlibwidget_pop_out_compare4,
                                 self.comboBox_compare_plot4)

        self.PopOutPlotWindow4.show()

    def compare_pop_out_plot5(self):
        """
        Pops out plot5 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow5 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare5 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow5.widget_plot,
                                 self.matplotlibwidget_pop_out_compare5,
                                 self.comboBox_compare_plot5)

        self.PopOutPlotWindow5.show()

    def compare_pop_out_plot6(self):
        """
        Pops out plot6 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow6 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare6 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow6.widget_plot,
                                 self.matplotlibwidget_pop_out_compare6,
                                 self.comboBox_compare_plot6)

        self.PopOutPlotWindow6.show()

    def compare_pop_out_plot7(self):
        """
        Pops out plot7 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow7 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare7 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow7.widget_plot,
                                 self.matplotlibwidget_pop_out_compare7,
                                 self.comboBox_compare_plot7)

        self.PopOutPlotWindow7.show()

    def compare_pop_out_plot8(self):
        """
        Pops out plot8 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow8 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare8 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow8.widget_plot,
                                 self.matplotlibwidget_pop_out_compare8,
                                 self.comboBox_compare_plot8)

        self.PopOutPlotWindow8.show()

    def compare_pop_out_plot9(self):
        """
        Pops out plot9 in Comparison Cools in a PopOutPlotWindow.

        :return: None
        """
        self.PopOutPlotWindow9 = PopOutPlotWindow()
        self.matplotlibwidget_pop_out_compare9 = MatplotlibWidget()

        self.compare_create_plot(self.PopOutPlotWindow9.widget_plot,
                                 self.matplotlibwidget_pop_out_compare9,
                                 self.comboBox_compare_plot9)

        self.PopOutPlotWindow9.show()

    def comboBox_compare_transistor_changed(self, comboBox_compare_transistor, comboBox_compare_v_g_on_transistor,
                                            comboBox_compare_v_g_off_transistor, lineEdit_compare_t_j_transistor,
                                            slider_compare_r_g_on_transistor, label_compare_r_g_on_value_transistor,
                                            slider_compare_r_g_off_transistor, label_compare_r_g_off_value_transistor):
        """
        Fill the comboBoxes for transistor with data based on the available data in the transistordatabase.

        :return: None
        """
        transistor = self.tdb.load_transistor(comboBox_compare_transistor.currentText())
        comboBox_compare_v_g_on_transistor.clear()
        comboBox_compare_v_g_off_transistor.clear()

        if transistor is not None:
            available_v_g_on_transistor = [str(channel.v_g) for channel in transistor.switch.channel]
            available_v_g_on_transistor_cleared = []

            for v_g in available_v_g_on_transistor:
                if v_g not in available_v_g_on_transistor_cleared and v_g != "None":
                    available_v_g_on_transistor_cleared.append(v_g)

            comboBox_compare_v_g_on_transistor.addItems(available_v_g_on_transistor_cleared)
            comboBox_compare_v_g_on_transistor.setCurrentText(
                str(max(channel.v_g for channel in transistor.switch.channel)))

            if transistor.type.lower() != "igbt":
                available_v_g_off_transistor = [str(channel.v_g) for channel in transistor.diode.channel]
                available_v_g_off_transistor_cleared = []

                for v_g in available_v_g_off_transistor:
                    if v_g not in available_v_g_off_transistor_cleared and v_g != "None":
                        available_v_g_off_transistor_cleared.append(v_g)

                comboBox_compare_v_g_off_transistor.addItems(available_v_g_off_transistor_cleared)
                comboBox_compare_v_g_off_transistor.setCurrentText(
                    str(min(channel.v_g for channel in transistor.diode.channel)))

            try:
                t_j_available_unfiltered = [i for i in [e_on.t_j for e_on in transistor.switch.e_on]]
                t_j_available_unfiltered = np.sort(t_j_available_unfiltered)
                t_j_available = []
                for i in t_j_available_unfiltered:
                    if i not in t_j_available:
                        t_j_available.append(i)

                r_g_on_max_list = np.zeros_like(t_j_available)

                for i in range(len(t_j_available)):
                    r_e_object_on = transistor.get_object_r_e_simplified(
                        e_on_off_rr="e_on",
                        t_j=t_j_available[i], v_g=max([i for i in [e_on.v_g for e_on in transistor.switch.e_on] if i is not None]),
                        v_supply=max([i for i in [e_on.v_supply for e_on in transistor.switch.e_on] if i is not None]),
                        normalize_t_to_v=10)

                    r_g_on_max_list[i] = np.amax(r_e_object_on.graph_r_e[0]) * 10000

                r_g_on_max = floor(10 * min(r_g_on_max_list) / 10000) / 10
                slider_compare_r_g_on_transistor.setMinimum(0)
                slider_compare_r_g_on_transistor.setMaximum(int(r_g_on_max * 100))
                slider_compare_r_g_on_transistor.setValue(int(r_g_on_max * 100))
                label_compare_r_g_on_value_transistor.setText(str(r_g_on_max))

            except:
                try:
                    r_g_on = max([i for i in [e_on.r_g for e_on in transistor.switch.e_on] if i is not None])
                    slider_compare_r_g_on_transistor.setMinimum(int(r_g_on * 100))
                    slider_compare_r_g_on_transistor.setMaximum(int(r_g_on * 100))
                    # self.show_popup_message(f"No energy data for different turn on gate resistor for <b>{transistor.name}</b> available!")
                except:
                    # self.show_popup_message(f"No turn-on energy data for <b>{transistor.name}</b> available!")
                    slider_compare_r_g_on_transistor.setMinimum(0)
                    slider_compare_r_g_on_transistor.setMaximum(0)

            try:
                t_j_available_unfiltered = [i for i in [e_off.t_j for e_off in transistor.switch.e_off]]
                t_j_available_unfiltered = np.sort(t_j_available_unfiltered)
                t_j_available = []
                for i in t_j_available_unfiltered:
                    if i not in t_j_available:
                        t_j_available.append(i)

                r_g_off_max_list = np.zeros_like(t_j_available)

                for i in range(len(t_j_available)):
                    r_e_object_off = transistor.get_object_r_e_simplified(
                        e_on_off_rr="e_off", t_j=t_j_available[i],
                        v_g=max([i for i in [e_off.v_g for e_off in transistor.switch.e_off] if i is not None]),
                        v_supply=max([i for i in [e_off.v_supply for e_off in transistor.switch.e_off] if i is not None]),
                        normalize_t_to_v=10)
                    r_g_off_max_list[i] = np.amax(r_e_object_off.graph_r_e[0]) * 10000

                r_g_off_max = floor(10 * min(r_g_off_max_list) / 10000) / 10

                if transistor.type == "IGBT":
                    t_j_available_unfiltered = [i for i in [e_rr.t_j for e_rr in transistor.diode.e_rr]]
                    t_j_available_unfiltered = np.sort(t_j_available_unfiltered)
                    t_j_available = []
                    for i in t_j_available_unfiltered:
                        if i not in t_j_available:
                            t_j_available.append(i)

                    r_g_rr_max_list = np.zeros_like(t_j_available)

                    for i in range(len(t_j_available)):
                        r_e_object_rr = transistor.get_object_r_e_simplified(
                            e_on_off_rr="e_rr",
                            t_j=t_j_available[i], v_g=min([i for i in [e_rr.v_g for e_rr in transistor.diode.e_rr] if i is not None]),
                            v_supply=max([i for i in [e_rr.v_supply for e_rr in transistor.diode.e_rr] if i is not None]),
                            normalize_t_to_v=10)
                        r_g_rr_max_list[i] = np.amax(r_e_object_rr.graph_r_e[0]) * 10000

                    r_g_rr_max = floor(10 * min(r_g_rr_max_list) / 10000) / 10

                r_g_off_max = min(r_g_off_max, r_g_rr_max)

                slider_compare_r_g_off_transistor.setMinimum(0)
                slider_compare_r_g_off_transistor.setMaximum(int(r_g_off_max * 100))
                slider_compare_r_g_off_transistor.setValue(int(r_g_off_max * 100))
                label_compare_r_g_off_value_transistor.setText(str(r_g_off_max))
            except:
                try:
                    r_g_off = max([i for i in [e_off.r_g for e_off in transistor.switch.e_off] if i is not None])
                    slider_compare_r_g_off_transistor.setMinimum(int(r_g_off * 100))
                    slider_compare_r_g_off_transistor.setMaximum(int(r_g_off * 100))
                    # self.show_popup_message(f"No energy data for different turn off gate resistor for <b>{transistor.name}</b> available!")
                except:
                    # self.show_popup_message(f"No turn-off energy data for <b>{transistor.name}</b> available!")
                    slider_compare_r_g_off_transistor.setMinimum(0)
                    slider_compare_r_g_off_transistor.setMaximum(0)

    def comboBox_compare_transistor1_changed(self):
        """
        Run the function to fill the comboBoxes and configurate the sliders for the comparison tools tab for transistor1.

        :return:
        """
        self.comboBox_compare_transistor_changed(self.comboBox_compare_transistor1,
                                                 self.comboBox_compare_v_g_on_transistor1,
                                                 self.comboBox_compare_v_g_off_transistor1,
                                                 self.lineEdit_compare_t_j_transistor1,
                                                 self.slider_compare_r_g_on_transistor1,
                                                 self.label_compare_r_g_on_value_transistor1,
                                                 self.slider_compare_r_g_off_transistor1,
                                                 self.label_compare_r_g_off_value_transistor1)

    def comboBox_compare_transistor2_changed(self):
        """
        Run the function to fill the comboBoxes and configurate the sliders for the comparison tools tab for transistor2.

        :return:
        """
        self.comboBox_compare_transistor_changed(self.comboBox_compare_transistor2,
                                                 self.comboBox_compare_v_g_on_transistor2,
                                                 self.comboBox_compare_v_g_off_transistor2,
                                                 self.lineEdit_compare_t_j_transistor2,
                                                 self.slider_compare_r_g_on_transistor2,
                                                 self.label_compare_r_g_on_value_transistor2,
                                                 self.slider_compare_r_g_off_transistor2,
                                                 self.label_compare_r_g_off_value_transistor2)

    def comboBox_compare_transistor3_changed(self):
        """
        Run the function to fill the comboBoxes and configurate the sliders for the comparison tools tab for transistor3.

        :return:
        """
        self.comboBox_compare_transistor_changed(self.comboBox_compare_transistor3,
                                                 self.comboBox_compare_v_g_on_transistor3,
                                                 self.comboBox_compare_v_g_off_transistor3,
                                                 self.lineEdit_compare_t_j_transistor3,
                                                 self.slider_compare_r_g_on_transistor3,
                                                 self.label_compare_r_g_on_value_transistor3,
                                                 self.slider_compare_r_g_off_transistor3,
                                                 self.label_compare_r_g_off_value_transistor3)

    def slider_compare_r_g_value_changed(self):
        """
        Set the labels below the sliders to choose gate resistors to show currently selected values.

        :return:
        """
        self.label_compare_r_g_on_value_transistor1.setText(
            str(round(self.slider_compare_r_g_on_transistor1.value() / 100, 1)))
        self.label_compare_r_g_off_value_transistor1.setText(
            str(round(self.slider_compare_r_g_off_transistor1.value() / 100, 1)))
        self.label_compare_r_g_on_value_transistor2.setText(
            str(round(self.slider_compare_r_g_on_transistor2.value() / 100, 1)))
        self.label_compare_r_g_off_value_transistor2.setText(
            str(round(self.slider_compare_r_g_off_transistor2.value() / 100, 1)))
        self.label_compare_r_g_on_value_transistor3.setText(
            str(round(self.slider_compare_r_g_on_transistor3.value() / 100, 1)))
        self.label_compare_r_g_off_value_transistor3.setText(
            str(round(self.slider_compare_r_g_off_transistor3.value() / 100, 1)))
