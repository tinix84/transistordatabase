"""Curve management mixin for MainWindow."""
from PyQt5.QtWidgets import QFileDialog

from transistordatabase.database_manager import DatabaseManager
from transistordatabase.gui._widgets import ViewCurveWindow


def _get_curve_checker_window():
    """Deferred import to avoid circular dependency with gui.py."""
    from transistordatabase.gui.gui import CurveCheckerWindow
    return CurveCheckerWindow


class CurveManagementMixin:
    """Mixin providing curve add/view/delete functionality for MainWindow."""

    def add_curve_switch_channel(self):
        """
        Add an item to the comboBox to store the switch channel curves for a new transistor.

        Each comboBox item consists of a text containing the boundary conditions and path and
        a value that is a dictionary(graph key will be filled by the CurveChecker), which will be used to fill the transistor template

        :return: None
        """
        t_j = float(
            self.lineEdit_create_transistor_switch_add_channel_data_t_j.text()) \
            if self.lineEdit_create_transistor_switch_add_channel_data_t_j.text() != "" else None
        t_j_entry_name = self.lineEdit_create_transistor_switch_add_channel_data_t_j.text() + \
            " °C" if self.lineEdit_create_transistor_switch_add_channel_data_t_j.text() != "" else None

        v_g = float(
            self.lineEdit_create_transistor_switch_add_channel_data_v_g.text()) \
            if self.lineEdit_create_transistor_switch_add_channel_data_v_g.text() != "" else None
        v_g_entry_name = self.lineEdit_create_transistor_switch_add_channel_data_v_g.text() + \
            " V" if self.lineEdit_create_transistor_switch_add_channel_data_v_g.text() != "" else None

        file_path = self.browse_file_csv()

        comboBox_entry_name = f"T_j = {t_j_entry_name}, V_g = {v_g_entry_name}\nPath: {file_path}"
        data_dict = {"t_j": t_j, "v_g": v_g, "graph_v_i": "", "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_channel_data)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_switch_added_curves_channel_data.addItem(comboBox_entry_name,
                                                                                         data_dict)

                self.comboBox_create_transistor_switch_added_curves_channel_data.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_switch_added_curves_channel_data.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch Channel Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curves_channel_data,
                    xlabel="Voltage in V",
                    ylabel="Current in A")

    def add_dpt_measurement_data_directory(self):
        """
        Add double pulse test measurement data to a comboBox so that the data can be saved to transistor.

        Once "Load Transistor into Local Database" button is pressed.

        :return: None
        """
        dataset_type = self.comboBox_create_transistor_add_data_dpt_dataset_type.currentText()
        comment = self.lineEdit_create_transistor_add_data_dpt_comment.text()
        load_inductance = float(
            self.lineEdit_create_transistor_add_data_dpt_load_inductance.text()) \
            if self.lineEdit_create_transistor_add_data_dpt_load_inductance.text() != "" else None
        commutation_inductance = float(
            self.lineEdit_create_transistor_add_data_commutation_inductance.text()) \
            if self.lineEdit_create_transistor_add_data_commutation_inductance.text() != "" else None
        measurement_date = self.lineEdit_create_transistor_add_data_dpt_measurement_date.text()
        measurement_testbench = self.comboBox_create_transistor_add_data_dpt_measurement_testbench.currentText()
        # v_g = float(
        #     self.lineEdit_create_transistor_add_data_dpt_v_g.text())
        #     if self.lineEdit_create_transistor_add_data_dpt_v_g.text() != "" else None
        # v_g_off = float(
        #     self.lineEdit_create_transistor_add_data_dpt_v_g_off.text())
        #     if self.lineEdit_create_transistor_add_data_dpt_v_g_off.text() != "" else None
        # r_g_on = float(
        #     self.lineEdit_create_transistor_add_data_dpt_r_g_on.text())
        #     if self.lineEdit_create_transistor_add_data_dpt_r_g_on.text() != "" else None
        # r_g_off = float(
        #     self.lineEdit_create_transistor_add_data_dpt_r_g_off.text())
        #     if self.lineEdit_create_transistor_add_data_dpt_r_g_off.text() != "" else None
        energies = self.comboBox_create_transistor_add_data_dpt_energies.currentText()
        commutation_device = self.lineEdit_create_transistor_add_data_dpt_commutation_device.text()
        integration_interval = self.comboBox_create_transistor_add_data_dpt_integration_interval.currentText()
        # t_j = float(
        #     self.lineEdit_create_transistor_add_data_dpt_t_j.text()) if self.lineEdit_create_transistor_add_data_dpt_t_j.text() != "" else None

        directory_path = QFileDialog.getExistingDirectory(self, caption="Open Directory")

        if directory_path != "":
            directory_path = directory_path + str("/*.csv")
            comboBox_entry_name = f"Directory: {directory_path}"

            if dataset_type == "I_E Curve":
                dataset_type = "graph_i_e"
            elif dataset_type == "R_E Curve":
                dataset_type = "graph_r_e"

            data_dict = {
                'path': directory_path,
                'dataset_type': dataset_type,
                'comment': comment,
                'load_inductance': load_inductance,
                'commutation_inductance': commutation_inductance,
                'measurement_date': measurement_date,
                'measurement_testbench': measurement_testbench,
                # 'v_g': v_g,
                # 'v_g_off': v_g_off,
                # 'r_g_on': r_g_on,
                # 'r_g_off': r_g_off,
                # 't_j': t_j,
                'energies': energies,
                'commutation_device': commutation_device,
                'integration_interval': list(self.translation_dict.keys())[
                    list(self.translation_dict.values()).index(integration_interval)],
                'mode': 'save'}

            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_added_dpt)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Data already added!")
            else:
                try:
                    new_dpt_dict = DatabaseManager.dpt_save_data(data_dict)
                    # if new_dpt_dict == ValueError:
                    #     self.show_popup_message("Name of the files not matching")
                    self.comboBox_create_transistor_added_dpt.addItem(comboBox_entry_name,
                                                                      {"new_dpt_dict": new_dpt_dict,
                                                                       "data_dict": data_dict})
                    self.comboBox_create_transistor_added_dpt.setCurrentText(comboBox_entry_name)
                except:
                    self.show_popup_message("Selected Directory is invalid! Possible Reasons <br>"
                                            "1. Values of file are not matching(Voltage, Resistance or Temperature) <br>"
                                            "2. Following keys in csv-file name must match <br>"
                                            "    - Temperature <b>_</b>xx<b>C_</b><br>"
                                            "    - Voltage <b>_</b>xxx<b>V_</b> <br>"
                                            "    - Resistance <b>_</b>x.xx<b>R_</b> <br>"
                                            "    - Gate Voltage <b>_</b>xx<b>vg_</b> <br>"
                                            "    - Current on/off <b> _ON_I</b> or <b> _OFF_I </b><br>"
                                            "    - Voltage on/off <b>ON_U</b> or <b>OFF_U</b>")

    def view_dpt_measurement_data(self):
        """
        Show added DPT measurement data in a ViewCurveWindow.

        :return: None
        """
        try:
            data = self.comboBox_create_transistor_added_dpt.itemData(
                self.comboBox_create_transistor_added_dpt.currentIndex())
            data_dict = data["data_dict"]
            DatabaseManager.dpt_save_data(data_dict)
        except:
            try:
                data_dict = self.comboBox_create_transistor_added_dpt.itemData(
                    self.comboBox_create_transistor_added_dpt.currentIndex())
                if data_dict["dataset_type"] == "graph_i_e":
                    xlabel = "Current in A"
                elif data_dict["dataset_type"] == "graph_r_e":
                    xlabel = "Gate Resistor in Ω"

                self.ViewCurveWindow = ViewCurveWindow()
                self.ViewCurveWindow.view_curve(curve_title="DPT Measurement Data Curve",
                                                xlabel=xlabel,
                                                ylabel="Energy in J",
                                                comboBox=self.comboBox_create_transistor_added_dpt)
            except:
                pass

    def delete_dpt_measurement_data(self):
        """
        Delete added DPT measurement data.

        :return: None
        """
        try:
            if "Directory" in self.comboBox_create_transistor_added_dpt.currentText():
                self.comboBox_create_transistor_added_dpt.removeItem(
                    self.comboBox_create_transistor_added_dpt.currentIndex())
            else:
                self.show_popup_message("Existing DPT Measurement Data can not be deleted!")
        except:
            pass

    def view_curve_switch_channel(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(curve_title="Switch Channel Curve",
                                        xlabel="Voltage in V",
                                        ylabel="Current in A",
                                        comboBox=self.comboBox_create_transistor_switch_added_curves_channel_data)

    def delete_curve_switch_channel(self):
        """
        Delete the current item from the comboBox to store switch channel curves.

        :return: None
        """
        self.comboBox_create_transistor_switch_added_curves_channel_data.removeItem(
            self.comboBox_create_transistor_switch_added_curves_channel_data.currentIndex())
        self.comboBox_create_transistor_switch_added_curves_channel_data.setDisabled(False)

    def add_curve_switch_switching_losses(self):
        """Add switching loss curve."""
        t_j = float(
            self.lineEdit_create_transistor_switch_add_switching_losses_t_j.text()) if \
            self.lineEdit_create_transistor_switch_add_switching_losses_t_j.text() != "" else None
        t_j_entry_name = self.lineEdit_create_transistor_switch_add_switching_losses_t_j.text() + " °C" if \
            self.lineEdit_create_transistor_switch_add_switching_losses_t_j.text() != "" else None

        v_g = float(
            self.lineEdit_create_transistor_switch_add_switching_losses_v_g.text()) if \
            self.lineEdit_create_transistor_switch_add_switching_losses_v_g.text() != "" else None
        v_g_entry_name = self.lineEdit_create_transistor_switch_add_switching_losses_v_g.text() + " V" if \
            self.lineEdit_create_transistor_switch_add_switching_losses_v_g.text() != "" else None

        r_g_i_x = float(
            self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text()) if \
            self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text() != "" else None
        if self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text() != "" and \
                self.comboBox_create_transistor_switch_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Channel Current":
            r_g_i_x_entry_name = self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text() + " Ω"
        elif self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text() != "" and \
                self.comboBox_create_transistor_switch_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Gate Resistor":
            r_g_i_x_entry_name = self.lineEdit_create_transistor_switch_add_switching_losses_r_g_i_x.text() + " A"

        v_supply = float(
            self.lineEdit_create_transistor_switch_add_switching_losses_v_supply.text()) \
            if self.lineEdit_create_transistor_switch_add_switching_losses_v_supply.text() != "" else None
        v_supply_entry_name = self.lineEdit_create_transistor_switch_add_switching_losses_v_supply.text() + \
            " V" if self.lineEdit_create_transistor_switch_add_switching_losses_v_supply.text() != "" else None

        e_on_off = self.comboBox_create_transistor_switch_add_switching_losses_on_off.currentText()
        curve_type = self.comboBox_create_transistor_switch_add_switching_losses_curve_type.currentText()

        file_path = self.browse_file_csv()

        if curve_type == "Switching Losses vs. Channel Current":
            data_dict = {"e_on_off": e_on_off.lower(), "dataset_type": "graph_i_e", "t_j": t_j, 'v_g': v_g,
                         'v_supply': v_supply, 'r_g': r_g_i_x, "graph_i_e": "", "path": file_path}
            comboBox_entry_name = f"{e_on_off}: T_j = {t_j_entry_name}, V_g = {v_g_entry_name}, " \
                                  f"R_g = {r_g_i_x_entry_name}, V_supply = {v_supply_entry_name}\nPath: {file_path}"
        if curve_type == "Switching Losses vs. Gate Resistor":
            data_dict = {"e_on_off": e_on_off.lower(), "dataset_type": "graph_r_e", "t_j": t_j, 'v_g': v_g,
                         'v_supply': v_supply, 'i_x': r_g_i_x, "graph_r_e": "", "path": file_path}
            comboBox_entry_name = f"{e_on_off}: T_j = {t_j_entry_name}, V_g = {v_g_entry_name}, " \
                                  f"I_x = {r_g_i_x_entry_name}, V_supply = {v_supply_entry_name}\nPath: {file_path}"

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_switching_losses)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_switch_added_curves_switching_losses.addItem(comboBox_entry_name,
                                                                                             data_dict)

                self.comboBox_create_transistor_switch_added_curves_switching_losses.setCurrentText(
                    comboBox_entry_name)
                self.comboBox_create_transistor_switch_added_curves_switching_losses.setDisabled(True)

                if data_dict["dataset_type"] == "graph_i_e":
                    xlabel = "Current in A"
                elif data_dict["dataset_type"] == "graph_r_e":
                    xlabel = "Gate Resistor in Ω"

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch Switching Losses Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curves_switching_losses,
                    xlabel=xlabel,
                    ylabel="Energy in J")

    def view_curve_switch_switching_losses(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        data_dict = self.comboBox_create_transistor_switch_added_curves_switching_losses.itemData(
            self.comboBox_create_transistor_switch_added_curves_switching_losses.currentIndex())
        if data_dict["dataset_type"] == "graph_i_e":
            xlabel = "Current in A"
        elif data_dict["dataset_type"] == "graph_r_e":
            xlabel = "Gate Resistor in Ω"

        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(curve_title="Switch Switching Losses Curve",
                                        xlabel=xlabel,
                                        ylabel="Energy in J",
                                        comboBox=self.comboBox_create_transistor_switch_added_curves_switching_losses)

    def delete_curve_switch_switching_losses(self):
        """Clear switching loss curve."""
        self.comboBox_create_transistor_switch_added_curves_switching_losses.removeItem(
            self.comboBox_create_transistor_switch_added_curves_switching_losses.currentIndex())
        self.comboBox_create_transistor_switch_added_curves_switching_losses.setDisabled(False)

    def add_curve_switch_gate_charge(self):
        """Add gate charge curve."""
        i_channel = float(
            self.lineEdit_create_transistor_switch_add_gate_charge_i_channel.text()) if \
            self.lineEdit_create_transistor_switch_add_gate_charge_i_channel.text() != "" else None
        i_channel_entry_name = self.lineEdit_create_transistor_switch_add_gate_charge_i_channel.text() + \
            " A" if self.lineEdit_create_transistor_switch_add_gate_charge_i_channel.text() != "" else None

        i_g = float(
            self.lineEdit_create_transistor_switch_add_gate_charge_i_g.text()) \
            if self.lineEdit_create_transistor_switch_add_gate_charge_i_g.text() != "" else None
        i_g_entry_name = self.lineEdit_create_transistor_switch_add_gate_charge_i_g.text() + \
            " A" if self.lineEdit_create_transistor_switch_add_gate_charge_i_g.text() != "" else None

        t_j = float(
            self.lineEdit_create_transistor_switch_add_gate_charge_t_j.text()) \
            if self.lineEdit_create_transistor_switch_add_gate_charge_t_j.text() != "" else None
        t_j_entry_name = self.lineEdit_create_transistor_switch_add_gate_charge_t_j.text() + \
            " °C" if self.lineEdit_create_transistor_switch_add_gate_charge_t_j.text() != "" else None

        v_supply = float(
            self.lineEdit_create_transistor_switch_add_gate_charge_v_supply.text()) \
            if self.lineEdit_create_transistor_switch_add_gate_charge_v_supply.text() != "" else None
        v_supply_entry_name = self.lineEdit_create_transistor_switch_add_gate_charge_v_supply.text() + \
            " V" if self.lineEdit_create_transistor_switch_add_gate_charge_v_supply.text() != "" else None

        file_path = self.browse_file_csv()
        comboBox_entry_name = f"I_channel ={i_channel_entry_name}, T_j = {t_j_entry_name}, " \
                              f"V_supply = {v_supply_entry_name}, I_g = {i_g_entry_name}\nPath: {file_path}"
        data_dict = {'i_channel': i_channel, 't_j': t_j, 'v_supply': v_supply, 'i_g': i_g, 'graph_q_v': "",
                     "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_gate_charge)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_switch_added_curves_gate_charge.addItem(comboBox_entry_name, data_dict)

                self.comboBox_create_transistor_switch_added_curves_gate_charge.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_switch_added_curves_gate_charge.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch Gate Charge Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curves_gate_charge,
                    xlabel="Gate Charge in nC",
                    ylabel="Gate Source Voltage in V")

    def view_curve_switch_gate_charge(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Switch Gate Charge Curve",
            comboBox=self.comboBox_create_transistor_switch_added_curves_gate_charge,
            xlabel="Gate Charge in nC",
            ylabel="Gate Source Voltage in V")

    def delete_curve_switch_gate_charge(self):
        """Clear gate charge curve."""
        self.comboBox_create_transistor_switch_added_curves_gate_charge.removeItem(
            self.comboBox_create_transistor_switch_added_curves_gate_charge.currentIndex())
        self.comboBox_create_transistor_switch_added_curves_gate_charge.setDisabled(False)

    def add_curve_switch_r_on(self):
        """Add R_on curve."""
        i_channel = float(
            self.lineEdit_create_transistor_switch_add_r_on_i_channel.text()) \
            if self.lineEdit_create_transistor_switch_add_r_on_i_channel.text() != "" else None
        i_channel_entry_name = self.lineEdit_create_transistor_switch_add_r_on_i_channel.text() + \
            " A" if self.lineEdit_create_transistor_switch_add_r_on_i_channel.text() != "" else None

        v_g = float(
            self.lineEdit_create_transistor_switch_add_r_on_v_g.text()) if self.lineEdit_create_transistor_switch_add_r_on_v_g.text() != "" else None
        v_g_entry_name = self.lineEdit_create_transistor_switch_add_r_on_v_g.text() + \
            " V" if self.lineEdit_create_transistor_switch_add_r_on_v_g.text() != "" else None

        r_channel_nominal = float(
            self.lineEdit_create_transistor_switch_add_r_on_r_channel_nominal.text()) \
            if self.lineEdit_create_transistor_switch_add_r_on_r_channel_nominal.text() != "" else None
        r_channel_nominal_entry_name = self.lineEdit_create_transistor_switch_add_r_on_r_channel_nominal.text() + \
            " Ω" if self.lineEdit_create_transistor_switch_add_r_on_r_channel_nominal.text() != "" else None

        file_path = self.browse_file_csv()
        comboBox_entry_name = f"I_channel ={i_channel_entry_name}, V_g = {v_g_entry_name}, " \
                              f"R_channel_nominal = {r_channel_nominal_entry_name}\nPath: {file_path}"
        data_dict = {'i_channel': i_channel, 'v_g': v_g, 'dataset_type': 't_r', 'r_channel_nominal': r_channel_nominal,
                     'graph_t_r': "", "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_r_on)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_switch_added_curves_r_on.addItem(comboBox_entry_name,
                                                                                 data_dict)

                self.comboBox_create_transistor_switch_added_curves_r_on.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_switch_added_curves_r_on.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch On Resistance Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curves_r_on,
                    xlabel="Junction Temperature in °C",
                    ylabel="On Resistance in Ω")

    def view_curve_switch_r_on(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Switch On Resistance Curve",
            comboBox=self.comboBox_create_transistor_switch_added_curves_r_on,
            xlabel="Junction Temperature in °C",
            ylabel="On Resistance in Ω")

    def delete_curve_switch_r_on(self):
        """Clear R_on resistance curve."""
        self.comboBox_create_transistor_switch_added_curves_r_on.removeItem(
            self.comboBox_create_transistor_switch_added_curves_r_on.currentIndex())
        self.comboBox_create_transistor_switch_added_curves_r_on.setDisabled(False)

    def add_curve_diode_channel(self):
        """Add channel curve."""
        t_j = float(
            self.lineEdit_create_transistor_diode_add_channel_data_t_j.text()) \
            if self.lineEdit_create_transistor_diode_add_channel_data_t_j.text() != "" else None
        t_j_entry_name = self.lineEdit_create_transistor_diode_add_channel_data_t_j.text() + \
            " °C" if self.lineEdit_create_transistor_diode_add_channel_data_t_j.text() != "" else None

        v_g = float(
            self.lineEdit_create_transistor_diode_add_channel_data_v_g.text()) \
            if self.lineEdit_create_transistor_diode_add_channel_data_v_g.text() != "" else None
        v_g_entry_name = self.lineEdit_create_transistor_diode_add_channel_data_v_g.text() + \
            " V" if self.lineEdit_create_transistor_diode_add_channel_data_v_g.text() != "" else None

        file_path = self.browse_file_csv()
        comboBox_entry_name = f"T_j = {t_j_entry_name}, V_g = {v_g_entry_name}\nPath: {file_path}"
        data_dict = {"t_j": t_j, 'v_g': v_g, "graph_v_i": "", "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_diode_added_curves_channel_data)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_diode_added_curves_channel_data.addItem(comboBox_entry_name,
                                                                                        data_dict)

                self.comboBox_create_transistor_diode_added_curves_channel_data.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_diode_added_curves_channel_data.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Diode Channel Curve",
                    comboBox=self.comboBox_create_transistor_diode_added_curves_channel_data,
                    xlabel="Voltage in V",
                    ylabel="Current in A")

    def view_curve_diode_channel(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Diode Channel Curve",
            comboBox=self.comboBox_create_transistor_diode_added_curves_channel_data,
            xlabel="Voltage in V",
            ylabel="Current in A")

    def delete_curve_diode_channel(self):
        """Clear channel (forward characteristics) curve."""
        self.comboBox_create_transistor_diode_added_curves_channel_data.removeItem(
            self.comboBox_create_transistor_diode_added_curves_channel_data.currentIndex())
        self.comboBox_create_transistor_diode_added_curves_channel_data.setDisabled(False)

    def add_curve_diode_switching_losses(self):
        """Add switching loss curve."""
        t_j = float(
            self.lineEdit_create_transistor_diode_add_switching_losses_t_j.text()) if \
            self.lineEdit_create_transistor_diode_add_switching_losses_t_j.text() != "" else None
        t_j_entry_name = self.lineEdit_create_transistor_diode_add_switching_losses_t_j.text() + \
            " °C" if self.lineEdit_create_transistor_diode_add_switching_losses_t_j.text() != "" else None

        v_g = float(
            self.lineEdit_create_transistor_diode_add_switching_losses_v_g.text()) if \
            self.lineEdit_create_transistor_diode_add_switching_losses_v_g.text() != "" else None
        v_g_entry_name = self.lineEdit_create_transistor_diode_add_switching_losses_v_g.text() + \
            " V" if self.lineEdit_create_transistor_diode_add_switching_losses_v_g.text() != "" else None

        r_g_i_x = float(
            self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text()) \
            if self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text() != "" else None

        if self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text() != "" and \
                self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Channel Current":
            r_g_i_x_entry_name = self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text() + " Ω"
        elif self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text() != "" and \
                self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Gate Resistor":
            r_g_i_x_entry_name = self.lineEdit_create_transistor_diode_add_switching_losses_r_g_i_x.text() + " A"

        v_supply = float(
            self.lineEdit_create_transistor_diode_add_switching_losses_v_supply.text()) \
            if self.lineEdit_create_transistor_diode_add_switching_losses_v_supply.text() != "" else None
        v_supply_entry_name = self.lineEdit_create_transistor_diode_add_switching_losses_v_supply.text() + \
            " V" if self.lineEdit_create_transistor_diode_add_switching_losses_v_supply.text() != "" else None

        file_path = self.browse_file_csv()

        if self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Channel Current":
            data_dict = {"dataset_type": "graph_i_e", "t_j": t_j, 'v_g': v_g, 'v_supply': v_supply, 'r_g': r_g_i_x,
                         "graph_i_e": "", "path": file_path}
            comboBox_entry_name = f"T_j = {t_j_entry_name}, V_g = {v_g_entry_name}, r_g = {r_g_i_x_entry_name}, " \
                                  f"V_supply = {v_supply_entry_name}\nPath: {file_path}"
        elif self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Gate Resistor":
            data_dict = {"dataset_type": "graph_r_e", "t_j": t_j, 'v_g': v_g, 'v_supply': v_supply, 'i_x': r_g_i_x,
                         "graph_r_e": "", "path": file_path}
            comboBox_entry_name = f"T_j = {t_j_entry_name}, V_g = {v_g_entry_name}, i_x = {r_g_i_x_entry_name}, " \
                                  f"V_supply = {v_supply_entry_name}\nPath: {file_path}"

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_diode_added_curves_switching_losses)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_diode_added_curves_switching_losses.addItem(comboBox_entry_name, data_dict)

                self.comboBox_create_transistor_diode_added_curves_switching_losses.setCurrentText(
                    comboBox_entry_name)
                self.comboBox_create_transistor_diode_added_curves_switching_losses.setDisabled(True)

                if data_dict["dataset_type"] == "graph_i_e":
                    xlabel = "Current in A"
                elif data_dict["dataset_type"] == "graph_r_e":
                    xlabel = "Gate Resistor in Ω"

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Diode Reverse Recovery Losses Curve",
                    comboBox=self.comboBox_create_transistor_diode_added_curves_switching_losses,
                    xlabel=xlabel,
                    ylabel="Energy in J")

    def view_curve_diode_switching_losses(self):
        """
        Run CurveCecker for currently selected curve to edit it.

        :return: None
        """
        data_dict = self.comboBox_create_transistor_diode_added_curves_switching_losses.itemData(
            self.comboBox_create_transistor_diode_added_curves_switching_losses.currentIndex())
        if data_dict["dataset_type"] == "graph_i_e":
            xlabel = "Current in A"
        elif data_dict["dataset_type"] == "graph_r_e":
            xlabel = "Gate Resistor in Ω"

        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Diode Reverse Recovery Losses Curve",
            comboBox=self.comboBox_create_transistor_diode_added_curves_switching_losses,
            xlabel=xlabel,
            ylabel="Energy in J")

    def delete_curve_diode_switching_losses(self):
        """Clear the SOA curve."""
        self.comboBox_create_transistor_diode_added_curves_switching_losses.removeItem(
            self.comboBox_create_transistor_diode_added_curves_switching_losses.currentIndex())
        self.comboBox_create_transistor_diode_added_curves_switching_losses.setDisabled(False)

    def add_curve_switch_soa_t_pulse(self):
        """Add the SOA curve."""
        t_c = float(
            self.lineEdit_create_transistor_switch_soa_t_pulse_t_c.text()) \
            if self.lineEdit_create_transistor_switch_soa_t_pulse_t_c.text() != "" else None
        t_c_entry_name = self.lineEdit_create_transistor_switch_soa_t_pulse_t_c.text() + " °C" \
            if self.lineEdit_create_transistor_switch_soa_t_pulse_t_c.text() != "" else None

        time_pulse = float(
            self.lineEdit_create_transistor_switch_soa_t_pulse_time_pulse.text()) \
            if self.lineEdit_create_transistor_switch_soa_t_pulse_time_pulse.text() != "" else None
        time_pulse_entry_name = self.lineEdit_create_transistor_switch_soa_t_pulse_time_pulse.text() + " s" \
            if self.lineEdit_create_transistor_switch_soa_t_pulse_time_pulse.text() != "" else None

        file_path = self.browse_file_csv()
        comboBox_entry_name = f"T_j = {t_c_entry_name}, Time_pulse = {time_pulse_entry_name}\nPath: {file_path}"
        data_dict = {'t_c': t_c, 'time_pulse': time_pulse, 'graph_i_v': "", "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_soa_t_pulse)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.addItem(comboBox_entry_name,
                                                                                        data_dict)

                self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch SOA Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curves_soa_t_pulse,
                    xlabel="V_ds/V_r in V",
                    ylabel="I_d/I_r in A")

    def view_curve_switch_soa_t_pulse(self):
        """View the SOA curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Switch SOA Curve",
            comboBox=self.comboBox_create_transistor_switch_added_curves_soa_t_pulse,
            xlabel="V_ds/V_r in V",
            ylabel="I_d/I_r in A")

    def delete_curve_switch_soa_t_pulse(self):
        """Clear the SOA curve."""
        self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.removeItem(
            self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.currentIndex())
        self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.setDisabled(False)

    def add_curve_diode_soa_t_pulse(self):
        """Add the SOA curve."""
        t_c = float(
            self.lineEdit_create_transistor_diode_soa_t_pulse_t_c.text()) \
            if self.lineEdit_create_transistor_diode_soa_t_pulse_t_c.text() != "" else None
        t_c_entry_name = self.lineEdit_create_transistor_diode_soa_t_pulse_t_c.text() + \
            " °C" if self.lineEdit_create_transistor_diode_soa_t_pulse_t_c.text() != "" else None

        time_pulse = float(
            self.lineEdit_create_transistor_diode_soa_t_pulse_time_pulse.text()) \
            if self.lineEdit_create_transistor_diode_soa_t_pulse_time_pulse.text() != "" else None
        time_pulse_entry_name = self.lineEdit_create_transistor_diode_soa_t_pulse_time_pulse.text() + \
            " s" if self.lineEdit_create_transistor_diode_soa_t_pulse_time_pulse.text() != "" else None

        file_path = self.browse_file_csv()
        comboBox_entry_name = f"T_j = {t_c_entry_name}, Time_pulse = {time_pulse_entry_name}\nPath: {file_path}"
        data_dict = {'t_c': t_c, 'time_pulse': time_pulse, 'graph_i_v': "", "path": file_path}

        if file_path != "":
            all_items_text = self.get_all_items_text_from_comboBox(
                self.comboBox_create_transistor_diode_added_curves_soa_t_pulse)
            if comboBox_entry_name in all_items_text:
                self.show_popup_message("Curve already added!")
            else:
                self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.addItem(comboBox_entry_name,
                                                                                       data_dict)

                self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.setCurrentText(comboBox_entry_name)
                self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Diode SOA Curve",
                    comboBox=self.comboBox_create_transistor_diode_added_curves_soa_t_pulse,
                    xlabel="V_ds/V_r in V",
                    ylabel="I_d/I_r in A")

    def view_curve_diode_soa_t_pulse(self):
        """View the SOA curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Diode SOA Curve",
            comboBox=self.comboBox_create_transistor_diode_added_curves_soa_t_pulse,
            xlabel="V_ds/V_r in V",
            ylabel="I_d/I_r in A")

    def delete_curve_diode_soa_t_pulse(self):
        """Clear the SOA curve."""
        self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.removeItem(
            self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.currentIndex())
        self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.setDisabled(False)

    def add_curve_switch_t_rthjc(self):
        """Add the R_th,jc (time) curve."""
        if self.comboBox_create_transistor_switch_added_curve_t_rthjc.count() == 0:
            file_path = self.browse_file_csv()
            comboBox_entry_name = f"Path: {file_path}"
            data_dict = {'graph_t_rthjc': "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_switch_added_curve_t_rthjc.addItem(comboBox_entry_name,
                                                                                   data_dict)

                self.comboBox_create_transistor_switch_added_curve_t_rthjc.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Switch T_Rthjc Curve",
                    comboBox=self.comboBox_create_transistor_switch_added_curve_t_rthjc,
                    xlabel="Junction Temperature in °C",
                    ylabel="T_Rthjc in Ω")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_switch_t_rthjc(self):
        """View the R_th,jc (time) curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Switch T_Rthjc Curve",
            comboBox=self.comboBox_create_transistor_switch_added_curve_t_rthjc,
            xlabel="Junction Temperature in °C",
            ylabel="T_Rthjc in Ω")

    def delete_curve_switch_t_rthjc(self):
        """Clear the R_th,jc (time) curve."""
        self.comboBox_create_transistor_switch_added_curve_t_rthjc.clear()
        self.comboBox_create_transistor_diode_added_curves_soa_t_pulse.setDisabled(False)

    def add_curve_diode_t_rthjc(self):
        """Add the R_th,jc (time) curve."""
        if self.comboBox_create_transistor_diode_added_curve_t_rthjc.count() == 0:
            file_path = self.browse_file_csv()
            comboBox_entry_name = f"Path: {file_path}"
            data_dict = {'graph_t_rthjc': "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_diode_added_curve_t_rthjc.addItem(comboBox_entry_name,
                                                                                  data_dict)

                self.comboBox_create_transistor_diode_added_curve_t_rthjc.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="Diode T_Rthjc Curve",
                    comboBox=self.comboBox_create_transistor_diode_added_curve_t_rthjc,
                    xlabel="Junction Temperature in °C",
                    ylabel="T_Rthjc in Ω")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_diode_t_rthjc(self):
        """View the R_th,jc (time) curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="Diode T_Rthjc Curve",
            comboBox=self.comboBox_create_transistor_diode_added_curve_t_rthjc,
            xlabel="Junction Temperature in °C",
            ylabel="T_Rthjc in Ω")

    def delete_curve_diode_t_rthjc(self):
        """Clear the R_th,jc (time) curve."""
        self.comboBox_create_transistor_diode_added_curve_t_rthjc.clear()
        self.comboBox_create_transistor_diode_added_curve_t_rthjc.setDisabled(False)

    def add_curve_v_ecoss(self):
        """Add the E_coss (V) curve."""
        if self.comboBox_create_transistor_added_curve_v_ecoss.count() == 0:
            file_path = self.browse_file_csv()
            comboBox_entry_name = f"Path: {file_path}"
            data_dict = {'graph_v_ecoss': "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_curve_v_ecoss.addItem(comboBox_entry_name,
                                                                            data_dict)

                self.comboBox_create_transistor_added_curve_v_ecoss.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="E_coss Curve",
                    comboBox=self.comboBox_create_transistor_added_curve_v_ecoss,
                    xlabel="Voltage in V",
                    ylabel="Energy in J")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_v_ecoss(self):
        """View the E_coss (V) curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="E_coss Curve",
            comboBox=self.comboBox_create_transistor_added_curve_v_ecoss,
            xlabel="Voltage in V",
            ylabel="Energy in J")

    def delete_curve_v_ecoss(self):
        """Clear the E_oss (V) curve."""
        self.comboBox_create_transistor_added_curve_v_ecoss.clear()
        self.comboBox_create_transistor_added_curve_v_ecoss.setDisabled(False)

    def add_curve_c_iss_normal(self):
        """Add the C_iss curve."""
        if self.comboBox_create_transistor_added_c_iss_normal.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_iss_t_j.text()) if self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() + " °C" \
                if self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}\nPath: {file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_iss_normal.addItem(comboBox_entry_name, data_dict)

                self.comboBox_create_transistor_added_c_iss_normal.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_iss_normal Curve",
                    comboBox=self.comboBox_create_transistor_added_c_iss_normal,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_iss_normal(self):
        """View the C_iss curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_iss_normal Curve",
            comboBox=self.comboBox_create_transistor_added_c_iss_normal,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_iss_normal(self):
        """Clear the C_iss curve."""
        self.comboBox_create_transistor_added_c_iss_normal.clear()
        self.comboBox_create_transistor_added_c_iss_normal.setDisabled(False)

    def add_curve_c_iss_detail(self):
        """Add the C_iss_detail curve."""
        if self.comboBox_create_transistor_added_c_iss_detail.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_iss_t_j.text()) \
                if self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() + \
                " °C" if self.lineEdit_create_transistor_add_curve_c_iss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}\nPath: {file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_iss_detail.addItem(comboBox_entry_name,
                                                                           data_dict)

                self.comboBox_create_transistor_added_c_iss_detail.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_iss_detail Curve",
                    comboBox=self.comboBox_create_transistor_added_c_iss_detail,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_iss_detail(self):
        """View the C_iss_detail curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_iss_detail Curve",
            comboBox=self.comboBox_create_transistor_added_c_iss_detail,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_iss_detail(self):
        """Clear the C_iss_detail curve."""
        self.comboBox_create_transistor_added_c_iss_detail.clear()
        self.comboBox_create_transistor_added_c_iss_detail.setDisabled(False)

    def add_curve_c_oss_normal(self):
        """Add the C_oss curve."""
        if self.comboBox_create_transistor_added_c_oss_normal.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_oss_t_j.text()) if self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() + " °C" \
                if self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}\nPath: {file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_oss_normal.addItem(comboBox_entry_name,
                                                                           data_dict)

                self.comboBox_create_transistor_added_c_oss_normal.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_oss_normal Curve",
                    comboBox=self.comboBox_create_transistor_added_c_oss_normal,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_oss_normal(self):
        """View the C_oss curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_oss_normal Curve",
            comboBox=self.comboBox_create_transistor_added_c_oss_normal,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_oss_normal(self):
        """Clear the C_oss curve."""
        self.comboBox_create_transistor_added_c_oss_normal.clear()
        self.comboBox_create_transistor_added_c_oss_normal.setDisabled(False)

    def add_curve_c_oss_detail(self):
        """Add the C_oss_detail curve."""
        if self.comboBox_create_transistor_added_c_oss_detail.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_oss_t_j.text()) if \
                self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() + " °C" \
                if self.lineEdit_create_transistor_add_curve_c_oss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}\nPath: {file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_oss_detail.addItem(comboBox_entry_name,
                                                                           data_dict)

                self.comboBox_create_transistor_added_c_oss_detail.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_oss_detail Curve",
                    comboBox=self.comboBox_create_transistor_added_c_oss_detail,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_oss_detail(self):
        """View the C_oss_detail curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_oss_detail Curve",
            comboBox=self.comboBox_create_transistor_added_c_oss_detail,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_oss_detail(self):
        """Clear the C_oss_detail curve."""
        self.comboBox_create_transistor_added_c_oss_detail.clear()
        self.comboBox_create_transistor_added_c_oss_detail.setDisabled(False)

    def add_curve_c_rss_normal(self):
        """Add the C_rss curve."""
        if self.comboBox_create_transistor_added_c_rss_normal.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_rss_t_j.text()) if self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() + " °C" \
                if self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}, Path\n{file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_rss_normal.addItem(comboBox_entry_name,
                                                                           data_dict)

                self.comboBox_create_transistor_added_c_rss_normal.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_rss_normal Curve",
                    comboBox=self.comboBox_create_transistor_added_c_rss_normal,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_rss_normal(self):
        """View the C_rss curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_rss_normal Curve",
            comboBox=self.comboBox_create_transistor_added_c_rss_normal,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_rss_normal(self):
        """Clear the C_rss curve."""
        self.comboBox_create_transistor_added_c_rss_normal.clear()
        self.comboBox_create_transistor_added_c_rss_normal.setDisabled(False)

    def add_curve_c_rss_detail(self):
        """Add the C_rss_detail curve."""
        if self.comboBox_create_transistor_added_c_rss_detail.count() == 0:
            t_j = float(
                self.lineEdit_create_transistor_add_curve_c_rss_t_j.text()) if \
                self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() != "" else None
            t_j_entry_name = self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() + " °C" \
                if self.lineEdit_create_transistor_add_curve_c_rss_t_j.text() != "" else None

            file_path = self.browse_file_csv()
            comboBox_entry_name = f"T_j = {t_j_entry_name}\nPath: {file_path}"
            data_dict = {"t_j": t_j, "graph_v_c": "", "path": file_path}

            if file_path != "":
                self.comboBox_create_transistor_added_c_rss_detail.addItem(comboBox_entry_name,
                                                                           data_dict)

                self.comboBox_create_transistor_added_c_rss_detail.setDisabled(True)

                self.CurveCheckerWindow = _get_curve_checker_window()()
                self.CurveCheckerWindow.run_curve_checker(
                    curve_title="C_rss_detail Curve",
                    comboBox=self.comboBox_create_transistor_added_c_rss_detail,
                    xlabel="Voltage in V",
                    ylabel="Capacitance in F")
        else:
            self.show_popup_message("Curve has already been added!")

    def view_curve_c_rss_detail(self):
        """View the C_rss_detail curve."""
        self.ViewCurveWindow = ViewCurveWindow()
        self.ViewCurveWindow.view_curve(
            curve_title="C_rss_detail Curve",
            comboBox=self.comboBox_create_transistor_added_c_rss_detail,
            xlabel="Voltage in V",
            ylabel="Capacitance in F")

    def delete_curve_c_rss_detail(self):
        """Clear crss_detail curve."""
        self.comboBox_create_transistor_added_c_rss_detail.clear()
        self.comboBox_create_transistor_added_c_rss_detail.setDisabled(False)
