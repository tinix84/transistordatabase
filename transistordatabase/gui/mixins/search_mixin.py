"""Search database mixin for MainWindow."""
import numpy as np

from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QLineEdit


class SearchDatabaseMixin:
    """Mixin providing search database functionality for MainWindow."""

    def update_database_from_fileexchange(self):
        """
        Update the local database from the github fileexchange.

        :return: None
        :rtype: None
        """
        self.tdb.update_from_fileexchange()
        self.label_updated_database.setText("Successfully Updated")

    def reset_filter_search_database(self):
        """
        Reset all filters in the search database tab.

        :return: None
        """
        for widget in self.scrollAreaWidgetContents_search_database.children():
            if isinstance(widget, QLineEdit):
                widget.clear()

    def get_marked_transistor(self):
        """
        Get the marked transistor object from the search-database tab.

        :return: Transistor
        :rtype: Transistor
        """
        try:
            for i in range(self.tableWidget_search_database.columnCount()):
                if self.tableWidget_search_database.horizontalHeaderItem(i).text() == "NAME":
                    column = i
            selected_transistor_name = self.tableWidget_search_database.item(
                self.tableWidget_search_database.currentRow(), column).text()

            return self.tdb.load_transistor(selected_transistor_name)

        except:
            self.show_popup_message("Error: No transistor selected!")

    def delete_marked_transistor_search_database_from_local_tdb(self):
        """Delete the marked transistor ('search transistor'-tab) from the local mongodb-database."""
        transistor = self.get_marked_transistor()
        self.tdb.delete_transistor(transistor.name)
        self.search_database_load_data()

    def load_from_search_database_into_create_transistor(self):
        """
        Load a selected transistor from search database into create transistor.

        :return: None
        """
        try:
            self.clear_create_transistor()

            for i in range(self.tableWidget_search_database.columnCount()):
                if self.tableWidget_search_database.horizontalHeaderItem(i).text() == "NAME":
                    column = i
            selected_transistor_name = self.tableWidget_search_database.item(
                self.tableWidget_search_database.currentRow(), column).text()

            transistor = self.tdb.load_transistor(selected_transistor_name)

            transistor_dict = transistor.convert_to_dict()
            transistor_switch_dict = transistor.switch.convert_to_dict()
            transistor_diode_dict = transistor.diode.convert_to_dict()

            self.lineEdit_create_transistor_name.setText(str(transistor_dict["name"]))
            self.comboBox_create_transistor_type.setCurrentText(str(transistor_dict["type"]))
            self.lineEdit_create_transistor_author.setText(str(transistor_dict["author"]))
            self.lineEdit_create_transistor_comment.setText(str(transistor_dict["comment"]))
            self.comboBox_create_transistor_manufacturer.setCurrentText(str(transistor_dict["manufacturer"]))
            self.lineEdit_create_transistor_datasheet_hyperlink.setText(str(transistor_dict["datasheet_hyperlink"]))
            self.lineEdit_create_transistor_datasheet_date.setText(str(transistor_dict["datasheet_date"]))
            self.lineEdit_create_transistor_datasheet_version.setText(str(transistor_dict["datasheet_version"]))
            self.lineEdit_create_transistor_housing_area.setText(str(transistor_dict["housing_area"]))
            self.lineEdit_create_transistor_cooling_area.setText(str(transistor_dict["cooling_area"]))
            self.comboBox_create_transistor_housing_type.setCurrentText(transistor_dict["housing_type"])
            self.lineEdit_create_transistor_v_abs_max.setText(str(transistor_dict["v_abs_max"]))
            self.lineEdit_create_transistor_i_abs_max.setText(str(transistor_dict["i_abs_max"]))
            self.lineEdit_create_transistor_i_cont.setText(str(transistor_dict["i_cont"]))

            if transistor_dict["c_oss_er"] is not None:
                self.lineEdit_create_transistor_transistor_c_oss_er_c_o.setText(str(transistor_dict["c_oss_er"]["c_o"]))
                self.lineEdit_create_transistor_transistor_c_oss_er_v_gs.setText(
                    str(transistor_dict["c_oss_er"]["v_gs"]))
                self.lineEdit_create_transistor_transistor_c_oss_er_v_ds.setText(
                    str(transistor_dict["c_oss_er"]["v_ds"]))

            if transistor_dict["c_oss_tr"] is not None:
                self.lineEdit_create_transistor_transistor_c_oss_tr_c_o.setText(str(transistor_dict["c_oss_tr"]["c_o"]))
                self.lineEdit_create_transistor_transistor_c_oss_tr_v_gs.setText(
                    str(transistor_dict["c_oss_tr"]["v_gs"]))
                self.lineEdit_create_transistor_transistor_c_oss_tr_v_ds.setText(
                    str(transistor_dict["c_oss_tr"]["v_ds"]))

            self.lineEdit_create_transistor_t_c_max.setText(str(transistor_dict["t_c_max"]))
            self.lineEdit_create_transistor_r_g_int.setText(str(transistor_dict["r_g_int"]))
            self.lineEdit_create_transistor_r_th_cs.setText(str(transistor_dict["r_th_cs"]))
            self.lineEdit_create_transistor_r_th_diode_cs.setText(str(transistor_dict["r_th_diode_cs"]))
            self.lineEdit_create_transistor_r_th_switch_cs.setText(str(transistor_dict["r_th_switch_cs"]))
            self.lineEdit_create_transistor_r_g_on_recommended.setText(str(transistor_dict["r_g_on_recommended"]))
            self.lineEdit_create_transistor_r_g_off_recommended.setText(str(transistor_dict["r_g_off_recommended"]))
            self.lineEdit_create_transistor_c_iss_fix.setText(str(transistor_dict["c_iss_fix"]))
            self.lineEdit_create_transistor_c_oss_fix.setText(str(transistor_dict["c_oss_fix"]))
            self.lineEdit_create_transistor_c_rss_fix.setText(str(transistor_dict["c_rss_fix"]))

            # SWITCH KEYS###
            self.comboBox_create_transistor_switch_manufacturer.setCurrentText(
                str(transistor_switch_dict["manufacturer"]))
            self.lineEdit_create_transistor_switch_technology.setText(str(transistor_switch_dict["technology"]))
            self.lineEdit_create_transistor_switch_comment.setText(str(transistor_switch_dict["comment"]))
            self.lineEdit_create_transistor_switch_t_j_max.setText(str(transistor_switch_dict["t_j_max"]))

            try:
                r_th_vector_list = transistor_switch_dict["thermal_foster"]["r_th_vector"]
                r_th_vector = ""
                for i in range(len(r_th_vector_list)):
                    r_th_vector = r_th_vector + str(r_th_vector_list[i]) + " "
                self.lineEdit_create_transistor_switch_r_th_vector.setText(r_th_vector)
            except:
                self.lineEdit_create_transistor_switch_r_th_vector.setText("None")

            self.lineEdit_create_transistor_switch_r_th_total.setText(
                str(transistor_switch_dict["thermal_foster"]["r_th_total"]))

            try:
                c_th_vector_list = transistor_switch_dict["thermal_foster"]["c_th_vector"]
                c_th_vector = ""
                for i in range(len(c_th_vector_list)):
                    c_th_vector = c_th_vector + str(c_th_vector_list[i]) + " "
                self.lineEdit_create_transistor_switch_c_th_vector.setText(c_th_vector)
            except:
                self.lineEdit_create_transistor_switch_c_th_vector.setText("None")

            self.lineEdit_create_transistor_switch_c_th_total.setText(
                str(transistor_switch_dict["thermal_foster"]["c_th_total"]))

            try:
                tau_vector_list = transistor_switch_dict["thermal_foster"]["tau_vector"]
                tau_vector = ""
                for i in range(len(tau_vector_list)):
                    tau_vector = tau_vector + str(tau_vector_list[i]) + " "
                self.lineEdit_create_transistor_switch_tau_vector.setText(tau_vector)
            except:
                self.lineEdit_create_transistor_switch_tau_vector.setText("None")

            self.lineEdit_create_transistor_switch_tau_total.setText(
                str(transistor_switch_dict["thermal_foster"]["tau_total"]))

            # DIODE KEYS#
            self.comboBox_create_transistor_diode_manufacturer.setCurrentText(
                str(transistor_diode_dict["manufacturer"]))
            self.lineEdit_create_transistor_diode_technology.setText(str(transistor_diode_dict["technology"]))
            self.lineEdit_create_transistor_diode_comment.setText(str(transistor_diode_dict["comment"]))
            self.lineEdit_create_transistor_diode_t_j_max.setText(str(transistor_diode_dict["t_j_max"]))

            try:
                r_th_vector_list = transistor_diode_dict["thermal_foster"]["r_th_vector"]
                r_th_vector = ""
                for i in range(len(r_th_vector_list)):
                    r_th_vector = r_th_vector + str(r_th_vector_list[i]) + " "
                self.lineEdit_create_transistor_diode_r_th_vector.setText(r_th_vector)
            except:
                self.lineEdit_create_transistor_diode_r_th_vector.setText("None")

            self.lineEdit_create_transistor_diode_r_th_total.setText(
                str(transistor_diode_dict["thermal_foster"]["r_th_total"]))

            try:
                c_th_vector_list = transistor_diode_dict["thermal_foster"]["c_th_vector"]
                c_th_vector = ""
                for i in range(len(c_th_vector_list)):
                    c_th_vector = c_th_vector + str(c_th_vector_list[i]) + " "
                self.lineEdit_create_transistor_diode_c_th_vector.setText(c_th_vector)
            except:
                self.lineEdit_create_transistor_diode_c_th_vector.setText("None")

            self.lineEdit_create_transistor_diode_c_th_total.setText(
                str(transistor_diode_dict["thermal_foster"]["c_th_total"]))

            try:
                tau_vector_list = transistor_diode_dict["thermal_foster"]["tau_vector"]
                tau_vector = ""
                for i in range(len(tau_vector_list)):
                    tau_vector = tau_vector + str(tau_vector_list[i]) + " "
                self.lineEdit_create_transistor_diode_tau_vector.setText(tau_vector)
            except:
                self.lineEdit_create_transistor_diode_tau_vector.setText("None")

            self.lineEdit_create_transistor_diode_tau_total.setText(
                str(transistor_diode_dict["thermal_foster"]["tau_total"]))

            # TRANSISTOR CURVES#
            graph_v_ecoss = transistor_dict["graph_v_ecoss"]
            if graph_v_ecoss is not None and graph_v_ecoss != []:
                self.comboBox_create_transistor_added_curve_v_ecoss.addItem("V_Ecoss Curve",
                                                                            {"graph_v_ecoss": np.array(graph_v_ecoss)})

            list = transistor_dict["c_iss"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_added_c_iss_normal.addItem(f"C_iss Curve, T_j = {t_j}", list[i])

            list = transistor_dict["c_oss"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_added_c_oss_normal.addItem(f"C_oss Curve, T_j = {t_j}", list[i])

            list = transistor_dict["c_rss"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_added_c_rss_normal.addItem(f"C_rss Curve, T_j = {t_j}", list[i])

            # SWITCH CURVES#

            thermal_foster_switch = transistor_switch_dict["thermal_foster"]
            if thermal_foster_switch["graph_t_rthjc"] is not None and thermal_foster_switch["graph_t_rthjc"] != []:
                self.comboBox_create_transistor_switch_added_curve_t_rthjc.addItem("T_Rthjc Curve", {
                    "graph_t_rthjc": np.array(thermal_foster_switch["graph_t_rthjc"])})

            list = transistor_switch_dict["channel"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_channel_data.addItem(
                        f"Switch Channel Curve, T_j = {t_j}, V_g = {v_g}", list[i])

            list = transistor_switch_dict["e_on"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                list[i].update({"e_on_off": "e_on"})
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                r_g = str(list[i]["r_g"]) + " Ω" if list[i]["r_g"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + " V" if list[i]["v_supply"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_switching_losses.addItem(
                        f"E_on: T_j = {t_j}, V_g = {v_g}, R_g = {r_g}, V_supply = {v_supply}", list[i])

            list = transistor_switch_dict["e_off"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                list[i].update({"e_on_off": "e_off"})
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                r_g = str(list[i]["r_g"]) + " Ω" if list[i]["r_g"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + "V" if list[i]["v_supply"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_switching_losses.addItem(
                        f"E_off: T_j = {t_j}, V_g = {v_g}, R_g = {r_g}, V_supply = {v_supply}", list[i])

            list = transistor_switch_dict["soa"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_c = str(list[i]["t_c"]) + " °C" if list[i]["t_c"] is not None else "None"
                time_pulse = str(list[i]["time_pulse"]) + " s" if list[i]["time_pulse"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_soa_t_pulse.addItem(
                        f"T_c = {t_c}, Time_pulse = {time_pulse}", list[i])

            list = transistor_switch_dict["charge_curve"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                i_channel = str(list[i]["i_channel"]) + " A" if list[i]["i_channel"] is not None else "None"
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + " V" if list[i]["v_supply"] is not None else "None"
                i_g = str(list[i]["i_g"]) + " A" if list[i]["i_g"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_gate_charge.addItem(
                        f"I_channel = {i_channel}, T_j = {t_j}, V_supply = {v_supply}, I_g = {i_g}", list[i])

            list = transistor_switch_dict["r_channel_th"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                i_channel = str(list[i]["i_channel"]) + " A" if list[i]["i_channel"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                r_channel_nominal = str(list[i]["r_channel_nominal"]) + " Ω" if list[i]["r_channel_nominal"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_switch_added_curves_r_on.addItem(
                        f"I_channel = {i_channel}, V_g = {v_g}, R_channel_nominal = {r_channel_nominal}", list[i])

            # DIODE CURVES #

            thermal_foster_diode = transistor_diode_dict["thermal_foster"]
            if thermal_foster_diode["graph_t_rthjc"] is not None and thermal_foster_diode["graph_t_rthjc"] != []:
                self.comboBox_create_transistor_diode_added_curve_t_rthjc.addItem("T_Rthjc Curve", {
                    "graph_t_rthjc": np.array(thermal_foster_diode["graph_t_rthjc"])})

            list = transistor_diode_dict["channel"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + " °C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_diode_added_curves_channel_data.addItem(
                        f"Diode Channel Curve, T_j = {t_j}, V_g = {v_g}", list[i])

            list = transistor_diode_dict["e_rr"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + "° C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                r_g = str(list[i]["r_g"]) + " Ω" if list[i]["r_g"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + " V" if list[i]["v_supply"] is not None else "None "
                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_diode_added_curves_switching_losses.addItem(
                        f"T_j = {t_j}, V_g = {v_g}, R_g = {r_g}, V_supply = {v_supply}", list[i])

            list = transistor_switch_dict["e_on_meas"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + "° C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                v_g_off = str(list[i]["v_g_off"]) + " V" if list[i]["v_g_off"] is not None else "None"
                r_g = str(list[i]["r_g"]) + " Ω" if list[i]["r_g"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + " V" if list[i]["v_supply"] is not None else "None "
                load_inductance = str(list[i]["load_inductance"]) + " F" if list[i]["load_inductance"] is not None else "None "
                commutation_inductance = str(list[i]["commutation_inductance"]) + " F" if list[i]["commutation_inductance"] is not None else "None "
                measurement_date = str(list[i]["measurement_date"]) if list[i]["measurement_date"] is not None else "None"
                measurement_testbench = str(list[i]["measurement_testbench"]) if list[i]["measurement_testbench"] is not None else "None"
                commutation_device = str(list[i]["commutation_device"]) if list[i]["commutation_device"] is not None else "None"
                comment = str(list[i]["comment"]) if list[i]["comment"] is not None else "None"

                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_added_dpt.addItem(
                        f"E_on: T_j = {t_j}, V_g = {v_g}, V_g_off = {v_g_off}, R_g = {r_g}, V_supply = {v_supply}\n"
                        f"Load Inductance = {load_inductance}, Commutation Inductance = {commutation_inductance}\n"
                        f"Measurement Date: {measurement_date}\nMeasurement Testbench: {measurement_testbench}\n"
                        f"Commutation Device: {commutation_device}\nComment: {comment}", list[i])

            list = transistor_switch_dict["e_off_meas"]
            list = self.convert_graph_key_in_list_to_array(list)
            for i in range(len(list)):
                t_j = str(list[i]["t_j"]) + "° C" if list[i]["t_j"] is not None else "None"
                v_g = str(list[i]["v_g"]) + " V" if list[i]["v_g"] is not None else "None"
                v_g_off = str(list[i]["v_g_off"]) + " V" if list[i]["v_g_off"] is not None else "None"
                r_g = str(list[i]["r_g"]) + " Ω" if list[i]["r_g"] is not None else "None"
                v_supply = str(list[i]["v_supply"]) + " V" if list[i]["v_supply"] is not None else "None "
                load_inductance = str(list[i]["load_inductance"]) + " F" if list[i]["load_inductance"] is not None else "None "
                commutation_inductance = str(list[i]["commutation_inductance"]) + " F" if list[i]["commutation_inductance"] is not None else "None "
                measurement_date = str(list[i]["measurement_date"]) if list[i]["measurement_date"] is not None else "None"
                measurement_testbench = str(list[i]["measurement_testbench"]) if list[i]["measurement_testbench"] is not None else "None"
                commutation_device = str(list[i]["commutation_device"]) if list[i]["commutation_device"] is not None else "None"
                comment = str(list[i]["comment"]) if list[i]["comment"] is not None else "None"

                if list[i] is not None and list[i] != []:
                    self.comboBox_create_transistor_added_dpt.addItem(
                        f"E_off: T_j = {t_j}, V_g = {v_g}, V_g_off = {v_g_off}, R_g = {r_g}, V_supply = {v_supply}\n"
                        f"Load Inductance = {load_inductance}, Commutation Inductance = {commutation_inductance}\n"
                        f"Measurement Date: {measurement_date}\nMeasurement Testbench: {measurement_testbench}\n"
                        f"Commutation Device: {commutation_device}\nComment: {comment}", list[i])

            raw_measurement_data = transistor_dict["raw_measurement_data"]
            e_on_meas = transistor_switch_dict["e_on_meas"]
            e_off_meas = transistor_switch_dict["e_off_meas"]
            dpt_data = {"raw_measurement_data": raw_measurement_data,
                        "e_on_meas": e_on_meas,
                        "e_off_meas": e_off_meas}
            if raw_measurement_data is not None and raw_measurement_data != []:
                self.comboBox_create_transistor_added_dpt.addItem("All DPT Measurement Data", dpt_data)

            # Clear all lineEdits containing "None"
            for widget in self.scrollAreaWidgetContents_create_transistor.children():
                if isinstance(widget, QLineEdit):
                    if widget.text() == "None":
                        widget.clear()

            self.show_popup_message(f"<b>{transistor.name}</b> successfully loaded into Create Transistor!")
        except:
            self.show_popup_message("Error: No transistor selected!")

    def load_from_search_database_into_exporting_tools(self):
        """
        Load a selected transistor from search database into exporting tools.

        :return: None
        """
        try:
            for i in range(self.tableWidget_search_database.columnCount()):
                if self.tableWidget_search_database.horizontalHeaderItem(i).text() == "NAME":
                    column = i
            selected_transistor_name = self.tableWidget_search_database.item(
                self.tableWidget_search_database.currentRow(), column).text()

            self.comboBox_export_transistor.setCurrentText(selected_transistor_name)
            self.show_popup_message(f"<b>{selected_transistor_name}</b> successfully loaded into Exporting Tools!")
        except:
            self.show_popup_message("Error: No transistor selected!")

    def load_from_search_database_into_comparison_tools(self):
        """
        Load a selected transistor from search database into comparison tools.

        :return: None
        """
        try:
            for i in range(self.tableWidget_search_database.columnCount()):
                if self.tableWidget_search_database.horizontalHeaderItem(i).text() == "NAME":
                    column = i
            selected_transistor_name = self.tableWidget_search_database.item(
                self.tableWidget_search_database.currentRow(), column).text()

            target_transistor = self.comboBox_search_database_load_comparison_tools.currentText()

            if target_transistor == "Transistor1":
                self.comboBox_compare_transistor1.setCurrentText(selected_transistor_name)

            if target_transistor == "Transistor2":
                self.comboBox_compare_transistor2.setCurrentText(selected_transistor_name)

            if target_transistor == "Transistor3":
                self.comboBox_compare_transistor3.setCurrentText(selected_transistor_name)

            self.show_popup_message(
                f"<b>{selected_transistor_name}</b> successfully loaded as {target_transistor} into Comparison Tools!")
        except:
            self.show_popup_message("Error: No transistor selected!")

    def load_from_search_database_into_topology_calculator(self):
        """
        Load a selected transistor from search database into topology calculator.

        :return: None
        """
        try:
            for i in range(self.tableWidget_search_database.columnCount()):
                if self.tableWidget_search_database.horizontalHeaderItem(i).text() == "NAME":
                    column = i
            selected_transistor_name = self.tableWidget_search_database.item(
                self.tableWidget_search_database.currentRow(), column).text()

            target_transistor = self.comboBox_search_database_load_topology_calculator.currentText()

            if target_transistor == "Transistor1":
                self.comboBox_topology_transistor1.setCurrentText(selected_transistor_name)

            if target_transistor == "Transistor2":
                self.comboBox_topology_transistor2.setCurrentText(selected_transistor_name)

            self.show_popup_message(
                f"<b>{selected_transistor_name}</b> successfully loaded as {target_transistor} into Topology Calculator!")

        except:
            self.show_popup_message("Error: No transistor selected!")

    def check_value_is_in_between(self, min, max, value):
        """Check if value is between min and max."""
        # TODO Could be put in tdb functions?
        if not value or (float(min) < float(value) < float(max)):
            return True

        return False

    def search_database_load_data(self):
        """
        Load the data from the transistordatabase into the tableWidget while taking into account all the set filters.

        :return: None
        """
        transistor_list = self.tdb.get_transistor_names_list()
        transistordatabase = []

        for i in range(len(transistor_list)):
            transistor = self.tdb.load_transistor(transistor_list[i])

            transistor_dict = transistor.convert_to_dict()
            switch_dict = transistor.switch.convert_to_dict()
            diode_dict = transistor.diode.convert_to_dict()

            switch_dict_keys = list(switch_dict.keys())
            for i in range(len(switch_dict_keys)):
                switch_dict_keys[i] = "switch_" + switch_dict_keys[i]

            switch_dict_keys_new = switch_dict_keys
            switch_dict_keys_old = list(switch_dict.keys())

            for i in range(len(switch_dict)):
                switch_dict[switch_dict_keys_new[i]] = switch_dict.pop(switch_dict_keys_old[i])

            diode_dict_keys = list(diode_dict.keys())
            for i in range(len(diode_dict_keys)):
                diode_dict_keys[i] = "diode_" + diode_dict_keys[i]

            diode_dict_keys_new = diode_dict_keys
            diode_dict_keys_old = list(diode_dict.keys())

            for i in range(len(diode_dict)):
                diode_dict[diode_dict_keys_new[i]] = diode_dict.pop(diode_dict_keys_old[i])
            transistor_dict = {**transistor_dict, **switch_dict, **diode_dict}

            transistordatabase.append(transistor_dict)

        transistordatabase_keys = list(transistor_dict.keys())

        keys_to_remove = ["c_oss", "c_iss", "c_rss", "raw_measurement_data", "graph_v_ecoss", "c_oss_tr",
                          "c_oss_er", "diode", "switch", "switch_channel", "switch_e_on",
                          "switch_e_off", "switch_e_on_meas", "switch_e_off_meas", "switch_linearized_switch",
                          "switch_r_channel_th", "switch_charge_curve", "switch_soa", "diode_thermal_foster",
                          "switch_thermal_foster",
                          "diode_channel", "diode_e_rr", "diode_linearized_diode", "diode_soa"]

        if self.checkBox_search_database_name.isChecked() is False:
            keys_to_remove.append("name")
        if self.checkBox_search_database_type.isChecked() is False:
            keys_to_remove.append("type")
        if self.checkBox_search_database_author.isChecked() is False:
            keys_to_remove.append("author")
        if self.checkBox_search_database_technology.isChecked() is False:
            keys_to_remove.append("technology")
        if self.checkBox_search_database_template_version.isChecked() is False:
            keys_to_remove.append("template_version")
        if self.checkBox_search_database_template_date.isChecked() is False:
            keys_to_remove.append("template_date")
        if self.checkBox_search_database_creation_date.isChecked() is False:
            keys_to_remove.append("creation_date")
        if self.checkBox_search_database_last_modified.isChecked() is False:
            keys_to_remove.append("last_modified")
        if self.checkBox_search_database_comment.isChecked() is False:
            keys_to_remove.append("comment")
        if self.checkBox_search_database_datasheet_hyperlink.isChecked() is False:
            keys_to_remove.append("datasheet_hyperlink")
        if self.checkBox_search_database_datasheet_date.isChecked() is False:
            keys_to_remove.append("datasheet_date")
        if self.checkBox_search_database_datasheet_version.isChecked() is False:
            keys_to_remove.append("datasheet_version")
        if self.checkBox_search_database_housing_area.isChecked() is False:
            keys_to_remove.append("housing_area")
        if self.checkBox_search_database_cooling_area.isChecked() is False:
            keys_to_remove.append("cooling_area")
        if self.checkBox_search_database_t_c_max.isChecked() is False:
            keys_to_remove.append("t_c_max")
        if self.checkBox_search_database_r_g_int.isChecked() is False:
            keys_to_remove.append("r_g_int")
        if self.checkBox_search_database_r_g_on_recommended.isChecked() is False:
            keys_to_remove.append("r_g_on_recommended")
        if self.checkBox_search_database_r_g_off_recommended.isChecked() is False:
            keys_to_remove.append("r_g_off_recommended")
        if self.checkBox_search_database_c_oss_fix.isChecked() is False:
            keys_to_remove.append("c_oss_fix")
        if self.checkBox_search_database_c_iss_fix.isChecked() is False:
            keys_to_remove.append("c_iss_fix")
        if self.checkBox_search_database_c_rss_fix.isChecked() is False:
            keys_to_remove.append("c_rss_fix")
        if self.checkBox_search_database_housing_type.isChecked() is False:
            keys_to_remove.append("housing_type")
        if self.checkBox_search_database_manufacturer.isChecked() is False:
            keys_to_remove.append("manufacturer")
        if self.checkBox_search_database_r_th_cs.isChecked() is False:
            keys_to_remove.append("r_th_cs")
        if self.checkBox_search_database_r_th_switch_cs.isChecked() is False:
            keys_to_remove.append("r_th_switch_cs")
        if self.checkBox_search_database_r_th_diode_cs.isChecked() is False:
            keys_to_remove.append("r_th_diode_cs")
        if self.checkBox_search_database_v_abs_max.isChecked() is False:
            keys_to_remove.append("v_abs_max")
        if self.checkBox_search_database_i_abs_max.isChecked() is False:
            keys_to_remove.append("i_abs_max")
        if self.checkBox_search_database_i_cont.isChecked() is False:
            keys_to_remove.append("i_cont")
        if self.checkBox_search_database_switch_t_j_max.isChecked() is False:
            keys_to_remove.append("switch_t_j_max")
        if self.checkBox_search_database_switch_comment.isChecked() is False:
            keys_to_remove.append("switch_comment")
        if self.checkBox_search_database_switch_manufacturer.isChecked() is False:
            keys_to_remove.append("switch_manufacturer")
        if self.checkBox_search_database_switch_technology.isChecked() is False:
            keys_to_remove.append("switch_technology")
        if self.checkBox_search_database_diode_comment.isChecked() is False:
            keys_to_remove.append("diode_comment")
        if self.checkBox_search_database_diode_manufacturer.isChecked() is False:
            keys_to_remove.append("diode_manufacturer")
        if self.checkBox_search_database_diode_technology.isChecked() is False:
            keys_to_remove.append("diode_technology")
        if self.checkBox_search_database_diode_t_j_max.isChecked() is False:
            keys_to_remove.append("diode_t_j_max")

        for key in keys_to_remove:
            if key in transistordatabase_keys:
                transistordatabase_keys.remove(key)

        for transistor in transistordatabase:
            for key in transistordatabase_keys:
                if transistor[key] is None:
                    transistor[key] = 0

        if self.lineEdit_search_database_name.text() != "":
            name = self.lineEdit_search_database_name.text()
        else:
            name = ""
        if self.lineEdit_search_database_type.text() != "":
            type = self.lineEdit_search_database_type.text()
        else:
            type = ""

        if self.lineEdit_search_database_author.text() != "":
            author = self.lineEdit_search_database_author.text()
        else:
            author = ""
        if self.lineEdit_search_database_technology.text() != "":
            technology = self.lineEdit_search_database_technology.text()
        else:
            technology = ""
        if self.lineEdit_search_database_template_version.text() != "":
            template_version = self.lineEdit_search_database_template_version.text()
        else:
            template_version = ""
        if self.lineEdit_search_database_template_date.text() != "":
            template_date = self.lineEdit_search_database_template_date.text()
        else:
            template_date = ""
        if self.lineEdit_search_database_creation_date.text() != "":
            creation_date = self.lineEdit_search_database_creation_date.text()
        else:
            creation_date = ""
        if self.lineEdit_search_database_last_modified.text() != "":
            last_modified = self.lineEdit_search_database_last_modified.text()
        else:
            last_modified = ""
        if self.lineEdit_search_database_comment.text() != "":
            comment = self.lineEdit_search_database_comment.text()
        else:
            comment = ""
        if self.lineEdit_search_database_datasheet_hyperlink.text() != "":
            datasheet_hyperlink = self.lineEdit_search_database_datasheet_hyperlink.text()
        else:
            datasheet_hyperlink = ""
        if self.lineEdit_search_database_datasheet_date.text() != "":
            datasheet_date = self.lineEdit_search_database_datasheet_date.text()
        else:
            datasheet_date = ""
        if self.lineEdit_search_database_datasheet_version.text() != "":
            datasheet_version = self.lineEdit_search_database_datasheet_version.text()
        else:
            datasheet_version = ""
        if self.lineEdit_search_database_housing_type.text() != "":
            housing_type = self.lineEdit_search_database_housing_type.text()
        else:
            housing_type = ""
        if self.lineEdit_search_database_manufacturer.text() != "":
            manufacturer = self.lineEdit_search_database_manufacturer.text()
        else:
            manufacturer = ""
        if self.lineEdit_search_database_switch_comment.text() != "":
            switch_comment = self.lineEdit_search_database_switch_comment.text()
        else:
            switch_comment = ""
        if self.lineEdit_search_database_switch_manufacturer.text() != "":
            switch_manufacturer = self.lineEdit_search_database_switch_manufacturer.text()
        else:
            switch_manufacturer = ""
        if self.lineEdit_search_database_switch_technology.text() != "":
            switch_technology = self.lineEdit_search_database_switch_technology.text()
        else:
            switch_technology = ""
        if self.lineEdit_search_database_diode_comment.text() != "":
            diode_comment = self.lineEdit_search_database_diode_comment.text()
        else:
            diode_comment = ""
        if self.lineEdit_search_database_diode_manufacturer.text() != "":
            diode_manufacturer = self.lineEdit_search_database_diode_manufacturer.text()
        else:
            diode_manufacturer = ""
        if self.lineEdit_search_database_diode_technology.text() != "":
            diode_technology = self.lineEdit_search_database_diode_technology.text()
        else:
            diode_technology = ""
        if self.lineEdit_search_database_t_c_max_min.text() != "":
            t_c_max_min = self.lineEdit_search_database_t_c_max_min.text()
        else:
            t_c_max_min = -100000
        if self.lineEdit_search_database_t_c_max_max.text() != "":
            t_c_max_max = self.lineEdit_search_database_t_c_max_max.text()
        else:
            t_c_max_max = 100000
        if self.lineEdit_search_database_r_g_int_min.text() != "":
            r_g_int_min = self.lineEdit_search_database_r_g_int_min.text()
        else:
            r_g_int_min = -100000
        if self.lineEdit_search_database_r_g_int_max.text() != "":
            r_g_int_max = self.lineEdit_search_database_r_g_int_max.text()
        else:
            r_g_int_max = 100000
        if self.lineEdit_search_database_r_g_on_recommended_min.text() != "":
            r_g_on_recommended_min = self.lineEdit_search_database_r_g_on_recommended_min.text()
        else:
            r_g_on_recommended_min = -100000
        if self.lineEdit_search_database_r_g_on_recommended_max.text() != "":
            r_g_on_recommended_max = self.lineEdit_search_database_r_g_on_recommended_max.text()
        else:
            r_g_on_recommended_max = 100000
        if self.lineEdit_search_database_r_g_off_recommended_min.text() != "":
            r_g_off_recommended_min = self.lineEdit_search_database_r_g_off_recommended_min.text()
        else:
            r_g_off_recommended_min = -100000
        if self.lineEdit_search_database_r_g_off_recommended_max.text() != "":
            r_g_off_recommended_max = self.lineEdit_search_database_r_g_off_recommended_max.text()
        else:
            r_g_off_recommended_max = 100000
        if self.lineEdit_search_database_c_oss_fix_min.text() != "":
            c_oss_fix_min = self.lineEdit_search_database_c_oss_fix_min.text()
        else:
            c_oss_fix_min = -100000
        if self.lineEdit_search_database_c_oss_fix_max.text() != "":
            c_oss_fix_max = self.lineEdit_search_database_c_oss_fix_max.text()
        else:
            c_oss_fix_max = 100000
        if self.lineEdit_search_database_c_iss_fix_min.text() != "":
            c_iss_fix_min = self.lineEdit_search_database_c_iss_fix_min.text()
        else:
            c_iss_fix_min = -100000
        if self.lineEdit_search_database_c_iss_fix_max.text() != "":
            c_iss_fix_max = self.lineEdit_search_database_c_iss_fix_max.text()
        else:
            c_iss_fix_max = 100000
        if self.lineEdit_search_database_c_rss_fix_min.text() != "":
            c_rss_fix_min = self.lineEdit_search_database_c_rss_fix_min.text()
        else:
            c_rss_fix_min = -100000
        if self.lineEdit_search_database_c_rss_fix_max.text() != "":
            c_rss_fix_max = self.lineEdit_search_database_c_rss_fix_max.text()
        else:
            c_rss_fix_max = 100000
        if self.lineEdit_search_database_r_th_cs_min.text() != "":
            r_th_cs_min = self.lineEdit_search_database_r_th_cs_min.text()
        else:
            r_th_cs_min = -100000
        if self.lineEdit_search_database_r_th_cs_max.text() != "":
            r_th_cs_max = self.lineEdit_search_database_r_th_cs_max.text()
        else:
            r_th_cs_max = 100000
        if self.lineEdit_search_database_r_th_switch_cs_min.text() != "":
            r_th_switch_cs_min = self.lineEdit_search_database_r_th_switch_cs_min.text()
        else:
            r_th_switch_cs_min = -100000
        if self.lineEdit_search_database_r_th_switch_cs_max.text() != "":
            r_th_switch_cs_max = self.lineEdit_search_database_r_th_switch_cs_max.text()
        else:
            r_th_switch_cs_max = 100000

        if self.lineEdit_search_database_r_th_diode_cs_min.text() != "":
            r_th_diode_cs_min = self.lineEdit_search_database_r_th_diode_cs_min.text()
        else:
            r_th_diode_cs_min = -100000
        if self.lineEdit_search_database_r_th_diode_cs_max.text() != "":
            r_th_diode_cs_max = self.lineEdit_search_database_r_th_diode_cs_max.text()
        else:
            r_th_diode_cs_max = 100000

        if self.lineEdit_search_database_v_abs_max_min.text() != "":
            v_abs_max_min = self.lineEdit_search_database_v_abs_max_min.text()
        else:
            v_abs_max_min = -100000
        if self.lineEdit_search_database_v_abs_max_max.text() != "":
            v_abs_max_max = self.lineEdit_search_database_v_abs_max_max.text()
        else:
            v_abs_max_max = 100000

        if self.lineEdit_search_database_i_abs_max_min.text() != "":
            i_abs_max_min = self.lineEdit_search_database_i_abs_max_min.text()
        else:
            i_abs_max_min = -100000
        if self.lineEdit_search_database_i_abs_max_max.text() != "":
            i_abs_max_max = self.lineEdit_search_database_i_abs_max_max.text()
        else:
            i_abs_max_max = 100000
        if self.lineEdit_search_database_i_cont_min.text() != "":
            i_cont_min = self.lineEdit_search_database_i_cont_min.text()
        else:
            i_cont_min = -100000
        if self.lineEdit_search_database_i_cont_max.text() != "":
            i_cont_max = self.lineEdit_search_database_i_cont_max.text()
        else:
            i_cont_max = 100000
        if self.lineEdit_search_database_switch_t_j_max_min.text() != "":
            switch_t_j_max_min = self.lineEdit_search_database_switch_t_j_max_min.text()
        else:
            switch_t_j_max_min = -100000
        if self.lineEdit_search_database_switch_t_j_max_max.text() != "":
            switch_t_j_max_max = self.lineEdit_search_database_switch_t_j_max_max.text()
        else:
            switch_t_j_max_max = 100000
        if self.lineEdit_search_database_diode_t_j_max_min.text() != "":
            diode_t_j_max_min = self.lineEdit_search_database_diode_t_j_max_min.text()
        else:
            diode_t_j_max_min = -100000
        if self.lineEdit_search_database_diode_t_j_max_max.text() != "":
            diode_t_j_max_max = self.lineEdit_search_database_diode_t_j_max_max.text()
        else:
            diode_t_j_max_max = 100000
        if self.lineEdit_search_database_housing_area_min.text() != "":
            housing_area_min = self.lineEdit_search_database_housing_area_min.text()
        else:
            housing_area_min = -100000
        if self.lineEdit_search_database_housing_area_max.text() != "":
            housing_area_max = self.lineEdit_search_database_housing_area_max.text()
        else:
            housing_area_max = 100000
        if self.lineEdit_search_database_cooling_area_min.text() != "":
            cooling_area_min = self.lineEdit_search_database_cooling_area_min.text()
        else:
            cooling_area_min = -100000
        if self.lineEdit_search_database_cooling_area_max.text() != "":
            cooling_area_max = self.lineEdit_search_database_cooling_area_max.text()
        else:
            cooling_area_max = 100000
        # TODO Currently it is not possible to filter for None values. Should this be an option?

        transistordatabase_filtered = []
        for i in range(len(transistordatabase)):
            if type.lower() in str(transistordatabase[i]["type"]).lower() and \
                    name.lower() in str(transistordatabase[i]["name"]).lower() and \
                    author.lower() in str(transistordatabase[i]["author"]).lower() and \
                    technology.lower() in str(transistordatabase[i]["technology"]).lower() and \
                    template_version.lower() in str(transistordatabase[i]["template_version"]).lower() and \
                    template_date.lower() in str(transistordatabase[i]["template_date"]).lower() and \
                    creation_date.lower() in str(transistordatabase[i]["creation_date"]).lower() and \
                    last_modified.lower() in str(transistordatabase[i]["last_modified"]).lower() and \
                    comment.lower() in str(transistordatabase[i]["comment"]).lower() and \
                    datasheet_hyperlink.lower() in str(transistordatabase[i]["datasheet_hyperlink"]).lower() and \
                    datasheet_date.lower() in str(transistordatabase[i]["datasheet_date"]).lower() and \
                    datasheet_version.lower() in str(transistordatabase[i]["datasheet_version"]).lower() and \
                    housing_type.lower() in str(transistordatabase[i]["housing_type"]).lower() and \
                    manufacturer.lower() in str(transistordatabase[i]["manufacturer"]).lower() and \
                    switch_comment.lower() in str(transistordatabase[i]["switch_comment"]).lower() and \
                    switch_manufacturer.lower() in str(transistordatabase[i]["switch_manufacturer"]).lower() and \
                    switch_technology.lower() in str(transistordatabase[i]["switch_technology"]).lower() and \
                    diode_comment.lower() in str(transistordatabase[i]["diode_comment"]).lower() and \
                    diode_manufacturer.lower() in str(transistordatabase[i]["diode_manufacturer"]).lower() and \
                    diode_technology.lower() in str(transistordatabase[i]["diode_technology"]).lower() and \
                    self.check_value_is_in_between(i_abs_max_min, i_abs_max_max, transistordatabase[i]["i_abs_max"]) and \
                    self.check_value_is_in_between(t_c_max_min, t_c_max_max, transistordatabase[i]["t_c_max"]) and \
                    self.check_value_is_in_between(r_g_int_min, r_g_int_max, transistordatabase[i]["r_g_int"]) and \
                    self.check_value_is_in_between(r_g_on_recommended_min, r_g_on_recommended_max, transistordatabase[i]["r_g_on_recommended"]) and \
                    self.check_value_is_in_between(r_g_off_recommended_min, r_g_off_recommended_max, transistordatabase[i]["r_g_off_recommended"]) and \
                    self.check_value_is_in_between(c_oss_fix_min, c_oss_fix_max, transistordatabase[i]["c_oss_fix"]) and \
                    self.check_value_is_in_between(c_iss_fix_min, c_iss_fix_max, transistordatabase[i]["c_iss_fix"]) and \
                    self.check_value_is_in_between(c_rss_fix_min, c_rss_fix_max, transistordatabase[i]["c_rss_fix"]) and \
                    self.check_value_is_in_between(r_th_cs_min, r_th_cs_max, transistordatabase[i]["r_th_cs"]) and \
                    self.check_value_is_in_between(r_th_switch_cs_min, r_th_switch_cs_max, transistordatabase[i]["r_th_switch_cs"]) and \
                    self.check_value_is_in_between(r_th_diode_cs_min, r_th_diode_cs_max, transistordatabase[i]["r_th_diode_cs"]) and \
                    self.check_value_is_in_between(v_abs_max_min, v_abs_max_max, transistordatabase[i]["v_abs_max"]) and \
                    self.check_value_is_in_between(i_cont_min, i_cont_max, transistordatabase[i]["i_cont"]) and \
                    self.check_value_is_in_between(switch_t_j_max_min, switch_t_j_max_max, transistordatabase[i]["switch_t_j_max"]) and \
                    self.check_value_is_in_between(diode_t_j_max_min, diode_t_j_max_max, transistordatabase[i]["diode_t_j_max"]) and \
                    self.check_value_is_in_between(housing_area_min, housing_area_max, transistordatabase[i]["housing_area"]) and \
                    self.check_value_is_in_between(cooling_area_min, cooling_area_max, transistordatabase[i]["cooling_area"]):
                transistordatabase_filtered.append(transistordatabase[i])

        print(len(transistordatabase_filtered))
        for transistor in transistordatabase_filtered:
            for key in keys_to_remove:
                del transistor[key]

            for key in transistordatabase_keys:
                if transistor[key] == 0:
                    transistor[key] = ""

        transistordatabase_keys_upper = []
        unit = ""
        for key in transistordatabase_keys:
            if key == "t_c_max" or key == "switch_t_j_max" or key == "diode_t_j_max":
                unit = " [°C]"
            if key == "housing_area" or key == "cooling_area":
                unit = " [m²]"
            if key == "r_g_int" or key == "r_g_on_recommended" or key == "r_g_off_recommended" or \
                    key == "r_th_cs" or key == "r_th_switch_cs" or key == "r_th_diode_cs":
                unit = " [Ω]"
            if key == "c_oss_fix" or key == "c_iss_fix" or key == "c_rss_fix":
                unit = " [F]"
            if key == "v_abs_max":
                unit = " [V]"
            if key == "i_abs_max" or key == "i_cont":
                unit = " [A]"
            transistordatabase_keys_upper.append(
                str(transistordatabase_keys[transistordatabase_keys.index(key)].upper()) + str(unit))
            unit = ""

        row = 0
        column = 0
        self.tableWidget_search_database.setRowCount(len(transistordatabase_filtered))
        self.tableWidget_search_database.setColumnCount(len(transistordatabase_keys))
        self.tableWidget_search_database.setHorizontalHeaderLabels(transistordatabase_keys_upper)
        for transistor in transistordatabase_filtered:
            for key in transistordatabase_keys:
                item = QtWidgets.QTableWidgetItem()
                try:
                    item.setData(QtCore.Qt.DisplayRole, float(transistor[key]))
                except:
                    item.setData(QtCore.Qt.DisplayRole, str(transistor[key]))
                self.tableWidget_search_database.setItem(row, column, item)
                column = column + 1
