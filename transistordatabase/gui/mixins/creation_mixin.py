"""Transistor creation mixin for MainWindow."""
import datetime
import os
import tempfile
import webbrowser

import numpy as np

from PyQt5.QtWidgets import QLineEdit, QComboBox

from transistordatabase.transistor import Transistor
from transistordatabase.checker_functions import csv2array
from transistordatabase.helper_functions import merge_curve


def _get_information_window():
    """Deferred import to avoid circular dependency with gui.py."""
    from transistordatabase.gui.gui import InformationWindow
    return InformationWindow


class TransistorCreationMixin:
    """Mixin providing transistor creation functionality for MainWindow."""

    def comboBox_create_transistor_switch_add_switching_losses_curve_type_changed(self):
        """Check for changed comboBox."""
        if self.comboBox_create_transistor_switch_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Channel Current":
            self.label_create_switch_switching_losses_r_g_i_x.setText("R_g [Ω]")
        elif self.comboBox_create_transistor_switch_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Gate Resistor":
            self.label_create_switch_switching_losses_r_g_i_x.setText(r"I [A]")

    def comboBox_create_transistor_diode_add_switching_losses_curve_type_changed(self):
        """Check for changed comboBox."""
        if self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Channel Current":
            self.label_create_diode_switching_losses_r_g_i_x.setText("R_g [Ω]")
        elif self.comboBox_create_transistor_diode_add_switching_losses_curve_type.currentText() == "Switching Losses vs. Gate Resistor":
            self.label_create_diode_switching_losses_r_g_i_x.setText(r"I [A]")

    def fill_comboBoxes_create_transistor(self):
        """
        Fill the ComboBoxes containing the transistor types, housing types and manufacturers list.

        :return: None
        """
        self.comboBox_create_transistor_add_data_dpt_dataset_type.addItems(["I_E Curve", "R_E Curve"])
        self.comboBox_create_transistor_add_data_dpt_energies.addItems(["both", "e_on", "e_off"])

        self.comboBox_create_transistor_switch_add_switching_losses_curve_type.addItems(
            ["Switching Losses vs. Channel Current", "Switching Losses vs. Gate Resistor"])
        self.comboBox_create_transistor_diode_add_switching_losses_curve_type.addItems(
            ["Switching Losses vs. Channel Current", "Switching Losses vs. Gate Resistor"])
        self.comboBox_create_transistor_switch_add_switching_losses_on_off.addItems(["E_on", "E_off"])

        self.comboBox_create_transistor_type.addItems(["MOSFET", "IGBT", "SiC-MOSFET", "GaN-Transistor"])

        self.comboBox_create_transistor_housing_type.addItems(self.tdb.housing_types)
        self.comboBox_create_transistor_manufacturer.addItems(self.tdb.module_manufacturers)
        self.comboBox_create_transistor_switch_manufacturer.addItems([""] + self.tdb.module_manufacturers)
        self.comboBox_create_transistor_diode_manufacturer.addItems([""] + self.tdb.module_manufacturers)
        self.comboBox_create_transistor_add_data_dpt_integration_interval.addItems(
            [self.translation_dict['IEC 60747-9'], self.translation_dict['IEC 60747-8'],
             self.translation_dict['Mitsubishi'], self.translation_dict['Infineon'],
             self.translation_dict['Wolfspeed']])

    def clear_create_transistor(self):
        """
        Clear all inputs on the Create Transistor tab.

        :return: None
        """
        for widget in self.scrollAreaWidgetContents_create_transistor.children():
            if isinstance(widget, QLineEdit) or isinstance(widget, QComboBox):
                widget.clear()
        self.fill_comboBoxes_create_transistor()

    def get_all_items_text_from_comboBox(self, comboBox):
        """
        Return a list of all items stored in a ComboBox.

        :param comboBox: comboBox object
        :return: list of all items text stored in a comboBox or empty list if comboBox is empty
        """
        if comboBox.count != 0:
            all_items_text = [comboBox.itemText(i) for i in range(comboBox.count())]
            return all_items_text
        else:
            return []

    def get_all_items_data_from_comboBox(self, comboBox):
        """
        Return a list of all values connected to the texts in a comboBox.

        :param comboBox: comboBox object
        :return: list of all values stored in a comboBox or empty list if comboBox is empty
        """
        if comboBox.count != 0:
            all_items_data = [comboBox.itemData(i) for i in range(comboBox.count())]
            return all_items_data
        else:
            return []

    def create_transistor(self):
        """
        Fill the transistor template with all inputs from the Create Transistor tab and creates a transistor object.

        Missing inputs are set to None or empty lists in case of curves.

        :return: transistor object
        """
        # TRANSISTOR PARAMETERS #

        try:
            c_iss_normal_dict = self.comboBox_create_transistor_added_c_iss_normal.itemData(
                self.comboBox_create_transistor_added_c_iss_normal.currentIndex())
            c_iss_normal = c_iss_normal_dict["graph_v_c"]

            c_iss_detail_dict = self.comboBox_create_transistor_added_c_iss_detail.itemData(
                self.comboBox_create_transistor_added_c_iss_detail.currentIndex())
            c_iss_detail = c_iss_detail_dict["graph_v_c"]

            c_iss_merged = merge_curve(c_iss_normal, c_iss_detail)

            c_iss = {"t_j": float(c_iss_normal_dict["t_j"]),
                     "graph_v_c": c_iss_merged}
        except:
            try:
                c_iss = c_iss_normal_dict
            except:
                c_iss = None

        try:
            c_oss_normal_dict = self.comboBox_create_transistor_added_c_oss_normal.itemData(
                self.comboBox_create_transistor_added_c_oss_normal.currentIndex())
            c_oss_normal = c_oss_normal_dict["graph_v_c"]

            c_oss_detail_dict = self.comboBox_create_transistor_added_c_oss_detail.itemData(
                self.comboBox_create_transistor_added_c_oss_detail.currentIndex())
            c_oss_detail = c_oss_detail_dict["graph_v_c"]

            c_oss_merged = merge_curve(c_oss_normal, c_oss_detail)

            c_oss = {"t_j": float(c_oss_normal_dict["t_j"]),
                     "graph_v_c": c_oss_merged}
        except:
            try:
                c_oss = c_oss_normal_dict
            except:
                c_oss = None

        try:
            c_rss_normal_dict = self.comboBox_create_transistor_added_c_rss_normal.itemData(
                self.comboBox_create_transistor_added_c_rss_normal.currentIndex())
            c_rss_normal = c_iss_normal_dict["graph_v_c"]

            c_rss_detail_dict = self.comboBox_create_transistor_added_c_rss_detail.itemData(
                self.comboBox_create_transistor_added_c_rss_detail.currentIndex())
            c_rss_detail = c_rss_detail_dict["graph_v_c"]

            c_rss_merged = merge_curve(c_rss_normal, c_rss_detail)

            c_rss = {"t_j": float(c_rss_normal_dict["t_j"]),
                     "graph_v_c": c_rss_merged}
        except:
            try:
                c_rss = c_rss_normal_dict
            except:
                c_rss = None

        try:
            c_oss_er = {"c_o": float(self.lineEdit_create_transistor_transistor_c_oss_er_c_o.text()),
                        "v_gs": float(self.lineEdit_create_transistor_transistor_c_oss_er_v_gs.text()),
                        "v_ds": float(self.lineEdit_create_transistor_transistor_c_oss_er_v_ds.text())}
            c_oss_tr = {"c_o": float(self.lineEdit_create_transistor_transistor_c_oss_tr_c_o.text()),
                        "v_gs": float(self.lineEdit_create_transistor_transistor_c_oss_tr_v_gs.text()),
                        "v_ds": float(self.lineEdit_create_transistor_transistor_c_oss_tr_v_ds.text())}
        except:
            c_oss_er = None
            c_oss_tr = None

        try:
            v_ecoss_dict = self.comboBox_create_transistor_added_curve_v_ecoss.itemData(
                self.comboBox_create_transistor_added_curve_v_ecoss.currentIndex())
            v_ecoss = v_ecoss_dict["graph_v_ecoss"]
        except:
            v_ecoss = None

        try:
            housing_area = float(self.lineEdit_create_transistor_housing_area.text())
        except:
            housing_area = None

        try:
            cooling_area = float(self.lineEdit_create_transistor_cooling_area.text())
        except:
            cooling_area = None

        try:
            v_abs_max = float(self.lineEdit_create_transistor_v_abs_max.text())
        except:
            v_abs_max = None

        try:
            i_abs_max = float(self.lineEdit_create_transistor_i_abs_max.text())
        except:
            i_abs_max = None

        try:
            i_cont = float(self.lineEdit_create_transistor_i_cont.text())
        except:
            i_cont = None

        try:
            c_iss_fix = float(self.lineEdit_create_transistor_c_iss_fix.text())
        except:
            c_iss_fix = None

        try:
            c_oss_fix = float(self.lineEdit_create_transistor_c_oss_fix.text())
        except:
            c_oss_fix = None

        try:
            c_rss_fix = float(self.lineEdit_create_transistor_c_rss_fix.text())
        except:
            c_rss_fix = None

        try:
            r_g_int = float(self.lineEdit_create_transistor_r_g_int.text())
        except:
            r_g_int = None

        try:
            r_th_cs = float(self.lineEdit_create_transistor_r_th_cs.text())
        except:
            r_th_cs = None

        try:
            r_th_diode_cs = float(self.lineEdit_create_transistor_r_th_diode_cs.text())
        except:
            r_th_diode_cs = None

        try:
            r_th_switch_cs = float(self.lineEdit_create_transistor_r_th_switch_cs.text())
        except:
            r_th_switch_cs = None

        try:
            r_g_on_recommended = float(self.lineEdit_create_transistor_r_g_on_recommended.text())
        except:
            r_g_on_recommended = None

        try:
            r_g_off_recommended = float(self.lineEdit_create_transistor_r_g_off_recommended.text())
        except:
            r_g_off_recommended = None

        try:
            t_c_max = float(self.lineEdit_create_transistor_t_c_max.text())
        except:
            t_c_max = None

        # get currently existing dpt data
        raw_measurement_data = None
        e_on_meas = None
        e_off_meas = None
        for i in range(self.comboBox_create_transistor_added_dpt.count()):
            if self.comboBox_create_transistor_added_dpt.itemText(i) == "All DPT Measurement Data":
                dpt_measurement_data = self.comboBox_create_transistor_added_dpt.itemData(i)
                raw_measurement_data = dpt_measurement_data["raw_measurement_data"]
                e_on_meas = dpt_measurement_data["e_on_meas"]
                e_off_meas = dpt_measurement_data["e_off_meas"]

        transistor_args = {'name': self.lineEdit_create_transistor_name.text(),
                           'type': self.comboBox_create_transistor_type.currentText(),
                           'author': self.lineEdit_create_transistor_author.text(),
                           'comment': self.lineEdit_create_transistor_comment.text(),
                           'manufacturer': self.comboBox_create_transistor_manufacturer.currentText(),
                           'datasheet_hyperlink': self.lineEdit_create_transistor_datasheet_hyperlink.text(),
                           'datasheet_date': self.lineEdit_create_transistor_datasheet_date.text(),
                           'datasheet_version': self.lineEdit_create_transistor_datasheet_version.text(),
                           'housing_area': housing_area,
                           'cooling_area': cooling_area,
                           'housing_type': self.comboBox_create_transistor_housing_type.currentText(),
                           'v_abs_max': v_abs_max,
                           'i_abs_max': i_abs_max,
                           'i_cont': i_cont,
                           'c_iss': c_iss,
                           'c_oss': c_oss,
                           'c_rss': c_rss,
                           'c_oss_er': c_oss_er,
                           'c_oss_tr': c_oss_tr,
                           'c_iss_fix': c_iss_fix,
                           'c_oss_fix': c_oss_fix,
                           'c_rss_fix': c_rss_fix,
                           'graph_v_ecoss': v_ecoss,
                           'r_g_int': r_g_int,
                           'r_th_cs': r_th_cs,
                           'r_th_diode_cs': r_th_diode_cs,
                           'r_th_switch_cs': r_th_switch_cs,
                           "r_g_on_recommended": r_g_on_recommended,
                           "r_g_off_recommended": r_g_off_recommended,
                           "t_c_max": t_c_max,
                           "raw_measurement_data": raw_measurement_data}

        # SWITCH PARAMETERS #
        try:
            t_rthjc_dict = self.comboBox_create_transistor_switch_added_curve_t_rthjc.itemData(
                self.comboBox_create_transistor_switch_added_curve_t_rthjc.currentIndex())
            t_rthjc = t_rthjc_dict["graph_t_rthjc"]
        except:
            t_rthjc = None

        try:
            r_th_total_switch = float(self.lineEdit_create_transistor_switch_r_th_total.text())
        except:
            r_th_total_switch = None

        try:
            c_th_total = float(self.lineEdit_create_transistor_switch_c_th_total.text())
        except:
            c_th_total = None

        try:
            tau_total = float(self.lineEdit_create_transistor_switch_tau_total.text())
        except:
            tau_total = None

        switch_foster_args = {
            'r_th_vector': list(map(float, self.lineEdit_create_transistor_switch_r_th_vector.text().split())),
            'r_th_total': r_th_total_switch,
            'c_th_vector': list(map(float, self.lineEdit_create_transistor_switch_c_th_vector.text().split())),
            'c_th_total': c_th_total,
            'tau_vector': list(map(float, self.lineEdit_create_transistor_switch_tau_vector.text().split())),
            'tau_total': tau_total,
            'graph_t_rthjc': t_rthjc}

        for key in switch_foster_args.keys():
            if switch_foster_args[key] == [] or switch_foster_args[key] == "":
                switch_foster_args[key] = None

        e_on_off_list = self.get_all_items_data_from_comboBox(
            self.comboBox_create_transistor_switch_added_curves_switching_losses)
        e_on_list = []
        e_off_list = []
        for i in range(len(e_on_off_list)):
            if e_on_off_list[i]["e_on_off"] == "e_on":
                del e_on_off_list[i]["e_on_off"]
                e_on_list.append(e_on_off_list[i])
            elif e_on_off_list[i]["e_on_off"] == "e_off":
                del e_on_off_list[i]["e_on_off"]
                e_off_list.append(e_on_off_list[i])

        try:
            t_j_max_switch = float(self.lineEdit_create_transistor_switch_t_j_max.text())
        except:
            t_j_max_switch = None

        switch_args = {
            'comment': self.lineEdit_create_transistor_switch_comment.text(),
            'manufacturer': self.comboBox_create_transistor_switch_manufacturer.currentText(),
            'technology': self.lineEdit_create_transistor_switch_technology.text(),
            't_j_max': t_j_max_switch,
            'channel': self.get_all_items_data_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_channel_data),
            'e_on': e_on_list,
            'e_off': e_off_list,
            'charge_curve': self.get_all_items_data_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_gate_charge),
            'r_channel_th': self.get_all_items_data_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_r_on),
            'thermal_foster': switch_foster_args,
            'soa': self.get_all_items_data_from_comboBox(
                self.comboBox_create_transistor_switch_added_curves_soa_t_pulse),
            "e_on_meas": e_on_meas,
            "e_off_meas": e_off_meas}

        # DIODE PARAMATERS #
        try:
            t_rthjc_dict = self.comboBox_create_transistor_diode_added_curve_t_rthjc.itemData(
                self.comboBox_create_transistor_diode_added_curve_t_rthjc.currentIndex())
            t_rthjc = t_rthjc_dict["graph_t_rthjc"]
        except:
            t_rthjc = None

        try:
            r_th_total_diode = float(self.lineEdit_create_transistor_diode_r_th_total.text())
        except:
            r_th_total_diode = None

        try:
            c_th_total = float(self.lineEdit_create_transistor_diode_c_th_total.text())
        except:
            c_th_total = None

        try:
            tau_total = float(self.lineEdit_create_transistor_diode_tau_total.text())
        except:
            tau_total = None

        diode_foster_args = {
            'r_th_vector': list(map(float, self.lineEdit_create_transistor_diode_r_th_vector.text().split())),
            'r_th_total': r_th_total_diode,
            'c_th_vector': list(map(float, self.lineEdit_create_transistor_diode_c_th_vector.text().split())),
            'c_th_total': c_th_total,
            'tau_vector': list(map(float, self.lineEdit_create_transistor_diode_tau_vector.text().split())),
            'tau_total': tau_total,
            'graph_t_rthjc': t_rthjc}

        for key in list(diode_foster_args.keys()):
            if diode_foster_args[key] == [] or diode_foster_args[key] == "":
                diode_foster_args[key] = None

        try:
            t_j_max_diode = float(self.lineEdit_create_transistor_diode_t_j_max.text())
        except:
            t_j_max_diode = None

        diode_args = {'comment': self.lineEdit_create_transistor_diode_comment.text(),
                      'manufacturer': self.comboBox_create_transistor_diode_manufacturer.currentText(),
                      'technology': self.lineEdit_create_transistor_diode_technology.text(),
                      't_j_max': t_j_max_diode,
                      'channel': self.get_all_items_data_from_comboBox(
                          self.comboBox_create_transistor_diode_added_curves_channel_data),
                      'e_rr': self.get_all_items_data_from_comboBox(
                          self.comboBox_create_transistor_diode_added_curves_switching_losses),
                      'thermal_foster': diode_foster_args,
                      'soa': self.get_all_items_data_from_comboBox(
                          self.comboBox_create_transistor_diode_added_curves_soa_t_pulse)}

        transistor = Transistor(transistor_args, switch_args, diode_args)

        # Add new DPT data to new transistor object
        data_list = self.get_all_items_data_from_comboBox(self.comboBox_create_transistor_added_dpt)
        for data in data_list:
            try:
                transistor.add_dpt_measurement(data["new_dpt_dict"])
            except:
                pass

        dict = {"transistor_args": transistor_args, "switch_args": switch_args, "diode_args": diode_args,
                "transistor": transistor}

        return dict

    def load_transistor_into_local_database(self):
        """
        Check if a transistor already exists and opens an information window to ask user if he wants to overwrite the transistor.

        Else saves the new transistor to local database.

        :return: None
        """
        try:
            dict = self.create_transistor()
            transistor = dict["transistor"]

            existing_transistor_list = self.get_transistor_list()
            check = False
            for existing_transistor in existing_transistor_list:
                if existing_transistor == transistor.name:
                    check = True
                    self.InformationWindow = _get_information_window()()
                    self.InformationWindow.run_information_window()
            if check is False:
                transistor.save()

                self.comboBox_export_transistor.addItem(transistor.name)
                self.comboBox_compare_transistor1.addItem(transistor.name)
                self.comboBox_compare_transistor2.addItem(transistor.name)
                self.comboBox_compare_transistor3.addItem(transistor.name)
                self.comboBox_topology_transistor1.addItem(transistor.name)
                self.comboBox_topology_transistor2.addItem(transistor.name)

                self.search_database_load_data()
                self.show_popup_message(f"<b>{transistor.name}</b> succsessfully created!")
                self.button_create_transistor_create.setDisabled(True)
        except:
            self.show_popup_message("Error: Transistor could not be created! Check if all inputs are correct!")

    def overwrite_transistor(self):
        """
        Overwrite a transistor stored in the local database.

        :return: None
        """
        dict = self.create_transistor()
        transistor_new = Transistor(dict["transistor_args"], dict["switch_args"], dict["diode_args"])

        # Add new DPT data to overwritten transistor object
        data_list = self.get_all_items_data_from_comboBox(self.comboBox_create_transistor_added_dpt)
        for data in data_list:
            try:
                transistor_new.add_dpt_measurement(data["new_dpt_dict"])
            except:
                pass

        self.tdb.save_transistor(transistor_new, True)
        self.search_database_load_data()
        self.show_popup_message(f"Transistor <b>{transistor_new.name}</b> succsessfully overwritten!")
        self.button_create_transistor_create.setDisabled(True)

    def preview_transistor_on_virtual_datasheet(self):
        """
        Previews the transistor with paramaters given in the Create Transistor Tab on a virtual datasheet.

        :return:
        """
        try:
            dict = self.create_transistor()
            transistor = dict["transistor"]

            html = transistor.export_datasheet(build_collection=True)
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
                url = 'file://' + f.name
                f.write(html)
            webbrowser.open(url)
            self.button_create_transistor_create.setDisabled(False)
        except:
            self.show_popup_message("Error: Transistor could not previewed! Check if all inputs are correct!")

    def convert_graph_key_in_list_to_array(self, curve_list):
        """
        Convert the graph key in a list of curves to an numpy array so that the edited transistor can be loaded into the database.

        :return: curve list with graph key converted to a numpy array
        """
        for i in range(len(curve_list)):
            for key in curve_list[i].keys():
                if "graph" in key and curve_list[i][key] is not None:
                    array = np.array(curve_list[i][key])
                    curve_list[i][key] = array
        return curve_list
