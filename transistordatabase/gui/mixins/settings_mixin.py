"""Settings mixin for MainWindow."""
import json
import pathlib

from PyQt5.QtWidgets import QFileDialog

from transistordatabase.gui._utils import resource_path


class SettingsMixin:
    """Mixin providing settings save/load functionality for MainWindow."""

    def get_settings_dict(self):
        """
        Create and return a dict containing user inputs and settings.

        :return: all_settings_dict: dict containing the name of each widget and its value
        """
        search_database_settings_dict = {"checkBox_search_database_name": self.checkBox_search_database_name.isChecked(),
                                         "checkBox_search_database_type": self.checkBox_search_database_type.isChecked(),
                                         "checkBox_search_database_author": self.checkBox_search_database_author.isChecked(),
                                         "checkBox_search_database_technology": self.checkBox_search_database_technology.isChecked(),
                                         "checkBox_search_database_template_version": self.checkBox_search_database_template_version.isChecked(),
                                         "checkBox_search_database_template_date": self.checkBox_search_database_template_date.isChecked(),
                                         "checkBox_search_database_creation_date": self.checkBox_search_database_creation_date.isChecked(),
                                         "checkBox_search_database_last_modified": self.checkBox_search_database_last_modified.isChecked(),
                                         "checkBox_search_database_comment": self.checkBox_search_database_comment.isChecked(),
                                         "checkBox_search_database_datasheet_hyperlink": self.checkBox_search_database_datasheet_hyperlink.isChecked(),
                                         "checkBox_search_database_datasheet_date": self.checkBox_search_database_datasheet_date.isChecked(),
                                         "checkBox_search_database_datasheet_version": self.checkBox_search_database_datasheet_version.isChecked(),
                                         "checkBox_search_database_manufacturer": self.checkBox_search_database_manufacturer.isChecked(),
                                         "checkBox_search_database_housing_type": self.checkBox_search_database_housing_type.isChecked(),
                                         "checkBox_search_database_housing_area": self.checkBox_search_database_housing_area.isChecked(),
                                         "checkBox_search_database_cooling_area": self.checkBox_search_database_cooling_area.isChecked(),
                                         "checkBox_search_database_t_c_max": self.checkBox_search_database_t_c_max.isChecked(),
                                         "checkBox_search_database_r_g_int": self.checkBox_search_database_r_g_int.isChecked(),
                                         "checkBox_search_database_r_g_on_recommended": self.checkBox_search_database_r_g_on_recommended.isChecked(),
                                         "checkBox_search_database_r_g_off_recommended": self.checkBox_search_database_r_g_off_recommended.isChecked(),
                                         "checkBox_search_database_c_oss_fix": self.checkBox_search_database_c_oss_fix.isChecked(),
                                         "checkBox_search_database_c_iss_fix": self.checkBox_search_database_c_iss_fix.isChecked(),
                                         "checkBox_search_database_c_rss_fix": self.checkBox_search_database_c_rss_fix.isChecked(),
                                         "checkBox_search_database_r_th_cs": self.checkBox_search_database_r_th_cs.isChecked(),
                                         "checkBox_search_database_r_th_switch_cs": self.checkBox_search_database_r_th_switch_cs.isChecked(),
                                         "checkBox_search_database_r_th_diode_cs": self.checkBox_search_database_r_th_diode_cs.isChecked(),
                                         "checkBox_search_database_v_abs_max": self.checkBox_search_database_v_abs_max.isChecked(),
                                         "checkBox_search_database_i_abs_max": self.checkBox_search_database_i_abs_max.isChecked(),
                                         "checkBox_search_database_i_cont": self.checkBox_search_database_i_cont.isChecked(),
                                         "checkBox_search_database_switch_comment": self.checkBox_search_database_switch_comment.isChecked(),
                                         "checkBox_search_database_switch_manufacturer": self.checkBox_search_database_switch_manufacturer.isChecked(),
                                         "checkBox_search_database_switch_technology": self.checkBox_search_database_switch_technology.isChecked(),
                                         "checkBox_search_database_switch_t_j_max": self.checkBox_search_database_switch_t_j_max.isChecked(),
                                         "checkBox_search_database_diode_comment": self.checkBox_search_database_diode_comment.isChecked(),
                                         "checkBox_search_database_diode_manufacturer": self.checkBox_search_database_diode_manufacturer.isChecked(),
                                         "checkBox_search_database_diode_technology": self.checkBox_search_database_diode_technology.isChecked(),
                                         "checkBox_search_database_diode_t_j_max": self.checkBox_search_database_diode_t_j_max.isChecked(),
                                         "lineEdit_search_database_name": self.lineEdit_search_database_name.text(),
                                         "lineEdit_search_database_type": self.lineEdit_search_database_type.text(),
                                         "lineEdit_search_database_author": self.lineEdit_search_database_author.text(),
                                         "lineEdit_search_database_technology": self.lineEdit_search_database_technology.text(),
                                         "lineEdit_search_database_template_version": self.lineEdit_search_database_template_version.text(),
                                         "lineEdit_search_database_template_date": self.lineEdit_search_database_template_date.text(),
                                         "lineEdit_search_database_creation_date": self.lineEdit_search_database_creation_date.text(),
                                         "lineEdit_search_database_last_modified": self.lineEdit_search_database_last_modified.text(),
                                         "lineEdit_search_database_comment": self.lineEdit_search_database_comment.text(),
                                         "lineEdit_search_database_datasheet_hyperlink": self.lineEdit_search_database_datasheet_hyperlink.text(),
                                         "lineEdit_search_database_datasheet_date": self.lineEdit_search_database_datasheet_date.text(),
                                         "lineEdit_search_database_datasheet_version": self.lineEdit_search_database_datasheet_version.text(),
                                         "lineEdit_search_database_manufacturer": self.lineEdit_search_database_manufacturer.text(),
                                         "lineEdit_search_database_housing_type": self.lineEdit_search_database_housing_type.text(),
                                         "lineEdit_search_database_switch_comment": self.lineEdit_search_database_switch_comment.text(),
                                         "lineEdit_search_database_switch_manufacturer": self.lineEdit_search_database_switch_manufacturer.text(),
                                         "lineEdit_search_database_switch_technology": self.lineEdit_search_database_switch_technology.text(),
                                         "lineEdit_search_database_diode_comment": self.lineEdit_search_database_diode_comment.text(),
                                         "lineEdit_search_database_diode_manufacturer": self.lineEdit_search_database_diode_manufacturer.text(),
                                         "lineEdit_search_database_housing_area_min": self.lineEdit_search_database_housing_area_min.text(),
                                         "lineEdit_search_database_housing_area_max": self.lineEdit_search_database_housing_area_max.text(),
                                         "lineEdit_search_database_cooling_area_min": self.lineEdit_search_database_cooling_area_min.text(),
                                         "lineEdit_search_database_cooling_area_max": self.lineEdit_search_database_cooling_area_max.text(),
                                         "lineEdit_search_database_t_c_max_min": self.lineEdit_search_database_t_c_max_min.text(),
                                         "lineEdit_search_database_t_c_max_max": self.lineEdit_search_database_t_c_max_max.text(),
                                         "lineEdit_search_database_r_g_int_min": self.lineEdit_search_database_r_g_int_min.text(),
                                         "lineEdit_search_database_r_g_int_max": self.lineEdit_search_database_r_g_int_max.text(),
                                         "lineEdit_search_database_r_g_on_recommended_min": self.lineEdit_search_database_r_g_on_recommended_min.text(),
                                         "lineEdit_search_database_r_g_on_recommended_max": self.lineEdit_search_database_r_g_on_recommended_max.text(),
                                         "lineEdit_search_database_r_g_off_recommended_min": self.lineEdit_search_database_r_g_off_recommended_min.text(),
                                         "lineEdit_search_database_r_g_off_recommended_max": self.lineEdit_search_database_r_g_off_recommended_max.text(),
                                         "lineEdit_search_database_c_oss_fix_min": self.lineEdit_search_database_c_oss_fix_min.text(),
                                         "lineEdit_search_database_c_oss_fix_max": self.lineEdit_search_database_c_oss_fix_max.text(),
                                         "lineEdit_search_database_c_iss_fix_min": self.lineEdit_search_database_c_iss_fix_min.text(),
                                         "lineEdit_search_database_c_iss_fix_max": self.lineEdit_search_database_c_iss_fix_max.text(),
                                         "lineEdit_search_database_c_rss_fix_min": self.lineEdit_search_database_c_rss_fix_min.text(),
                                         "lineEdit_search_database_c_rss_fix_max": self.lineEdit_search_database_c_rss_fix_max.text(),
                                         "lineEdit_search_database_r_th_cs_min": self.lineEdit_search_database_r_th_cs_min.text(),
                                         "lineEdit_search_database_r_th_cs_max": self.lineEdit_search_database_r_th_cs_max.text(),
                                         "lineEdit_search_database_r_th_switch_cs_min": self.lineEdit_search_database_r_th_switch_cs_min.text(),
                                         "lineEdit_search_database_r_th_switch_cs_max": self.lineEdit_search_database_r_th_switch_cs_max.text(),
                                         "lineEdit_search_database_r_th_diode_cs_min": self.lineEdit_search_database_r_th_diode_cs_min.text(),
                                         "lineEdit_search_database_r_th_diode_cs_max": self.lineEdit_search_database_r_th_diode_cs_max.text(),
                                         "lineEdit_search_database_v_abs_max_min": self.lineEdit_search_database_v_abs_max_min.text(),
                                         "lineEdit_search_database_v_abs_max_max": self.lineEdit_search_database_v_abs_max_max.text(),
                                         "lineEdit_search_database_i_abs_max_min": self.lineEdit_search_database_i_abs_max_min.text(),
                                         "lineEdit_search_database_i_abs_max_max": self.lineEdit_search_database_i_abs_max_max.text(),
                                         "lineEdit_search_database_i_cont_min": self.lineEdit_search_database_i_cont_min.text(),
                                         "lineEdit_search_database_i_cont_max": self.lineEdit_search_database_i_cont_max.text(),
                                         "lineEdit_search_database_switch_t_j_max_min": self.lineEdit_search_database_switch_t_j_max_min.text(),
                                         "lineEdit_search_database_switch_t_j_max_max": self.lineEdit_search_database_switch_t_j_max_max.text(),
                                         "lineEdit_search_database_diode_t_j_max_min": self.lineEdit_search_database_diode_t_j_max_min.text(),
                                         "lineEdit_search_database_diode_t_j_max_max": self.lineEdit_search_database_diode_t_j_max_max.text()}

        exporting_tools_settings_dict = {"comboBox_export_transistor": self.comboBox_export_transistor.currentText(),
                                         "lineEdit_export_number_parallel_transistors": self.lineEdit_export_number_parallel_transistors.text(),
                                         "lineEdit_export_simulink_v_supply": self.lineEdit_export_simulink_v_supply.text(),
                                         "lineEdit_export_simulink_r_g_on": self.lineEdit_export_simulink_r_g_on.text(),
                                         "lineEdit_export_simulink_r_g_off": self.lineEdit_export_simulink_r_g_off.text(),
                                         "lineEdit_export_simulink_normalize_t_to_v": self.lineEdit_export_simulink_normalize_t_to_v.text(),
                                         "lineEdit_export_gecko_v_supply": self.lineEdit_export_gecko_v_supply.text(),
                                         "lineEdit_export_gecko_r_g_on": self.lineEdit_export_gecko_r_g_on.text(),
                                         "lineEdit_export_gecko_r_g_off": self.lineEdit_export_gecko_r_g_off.text(),
                                         "lineEdit_export_gecko_v_g_on": self.lineEdit_export_gecko_v_g_on.text(),
                                         "lineEdit_export_gecko_v_g_off": self.lineEdit_export_gecko_v_g_off.text()}

        comparison_tools_settings_dict = {
            "comboBox_compare_transistor1": self.comboBox_compare_transistor1.currentText(),
            "comboBox_compare_transistor2": self.comboBox_compare_transistor2.currentText(),
            "comboBox_compare_transistor3": self.comboBox_compare_transistor3.currentText(),
            "comboBox_compare_v_g_on_transistor1": self.comboBox_compare_v_g_on_transistor1.currentText(),
            "comboBox_compare_v_g_off_transistor1": self.comboBox_compare_v_g_off_transistor1.currentText(),
            "comboBox_compare_v_g_on_transistor2": self.comboBox_compare_v_g_on_transistor2.currentText(),
            "comboBox_compare_v_g_off_transistor2": self.comboBox_compare_v_g_off_transistor2.currentText(),
            "comboBox_compare_v_g_on_transistor3": self.comboBox_compare_v_g_on_transistor3.currentText(),
            "comboBox_compare_v_g_off_transistor3": self.comboBox_compare_v_g_off_transistor3.currentText(),
            "comboBox_compare_plot1": self.comboBox_compare_plot1.currentText(),
            "comboBox_compare_plot2": self.comboBox_compare_plot2.currentText(),
            "comboBox_compare_plot3": self.comboBox_compare_plot3.currentText(),
            "comboBox_compare_plot4": self.comboBox_compare_plot4.currentText(),
            "comboBox_compare_plot5": self.comboBox_compare_plot5.currentText(),
            "comboBox_compare_plot6": self.comboBox_compare_plot6.currentText(),
            "comboBox_compare_plot7": self.comboBox_compare_plot7.currentText(),
            "comboBox_compare_plot8": self.comboBox_compare_plot8.currentText(),
            "comboBox_compare_plot9": self.comboBox_compare_plot9.currentText(),
            "lineEdit_compare_t_j_transistor1": self.lineEdit_compare_t_j_transistor1.text(),
            "lineEdit_compare_t_j_transistor2": self.lineEdit_compare_t_j_transistor2.text(),
            "lineEdit_compare_t_j_transistor3": self.lineEdit_compare_t_j_transistor3.text(),
            "lineEdit_compare_v_supply_transistor1": self.lineEdit_compare_v_supply_transistor1.text(),
            "lineEdit_compare_number_parallel_transistor1": self.lineEdit_compare_number_parallel_transistor1.text(),
            "lineEdit_compare_v_supply_transistor2": self.lineEdit_compare_v_supply_transistor2.text(),
            "lineEdit_compare_number_parallel_transistor2": self.lineEdit_compare_number_parallel_transistor2.text(),
            "lineEdit_compare_v_supply_transistor3": self.lineEdit_compare_v_supply_transistor3.text(),
            "lineEdit_compare_number_parallel_transistor3": self.lineEdit_compare_number_parallel_transistor3.text(),
            "label_compare_r_g_on_value_transistor1": self.label_compare_r_g_on_value_transistor1.text(),
            "label_compare_r_g_off_value_transistor1": self.label_compare_r_g_off_value_transistor1.text(),
            "label_compare_r_g_on_value_transistor2": self.label_compare_r_g_on_value_transistor2.text(),
            "label_compare_r_g_off_value_transistor2": self.label_compare_r_g_off_value_transistor2.text(),
            "label_compare_r_g_on_value_transistor3": self.label_compare_r_g_on_value_transistor3.text(),
            "label_compare_r_g_off_value_transistor3": self.label_compare_r_g_off_value_transistor3.text(),
            "slider_compare_r_g_on_transistor1": self.slider_compare_r_g_on_transistor1.value(),
            "slider_compare_r_g_off_transistor1": self.slider_compare_r_g_off_transistor1.value(),
            "slider_compare_r_g_on_transistor2": self.slider_compare_r_g_on_transistor2.value(),
            "slider_compare_r_g_off_transistor2": self.slider_compare_r_g_off_transistor2.value(),
            "slider_compare_r_g_on_transistor3": self.slider_compare_r_g_on_transistor3.value(),
            "slider_compare_r_g_off_transistor3": self.slider_compare_r_g_off_transistor3.value()}

        topology_calculator_settings_dict = {
            "comboBox_topology_topology": self.comboBox_topology_topology.currentText(),
            "comboBox_topology_transistor1": self.comboBox_topology_transistor1.currentText(),
            "comboBox_topology_transistor2": self.comboBox_topology_transistor2.currentText(),
            "comboBox_topology_v_g_on_transistor1": self.comboBox_topology_v_g_on_transistor1.currentText(),
            "comboBox_topology_plot1_line_contour": self.comboBox_topology_plot1_line_contour.currentText(),
            "comboBox_topology_plot1_x_axis": self.comboBox_topology_plot1_x_axis.currentText(),
            "comboBox_topology_plot1_y_axis": self.comboBox_topology_plot1_y_axis.currentText(),
            "comboBox_topology_plot1_z_axis": self.comboBox_topology_plot1_z_axis.currentText(),
            "comboBox_topology_plot2_line_contour": self.comboBox_topology_plot2_line_contour.currentText(),
            "comboBox_topology_plot2_x_axis": self.comboBox_topology_plot2_x_axis.currentText(),
            "comboBox_topology_plot2_y_axis": self.comboBox_topology_plot2_y_axis.currentText(),
            "comboBox_topology_plot2_z_axis": self.comboBox_topology_plot2_z_axis.currentText(),
            "comboBox_topology_plot3_line_contour": self.comboBox_topology_plot3_line_contour.currentText(),
            "comboBox_topology_plot3_x_axis": self.comboBox_topology_plot3_x_axis.currentText(),
            "comboBox_topology_plot3_y_axis": self.comboBox_topology_plot3_y_axis.currentText(),
            "comboBox_topology_plot3_z_axis": self.comboBox_topology_plot3_z_axis.currentText(),
            "comboBox_topology_plot4_line_contour": self.comboBox_topology_plot4_line_contour.currentText(),
            "comboBox_topology_plot4_x_axis": self.comboBox_topology_plot4_x_axis.currentText(),
            "comboBox_topology_plot4_y_axis": self.comboBox_topology_plot4_y_axis.currentText(),
            "comboBox_topology_plot4_z_axis": self.comboBox_topology_plot4_z_axis.currentText(),
            "comboBox_topology_plot5_line_contour": self.comboBox_topology_plot5_line_contour.currentText(),
            "comboBox_topology_plot5_x_axis": self.comboBox_topology_plot5_x_axis.currentText(),
            "comboBox_topology_plot5_y_axis": self.comboBox_topology_plot5_y_axis.currentText(),
            "comboBox_topology_plot5_z_axis": self.comboBox_topology_plot5_z_axis.currentText(),
            "comboBox_topology_plot6_line_contour": self.comboBox_topology_plot6_line_contour.currentText(),
            "comboBox_topology_plot6_x_axis": self.comboBox_topology_plot6_x_axis.currentText(),
            "comboBox_topology_plot6_y_axis": self.comboBox_topology_plot6_y_axis.currentText(),
            "comboBox_topology_plot6_z_axis": self.comboBox_topology_plot6_z_axis.currentText(),
            "lineEdit_topology_number_parallel_transistor1": self.lineEdit_topology_number_parallel_transistor1.text(),
            "lineEdit_topology_number_parallel_transistor2": self.lineEdit_topology_number_parallel_transistor2.text(),
            "lineEdit_topology_output_power": self.lineEdit_topology_output_power.text(),
            "lineEdit_topology_v_in": self.lineEdit_topology_v_in.text(),
            "lineEdit_topology_v_out": self.lineEdit_topology_v_out.text(),
            "lineEdit_topology_frequency": self.lineEdit_topology_frequency.text(),
            "lineEdit_topology_zeta": self.lineEdit_topology_zeta.text(),
            "lineEdit_topology_temperature_heatsink": self.lineEdit_topology_temperature_heatsink.text(),
            "lineEdit_topology_thermal_resistance_heatsink": self.lineEdit_topology_thermal_resistance_heatsink.text(),
            "lineEdit_topology_output_power_min": self.lineEdit_topology_output_power_min.text(),
            "lineEdit_topology_v_in_min": self.lineEdit_topology_v_in_min.text(),
            "lineEdit_topology_v_out_min": self.lineEdit_topology_v_out_min.text(),
            "lineEdit_topology_frequency_min": self.lineEdit_topology_frequency_min.text(),
            "lineEdit_topology_zeta_min": self.lineEdit_topology_zeta_min.text(),
            "lineEdit_topology_output_power_max": self.lineEdit_topology_output_power_max.text(),
            "lineEdit_topology_v_in_max": self.lineEdit_topology_v_in_max.text(),
            "lineEdit_topology_v_out_max": self.lineEdit_topology_v_out_max.text(),
            "lineEdit_topology_frequency_max": self.lineEdit_topology_frequency_max.text(),
            "lineEdit_topology_zeta_max": self.lineEdit_topology_zeta_max.text(),
            "label_topology_slider_r_g_on_value_transistor1": self.label_topology_slider_r_g_on_value_transistor1.text(),
            "label_topology_slider_r_g_off_value_transistor1": self.label_topology_slider_r_g_off_value_transistor1.text(),
            "slider_topology_r_g_on_transistor1": self.slider_topology_r_g_on_transistor1.value(),
            "slider_topology_r_g_off_transistor1": self.slider_topology_r_g_off_transistor1.value()}

        all_settings_dict = {**search_database_settings_dict, **exporting_tools_settings_dict,
                             **comparison_tools_settings_dict, **topology_calculator_settings_dict}
        return all_settings_dict

    def export_settings(self):
        """
        Save all user settings as json file to a directory selected by the user.

        :return: None
        """
        all_settings_dict = self.get_settings_dict()
        try:
            path = QFileDialog.getExistingDirectory(self, caption="Open Directory")
            with open(path + "/settings.json", 'w') as fp:
                json.dump(all_settings_dict, fp)
            self.show_popup_message(
                f"Saved settings as settings.json to <a href={path}>{path}</a>!")
        except:
            pass

    def save_settings(self):
        """
        Save all user settings to the current working directory.

        :return: None
        """
        all_settings_dict = self.get_settings_dict()
        path = pathlib.Path.cwd()
        with open(path.joinpath(resource_path("settings.json")), 'w') as fp:
            json.dump(all_settings_dict, fp, indent=2)
        self.show_popup_message("Settings saved succsessfully!")

    def load_settings(self):
        """
        Load a json file selected by the user containing all settings and sets it using the function set_all_settings.

        :return: None
        """
        try:
            path = QFileDialog.getOpenFileName(self, "Open File", "", resource_path("(*.json)"))
            with open(path[0], 'r') as fp:
                all_settings_dict = json.load(fp)
            self.set_all_settings(all_settings_dict)
            self.show_popup_message("Settings loaded successfully!")
        except:
            if path[0] != "":
                self.show_popup_message("Loading the settings failed! Check if the correct file was selected!")

    def load_local_settings(self):
        """
        Load the settings.json file in the current working directory and sets all settings.

        :return: None
        """
        try:
            with open(resource_path("settings.json"), 'r') as fp:
                all_settings_dict = json.load(fp)
            self.set_all_settings(all_settings_dict)
        except:
            pass

    def set_all_settings(self, all_settings_dict):
        """
        Set all settings given in the dict all_settings_dict.

        :param all_settings_dict: dict containing all settings
        :return: None
        """
        self.checkBox_search_database_name.setChecked(all_settings_dict["checkBox_search_database_name"])
        self.checkBox_search_database_type.setChecked(all_settings_dict["checkBox_search_database_type"])
        self.checkBox_search_database_author.setChecked(all_settings_dict["checkBox_search_database_author"])
        self.checkBox_search_database_technology.setChecked(all_settings_dict["checkBox_search_database_technology"])
        self.checkBox_search_database_template_version.setChecked(
            all_settings_dict["checkBox_search_database_template_version"])
        self.checkBox_search_database_template_date.setChecked(
            all_settings_dict["checkBox_search_database_template_date"])
        self.checkBox_search_database_creation_date.setChecked(
            all_settings_dict["checkBox_search_database_creation_date"])
        self.checkBox_search_database_last_modified.setChecked(
            all_settings_dict["checkBox_search_database_last_modified"])
        self.checkBox_search_database_comment.setChecked(all_settings_dict["checkBox_search_database_comment"])
        self.checkBox_search_database_datasheet_hyperlink.setChecked(
            all_settings_dict["checkBox_search_database_datasheet_hyperlink"])
        self.checkBox_search_database_datasheet_date.setChecked(
            all_settings_dict["checkBox_search_database_datasheet_date"])
        self.checkBox_search_database_datasheet_version.setChecked(
            all_settings_dict["checkBox_search_database_datasheet_version"])
        self.checkBox_search_database_manufacturer.setChecked(
            all_settings_dict["checkBox_search_database_manufacturer"])
        self.checkBox_search_database_housing_type.setChecked(
            all_settings_dict["checkBox_search_database_housing_type"])
        self.checkBox_search_database_housing_area.setChecked(
            all_settings_dict["checkBox_search_database_housing_area"])
        self.checkBox_search_database_cooling_area.setChecked(
            all_settings_dict["checkBox_search_database_cooling_area"])
        self.checkBox_search_database_t_c_max.setChecked(all_settings_dict["checkBox_search_database_t_c_max"])
        self.checkBox_search_database_r_g_int.setChecked(all_settings_dict["checkBox_search_database_r_g_int"])
        self.checkBox_search_database_r_g_on_recommended.setChecked(
            all_settings_dict["checkBox_search_database_r_g_on_recommended"])
        self.checkBox_search_database_r_g_off_recommended.setChecked(
            all_settings_dict["checkBox_search_database_r_g_off_recommended"])
        self.checkBox_search_database_c_oss_fix.setChecked(all_settings_dict["checkBox_search_database_c_oss_fix"])
        self.checkBox_search_database_c_iss_fix.setChecked(all_settings_dict["checkBox_search_database_c_iss_fix"])
        self.checkBox_search_database_c_rss_fix.setChecked(all_settings_dict["checkBox_search_database_c_rss_fix"])
        self.checkBox_search_database_r_th_cs.setChecked(all_settings_dict["checkBox_search_database_r_th_cs"])
        self.checkBox_search_database_r_th_switch_cs.setChecked(
            all_settings_dict["checkBox_search_database_r_th_switch_cs"])
        self.checkBox_search_database_r_th_diode_cs.setChecked(
            all_settings_dict["checkBox_search_database_r_th_diode_cs"])
        self.checkBox_search_database_v_abs_max.setChecked(all_settings_dict["checkBox_search_database_v_abs_max"])
        self.checkBox_search_database_i_abs_max.setChecked(all_settings_dict["checkBox_search_database_i_abs_max"])
        self.checkBox_search_database_i_cont.setChecked(all_settings_dict["checkBox_search_database_i_cont"])
        self.checkBox_search_database_switch_comment.setChecked(
            all_settings_dict["checkBox_search_database_switch_comment"])
        self.checkBox_search_database_switch_manufacturer.setChecked(
            all_settings_dict["checkBox_search_database_switch_manufacturer"])
        self.checkBox_search_database_switch_technology.setChecked(
            all_settings_dict["checkBox_search_database_switch_technology"])
        self.checkBox_search_database_switch_t_j_max.setChecked(
            all_settings_dict["checkBox_search_database_switch_t_j_max"])
        self.checkBox_search_database_diode_comment.setChecked(
            all_settings_dict["checkBox_search_database_diode_comment"])
        self.checkBox_search_database_diode_manufacturer.setChecked(
            all_settings_dict["checkBox_search_database_diode_manufacturer"])
        self.checkBox_search_database_diode_technology.setChecked(
            all_settings_dict["checkBox_search_database_diode_technology"])
        self.checkBox_search_database_diode_t_j_max.setChecked(
            all_settings_dict["checkBox_search_database_diode_t_j_max"])
        self.lineEdit_search_database_name.setText(all_settings_dict["lineEdit_search_database_name"])
        self.lineEdit_search_database_type.setText(all_settings_dict["lineEdit_search_database_type"])
        self.lineEdit_search_database_author.setText(all_settings_dict["lineEdit_search_database_author"])
        self.lineEdit_search_database_technology.setText(all_settings_dict["lineEdit_search_database_technology"])
        self.lineEdit_search_database_template_version.setText(
            all_settings_dict["lineEdit_search_database_template_version"])
        self.lineEdit_search_database_template_date.setText(
            all_settings_dict["lineEdit_search_database_template_date"])
        self.lineEdit_search_database_creation_date.setText(
            all_settings_dict["lineEdit_search_database_creation_date"])
        self.lineEdit_search_database_last_modified.setText(
            all_settings_dict["lineEdit_search_database_last_modified"])
        self.lineEdit_search_database_comment.setText(all_settings_dict["lineEdit_search_database_comment"])
        self.lineEdit_search_database_datasheet_hyperlink.setText(
            all_settings_dict["lineEdit_search_database_datasheet_hyperlink"])
        self.lineEdit_search_database_datasheet_date.setText(
            all_settings_dict["lineEdit_search_database_datasheet_date"])
        self.lineEdit_search_database_datasheet_version.setText(
            all_settings_dict["lineEdit_search_database_datasheet_version"])
        self.lineEdit_search_database_manufacturer.setText(all_settings_dict["lineEdit_search_database_manufacturer"])
        self.lineEdit_search_database_housing_type.setText(all_settings_dict["lineEdit_search_database_housing_type"])
        self.lineEdit_search_database_switch_comment.setText(
            all_settings_dict["lineEdit_search_database_switch_comment"])
        self.lineEdit_search_database_switch_manufacturer.setText(
            all_settings_dict["lineEdit_search_database_switch_manufacturer"])
        self.lineEdit_search_database_switch_technology.setText(
            all_settings_dict["lineEdit_search_database_switch_technology"])
        self.lineEdit_search_database_diode_comment.setText(
            all_settings_dict["lineEdit_search_database_diode_comment"])
        self.lineEdit_search_database_diode_manufacturer.setText(
            all_settings_dict["lineEdit_search_database_diode_manufacturer"])
        self.lineEdit_search_database_housing_area_min.setText(
            all_settings_dict["lineEdit_search_database_housing_area_min"])
        self.lineEdit_search_database_housing_area_max.setText(
            all_settings_dict["lineEdit_search_database_housing_area_max"])
        self.lineEdit_search_database_cooling_area_min.setText(
            all_settings_dict["lineEdit_search_database_cooling_area_min"])
        self.lineEdit_search_database_cooling_area_max.setText(
            all_settings_dict["lineEdit_search_database_cooling_area_max"])
        self.lineEdit_search_database_t_c_max_min.setText(all_settings_dict["lineEdit_search_database_t_c_max_min"])
        self.lineEdit_search_database_t_c_max_max.setText(all_settings_dict["lineEdit_search_database_t_c_max_max"])
        self.lineEdit_search_database_r_g_int_min.setText(all_settings_dict["lineEdit_search_database_r_g_int_min"])
        self.lineEdit_search_database_r_g_int_max.setText(all_settings_dict["lineEdit_search_database_r_g_int_max"])
        self.lineEdit_search_database_r_g_on_recommended_min.setText(
            all_settings_dict["lineEdit_search_database_r_g_on_recommended_min"])
        self.lineEdit_search_database_r_g_on_recommended_max.setText(
            all_settings_dict["lineEdit_search_database_r_g_on_recommended_max"])
        self.lineEdit_search_database_r_g_off_recommended_min.setText(
            all_settings_dict["lineEdit_search_database_r_g_off_recommended_min"])
        self.lineEdit_search_database_r_g_off_recommended_max.setText(
            all_settings_dict["lineEdit_search_database_r_g_off_recommended_max"])
        self.lineEdit_search_database_c_oss_fix_min.setText(
            all_settings_dict["lineEdit_search_database_c_oss_fix_min"])
        self.lineEdit_search_database_c_oss_fix_max.setText(
            all_settings_dict["lineEdit_search_database_c_oss_fix_max"])
        self.lineEdit_search_database_c_iss_fix_min.setText(
            all_settings_dict["lineEdit_search_database_c_iss_fix_min"])
        self.lineEdit_search_database_c_iss_fix_max.setText(
            all_settings_dict["lineEdit_search_database_c_iss_fix_max"])
        self.lineEdit_search_database_c_rss_fix_min.setText(
            all_settings_dict["lineEdit_search_database_c_rss_fix_min"])
        self.lineEdit_search_database_c_rss_fix_max.setText(
            all_settings_dict["lineEdit_search_database_c_rss_fix_max"])
        self.lineEdit_search_database_r_th_cs_min.setText(all_settings_dict["lineEdit_search_database_r_th_cs_min"])
        self.lineEdit_search_database_r_th_cs_max.setText(all_settings_dict["lineEdit_search_database_r_th_cs_max"])
        self.lineEdit_search_database_r_th_switch_cs_min.setText(
            all_settings_dict["lineEdit_search_database_r_th_switch_cs_min"])
        self.lineEdit_search_database_r_th_switch_cs_max.setText(
            all_settings_dict["lineEdit_search_database_r_th_switch_cs_max"])
        self.lineEdit_search_database_r_th_diode_cs_min.setText(
            all_settings_dict["lineEdit_search_database_r_th_diode_cs_min"])
        self.lineEdit_search_database_r_th_diode_cs_max.setText(
            all_settings_dict["lineEdit_search_database_r_th_diode_cs_max"])
        self.lineEdit_search_database_v_abs_max_min.setText(
            all_settings_dict["lineEdit_search_database_v_abs_max_min"])
        self.lineEdit_search_database_v_abs_max_max.setText(
            all_settings_dict["lineEdit_search_database_v_abs_max_max"])
        self.lineEdit_search_database_i_abs_max_min.setText(
            all_settings_dict["lineEdit_search_database_i_abs_max_min"])
        self.lineEdit_search_database_i_abs_max_max.setText(
            all_settings_dict["lineEdit_search_database_i_abs_max_max"])
        self.lineEdit_search_database_i_cont_min.setText(all_settings_dict["lineEdit_search_database_i_cont_min"])
        self.lineEdit_search_database_i_cont_max.setText(all_settings_dict["lineEdit_search_database_i_cont_max"])
        self.lineEdit_search_database_switch_t_j_max_min.setText(
            all_settings_dict["lineEdit_search_database_switch_t_j_max_min"])
        self.lineEdit_search_database_switch_t_j_max_max.setText(
            all_settings_dict["lineEdit_search_database_switch_t_j_max_max"])
        self.lineEdit_search_database_diode_t_j_max_min.setText(
            all_settings_dict["lineEdit_search_database_diode_t_j_max_min"])
        self.lineEdit_search_database_diode_t_j_max_max.setText(
            all_settings_dict["lineEdit_search_database_diode_t_j_max_max"])

        self.comboBox_export_transistor.setCurrentText(all_settings_dict["comboBox_export_transistor"])
        self.lineEdit_export_number_parallel_transistors.setText(
            all_settings_dict["lineEdit_export_number_parallel_transistors"])
        self.lineEdit_export_simulink_v_supply.setText(all_settings_dict["lineEdit_export_simulink_v_supply"])
        self.lineEdit_export_simulink_r_g_on.setText(all_settings_dict["lineEdit_export_simulink_r_g_on"])
        self.lineEdit_export_simulink_r_g_off.setText(all_settings_dict["lineEdit_export_simulink_r_g_off"])
        self.lineEdit_export_simulink_normalize_t_to_v.setText(
            all_settings_dict["lineEdit_export_simulink_normalize_t_to_v"])
        self.lineEdit_export_gecko_v_supply.setText(all_settings_dict["lineEdit_export_gecko_v_supply"])
        self.lineEdit_export_gecko_r_g_on.setText(all_settings_dict["lineEdit_export_gecko_r_g_on"])
        self.lineEdit_export_gecko_r_g_off.setText(all_settings_dict["lineEdit_export_gecko_r_g_off"])
        self.lineEdit_export_gecko_v_g_on.setText(all_settings_dict["lineEdit_export_gecko_v_g_on"])
        self.lineEdit_export_gecko_v_g_off.setText(all_settings_dict["lineEdit_export_gecko_v_g_off"])

        self.comboBox_compare_transistor1.setCurrentText(all_settings_dict["comboBox_compare_transistor1"])
        self.comboBox_compare_transistor2.setCurrentText(all_settings_dict["comboBox_compare_transistor2"])
        self.comboBox_compare_transistor3.setCurrentText(all_settings_dict["comboBox_compare_transistor3"])
        self.comboBox_compare_v_g_on_transistor1.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_on_transistor1"])
        self.comboBox_compare_v_g_off_transistor1.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_off_transistor1"])
        self.comboBox_compare_v_g_on_transistor2.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_on_transistor2"])
        self.comboBox_compare_v_g_off_transistor2.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_off_transistor2"])
        self.comboBox_compare_v_g_on_transistor3.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_on_transistor3"])
        self.comboBox_compare_v_g_off_transistor3.setCurrentText(
            all_settings_dict["comboBox_compare_v_g_off_transistor3"])
        self.comboBox_compare_plot1.setCurrentText(all_settings_dict["comboBox_compare_plot1"])
        self.comboBox_compare_plot2.setCurrentText(all_settings_dict["comboBox_compare_plot2"])
        self.comboBox_compare_plot3.setCurrentText(all_settings_dict["comboBox_compare_plot3"])
        self.comboBox_compare_plot4.setCurrentText(all_settings_dict["comboBox_compare_plot4"])
        self.comboBox_compare_plot5.setCurrentText(all_settings_dict["comboBox_compare_plot5"])
        self.comboBox_compare_plot6.setCurrentText(all_settings_dict["comboBox_compare_plot6"])
        self.comboBox_compare_plot7.setCurrentText(all_settings_dict["comboBox_compare_plot7"])
        self.comboBox_compare_plot8.setCurrentText(all_settings_dict["comboBox_compare_plot8"])
        self.comboBox_compare_plot9.setCurrentText(all_settings_dict["comboBox_compare_plot9"])
        self.lineEdit_compare_t_j_transistor1.setText(all_settings_dict["lineEdit_compare_t_j_transistor1"])
        self.lineEdit_compare_t_j_transistor2.setText(all_settings_dict["lineEdit_compare_t_j_transistor2"])
        self.lineEdit_compare_t_j_transistor3.setText(all_settings_dict["lineEdit_compare_t_j_transistor3"])
        self.lineEdit_compare_v_supply_transistor1.setText(all_settings_dict["lineEdit_compare_v_supply_transistor1"])
        self.lineEdit_compare_number_parallel_transistor1.setText(
            all_settings_dict["lineEdit_compare_number_parallel_transistor1"])
        self.lineEdit_compare_v_supply_transistor2.setText(all_settings_dict["lineEdit_compare_v_supply_transistor2"])
        self.lineEdit_compare_number_parallel_transistor2.setText(
            all_settings_dict["lineEdit_compare_number_parallel_transistor2"])
        self.lineEdit_compare_v_supply_transistor3.setText(all_settings_dict["lineEdit_compare_v_supply_transistor3"])
        self.lineEdit_compare_number_parallel_transistor3.setText(
            all_settings_dict["lineEdit_compare_number_parallel_transistor3"])
        self.label_compare_r_g_on_value_transistor1.setText(
            all_settings_dict["label_compare_r_g_on_value_transistor1"])
        self.label_compare_r_g_off_value_transistor1.setText(
            all_settings_dict["label_compare_r_g_off_value_transistor1"])
        self.label_compare_r_g_on_value_transistor2.setText(
            all_settings_dict["label_compare_r_g_on_value_transistor2"])
        self.label_compare_r_g_off_value_transistor2.setText(
            all_settings_dict["label_compare_r_g_off_value_transistor2"])
        self.label_compare_r_g_on_value_transistor3.setText(
            all_settings_dict["label_compare_r_g_on_value_transistor3"])
        self.label_compare_r_g_off_value_transistor3.setText(
            all_settings_dict["label_compare_r_g_off_value_transistor3"])
        self.slider_compare_r_g_on_transistor1.setValue(int(all_settings_dict["slider_compare_r_g_on_transistor1"]))
        self.slider_compare_r_g_off_transistor1.setValue(int(all_settings_dict["slider_compare_r_g_off_transistor1"]))
        self.slider_compare_r_g_on_transistor2.setValue(int(all_settings_dict["slider_compare_r_g_on_transistor2"]))
        self.slider_compare_r_g_off_transistor2.setValue(int(all_settings_dict["slider_compare_r_g_off_transistor2"]))
        self.slider_compare_r_g_on_transistor3.setValue(int(all_settings_dict["slider_compare_r_g_on_transistor3"]))
        self.slider_compare_r_g_off_transistor3.setValue(int(all_settings_dict["slider_compare_r_g_off_transistor3"]))

        self.comboBox_topology_topology.setCurrentText(all_settings_dict["comboBox_topology_topology"])
        self.comboBox_topology_transistor1.setCurrentText(all_settings_dict["comboBox_topology_transistor1"])
        self.comboBox_topology_transistor2.setCurrentText(all_settings_dict["comboBox_topology_transistor2"])
        self.comboBox_topology_v_g_on_transistor1.setCurrentText(
            all_settings_dict["comboBox_topology_v_g_on_transistor1"])
        self.comboBox_topology_plot1_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot1_line_contour"])
        self.comboBox_topology_plot1_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot1_x_axis"])
        self.comboBox_topology_plot1_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot1_y_axis"])
        self.comboBox_topology_plot1_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot1_z_axis"])
        self.comboBox_topology_plot2_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot2_line_contour"])
        self.comboBox_topology_plot2_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot2_x_axis"])
        self.comboBox_topology_plot2_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot2_y_axis"])
        self.comboBox_topology_plot2_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot2_z_axis"])
        self.comboBox_topology_plot3_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot3_line_contour"])
        self.comboBox_topology_plot3_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot3_x_axis"])
        self.comboBox_topology_plot3_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot3_y_axis"])
        self.comboBox_topology_plot3_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot3_z_axis"])
        self.comboBox_topology_plot4_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot4_line_contour"])
        self.comboBox_topology_plot4_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot4_x_axis"])
        self.comboBox_topology_plot4_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot4_y_axis"])
        self.comboBox_topology_plot4_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot4_z_axis"])
        self.comboBox_topology_plot5_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot5_line_contour"])
        self.comboBox_topology_plot5_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot5_x_axis"])
        self.comboBox_topology_plot5_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot5_y_axis"])
        self.comboBox_topology_plot5_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot5_z_axis"])
        self.comboBox_topology_plot6_line_contour.setCurrentText(
            all_settings_dict["comboBox_topology_plot6_line_contour"])
        self.comboBox_topology_plot6_x_axis.setCurrentText(all_settings_dict["comboBox_topology_plot6_x_axis"])
        self.comboBox_topology_plot6_y_axis.setCurrentText(all_settings_dict["comboBox_topology_plot6_y_axis"])
        self.comboBox_topology_plot6_z_axis.setCurrentText(all_settings_dict["comboBox_topology_plot6_z_axis"])
        self.lineEdit_topology_number_parallel_transistor1.setText(
            all_settings_dict["lineEdit_topology_number_parallel_transistor1"])
        self.lineEdit_topology_number_parallel_transistor2.setText(
            all_settings_dict["lineEdit_topology_number_parallel_transistor2"])
        self.lineEdit_topology_output_power.setText(all_settings_dict["lineEdit_topology_output_power"])
        self.lineEdit_topology_v_in.setText(all_settings_dict["lineEdit_topology_v_in"])
        self.lineEdit_topology_v_out.setText(all_settings_dict["lineEdit_topology_v_out"])
        self.lineEdit_topology_frequency.setText(all_settings_dict["lineEdit_topology_frequency"])
        self.lineEdit_topology_zeta.setText(all_settings_dict["lineEdit_topology_zeta"])
        self.lineEdit_topology_temperature_heatsink.setText(
            all_settings_dict["lineEdit_topology_temperature_heatsink"])
        self.lineEdit_topology_thermal_resistance_heatsink.setText(
            all_settings_dict["lineEdit_topology_thermal_resistance_heatsink"])
        self.lineEdit_topology_output_power_min.setText(all_settings_dict["lineEdit_topology_output_power_min"])
        self.lineEdit_topology_v_in_min.setText(all_settings_dict["lineEdit_topology_v_in_min"])
        self.lineEdit_topology_v_out_min.setText(all_settings_dict["lineEdit_topology_v_out_min"])
        self.lineEdit_topology_frequency_min.setText(all_settings_dict["lineEdit_topology_frequency_min"])
        self.lineEdit_topology_zeta_min.setText(all_settings_dict["lineEdit_topology_zeta_min"])
        self.lineEdit_topology_output_power_max.setText(all_settings_dict["lineEdit_topology_output_power_max"])
        self.lineEdit_topology_v_in_max.setText(all_settings_dict["lineEdit_topology_v_in_max"])
        self.lineEdit_topology_v_out_max.setText(all_settings_dict["lineEdit_topology_v_out_max"])
        self.lineEdit_topology_frequency_max.setText(all_settings_dict["lineEdit_topology_frequency_max"])
        self.lineEdit_topology_zeta_max.setText(all_settings_dict["lineEdit_topology_zeta_max"])
        self.label_topology_slider_r_g_on_value_transistor1.setText(
            all_settings_dict["label_topology_slider_r_g_on_value_transistor1"])
        self.label_topology_slider_r_g_off_value_transistor1.setText(
            all_settings_dict["label_topology_slider_r_g_off_value_transistor1"])
        self.slider_topology_r_g_on_transistor1.setValue(int(all_settings_dict["slider_topology_r_g_on_transistor1"])),
        self.slider_topology_r_g_off_transistor1.setValue(
            int(all_settings_dict["slider_topology_r_g_off_transistor1"])),
