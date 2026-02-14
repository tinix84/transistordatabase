"""Export tools mixin for MainWindow."""
import json
import os


class ExportToolsMixin:
    """Mixin providing export tools functionality for MainWindow."""

    def export_datasheet(self):
        """
        Export the virtual datasheet in form of pdf in the current working directory.

        :return: pdf file is created in the current working directory
        """
        # Maybe a path for the datasheet file can be given as a parameter?
        transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())
        transistor.export_datasheet()

        self.show_popup_message(
            f"Exported a virtual datasheet for {transistor.name} to {os.getcwd()}")

    def export_json(self):
        """
        Export a json file in the current working directory.

        :return: json file is created in the current working directory
        """
        # Maybe a path for the json file can be given as a parameter?
        transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())
        json_path = os.path.join(os.getcwd(), f"{transistor.name}.json")
        with open(json_path, "w") as fd:
            json.dump(transistor.convert_to_dict(), fd)
        self.show_popup_message(
            f"Exported a json file for {transistor.name} to {json_path}")

    def export_matlab(self):
        """
        Export a matlab file in the current working directory.

        :return: matlab file is created in the current working directory
        """
        try:
            transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())
            transistor_parallel = self.tdb.parallel_transistors(transistor, int(self.lineEdit_export_number_parallel_transistors.text()))
            transistor_parallel.export_matlab()

            self.show_popup_message(
                f"Exported a MATLAB file for {transistor_parallel.name} to {os.getcwd()}")
        except:
            self.show_popup_message("Invalid input for number of parallel transistors!")

    def export_simulink(self):
        """
        Export a simulink file in the current working directory.

        :return: simulink file is created in the current working directory
        """
        try:
            transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())
            if transistor.type == "IGBT":
                transistor_parallel = self.tdb.parallel_transistors(transistor, int(self.lineEdit_export_number_parallel_transistors.text()))
                transistor_parallel.export_simulink_loss_model(
                    r_g_on=float(self.lineEdit_export_simulink_r_g_on.text()),
                    r_g_off=float(self.lineEdit_export_simulink_r_g_off.text()),
                    v_supply=float(self.lineEdit_export_simulink_v_supply.text()),
                    normalize_t_to_v=float(self.lineEdit_export_simulink_normalize_t_to_v.text()))

                self.show_popup_message(
                    f"Exported a Simulink file for {transistor_parallel.name} to {os.getcwd()}")
            else:
                self.show_popup_message("Error: Exporting simulink files is working for IGBTs only!")

        except:
            self.show_popup_message("Error: One or more invalid inputs!")

    def export_plecs(self):
        """
        Export a PLECS file in the current working directory.

        :return: PLECS file is created in the current working directory
        """
        transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())

        transistor_parallel = self.tdb.parallel_transistors(transistor, int(self.lineEdit_export_number_parallel_transistors.text()))

        try:
            gate_voltages = [float(self.lineEdit_export_plecs_v_g_on.text()),
                             float(self.lineEdit_export_plecs_v_g_off.text()),
                             float(self.lineEdit_export_plecs_v_d_on.text()),
                             float(self.lineEdit_export_plecs_v_d_off.text())]
            transistor_parallel.export_plecs(gate_voltages)
        except:
            transistor_parallel.export_plecs()

        self.show_popup_message(
            f"Exported a PLECS file for {transistor_parallel.name} to {os.getcwd()}")

    def export_gecko(self):
        """
        Export GeckoCircuits files in the current working directory.

        :return: GeckoCircuits files are created in the current working directory
        """
        try:
            transistor = self.tdb.load_transistor(self.comboBox_export_transistor.currentText())
            transistor_parallel = self.tdb.parallel_transistors(transistor, int(self.lineEdit_export_number_parallel_transistors.text()))
            transistor_parallel.export_geckocircuits(v_supply=float(self.lineEdit_export_gecko_v_supply.text()),
                                                     r_g_on=float(self.lineEdit_export_gecko_r_g_on.text()),
                                                     r_g_off=float(self.lineEdit_export_gecko_r_g_off.text()),
                                                     v_g_on=float(self.lineEdit_export_gecko_v_g_on.text()),
                                                     v_g_off=float(self.lineEdit_export_gecko_v_g_off.text()))

            self.show_popup_message(
                f"Exported GeckoCircuits files for {transistor_parallel.name} to {os.getcwd()}")
        except:
            self.show_popup_message("Error: One or more invalid inputs!")
