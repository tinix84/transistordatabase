"""Utilities mixin for MainWindow."""
import pathlib
import tempfile
import webbrowser

from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5 import QtGui


class UtilitiesMixin:
    """Mixin providing utility methods for MainWindow."""

    def show_popup_message(self, message):
        """
        Pop up a notification window with a specific message.

        :param message: notification message
        :return: None
        """
        MessageBox = QMessageBox()
        MessageBox.setWindowTitle("Information")
        MessageBox.setText(message)
        MessageBox.setIcon(QMessageBox.Warning)
        MessageBox.StandardButtons(QMessageBox.Cancel)
        MessageBox.exec_()

    def browse_file_csv(self):
        """
        Open up a window to browse a csv file.

        :return: List containing the file path and type of the selected file
        """
        path = QFileDialog.getOpenFileName(self, "Open File", "", "(*.csv)")
        return path[0]

    def webbrowser_original_datasheet(self):
        """Open the web browser to view the original datasheet."""
        transistor = self.get_marked_transistor()
        webbrowser.open(transistor.datasheet_hyperlink)

    def webbrowser_virtual_datasheet(self):
        """Open the web browser to view the virtual datasheet.."""
        transistor = self.get_marked_transistor()
        html = transistor.export_datasheet(build_collection=True)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(html)
        webbrowser.open(url)

    def email_add_transistor_to_transistordatabase_file_exchange(self):
        """
        Email workflow to start a request for adding a new transistor to the transistordatabase file exchange.

        This routine will open the mailprogram with predefined adresses. The .json file needs to be added manually.
        """
        transistor = self.get_marked_transistor()
        transistor.export_json()

        self.show_popup_message(
            f'Workflow to start request for upload <b>{transistor.name}</b> to the transistordatabase file exchange: '
            f'<br> <br> 1. The browser opens and wants to access the mail program. Allow this.  <br> 2. '
            f'The email program opens with the addressee pre-filled. <br> 3. Add the transistor file {transistor.name}.json '
            f'as attachment from this filder: <a href={pathlib.Path.cwd().as_uri()}>{pathlib.Path.cwd().as_uri()}</a> <br> 4. Send Email.')

        email_body = f"Do not forget to attach the transistor file <b>{transistor.name}</b> to this email!! " \
                     f"Link to File: <a href={pathlib.Path.cwd().as_uri()}>{pathlib.Path.cwd().as_uri()}</a>"
        email_subject = 'Request to add transistor {transistor.name} to the transistordatabase file exchange (TDB-FE)'
        webbrowser.open('mailto:?to=tdb@lea.upb.de&subject=' + email_subject + '&body=' + email_body, new=2)

    # Help actions #
    def webbrowser_contribute(self):
        """Open the web browser to the contributing guide."""
        webbrowser.open('https://github.com/upb-lea/transistordatabase/blob/main/Contributing.rst')

    def webbrowser_bugreport(self):
        """Open the web browser to the GitHub issues."""
        webbrowser.open('https://github.com/upb-lea/transistordatabase/issues')

    def webbrowser_documentation(self):
        """Open the web browser to view the transistor database documentation."""
        webbrowser.open('https://upb-lea.github.io/transistordatabase/main/transistordatabase.html')

    def standard_output_written(self, text):
        """For the GUI integrated console output."""
        cursor = self.textEdit_stdout.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_stdout.setTextCursor(cursor)
        self.textEdit_stdout.ensureCursorVisible()

    def show_stdout(self):
        """
        Show stdout-textEdit if the action in the menu bar is checked.

        :return: None
        """
        if self.action_stdout.isChecked() is True:
            self.textEdit_stdout.setMaximumSize(16777215, 100)
        else:
            self.textEdit_stdout.setMaximumSize(16777215, 0)
