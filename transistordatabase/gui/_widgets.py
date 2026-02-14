"""Reusable GUI widget classes that have no dependency on MainWindow."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
import matplotlib.pyplot as plt
from matplotlib import cm

from PyQt5.QtWidgets import QWidget, QMainWindow, QVBoxLayout
from PyQt5 import uic, QtCore, QtGui
from mpl_toolkits.axes_grid1 import make_axes_locatable


class MatplotlibWidget(QWidget):
    """Matplotlib figure embedded inside a QWidget."""

    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot(111)
        self.layout = QVBoxLayout(self)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.toolbar)
        self.divider = make_axes_locatable(self.axis)
        self.axis_cm = self.divider.append_axes("right", size="3%", pad=0.03)
        self.sm = plt.cm.ScalarMappable(cmap=cm.inferno)
        self.figure.colorbar(mappable=self.sm, cax=self.axis_cm)


class PopOutPlotWindow(QMainWindow):
    """Pop-out window for displaying a single plot."""

    def __init__(self):
        super(PopOutPlotWindow, self).__init__()
        uic.loadUi("PopOutPlotWindow.ui", self)

        self.setWindowIcon(QtGui.QIcon("window_icon"))

        self.matplotlibwidget = MatplotlibWidget


class ViewCurveWindow(QMainWindow):
    """Window for viewing a single curve with scaling options."""

    def __init__(self):
        super(ViewCurveWindow, self).__init__()
        uic.loadUi("ViewCurveWindow.ui", self)

        self.setWindowIcon(QtGui.QIcon("window_icon"))

        self.matplotlibwidget = MatplotlibWidget()
        self.matplotlibwidget.axis_cm.remove()

        self.radioButton_scale_linear.toggled.connect(self.update_curve)
        self.radioButton_scale_log_x.toggled.connect(self.update_curve)
        self.radioButton_scale_log_y.toggled.connect(self.update_curve)
        self.radioButton_scale_log_xy.toggled.connect(self.update_curve)

    def update_curve(self):
        """
        Update the scaling of the curve when the radio buttons to scale linear/log(x)/log(y)/log(x,y) is pressed.

        :return: None
        """
        graph = self.comboBox_data.itemData(self.comboBox_data.currentIndex())

        xlabel = self.matplotlibwidget.axis.get_xlabel()
        ylabel = self.matplotlibwidget.axis.get_ylabel()
        curve_title = self.matplotlibwidget.axis.get_title()
        curve_label = self.comboBox_data.currentText()
        self.matplotlibwidget.axis.clear()

        if self.radioButton_scale_log_xy.isChecked() is True:
            self.matplotlibwidget.axis.loglog(graph[0], graph[1], label=curve_label)
        elif self.radioButton_scale_linear.isChecked() is True:
            self.matplotlibwidget.axis.plot(graph[0], graph[1], label=curve_label)
        elif self.radioButton_scale_log_x.isChecked() is True:
            self.matplotlibwidget.axis.semilogx(graph[0], graph[1], label=curve_label)
        elif self.radioButton_scale_log_y.isChecked() is True:
            self.matplotlibwidget.axis.semilogy(graph[0], graph[1], label=curve_label)

        self.matplotlibwidget.axis.set(xlabel=xlabel,
                                       ylabel=ylabel,
                                       title=curve_title)

        self.matplotlibwidget.axis.legend(fontsize=6)
        self.matplotlibwidget.axis.grid()

        self.matplotlibwidget.figure.canvas.draw_idle()

        self.matplotlibwidget.cursor = Cursor(self.matplotlibwidget.axis, horizOn=True, vertOn=True, useblit=True,
                                              color="Green", linewidth=1)

    def view_curve(self, comboBox, curve_title, xlabel, ylabel):
        """
        Show currently selected curve in a ViewCurveWindow.

        :return: None
        """
        try:
            self.layout = QVBoxLayout(self.widget_plot)
            self.layout.addWidget(self.matplotlibwidget)

            data_dict = comboBox.itemData(comboBox.currentIndex())
            curve_label = comboBox.currentText()

            for key in data_dict.keys():
                if data_dict[key] is not None and "graph" in key:
                    graph = data_dict[key]

            self.comboBox_data.addItem(curve_label, graph)
            self.comboBox_data.setDisabled(True)

            self.matplotlibwidget.axis.set(xlabel=xlabel,
                                           ylabel=ylabel,
                                           title=curve_title)
            # set linear scale as default
            self.radioButton_scale_linear.setChecked(True)
            self.update_curve()

            self.show()
        except:
            pass


class EmittingStream(QtCore.QObject):
    """Catch standard output and show it within the stdout-textEdit."""

    text_written = QtCore.pyqtSignal(str)

    def write(self, text):
        """
        Write standard output in stdout-textEdit.

        :return: None
        """
        self.text_written.emit(str(text))
