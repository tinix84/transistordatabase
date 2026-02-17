"""Entry point for the TDB PyQt5 desktop GUI."""
from __future__ import annotations

import sys


def main() -> None:
    """Launch the PyQt5 MainWindow."""
    from PyQt5.QtWidgets import QApplication
    from transistordatabase.gui.gui import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
