from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from sable_gui.adapter import SableAdapter
from sable_gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SABLE")

    root = Path(__file__).resolve().parent
    adapter = SableAdapter(root)

    window = MainWindow(adapter)
    window.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
