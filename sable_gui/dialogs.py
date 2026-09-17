from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

class WorkspaceDialog(QDialog):
    def __init__(self, files: list[Path], root: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggest Workspace File")
        self.resize(600, 450)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a workspace file to suggest:"))

        self.list = QListWidget()
        for path in files:
            item = QListWidgetItem(str(path.relative_to(root)))
            item.setData(256, str(path))
            self.list.addItem(item)
        layout.addWidget(self.list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected(self) -> str | None:
        item = self.list.currentItem()
        return item.data(256) if item else None


class UrlDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggest Web URL")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("URL:"))
        self.text = QLineEdit()
        self.text.setPlaceholderText("https://example.com/")
        layout.addWidget(self.text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def url(self) -> str:
        return self.text.text().strip()


from PyQt6.QtWidgets import QLineEdit
