from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .adapter import SableAdapter
from .dialogs import UrlDialog, WorkspaceDialog
from .models import MessageView, Resource, ResourceType
from .widgets import ChatView


class MainWindow(QMainWindow):
    def __init__(self, adapter: SableAdapter):
        super().__init__()
        self.adapter = adapter
        self.resources: list[Resource] = []
        self._closing = False
        self._generating = False

        self.setWindowTitle("SABLE")
        self.resize(1100, 800)

        self._build_ui()
        self._connect()
        self._load_workspace()

    def _build_ui(self):
        toolbar = QToolBar("Workspace")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        add = QAction("Add File", self)
        add.triggered.connect(self.add_file)
        toolbar.addAction(add)

        remove = QAction("Remove File", self)
        remove.triggered.connect(self.remove_file)
        toolbar.addAction(remove)

        open_ = QAction("Open File", self)
        open_.triggered.connect(self.open_file)
        toolbar.addAction(open_)

        toolbar.addSeparator()

        save = QAction("Save Chat", self)
        save.triggered.connect(self.adapter.save)
        toolbar.addAction(save)

        self.close_action = QAction("Close", self)
        self.close_action.triggered.connect(self.close)
        toolbar.addAction(self.close_action)

        self.chat = ChatView()

        self.status = QLabel("Initializing SABLE…")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Message SABLE…")
        self.prompt.returnPressed.connect(self.send)

        self.resource_button = QPushButton("+")
        self.resource_button.setFixedWidth(38)
        self.resource_button.setToolTip("Add resource")

        menu = QMenu(self)
        menu.addAction("Suggest workspace file", self.suggest_file)
        menu.addAction("Suggest web URL", self.suggest_url)
        self.resource_button.setMenu(menu)

        self.resource_bar = QHBoxLayout()
        self.resource_bar.setSpacing(5)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self.resource_button)
        prompt_row.addWidget(self.prompt, 1)
        prompt_row.addWidget(self.send_button)

        footer = QVBoxLayout()
        footer.addWidget(self.status)
        footer.addLayout(self.resource_bar)
        footer.addLayout(prompt_row)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.chat, 1)
        layout.addLayout(footer)
        self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())

    def _connect(self):
        self.adapter.ready.connect(
            lambda: self.status.setText("Ready")
        )
        self.adapter.generating.connect(self.set_generating)
        self.adapter.message.connect(self.receive)
        self.adapter.error.connect(self.show_error)
        self.adapter.closed.connect(self._closed)
        self.chat.editRequested.connect(self.edit_message)

    def _load_workspace(self):
        self.workspace_files = self.adapter.workspace.files()

    def set_generating(self, generating: bool):
        self._generating = generating
        self.send_button.setEnabled(not generating)
        self.prompt.setEnabled(not generating)
        self.status.setText("Generating…" if generating else "Ready")

    def send(self):
        if self._generating:
            return
        prompt = self.prompt.text().strip()
        if not prompt:
            return

        self.prompt.clear()
        self._add_local_user(prompt)
        self.adapter.submit(prompt, self.resources)

    def _add_local_user(self, prompt: str):
        from datetime import datetime

        item = MessageView(
            uid=f"gui-{datetime.now().timestamp()}",
            role="user",
            text=prompt,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            tokens=0,
            suggested=list(self.resources),
            editable=True,
        )
        self.chat.add_message(item)

    def receive(self, value):
        if isinstance(value, MessageView):
            self.chat.add_message(value)
            return

        if isinstance(value, dict) and value.get("edit"):
            self.prompt.setText(value["prompt"])
            self.resources.clear()
            for path in value["resources"]:
                self._add_resource(
                    Resource(
                        id=str(path),
                        name=Path(path).name,
                        kind=ResourceType.WORKSPACE_FILE,
                        location=str(path),
                    )
                )
            self.chat.model_obj.remove_uids(set(value["removed"]))
            self.prompt.setFocus()

    def edit_message(self, uid: str):
        if uid.startswith("gui-"):
            # The just-created local representation has not necessarily been
            # synchronized yet, so use the visible message directly.
            for message in self.chat.model_obj.items:
                if message.uid == uid:
                    self.prompt.setText(message.text)
                    self.resources = list(message.suggested)
                    self.chat.model_obj.remove_uids(
                        {m.uid for m in self.chat.model_obj.items
                         if self.chat.model_obj.items.index(m) >= self.chat.model_obj.items.index(message)}
                    )
                    self.prompt.setFocus()
                    return

        self.adapter.edit(uid)

    def _add_resource(self, resource: Resource):
        if any(r.id == resource.id for r in self.resources):
            return
        self.resources.append(resource)

        button = QPushButton(f"{resource.icon} {resource.name} ×")
        button.setToolTip(resource.location)
        button.clicked.connect(
            lambda checked=False, rid=resource.id: self.remove_resource(rid, button)
        )
        self.resource_bar.addWidget(button)

    def remove_resource(self, resource_id: str, button: QPushButton):
        self.resources[:] = [r for r in self.resources if r.id != resource_id]
        button.deleteLater()

    def suggest_file(self):
        self._load_workspace()
        dialog = WorkspaceDialog(
            self.workspace_files,
            self.adapter.workspace.root,
            self,
        )
        if dialog.exec():
            path = dialog.selected()
            if path:
                self._add_resource(
                    Resource(
                        id=str(Path(path).resolve()),
                        name=Path(path).name,
                        kind=ResourceType.WORKSPACE_FILE,
                        location=path,
                    )
                )

    def suggest_url(self):
        dialog = UrlDialog(self)
        if dialog.exec():
            url = dialog.url()
            parsed = QUrl(url)
            if not parsed.isValid() or parsed.scheme() not in ("http", "https"):
                QMessageBox.warning(self, "Invalid URL", "Please enter an HTTP(S) URL.")
                return

            self._add_resource(
                Resource(
                    id=url,
                    name=url,
                    kind=ResourceType.WEB_URL,
                    location=url,
                )
            )

    def add_file(self):
        source, _ = QFileDialog.getOpenFileName(self, "Add File")
        if not source:
            return
        try:
            destination = self.adapter.workspace.add_file(source)
            self._load_workspace()
            self.statusBar().showMessage(
                f"Copied {Path(source).name} → {self.adapter.workspace.relative(destination)}",
                3000,
            )
        except Exception as exc:
            self.show_error(str(exc))

    def remove_file(self):
        self._load_workspace()
        dialog = WorkspaceDialog(
            self.workspace_files,
            self.adapter.workspace.root,
            self,
        )
        dialog.setWindowTitle("Remove Workspace File")
        if dialog.exec():
            path = dialog.selected()
            if path:
                try:
                    self.adapter.workspace.remove_file(path)
                    self._load_workspace()
                except Exception as exc:
                    self.show_error(str(exc))

    def open_file(self):
        self._load_workspace()
        dialog = WorkspaceDialog(
            self.workspace_files,
            self.adapter.workspace.root,
            self,
        )
        dialog.setWindowTitle("Open Workspace File")
        if dialog.exec():
            path = dialog.selected()
            if path:
                try:
                    self.adapter.workspace.open_file(path)
                except Exception as exc:
                    self.show_error(str(exc))

    def show_error(self, text: str):
        self.status.setText("Error")
        QMessageBox.critical(self, "SABLE", text)

    def closeEvent(self, event):
        if self._closing:
            event.accept()
            return

        self._closing = True
        self.status.setText("Saving and closing…")
        self.close_action.setEnabled(False)
        self.adapter.closed.connect(lambda: event.accept())
        self.adapter.close()
        event.ignore()

    def _closed(self):
        if self._closing:
            self.deleteLater()
