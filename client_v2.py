from __future__ import annotations

import atexit
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QUrl, Qt, QSize, pyqtSignal
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ai.intelligence_v4 import Role, Sable
from ai_tools.manager import ToolManagerReport

# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

@dataclass
class Resource:
    name: str
    kind: str
    location: str

@dataclass
class MessageData:
    message_id: int
    is_user: bool
    text: str
    timestamp: datetime
    tokens: int | None = None
    resources: list[Resource] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)

class SableAdapter:
    """
    UI/backend seam.

    Replace these implementations with calls into Sable and the workspace
    subsystem. The UI should not need to know how those operations work.
    """
    
    def __init__(self, agent: Sable):
        self.agent = agent

    def submit(
        self,
        prompt: str,
        resources: list[Resource],
    ) -> bool:
        if not prompt.strip():
            return False
        
        if resources:
            prompt += f' Please take a look at: {", ".join(f"'{res.name}'" for res in resources)}'
            
        self.agent.submit(prompt, [res.location for res in resources if res.kind.startswith('file')])
        
        return True

    def generate(self) -> MessageData:
        msg = self.agent.generate()
        report: ToolManagerReport = msg.ui_meta["tool-report"]
        
        resources = [Resource(res.name, res.suffix, str(res.resolve())) for res in report.resource_files]
        # TODO fix back end dataflow to support this metadata 
        #resources += [Resource(res.t, res.suffix, str(res.resolve())) for res in report.resource_urls]
        
        message = MessageData(
            message_id=msg.uid.int,
            is_user=msg.role == Role.USER,
            text=msg.content,
            timestamp=msg.t,
            tokens=msg.ntokens,
            resources=resources,
            tool_calls=[f'{name} {f"X{count}" if count > 1 else ""}' for name, count in report.tool_calls.items()],
        )

        return message

    def save_chat(self) -> None:
        self.agent.append_history()

    def close(self) -> None:
        self.agent.close()

    def edit_message(self, message_id: int, prompt: str) -> None:
        # TODO, requires substantial serialization revamp to switch from append only to bidirectional modification of file end
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Resource widgets
# ---------------------------------------------------------------------------

class ResourceChip(QFrame):
    removed = pyqtSignal(object)
    activated = pyqtSignal(object)

    def __init__(
        self,
        resource: Resource,
        removable: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.resource = resource
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 3, 4, 3)
        layout.setSpacing(4)

        icon = QLabel(self._icon())
        name = QPushButton(resource.name)
        name.setFlat(True)
        name.setCursor(Qt.CursorShape.PointingHandCursor)
        name.clicked.connect(
            lambda: self.activated.emit(self.resource)
        )

        layout.addWidget(icon)
        layout.addWidget(name)

        if removable:
            remove = QToolButton()
            remove.setText("×")
            remove.setAutoRaise(True)
            remove.clicked.connect(
                lambda: self.removed.emit(self.resource)
            )
            layout.addWidget(remove)

    def _icon(self) -> str:
        return {
            "web-text": "🌐",
            "web-book": "📘",
            "web-news": "📰",
            ".txt": "📄",
            ".csv": "🗎",
            ".json": "🗐",
            ".markdown": "📑",
        }.get(self.resource.kind, "•")


# ---------------------------------------------------------------------------
# Message widgets
# ---------------------------------------------------------------------------

class MessageWidget(QFrame):
    edit_requested = pyqtSignal(object)

    def __init__(
        self,
        message: MessageData,
        adapter: SableAdapter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.message = message
        self.adapter = adapter

        self.setFrameShape(QFrame.Shape.NoFrame)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(5)

        header = QHBoxLayout()

        author = QLabel("You" if message.is_user else "Sable")
        author.setStyleSheet("font-weight: bold;")

        timestamp = QLabel(
            message.timestamp.strftime("%H:%M")
        )
        timestamp.setStyleSheet("color: gray;")

        header.addWidget(author)
        header.addStretch()
        header.addWidget(timestamp)

        outer.addLayout(header)

        body = QFrame()
        body.setFrameShape(QFrame.Shape.StyledPanel)

        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(10, 8, 10, 8)

        text = QLabel(message.text)
        text.setWordWrap(True)
        text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        body_layout.addWidget(text)
        outer.addWidget(body)

        controls = QHBoxLayout()
        controls.setSpacing(4)

        clipboard = QPushButton("Copy")
        clipboard.setFixedWidth(55)
        clipboard.clicked.connect(
            lambda: QApplication.clipboard().setText(message.text)
        )

        details = QToolButton()
        details.setText("Details")
        details.setCheckable(True)
        details.setArrowType(Qt.ArrowType.RightArrow)

        controls.addWidget(clipboard)

        if message.is_user:
            edit = QPushButton("Edit")
            edit.clicked.connect(
                lambda: self.edit_requested.emit(self.message)
            )
            controls.addWidget(edit)

        controls.addStretch()
        controls.addWidget(details)

        outer.addLayout(controls)

        self.details = QFrame()
        self.details.setVisible(False)

        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(25, 2, 0, 2)
        details_layout.setSpacing(3)

        tokens = (
            f"Tokens consumed: {message.tokens}"
            if message.tokens is not None
            else "Tokens consumed: unknown"
        )

        details_layout.addWidget(QLabel(tokens))

        if message.resources:
            label = (
                "Suggested Resources:"
                if message.is_user
                else "Processed Resources:"
            )
            details_layout.addWidget(QLabel(label))

            for resource in message.resources:
                chip = ResourceChip(resource)
                chip.activated.connect(adapter.open_resource)
                details_layout.addWidget(chip)

        if not message.is_user and message.tool_calls:
            details_layout.addWidget(QLabel("Tool Calls:"))

            for call in message.tool_calls:
                details_layout.addWidget(QLabel(f"  • {call}"))

        outer.addWidget(self.details)

        details.toggled.connect(self._toggle_details)

    def _toggle_details(self, visible: bool) -> None:
        self.details.setVisible(visible)

    def sizeHint(self) -> QSize:
        return QSize(0, self.minimumSizeHint().height())


# ---------------------------------------------------------------------------
# Virtual chat log
# ---------------------------------------------------------------------------

class ChatLog(QScrollArea):
    edit_requested = pyqtSignal(object)

    def __init__(
        self,
        adapter: SableAdapter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.adapter = adapter

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.addStretch()

        self.setWidget(self.container)

    def add_message(self, message: MessageData) -> None:
        widget = MessageWidget(
            message,
            self.adapter,
        )
        widget.edit_requested.connect(self.edit_requested)

        self.layout.insertWidget(
            self.layout.count() - 1,
            widget,
        )

        self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        )

    def clear_messages(self) -> None:
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            widget = item.widget()

            if widget:
                widget.deleteLater()


# ---------------------------------------------------------------------------
# Resource selection dialog
# ---------------------------------------------------------------------------

class ResourceDialog(QDialog):
    def __init__(
        self,
        workspace: list[Resource],
        web_urls: list[Resource],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.selected: list[Resource] = []

        self.setWindowTitle("Add Resources")
        self.resize(500, 350)

        layout = QVBoxLayout(self)

        tabs = QSplitter(Qt.Orientation.Vertical)

        self.workspace = QListWidget()
        self.web = QListWidget()

        workspace_label = QLabel("Workspace Files")
        web_label = QLabel("Web URLs")

        workspace_box = QWidget()
        workspace_layout = QVBoxLayout(workspace_box)
        workspace_layout.addWidget(workspace_label)
        workspace_layout.addWidget(self.workspace)

        web_box = QWidget()
        web_layout = QVBoxLayout(web_box)
        web_layout.addWidget(web_label)
        web_layout.addWidget(self.web)

        tabs.addWidget(workspace_box)
        tabs.addWidget(web_box)

        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self._populate(workspace, web_urls)

    def _populate(
        self,
        workspace: list[Resource],
        web_urls: list[Resource],
    ) -> None:
        for resource in workspace:
            item = QListWidgetItem(resource.name)
            item.setData(Qt.ItemDataRole.UserRole, resource)
            self.workspace.addItem(item)

        for resource in web_urls:
            item = QListWidgetItem(resource.name)
            item.setData(Qt.ItemDataRole.UserRole, resource)
            self.web.addItem(item)

        self.workspace.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.web.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )

    def accept(self) -> None:
        self.selected = []

        for widget in (self.workspace, self.web):
            for item in widget.selectedItems():
                self.selected.append(
                    item.data(Qt.ItemDataRole.UserRole)
                )

        super().accept()


class SableClient(QMainWindow):
    ROOT = Path(__file__).parents[0].resolve()
    WORKSPACE_ROOT = ROOT / "file-workspace"
    GENERATED_ROOT = WORKSPACE_ROOT / "generated"

    def __init__(
        self,
        adapter: SableAdapter,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.adapter = adapter

        # Files available to the application.
        self.workspace: list[Resource] = []

        # Resources suggested for the pending prompt.
        self.resources: list[Resource] = []

        self.setWindowTitle("Sable")
        self.resize(1000, 750)

        self._load_workspace()
        self._build_ui()

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def _load_workspace(self) -> None:
        self.WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

        self.workspace.clear()

        for path in self.WORKSPACE_ROOT.iterdir():
            if path.is_file():
                self.workspace.append(
                    Resource(
                        path.name,
                        path.suffix,
                        str(path),
                    )
                )

    def _add_workspace_file(self) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Workspace File",
            filter=(
                "Text files (*.txt *.md *.html);;"
                "Data files (*.csv *.json);;"
                "All Files (*)"
            ),
            initialFilter="Text files (*.txt *.md *.html)",
        )

        for file_path in file_paths:
            source_path = Path(file_path)
            out_path = Path(
                shutil.copy(source_path, self.WORKSPACE_ROOT)
            )

            resource = Resource(
                out_path.name,
                out_path.suffix,
                str(out_path),
            )

            # Replace the existing workspace entry if the copied file
            # overwrote a file with the same name.
            self.workspace = [
                res
                for res in self.workspace
                if res.name != resource.name
            ]
            self.workspace.append(resource)

    def _remove_workspace_file(self) -> None:
        names = [res.name for res in self.workspace]

        if not names:
            return

        name, ok = QInputDialog.getItem(
            self,
            "Remove Workspace File",
            "Choose Workspace File:",
            names,
            0,
            False,
        )

        if not ok:
            return

        resource = self.workspace[names.index(name)]

        Path(resource.location).unlink(missing_ok=True)
        self.workspace.remove(resource)

        # A deleted workspace file can no longer be a pending suggestion.
        if resource in self.resources:
            self.resources.remove(resource)
            self._refresh_resource_chips()

    def _open_workspace_file(self) -> None:
        names = [res.name for res in self.workspace]

        if not names:
            return

        name, ok = QInputDialog.getItem(
            self,
            "Open Workspace File",
            "Choose Workspace File:",
            names,
            0,
            False,
        )

        if not ok:
            return

        resource = self.workspace[names.index(name)]

        QDesktopServices.openUrl(
            QUrl.fromLocalFile(resource.location)
        )

    # ------------------------------------------------------------------
    # Resources suggested for the pending prompt
    # ------------------------------------------------------------------

    def _add_resource(self) -> None:
        dialog = ResourceDialog(
            workspace=self.workspace,
            web_urls=self.adapter.suggest_web_urls(),
            parent=self,
        )

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        for resource in dialog.selected:
            if resource not in self.resources:
                self.resources.append(resource)

        self._refresh_resource_chips()

    def _refresh_resource_chips(self) -> None:
        while self.resource_layout.count():
            item = self.resource_layout.takeAt(0)
            widget = item.widget()

            if widget:
                widget.deleteLater()

        for resource in self.resources:
            chip = ResourceChip(
                resource,
                removable=True,
            )
            chip.removed.connect(self._remove_resource)
            chip.activated.connect(self.adapter.open_resource)

            self.resource_layout.addWidget(chip)

        self.resource_layout.addStretch()

    def _remove_resource(self, resource: Resource) -> None:
        self.resources.remove(resource)
        self._refresh_resource_chips()

    def _clear_resources(self) -> None:
        self.resources.clear()
        self._refresh_resource_chips()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        self.chat = ChatLog(self.adapter)
        self.chat.edit_requested.connect(self._edit_message)

        root_layout.addWidget(self.chat, stretch=1)
        root_layout.addWidget(self._build_footer())

        self.setCentralWidget(root)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QHBoxLayout(bar)

        workspace = QToolButton()
        workspace.setText("Workspace")
        workspace.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )

        menu = workspace.menu()

        if menu is None:
            from PyQt6.QtWidgets import QMenu

            menu = QMenu(workspace)
            workspace.setMenu(menu)

        add_file = QAction("Add File", self)
        remove_file = QAction("Remove File", self)
        open_file = QAction(
            "Open File using System Default App",
            self,
        )

        add_file.triggered.connect(self._add_workspace_file)
        remove_file.triggered.connect(self._remove_workspace_file)
        open_file.triggered.connect(self._open_workspace_file)

        menu.addAction(add_file)
        menu.addAction(remove_file)
        menu.addSeparator()
        menu.addAction(open_file)

        save = QPushButton("Save Chat")
        save.clicked.connect(self._save_chat)

        close = QPushButton("Close")
        close.clicked.connect(self.close)

        layout.addWidget(workspace)
        layout.addStretch()
        layout.addWidget(save)
        layout.addWidget(close)

        return bar

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(footer)
        layout.setContentsMargins(10, 8, 10, 8)

        self.resource_layout = QHBoxLayout()
        self.resource_layout.setSpacing(4)

        layout.addLayout(self.resource_layout)

        controls = QHBoxLayout()

        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Message Sable...")
        self.prompt.setFixedHeight(70)

        add_resource = QPushButton("+")
        add_resource.setFixedWidth(35)
        add_resource.setToolTip("Add Resource")
        add_resource.clicked.connect(self._add_resource)

        self.send = QPushButton("Send")
        self.send.setFixedWidth(70)
        self.send.clicked.connect(self._submit)

        controls.addWidget(self.prompt, stretch=1)
        controls.addWidget(add_resource)
        controls.addWidget(self.send)

        layout.addLayout(controls)

        self.status = QLabel()
        self.status.setVisible(False)
        self.status.setStyleSheet("color: gray;")

        layout.addWidget(self.status)

        return footer

    # ------------------------------------------------------------------
    # Submission / generation
    # ------------------------------------------------------------------

    def _submit(self) -> None:
        prompt = self.prompt.toPlainText().strip()

        if not prompt:
            return

        # Snapshot the suggestions for this particular message.
        resources = list(self.resources)

        if not self.adapter.submit(prompt, resources):
            return

        user_message = MessageData(
            message_id=0,
            is_user=True,
            text=prompt,
            timestamp=datetime.now(),
            resources=resources,
        )

        self.chat.add_message(user_message)

        self.prompt.clear()
        self._clear_resources()
        self._set_generating(True)

        try:
            response = self.adapter.generate()
            self.chat.add_message(response)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Generation Error",
                str(exc),
            )
        finally:
            self._set_generating(False)

    def _set_generating(self, generating: bool) -> None:
        self.send.setEnabled(not generating)
        self.prompt.setEnabled(not generating)

        if generating:
            self.status.setText("Generating...")
            self.status.setVisible(True)
        else:
            self.status.setVisible(False)

    # ------------------------------------------------------------------
    # Chat editing
    # ------------------------------------------------------------------

    def _edit_message(self, message: MessageData) -> None:
        self.prompt.setPlainText(message.text)
        self.prompt.setFocus()

        self.adapter.edit_message(
            message.message_id,
            message.text,
        )

    # ------------------------------------------------------------------
    # Persistence / shutdown
    # ------------------------------------------------------------------

    def _save_chat(self) -> None:
        self.adapter.save_chat()

    def closeEvent(self, event) -> None:
        try:
            self.adapter.save_chat()
            self.adapter.close()
        finally:
            event.accept()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)

    sable = Sable()
    atexit.register(sable.close)

    client = SableClient(SableAdapter(sable))
    client.show()

    sys.exit(app.exec())