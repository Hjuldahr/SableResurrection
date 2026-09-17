from __future__ import annotations

import os
import shutil
import signal
import sys
import uuid
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from ai.intelligence_v4 import Sable

from .models import MessageView, Resource, ResourceType

class WorkspaceAdapter:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.generated = self.root / "generated"
        self.generated.mkdir(parents=True, exist_ok=True)

    def files(self) -> list[Path]:
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        )

    def relative(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root))

    def add_file(self, source: str | Path) -> Path:
        source = Path(source)
        destination = self.root / source.name
        if destination.exists():
            stem, suffix = source.stem, source.suffix
            i = 1
            while destination.exists():
                destination = self.root / f"{stem}_{i}{suffix}"
                i += 1
        return Path(shutil.copy2(source, destination))

    def remove_file(self, path: str | Path) -> None:
        path = Path(path).resolve()
        path.relative_to(self.root)
        if path.is_file():
            path.unlink()

    def open_file(self, path: str | Path) -> None:
        path = Path(path).resolve()
        path.relative_to(self.root)
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])


class SableWorker(QObject):
    ready = pyqtSignal()
    generating = pyqtSignal(bool)
    message = pyqtSignal(object)
    error = pyqtSignal(str)
    saved = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.sable = None

    @pyqtSlot()
    def initialize(self):
        try:
            # The supplied files are expected to be importable from the project.
            self.sable = Sable()
            self.ready.emit()
        except Exception as exc:
            self.error.emit(f"Failed to initialize SABLE: {exc}")

    @pyqtSlot(str, list)
    def submit(self, prompt: str, resources: list[dict[str, Any]]):
        if self.sable is None:
            return

        self.generating.emit(True)
        try:
            attachments = [
                r["location"]
                for r in resources
                if r["kind"] == ResourceType.WORKSPACE_FILE.value
            ]

            self.sable.submit(prompt, attachments or None)
            result = self.sable.generate()

            self.message.emit(self._message_view(result))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.generating.emit(False)

    @pyqtSlot()
    def save(self):
        if self.sable is None:
            return
        try:
            self.sable.append_history()
            self.saved.emit()
        except Exception as exc:
            self.error.emit(f"Save failed: {exc}")

    @pyqtSlot(str)
    def edit(self, uid: str):
        if self.sable is None:
            return
        try:
            index = next(
                i for i, m in enumerate(self.sable.history)
                if str(m.uid) == uid
            )
            msg = self.sable.history[index]
            if str(msg.role) != "user":
                return

            # Editing a user message removes it and every later message from
            # the active branch. The prompt itself is returned to the GUI.
            removed = self.sable.history[index:]
            del self.sable.history[index:]
            self.sable.start_of_new_history = min(
                self.sable.start_of_new_history,
                len(self.sable.history),
            )
            self.message.emit({
                "edit": True,
                "prompt": msg.content,
                "resources": msg.ui_meta.get("file_attachments", []),
                "removed": [str(m.uid) for m in removed],
            })
        except Exception as exc:
            self.error.emit(f"Edit failed: {exc}")

    @pyqtSlot()
    def close(self):
        try:
            if self.sable is not None:
                self.sable.append_history()
                self.sable.close()
        except Exception as exc:
            self.error.emit(f"Shutdown save failed: {exc}")
        finally:
            self.closed.emit()

    def _message_view(self, msg) -> MessageView:
        ui = msg.ui_meta or {}
        chat = msg.chat_meta or {}

        suggested = [
            self._resource_from_path(p)
            for p in ui.get("suggested_resources", [])
        ]
        processed = [
            self._resource_from_dict(r)
            for r in ui.get("processed_resources", [])
        ]
        browsed = [
            self._resource_from_dict(r)
            for r in ui.get("browsed_resources", [])
        ]

        tool_calls = []
        report = ui.get("tool-report")
        if report is not None:
            try:
                tool_calls.append(str(report))
            except Exception:
                pass

        return MessageView(
            uid=str(msg.uid),
            role=str(msg.role),
            text=msg.content,
            timestamp=msg.dt.astimezone().strftime("%H:%M:%S"),
            tokens=msg.ntokens,
            suggested=suggested,
            processed=processed,
            browsed=browsed,
            tool_calls=tool_calls,
            editable=str(msg.role) == "user",
            raw_message=msg,
        )

    @staticmethod
    def _resource_from_path(path: str | Path) -> Resource:
        path = Path(path)
        return Resource(
            id=str(path.resolve()),
            name=path.name,
            kind=ResourceType.WORKSPACE_FILE,
            location=str(path),
        )

    @staticmethod
    def _resource_from_dict(data: dict[str, Any]) -> Resource:
        kind = ResourceType(data["kind"])
        location = str(data["location"])
        return Resource(
            id=str(data.get("id", uuid.uuid5(uuid.NAMESPACE_URL, location))),
            name=str(data.get("name", location)),
            kind=kind,
            location=location,
        )


class SableAdapter(QObject):
    ready = pyqtSignal()
    generating = pyqtSignal(bool)
    message = pyqtSignal(object)
    error = pyqtSignal(str)
    saved = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, root: Path):
        super().__init__()
        self.workspace = WorkspaceAdapter(root)

        self.thread = QThread()
        self.worker = SableWorker(root)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.initialize)
        self.worker.ready.connect(self.ready)
        self.worker.generating.connect(self.generating)
        self.worker.message.connect(self.message)
        self.worker.error.connect(self.error)
        self.worker.saved.connect(self.saved)
        self.worker.closed.connect(self._on_closed)

        self.thread.start()
        self._closing = False

        self._install_shutdown_hooks()

    def submit(self, prompt: str, resources: list[Resource]):
        payload = [
            {
                "id": r.id,
                "name": r.name,
                "kind": r.kind.value,
                "location": r.location,
            }
            for r in resources
        ]
        QMetaObject.invokeMethod(
            self.worker,
            "submit",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, prompt),
            Q_ARG(list, payload),
        )

    def save(self):
        QMetaObject.invokeMethod(
            self.worker, "save", Qt.ConnectionType.QueuedConnection
        )

    def edit(self, uid: str):
        QMetaObject.invokeMethod(
            self.worker,
            "edit",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, uid),
        )

    def close(self):
        if self._closing:
            return
        self._closing = True
        QMetaObject.invokeMethod(
            self.worker, "close", Qt.ConnectionType.QueuedConnection
        )

    def _on_closed(self):
        self.closed.emit()
        self.thread.quit()
        self.thread.wait()

    def _install_shutdown_hooks(self):
        previous = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            self.close()
            if callable(previous):
                previous(signum, frame)

        signal.signal(signal.SIGINT, handler)

        if sys.platform == "win32":
            try:
                import ctypes

                HANDLER = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

                def console_handler(event):
                    if event in (0, 2, 5, 6):  # Ctrl-C, close, logoff, shutdown
                        self.close()
                    return False

                self._console_handler = HANDLER(console_handler)
                ctypes.windll.kernel32.SetConsoleCtrlHandler(
                    self._console_handler, True
                )
            except Exception:
                pass


# Imports kept at module bottom to make the adapter's public surface obvious.
from PyQt6.QtCore import Q_ARG, QMetaObject, Qt
