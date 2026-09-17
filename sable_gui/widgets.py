from __future__ import annotations

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt, QRect, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import QListView, QStyledItemDelegate, QStyle

from .models import MessageView


class ChatModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.items: list[MessageView] = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self.items[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return item
        return None

    def append(self, item: MessageView):
        row = len(self.items)
        self.beginInsertRows(QModelIndex(), row, row)
        self.items.append(item)
        self.endInsertRows()

    def remove_uids(self, uids: set[str]):
        keep = [m for m in self.items if m.uid not in uids]
        self.beginResetModel()
        self.items = keep
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self.items.clear()
        self.endResetModel()


class ChatDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.line_height = 22
        self.padding = 12

    def sizeHint(self, option, index):
        msg: MessageView = index.data(Qt.ItemDataRole.UserRole)
        width = max(option.rect.width() - 2 * self.padding, 300)
        chars_per_line = max(width // 8, 20)
        lines = max(1, (len(msg.text) + chars_per_line - 1) // chars_per_line)
        height = 58 + lines * self.line_height

        if msg.expanded:
            height += 30
            height += 24 * (
                len(msg.suggested)
                + len(msg.processed)
                + len(msg.browsed)
                + len(msg.tool_calls)
            )

        return QSize(option.rect.width(), height)

    def paint(self, painter: QPainter, option, index):
        msg: MessageView = index.data(Qt.ItemDataRole.UserRole)
        r = option.rect.adjusted(8, 6, -8, -6)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Background is deliberately left to the platform palette.
        painter.setPen(QPen(option.palette.mid()))
        painter.drawRoundedRect(r, 7, 7)

        title = "You" if msg.role == "user" else "SABLE"
        title_rect = QRect(r.left() + 10, r.top() + 7, r.width() - 100, 22)
        painter.setPen(option.palette.text().color())
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft, title)
        painter.drawText(
            QRect(r.right() - 90, r.top() + 7, 80, 22),
            Qt.AlignmentFlag.AlignRight,
            msg.timestamp,
        )

        body = QRect(r.left() + 10, r.top() + 30, r.width() - 20, r.height() - 54)
        painter.drawText(
            body,
            Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop,
            msg.text,
        )

        footer = QRect(r.left() + 10, r.bottom() - 22, r.width() - 20, 20)
        marker = "▼" if msg.expanded else "▶"
        painter.drawText(
            footer,
            Qt.AlignmentFlag.AlignLeft,
            f"{marker} {msg.tokens} tokens",
        )
        painter.drawText(
            footer,
            Qt.AlignmentFlag.AlignRight,
            "📋" + ("   ✎" if msg.editable else ""),
        )

        if msg.expanded:
            y = r.top() + 30
            # Metadata starts below the message body. Reuse compact text rather
            # than creating child widgets for every row.
            y = body.bottom() + 5
            entries = [
                ("Suggested", msg.suggested),
                ("Processed", msg.processed),
                ("Browsed", msg.browsed),
            ]
            for label, resources in entries:
                for resource in resources:
                    painter.drawText(
                        QRect(r.left() + 14, y, r.width() - 28, 20),
                        Qt.AlignmentFlag.AlignLeft,
                        f"{label}: {resource.icon} {resource.name}",
                    )
                    y += 24
            for call in msg.tool_calls:
                painter.drawText(
                    QRect(r.left() + 14, y, r.width() - 28, 20),
                    Qt.AlignmentFlag.AlignLeft,
                    f"Tool: {call}",
                )
                y += 24

        painter.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() != event.Type.MouseButtonRelease:
            return False

        msg: MessageView = index.data(Qt.ItemDataRole.UserRole)
        x = event.position().x()
        y = event.position().y()

        if x > option.rect.right() - 100:
            if msg.editable and x > option.rect.right() - 55:
                model.parent().editRequested.emit(msg.uid)
                return True

            QApplication.clipboard().setText(msg.text)
            return True

        # Footer expansion click.
        if y > option.rect.bottom() - 35:
            msg.expanded = not msg.expanded
            model.dataChanged.emit(index, index)
            model.layoutChanged.emit()
            return True

        # Resource clicks are handled by the main window via a future
        # resource-open signal; ordinary message clicks remain inert.
        return False


from PyQt6.QtWidgets import QApplication


class ChatView(QListView):
    editRequested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.model_obj = ChatModel()
        self.setModel(self.model_obj)
        self.setItemDelegate(ChatDelegate(self))
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setSelectionMode(QListView.SelectionMode.NoSelection)
        self.setUniformItemSizes(False)
        self.setWordWrap(True)
        self.setSpacing(4)
        self.setMouseTracking(True)

    def add_message(self, message: MessageView):
        self.model_obj.append(message)
        self.scrollToBottom()
