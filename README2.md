# SABLE PyQt6 Client

A PyQt6 client for the supplied `Sable` core and `NoteKeeper` DB.

## Layout

- `sable_gui/adapter.py` — GUI/backend boundary and worker thread.
- `sable_gui/models.py` — resource/message view models.
- `sable_gui/widgets.py` — chat message rendering and virtual list.
- `sable_gui/dialogs.py` — workspace/resource dialogs.
- `sable_gui/main_window.py` — application window.
- `run_gui.py` — entry point.

Copy these files into the SABLE project so `intelligence_v4.py` and `db.py` are importable.

## Resource semantics

Suggested, processed, and browsed are independent sets.

The supplied core currently exposes file attachments through `Message.ui_meta`; it does not expose a structured processed/browsed resource event stream. The adapter therefore leaves those fields empty until the core/tool layer supplies them. It does not incorrectly infer "processed" from "suggested".

## Workspace policy

The GUI may add/delete/open any workspace file. AI-side file creation/deletion remains a backend policy: the adapter only exposes the workspace root and generated directory to the GUI; it does not grant the model filesystem mutation privileges.

## Required core fix

The supplied `append_history()` opens `session.bin` with `wb` and then seeks backwards from the end. Truncating before that seek makes the negative seek invalid. Before relying on manual Save/Close, change the persistence implementation to write a complete replacement file (or open without truncation and explicitly truncate after writing).

The GUI does not silently patch this core behavior.
