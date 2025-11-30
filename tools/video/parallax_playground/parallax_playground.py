#!/usr/bin/env python3
"""Interactive scene builder for experimenting with parallax layers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:  # Optional drag-and-drop support
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TkinterDnD = None
    DND_FILES = "DND_Files"


ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


@dataclass
class ParallaxItem:
    """Represents a single image layer with distance/frequency settings."""

    image_path: Path
    min_distance: float = 1.0
    max_distance: float = 5.0
    frequency: float = 1.0

    def as_row(self) -> tuple[str, str, str, str]:
        return (
            str(self.image_path),
            f"{self.min_distance:.2f}",
            f"{self.max_distance:.2f}",
            f"{self.frequency:.2f}",
        )


BaseTkClass = TkinterDnD.Tk if TkinterDnD else tk.Tk


class ParallaxPlaygroundApp(BaseTkClass):
    """UI for configuring parallax items."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.title("Parallax Playground")
        self.geometry("820x560")

        self.repo_root = Path(__file__).resolve().parents[3]
        self.project_assets_dir = self._resolve_assets_dir(args.project)
        self.items: Dict[str, ParallaxItem] = {}
        self._editor: Optional[ttk.Entry] = None
        self._editor_var = tk.StringVar()

        self._build_ui()
        self._configure_drop()

    # ------------------------------------------------------------------
    def _resolve_assets_dir(self, project_root: Optional[str]) -> Path:
        """Determine the default folder to browse for images."""

        if project_root:
            project_path = Path(project_root).expanduser().resolve()
            candidate = project_path / "assets"
            if candidate.exists():
                return candidate
            if project_path.exists():
                return project_path

        fallback = self.repo_root / "assets"
        return fallback if fallback.exists() else Path.cwd()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=20)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Parallax Playground", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Build a set of parallax layers by pairing images with the distances they can "
                "appear at and how frequently they spawn."
            ),
            justify="left",
            wraplength=760,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Separator(container).pack(fill="x", pady=8)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 12))

        ttk.Label(controls, text="Items", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(controls, text="Load image...", command=self._open_file_dialog).pack(side="right")

        assets_hint = ttk.Label(
            container,
            text=f"Default image folder: {self.project_assets_dir}",
            foreground="#666",
        )
        assets_hint.pack(fill="x", pady=(0, 8))

        table_frame = ttk.Frame(container)
        table_frame.pack(fill="both", expand=True)
        columns = ("image", "min", "max", "frequency")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=12,
            selectmode="browse",
        )
        self.tree.heading("image", text="Image path")
        self.tree.heading("min", text="Min distance")
        self.tree.heading("max", text="Max distance")
        self.tree.heading("frequency", text="Frequency")
        self.tree.column("image", anchor="w", width=360)
        self.tree.column("min", anchor="center", width=100)
        self.tree.column("max", anchor="center", width=100)
        self.tree.column("frequency", anchor="center", width=100)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=y_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")

        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        hint = ttk.Label(
            container,
            text="Drop image files anywhere in this window or use the button above to add them.",
            foreground="#666",
        )
        hint.pack(fill="x", pady=(8, 0))

        self.tree.bind("<Double-1>", self._start_edit)

    # ------------------------------------------------------------------
    def _configure_drop(self) -> None:
        if TkinterDnD:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._handle_drop)
                self.tree.drop_target_register(DND_FILES)
                self.tree.dnd_bind("<<Drop>>", self._handle_drop)
            except Exception:
                # Fail silently if tkdnd is unavailable at runtime
                pass

    # ------------------------------------------------------------------
    def _handle_drop(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        paths = self.tk.splitlist(event.data) if event.data else []
        if not paths:
            return
        for raw in paths:
            path = self._normalize_path(raw)
            if path:
                self._add_image(path)

    # ------------------------------------------------------------------
    def _normalize_path(self, raw: str) -> Optional[Path]:
        """Convert drag-and-drop payloads into usable file paths.

        Some Linux desktops (e.g., Linux Mint with tkdnd) deliver file URIs
        like ``file:///home/user/image.png`` instead of plain paths, and these
        fail the ``Path.exists`` check.  Normalising here keeps drop support
        consistent across platforms.
        """

        if not raw:
            return None

        raw = raw.strip()
        if raw.startswith("file://"):
            parsed = urlparse(raw)
            if parsed.scheme == "file" and parsed.path:
                return Path(unquote(parsed.path))
            return None

        return Path(raw)

    # ------------------------------------------------------------------
    def _open_file_dialog(self) -> None:  # pragma: no cover - UI callback
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"), ("All files", "*.*")]
        initialdir = str(self.project_assets_dir)
        selections = filedialog.askopenfilenames(
            parent=self,
            title="Select image(s)",
            filetypes=filetypes,
            initialdir=initialdir,
        )
        for raw in self.tk.splitlist(selections):
            self._add_image(Path(raw))

    # ------------------------------------------------------------------
    def _add_image(self, path: Path) -> None:
        if not path or not path.exists():
            return
        if path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            messagebox.showwarning("Unsupported file", f"{path} is not a supported image type.")
            return

        item = ParallaxItem(path)
        iid = self.tree.insert("", "end", values=item.as_row())
        self.items[iid] = item
        self.tree.selection_set(iid)

    # ------------------------------------------------------------------
    def _start_edit(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        iid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not iid or column not in {"#2", "#3", "#4"}:  # min, max, frequency
            return

        self._teardown_editor()
        item_bbox = self.tree.bbox(iid, column)
        if not item_bbox:
            return
        x, y, width, height = item_bbox
        current_value = self.tree.set(iid, column)
        self._editor_var.set(current_value)

        self._editor = ttk.Entry(self.tree, textvariable=self._editor_var)
        self._editor.place(x=x, y=y, width=width, height=height)
        self._editor.focus()
        self._editor.select_range(0, tk.END)
        self._editor.bind("<Return>", lambda e: self._commit_edit(iid, column))
        self._editor.bind("<FocusOut>", lambda e: self._commit_edit(iid, column))

    # ------------------------------------------------------------------
    def _commit_edit(self, iid: str, column: str) -> None:  # pragma: no cover - UI callback
        if not self._editor:
            return
        raw_value = self._editor_var.get().strip()
        try:
            numeric = float(raw_value)
        except ValueError:
            messagebox.showerror("Invalid value", "Please enter a numeric value.")
            self._editor.focus_set()
            return

        item = self.items.get(iid)
        if not item:
            self._teardown_editor()
            return

        attr = {"#2": "min_distance", "#3": "max_distance", "#4": "frequency"}[column]
        setattr(item, attr, numeric)
        self.tree.set(iid, column, f"{numeric:.2f}")
        self._teardown_editor()

    # ------------------------------------------------------------------
    def _teardown_editor(self) -> None:
        if self._editor is not None:
            self._editor.destroy()
            self._editor = None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallax Playground")
    parser.add_argument("--project", help="Path to the active project root")
    parser.add_argument("--project-name", help="Display name for the project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ParallaxPlaygroundApp(args)
    app.mainloop()


if __name__ == "__main__":
    main()
