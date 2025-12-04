#!/usr/bin/env python3
"""Interactive scene builder for experimenting with parallax layers."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

import imageio.v2 as imageio
import numpy as np
import tkinter as tk
from PIL import Image, ImageColor, ImageDraw, ImageTk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:  # Optional drag-and-drop support
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TkinterDnD = None
    DND_FILES = "DND_Files"


ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

if hasattr(Image, "Resampling"):
    RESAMPLE = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow < 9 fallback
    RESAMPLE = Image.LANCZOS


@dataclass
class ActiveInstance:
    """State for a single spawned sprite in the animation."""

    item_id: str
    depth: float
    x: float
    lateral_offset: float
    landscape_fraction: float


@dataclass
class ParallaxItem:
    """Represents a single image layer with distance/frequency settings."""

    image_path: Path
    min_distance: float = 1.0
    max_distance: float = 5.0
    frequency: float = 1.0
    scale: float = 1.0

    def as_row(self) -> tuple[str, str, str, str, str]:
        return (
            str(self.image_path),
            f"{self.min_distance:.2f}",
            f"{self.max_distance:.2f}",
            f"{self.frequency:.2f}",
            f"{self.scale:.2f}",
        )


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    fps: int = 24
    codec: str = "libx264"
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


BaseTkClass = TkinterDnD.Tk if TkinterDnD else tk.Tk


@dataclass
class PlacedItem:
    """Snapshot of an item placed on the preview landscape."""

    item_id: str
    landscape_fraction: float
    depth: float


@dataclass
class MapPoint:
    """Represents a persistent placement on the map canvas."""

    item_id: str
    x: float
    depth: float


@dataclass
class PerspectiveMath:
    """Shared helpers for projecting depth onto the 2D canvas."""

    width: int
    height: int
    horizon_height: float
    horizon_distance: float
    field_of_view: float
    foreground_cutoff: float = 0.0
    camera_height: Optional[float] = None

    def __post_init__(self) -> None:
        """Derive a sensible camera height if one was not provided."""

        if self.camera_height is None:
            # Keep the camera above the landscape so items can slide underneath.
            self.camera_height = max(0.5, self.horizon_distance * 0.18)

    @property
    def fov_radians(self) -> float:
        return math.radians(max(1e-3, self.field_of_view))

    @property
    def pitch_radians(self) -> float:
        return (self.horizon_height - 0.5) * self.fov_radians

    def horizon_y(self) -> float:
        return self.height * self.horizon_height

    def depth_progress(self, depth: float) -> float:
        span = self.height - self.horizon_y()
        if span <= 0:
            return 0.0
        return (self.depth_to_screen_y(depth) - self.horizon_y()) / span

    def depth_to_screen_y(self, depth: float) -> float:
        camera_height = self.camera_height or 1.0
        angle = math.atan2(-camera_height, max(depth, 1e-5)) + self.pitch_radians
        screen_fraction = 0.5 - angle / self.fov_radians
        return screen_fraction * self.height

    def angular_span(self, distance: float, unit_height: float = 1.0) -> float:
        """Return the angular size of a vertical unit at the given distance."""

        camera_height = self.camera_height or 1.0
        depth = max(distance, 1e-5)
        bottom_angle = math.atan2(-camera_height, depth) + self.pitch_radians
        top_angle = math.atan2(unit_height - camera_height, depth) + self.pitch_radians
        return max(1e-6, top_angle - bottom_angle)

    def projected_scale(self, depth: float, base_scale: float = 0.1) -> float:
        zero_distance = max(self.foreground_cutoff, 0.01)
        reference_span = self.angular_span(zero_distance)
        span = self.angular_span(max(depth, 0.01))
        fov_factor = max(0.4, min(1.6, self.field_of_view / 60.0))
        base = (span / reference_span) * base_scale * fov_factor
        return max(0.02, min(base, 6.0))

    def depth_step_for_spacing(self, pixel_spacing: float) -> float:
        return pixel_spacing * self.horizon_distance / max(self.height - self.horizon_y(), 1e-3)

    def landscape_width_at_depth(self, depth: float) -> float:
        """Estimate the visible landscape width at a given depth.

        This lets us translate between a placement point on the ground plane and the
        screen-space x coordinate for the current camera position.
        """

        depth = max(depth, 1e-3)
        return self.width * max(1.0, self.horizon_distance / depth)


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
        self.placed_items: list[PlacedItem] = []
        self._item_choice_lookup: Dict[str, str] = {}
        self.map_points: list[MapPoint] = []
        self.map_width: float = 0.0
        self.map_depth: float = 0.0
        self.map_horizon_distance: float = 0.0
        self.map_camera_start_x: float = 0.0
        self.map_camera_end_x: float = 0.0
        self.map_camera_start_depth: float = 0.0
        self.map_camera_end_depth: float = 0.0
        self.item_colors: Dict[str, str] = {}
        self._editor: Optional[ttk.Entry] = None
        self._editor_var = tk.StringVar()

        self.direction_var = tk.StringVar(value="right")
        self.distance_var = tk.DoubleVar(value=50.0)
        self.rate_var = tk.DoubleVar(value=10.0)
        self.horizon_height_var = tk.DoubleVar(value=0.5)
        self.distance_to_horizon_var = tk.DoubleVar(value=100.0)
        self.field_of_view_var = tk.DoubleVar(value=60.0)
        self.horizon_fog_var = tk.DoubleVar(value=0.25)
        self.horizon_fog_depth_var = tk.DoubleVar(value=50.0)
        self.foreground_cutoff_var = tk.DoubleVar(value=5.0)
        self.grid_enabled_var = tk.BooleanVar(value=True)
        self.grid_color_var = tk.StringVar(value="#0b3d0b")
        self.grid_background_color_var = tk.StringVar(value="#050505")
        self.grid_vertical_spacing_var = tk.DoubleVar(value=160.0)
        self.grid_depth_spacing_var = tk.DoubleVar(value=10.0)
        self.duration_var = tk.StringVar()
        self.preview_distance_var = tk.DoubleVar(value=10.0)
        self.view_mode_var = tk.StringVar(value="preview")
        self.map_zoom_var = tk.DoubleVar(value=1.0)
        self.brush_size_var = tk.DoubleVar(value=120.0)
        self.density_var = tk.DoubleVar(value=1.0)
        self.conformity_var = tk.DoubleVar(value=0.6)
        self.min_separation_var = tk.DoubleVar(value=18.0)
        self.draw_tool_var = tk.StringVar(value="single")
        self.selected_item_var = tk.StringVar()
        self.preview_frame_var = tk.IntVar(value=1)
        self.preview_frame_label_var = tk.StringVar(value="Frame: –")
        self.render_settings = RenderSettings()
        self.last_render_path: Optional[Path] = None
        self._preview_after_id: Optional[str] = None
        self._preview_image: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None
        self._map_canvas: Optional[tk.Canvas] = None
        self._map_image_id: Optional[int] = None
        self._map_start: Optional[tuple[float, float]] = None
        self._total_frames: int = 1
        self._render_start_frame: int = 1
        self._render_end_frame: int = 1

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
        columns = ("image", "min", "max", "frequency", "scale")
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
        self.tree.heading("scale", text="Scale")
        self.tree.column("image", anchor="w", width=320)
        self.tree.column("min", anchor="center", width=100)
        self.tree.column("max", anchor="center", width=100)
        self.tree.column("frequency", anchor="center", width=100)
        self.tree.column("scale", anchor="center", width=100)

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

        self._build_animation_settings(container)
        self._build_render_controls(container)

        self.tree.bind("<Double-1>", self._start_edit)

        self._update_duration()
        self._refresh_item_choices()

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
    def _build_animation_settings(self, container: ttk.Frame) -> None:
        settings = ttk.LabelFrame(container, text="Animation settings", padding=12)
        settings.pack(fill="x", pady=(12, 0))

        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)

        ttk.Label(settings, text="Direction of travel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        direction_choices = [
            "up",
            "down",
            "left",
            "right",
            "clock-wise",
            "counter clock-wise",
        ]
        direction_menu = ttk.Combobox(
            settings,
            textvariable=self.direction_var,
            values=direction_choices,
            state="readonly",
            width=18,
        )
        direction_menu.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(settings, text="Distance to travel").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.distance_var).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(settings, text="Rate of travel").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.rate_var).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(settings, text="Horizon height (0-1)").grid(row=0, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.horizon_height_var).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(settings, text="Distance to horizon").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.distance_to_horizon_var).grid(row=1, column=3, sticky="ew", pady=4)

        ttk.Label(settings, text="Horizon fog (0-1)").grid(row=2, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.horizon_fog_var).grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Label(settings, text="Horizon fog depth").grid(row=3, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.horizon_fog_depth_var).grid(row=3, column=3, sticky="ew", pady=4)

        ttk.Label(settings, text="Foreground cut-off").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.foreground_cutoff_var).grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(settings, text="Show perspective grid", variable=self.grid_enabled_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(settings, text="Grid color").grid(row=4, column=2, sticky="w", padx=(16, 8), pady=(4, 0))
        ttk.Entry(settings, textvariable=self.grid_color_var).grid(row=4, column=3, sticky="ew", pady=(4, 0))

        ttk.Label(settings, text="Grid background").grid(row=5, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.grid_background_color_var).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Field of view").grid(row=5, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.field_of_view_var).grid(row=5, column=3, sticky="ew", pady=4)
        ttk.Label(settings, text="Vertical line spacing").grid(row=6, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.grid_vertical_spacing_var).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Depth line spacing").grid(row=6, column=2, sticky="w", padx=(16, 8), pady=4)
        ttk.Entry(settings, textvariable=self.grid_depth_spacing_var).grid(row=6, column=3, sticky="ew", pady=4)

        self.distance_var.trace_add("write", lambda *_: self._update_duration())
        self.rate_var.trace_add("write", lambda *_: self._update_duration())
        for var in (
            self.horizon_height_var,
            self.distance_to_horizon_var,
            self.field_of_view_var,
            self.grid_vertical_spacing_var,
            self.grid_depth_spacing_var,
            self.grid_color_var,
            self.grid_background_color_var,
            self.grid_enabled_var,
        ):
            var.trace_add("write", lambda *_: self._schedule_preview_refresh())

        info_bar = ttk.Frame(container)
        info_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(info_bar, textvariable=self.duration_var).pack(anchor="w")

        self._build_preview_panel(container)

    def _build_preview_panel(self, container: ttk.Frame) -> None:
        preview = ttk.LabelFrame(container, text="Preview and map", padding=12)
        preview.pack(fill="both", pady=(12, 0))

        notebook = ttk.Notebook(preview)
        notebook.pack(fill="both", expand=True)
        preview_tab = ttk.Frame(notebook)
        map_tab = ttk.Frame(notebook)
        notebook.add(preview_tab, text="Preview")
        notebook.add(map_tab, text="Map")
        notebook.bind("<<NotebookTabChanged>>", lambda _evt=None: self._on_view_changed(notebook))

        self._build_preview_tab(preview_tab)
        self._build_map_tab(map_tab)

    def _build_preview_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Item distance").pack(side="left")
        self.preview_distance_scale = ttk.Scale(
            controls,
            variable=self.preview_distance_var,
            from_=0.0,
            to=self.distance_to_horizon_var.get(),
            command=lambda _evt=None: self._update_preview_distance_label(),
        )
        self.preview_distance_scale.pack(side="left", fill="x", expand=True, padx=8)
        self.preview_distance_label = ttk.Label(controls, text="0.00")
        self.preview_distance_label.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Place items", command=self._preview_place_items).pack(side="left")

        frame_controls = ttk.Frame(parent)
        frame_controls.pack(fill="x", pady=(0, 8))
        ttk.Label(frame_controls, text="Frame").pack(side="left")
        self.preview_frame_scale = ttk.Scale(
            frame_controls,
            variable=self.preview_frame_var,
            from_=1,
            to=1,
            command=lambda _evt=None: self._update_frame_selection_label(),
        )
        self.preview_frame_scale.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(frame_controls, textvariable=self.preview_frame_label_var).pack(side="left", padx=(0, 8))
        ttk.Button(frame_controls, text="Draw frame", command=self._draw_frame_preview).pack(side="left")

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        self.preview_canvas = tk.Canvas(canvas_frame, width=780, height=260, highlightthickness=1, highlightbackground="#333")
        self.preview_canvas.pack(fill="both", expand=True)
        self._update_preview_distance_label()
        self._update_duration()
        self._schedule_preview_refresh()

    def _build_map_tab(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Create map", command=self._create_map).pack(side="left")
        ttk.Button(controls, text="Auto generate", command=self._auto_generate_map).pack(side="left", padx=6)
        ttk.Button(controls, text="Save map", command=self._save_map).pack(side="left", padx=6)
        ttk.Button(controls, text="Load map", command=self._load_map).pack(side="left", padx=6)
        ttk.Label(controls, textvariable=self.duration_var).pack(side="right")

        item_row = ttk.Frame(parent)
        item_row.pack(fill="x", pady=(0, 8))
        ttk.Label(item_row, text="Item").grid(row=0, column=0, padx=(0, 6))
        self.item_choice = ttk.Combobox(item_row, textvariable=self.selected_item_var, state="readonly", width=30)
        self.item_choice.grid(row=0, column=1, padx=(0, 12), sticky="ew")
        ttk.Label(item_row, text="Tool").grid(row=0, column=2, padx=(0, 6))
        ttk.Combobox(
            item_row,
            textvariable=self.draw_tool_var,
            values=["single", "spray", "line", "square"],
            state="readonly",
            width=12,
        ).grid(row=0, column=3, padx=(0, 12))
        ttk.Label(item_row, text="Brush").grid(row=0, column=4, padx=(0, 6))
        ttk.Entry(item_row, textvariable=self.brush_size_var, width=8).grid(row=0, column=5)
        ttk.Label(item_row, text="Density").grid(row=0, column=6, padx=(12, 6))
        ttk.Entry(item_row, textvariable=self.density_var, width=8).grid(row=0, column=7)
        ttk.Label(item_row, text="Conformity (0-1)").grid(row=0, column=8, padx=(12, 6))
        ttk.Entry(item_row, textvariable=self.conformity_var, width=8).grid(row=0, column=9)
        ttk.Label(item_row, text="Min spacing").grid(row=0, column=10, padx=(12, 6))
        ttk.Entry(item_row, textvariable=self.min_separation_var, width=8).grid(row=0, column=11)

        zoom_row = ttk.Frame(parent)
        zoom_row.pack(fill="x", pady=(0, 8))
        ttk.Button(zoom_row, text="Zoom in", command=lambda: self._adjust_map_zoom(1.2)).pack(side="left")
        ttk.Button(zoom_row, text="Zoom out", command=lambda: self._adjust_map_zoom(0.8)).pack(side="left", padx=6)
        ttk.Label(zoom_row, text="View mode: map shows positions used for rendering").pack(side="left", padx=12)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True)
        self._map_canvas = tk.Canvas(
            canvas_frame,
            width=780,
            height=320,
            highlightthickness=1,
            highlightbackground="#333",
            background="#0a0a0a",
        )
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self._map_canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self._map_canvas.yview)
        self._map_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self._map_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self._map_canvas.bind("<ButtonPress-1>", self._on_map_press)
        self._map_canvas.bind("<B1-Motion>", self._on_map_drag)
        self._map_canvas.bind("<ButtonRelease-1>", self._on_map_release)

        legend = ttk.Frame(parent)
        legend.pack(fill="x", pady=(8, 0))
        ttk.Label(legend, text="Item key:").pack(side="left")
        self._legend_canvas = tk.Canvas(legend, height=28, highlightthickness=0, background=self._map_canvas["background"])
        self._legend_canvas.pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _on_view_changed(self, notebook: ttk.Notebook) -> None:
        tab_text = notebook.tab(notebook.select(), "text")
        self.view_mode_var.set("map" if tab_text == "Map" else "preview")
        if self.view_mode_var.get() == "map":
            self._render_map()
        else:
            self._schedule_preview_refresh()

    def _assign_color(self, item_id: str) -> str:
        if item_id in self.item_colors:
            return self.item_colors[item_id]
        palette = [
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
        ]
        rng = random.Random(hash(item_id) & 0xFFFFFFFF)
        base_color = rng.choice(palette)
        self.item_colors[item_id] = base_color
        return base_color

    def _refresh_item_choices(self) -> None:
        labels = []
        self._item_choice_lookup = {}
        for iid, item in self.items.items():
            label = f"{Path(item.image_path).name} ({iid})"
            labels.append(label)
            self._item_choice_lookup[label] = iid
            self._assign_color(iid)
        self.item_choice.configure(values=labels)
        if labels and self.selected_item_var.get() not in labels:
            self.selected_item_var.set(labels[0])
        self._render_item_legend()

    def _render_item_legend(self) -> None:
        if not hasattr(self, "_legend_canvas"):
            return
        canvas: tk.Canvas = self._legend_canvas
        canvas.delete("all")
        x = 6
        y = 4
        values = self.item_choice.cget("values") or []
        for label in values:
            iid = self._item_choice_lookup.get(label)
            if not iid:
                continue
            color = self.item_colors.get(iid, "#ccc")
            canvas.create_rectangle(x, y, x + 18, y + 18, fill=color, outline="")
            canvas.create_text(x + 24, y + 9, text=label, anchor="w", fill="#ddd")
            x += 160
        canvas.configure(scrollregion=(0, 0, x + 20, 28))

    def _adjust_map_zoom(self, factor: float) -> None:
        new_zoom = min(6.0, max(0.2, self.map_zoom_var.get() * factor))
        self.map_zoom_var.set(new_zoom)
        self._render_map()

    def _view_width_at_distance(self, distance: float, fov_degrees: float) -> float:
        half_angle = math.radians(max(1.0, fov_degrees) / 2)
        return max(20.0, math.tan(half_angle) * distance * 2)

    def _map_to_canvas(self, x: float, depth: float) -> tuple[float, float]:
        scale = self.map_zoom_var.get()
        return x * scale, depth * scale

    def _map_coords_from_event(self, event: tk.Event) -> tuple[float, float]:
        if not self._map_canvas:
            return (0.0, 0.0)
        scale = self.map_zoom_var.get() or 1.0
        canvas_x = self._map_canvas.canvasx(event.x)
        canvas_y = self._map_canvas.canvasy(event.y)
        return canvas_x / scale, canvas_y / scale

    def _create_map(self) -> None:
        horizon_distance = max(1.0, self._safe_float(self.distance_to_horizon_var) or 1.0)
        travel_distance = abs(self._safe_float(self.distance_var) or 0.0)
        fov_value = max(1.0, self._safe_float(self.field_of_view_var) or 60.0)
        foreground_depth = max(0.01, self._safe_float(self.foreground_cutoff_var) or 1.0)

        horizon_width = self._view_width_at_distance(horizon_distance, fov_value)
        foreground_width = self._view_width_at_distance(foreground_depth, fov_value)
        half_width = max(horizon_width, foreground_width) / 2

        self.map_horizon_distance = horizon_distance
        self.map_points = []

        direction = self.direction_var.get()
        start_depth = 0.0
        end_depth = 0.0
        if direction == "up":
            end_depth = travel_distance
        elif direction == "down":
            start_depth = travel_distance

        if direction in {"right", "clock-wise"}:
            start_x = half_width
            end_x = start_x + travel_distance
        elif direction in {"left", "counter clock-wise"}:
            end_x = half_width
            start_x = end_x + travel_distance
        else:
            start_x = end_x = half_width

        max_depth = max(start_depth, end_depth)
        max_center = max(start_x, end_x)
        self.map_width = max_center + half_width
        self.map_depth = max_depth + horizon_distance
        self.map_camera_start_x = start_x
        self.map_camera_end_x = end_x
        self.map_camera_start_depth = start_depth
        self.map_camera_end_depth = end_depth
        self._render_map()
        self._update_duration()

    def _auto_generate_map(self) -> None:
        if not self._map_ready():
            messagebox.showwarning("Auto generate", "Create a map first so we know its bounds.", parent=self)
            return
        rng = random.Random()
        min_spacing = max(0.0, self.min_separation_var.get())
        for iid, item in self.items.items():
            count = max(1, int(round(item.frequency)))
            for _ in range(count):
                attempts = 0
                while attempts < 50:
                    depth = rng.uniform(item.min_distance, item.max_distance)
                    x = rng.uniform(0, self.map_width)
                    if depth > self.map_depth:
                        depth = self.map_depth
                    if self._map_point_too_close(x, depth, min_spacing):
                        attempts += 1
                        continue
                    self.map_points.append(MapPoint(item_id=iid, x=x, depth=depth))
                    break
        self._render_map()

    def _map_ready(self) -> bool:
        return self.map_width > 0 and self.map_depth > 0 and self._map_canvas is not None

    def _save_map(self) -> None:  # pragma: no cover - UI callback
        if not self.map_points:
            messagebox.showinfo("Save map", "No points to save yet.", parent=self)
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            parent=self,
            title="Save map",
        )
        if not file_path:
            return
        data = {
            "map_width": self.map_width,
            "map_depth": self.map_depth,
            "points": [
                {
                    "item_path": str(self.items[pt.item_id].image_path),
                    "x": pt.x,
                    "depth": pt.depth,
                }
                for pt in self.map_points
                if pt.item_id in self.items
            ],
        }
        Path(file_path).write_text(json.dumps(data, indent=2))
        messagebox.showinfo("Save map", f"Saved {len(data['points'])} points to {file_path}", parent=self)

    def _load_map(self) -> None:  # pragma: no cover - UI callback
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            parent=self,
            title="Load map",
        )
        if not file_path:
            return
        try:
            data = json.loads(Path(file_path).read_text())
        except Exception as exc:
            messagebox.showerror("Load map", f"Failed to read map: {exc}", parent=self)
            return
        self.map_width = float(data.get("map_width", 0.0))
        self.map_depth = float(data.get("map_depth", 0.0))
        self.map_points = []
        for entry in data.get("points", []):
            path = Path(entry.get("item_path", ""))
            existing_iid = next((iid for iid, item in self.items.items() if item.image_path == path), None)
            if existing_iid is None:
                if path.exists():
                    self._add_image(path)
                    existing_iid = next((iid for iid, item in self.items.items() if item.image_path == path), None)
                else:
                    continue
            if existing_iid:
                self.map_points.append(
                    MapPoint(item_id=existing_iid, x=float(entry.get("x", 0.0)), depth=float(entry.get("depth", 0.0)))
                )
        self._render_map()
        self._update_duration()

    def _map_point_too_close(self, x: float, depth: float, spacing: float) -> bool:
        for point in self.map_points:
            if math.hypot(point.x - x, point.depth - depth) < spacing:
                return True
        return False

    def _on_map_press(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        if not self._map_ready():
            return
        self._map_start = self._map_coords_from_event(event)
        if self._map_canvas:
            self._map_canvas.delete("preview")

    def _on_map_drag(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        if not self._map_ready() or not self._map_start:
            return
        current = self._map_coords_from_event(event)
        if not self._map_canvas:
            return
        self._map_canvas.delete("preview")
        scale = self.map_zoom_var.get()
        start_canvas = self._map_to_canvas(*self._map_start)
        current_canvas = self._map_to_canvas(*current)
        tool = self.draw_tool_var.get()
        if tool == "line":
            self._map_canvas.create_line(*start_canvas, *current_canvas, fill="#bbbbbb", dash=(4, 2), tags="preview")
        elif tool == "square":
            self._map_canvas.create_rectangle(
                start_canvas[0],
                start_canvas[1],
                current_canvas[0],
                current_canvas[1],
                outline="#bbbbbb",
                dash=(4, 2),
                tags="preview",
            )
        elif tool == "spray":
            radius = max(6.0, (self.brush_size_var.get() or 0.0) * scale / 2)
            self._map_canvas.create_oval(
                current_canvas[0] - radius,
                current_canvas[1] - radius,
                current_canvas[0] + radius,
                current_canvas[1] + radius,
                outline="#bbbbbb",
                tags="preview",
            )

    def _on_map_release(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        if not self._map_ready() or self._map_start is None:
            return
        end = self._map_coords_from_event(event)
        start = self._map_start
        self._map_start = None
        if self._map_canvas:
            self._map_canvas.delete("preview")
        self._apply_draw_tool(start, end)

    def _resolve_selected_item_id(self) -> Optional[str]:
        label = self.selected_item_var.get()
        if label in self._item_choice_lookup:
            return self._item_choice_lookup[label]
        if label in self.items:
            return label
        return None

    def _apply_draw_tool(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        item_id = self._resolve_selected_item_id()
        if not item_id:
            messagebox.showwarning("Map", "Select an item to place on the map first.", parent=self)
            return
        if not self._map_ready():
            return
        density = max(0.1, float(self.density_var.get()))
        conformity = min(1.0, max(0.0, float(self.conformity_var.get())))
        spacing = max(0.0, float(self.min_separation_var.get()))
        brush = max(4.0, float(self.brush_size_var.get()))
        tool = self.draw_tool_var.get()

        def add_point(x: float, depth: float) -> None:
            x = min(max(0.0, x), self.map_width)
            depth = min(max(0.0, depth), self.map_depth)
            if not self._map_point_too_close(x, depth, spacing):
                self.map_points.append(MapPoint(item_id=item_id, x=x, depth=depth))

        if tool == "single":
            add_point(*end)
        elif tool == "spray":
            cx, cy = end
            radius = brush / 2
            area = math.pi * radius * radius
            count = max(1, int(area / 800.0 * density))
            rng = random.Random()
            for _ in range(count):
                angle = rng.random() * math.tau
                dist = radius * math.pow(rng.random(), conformity)
                px = cx + math.cos(angle) * dist
                py = cy + math.sin(angle) * dist
                add_point(px, py)
        elif tool == "line":
            sx, sy = start
            ex, ey = end
            length = max(1e-3, math.hypot(ex - sx, ey - sy))
            step = max(6.0, (50.0 / density))
            count = max(2, int(length / step))
            for idx in range(count + 1):
                t = idx / max(count, 1)
                px = sx + (ex - sx) * t
                py = sy + (ey - sy) * t
                jitter = (1 - conformity) * brush * 0.1
                if jitter:
                    px += random.uniform(-jitter, jitter)
                    py += random.uniform(-jitter, jitter)
                add_point(px, py)
        elif tool == "square":
            min_x, max_x = sorted([start[0], end[0]])
            min_y, max_y = sorted([start[1], end[1]])
            width = max(1.0, max_x - min_x)
            height = max(1.0, max_y - min_y)
            spacing_step = max(10.0, 80.0 / density)
            jitter = (1 - conformity) * spacing_step * 0.5
            y = min_y
            while y <= max_y:
                x = min_x
                while x <= max_x:
                    px = x + random.uniform(-jitter, jitter)
                    py = y + random.uniform(-jitter, jitter)
                    add_point(px, py)
                    x += spacing_step
                y += spacing_step
        self._render_map()

    def _blend_canvas_color(self, color: str, background: str) -> str:
        """Convert RGBA hex colors to a Tk-friendly RGB string by blending with a background."""

        def hex_to_rgb(value: str) -> tuple[int, int, int]:
            value = value.lstrip("#")
            if len(value) != 6:
                raise ValueError("Expected #RRGGBB format")
            return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

        color = (color or "").strip()
        background = (background or "#000000").strip()

        try:
            if color.startswith("#") and len(color) == 9:
                fg = color[1:7]
                alpha = int(color[7:9], 16) / 255.0
                fg_r, fg_g, fg_b = hex_to_rgb(fg)
                bg_r, bg_g, bg_b = hex_to_rgb(background)
                blended = (
                    round(fg_r * alpha + bg_r * (1 - alpha)),
                    round(fg_g * alpha + bg_g * (1 - alpha)),
                    round(fg_b * alpha + bg_b * (1 - alpha)),
                )
                return "#%02x%02x%02x" % blended

            if color.startswith("#") and len(color) == 7:
                return color
        except ValueError:
            pass

        return background

    def _render_map(self) -> None:
        if not self._map_canvas:
            return
        canvas = self._map_canvas
        canvas.delete("all")
        if self.map_width <= 0 or self.map_depth <= 0:
            canvas.create_text(10, 10, text="Click 'Create map' to size the map.", anchor="nw", fill="#b0b0b0")
            return
        scale = self.map_zoom_var.get()
        canvas.configure(scrollregion=(0, 0, self.map_width * scale, self.map_depth * scale))
        map_background = "#0a0a0a"
        canvas.create_rectangle(0, 0, self.map_width * scale, self.map_depth * scale, fill=map_background, outline="#222")
        def to_canvas(pt: tuple[float, float]) -> tuple[float, float]:
            return self._map_to_canvas(pt[0], pt[1])

        fov_value = self._safe_float(self.field_of_view_var) or 60.0
        foreground_depth = max(0.01, self._safe_float(self.foreground_cutoff_var) or 1.0)
        horizon_distance = self.map_horizon_distance or self.map_depth
        foreground_width = self._view_width_at_distance(foreground_depth, fov_value)
        horizon_width = self._view_width_at_distance(horizon_distance, fov_value)
        start = (self.map_camera_start_x, max(0.0, self.map_camera_start_depth))
        end = (self.map_camera_end_x, max(0.0, self.map_camera_end_depth))

        def frustum_points(center: tuple[float, float]) -> list[tuple[float, float]]:
            center_x, camera_depth = center
            near_depth = camera_depth + foreground_depth
            far_depth = camera_depth + horizon_distance
            left_bottom = to_canvas((center_x - foreground_width * 0.5, near_depth))
            right_bottom = to_canvas((center_x + foreground_width * 0.5, near_depth))
            right_top = to_canvas((center_x + horizon_width * 0.5, far_depth))
            left_top = to_canvas((center_x - horizon_width * 0.5, far_depth))
            return [left_bottom, right_bottom, right_top, left_top]

        start_points = frustum_points(start)
        end_points = frustum_points(end)
        triangle_color = self._blend_canvas_color("#f2e55ca0", map_background)
        end_color = self._blend_canvas_color("#fff7b54a", map_background)
        canvas.create_polygon(*[coord for pt in start_points for coord in pt], fill=triangle_color, outline="")
        canvas.create_polygon(*[coord for pt in end_points for coord in pt], fill=end_color, outline="")

        line_color = self._blend_canvas_color("#f8f3c560", map_background)
        start_canvas = to_canvas(start)
        end_canvas = to_canvas(end)
        canvas.create_line(start_canvas[0], start_canvas[1], end_canvas[0], end_canvas[1], fill=line_color, dash=(3, 2))
        for idx in range(4):
            canvas.create_line(*start_points[idx], *end_points[idx], fill=line_color, dash=(3, 2))

        for point in self.map_points:
            color = self.item_colors.get(point.item_id, "#cccccc")
            cx, cy = to_canvas((point.x, point.depth))
            radius = max(4.0, 7.0 * scale * 0.6)
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=color, outline="#111")
        self._render_item_legend()

    # ------------------------------------------------------------------
    def _build_render_controls(self, container: ttk.Frame) -> None:
        ttk.Separator(container).pack(fill="x", pady=(12, 8))

        render_bar = ttk.Frame(container)
        render_bar.pack(fill="x")

        ttk.Button(render_bar, text="Render settings", command=self._open_render_settings).pack(side="left", padx=(0, 8))
        ttk.Button(render_bar, text="Render animation", command=self._render_animation).pack(side="left")
        ttk.Button(render_bar, text="Play last render", command=self._play_last_render).pack(side="left", padx=(8, 0))
        ttk.Button(render_bar, text="Save to assets", command=self._save_to_assets).pack(side="left", padx=(8, 0))

    # ------------------------------------------------------------------
    def _safe_float(self, var: tk.Variable) -> Optional[float]:
        try:
            return float(var.get())
        except (tk.TclError, ValueError):
            return None

    # ------------------------------------------------------------------
    def _collect_render_parameters(self, *, context: str) -> Optional[dict]:
        distance = self._safe_float(self.distance_var)
        rate = self._safe_float(self.rate_var)
        horizon_height = self._safe_float(self.horizon_height_var)
        horizon_distance = self._safe_float(self.distance_to_horizon_var)
        field_of_view = self._safe_float(self.field_of_view_var)
        fog_amount = self._safe_float(self.horizon_fog_var)
        fog_depth = self._safe_float(self.horizon_fog_depth_var)
        foreground_cutoff = self._safe_float(self.foreground_cutoff_var)
        grid_enabled = bool(self.grid_enabled_var.get())
        grid_color_raw = self.grid_color_var.get().strip() or "#0b3d0b"
        grid_background_raw = self.grid_background_color_var.get().strip() or "#050505"

        if None in {
            distance,
            rate,
            horizon_height,
            horizon_distance,
            fog_amount,
            fog_depth,
            foreground_cutoff,
            field_of_view,
        }:
            messagebox.showerror(context, "Please enter numeric animation settings before rendering.", parent=self)
            return None
        assert distance is not None and rate is not None
        assert horizon_height is not None and horizon_distance is not None and field_of_view is not None
        assert fog_amount is not None and fog_depth is not None and foreground_cutoff is not None

        if rate == 0:
            messagebox.showerror(context, "Rate of travel must be non-zero.", parent=self)
            return None
        if horizon_distance <= 0:
            messagebox.showerror(context, "Distance to horizon must be greater than zero.", parent=self)
            return None
        if not 0 <= horizon_height <= 1:
            messagebox.showerror(context, "Horizon height must be between 0 and 1.", parent=self)
            return None

        return {
            "distance": distance,
            "rate": rate,
            "horizon_height": horizon_height,
            "horizon_distance": horizon_distance,
            "field_of_view": field_of_view,
            "fog_amount": fog_amount,
            "fog_depth": fog_depth,
            "foreground_cutoff": foreground_cutoff,
            "grid_enabled": grid_enabled,
            "grid_color_raw": grid_color_raw,
            "grid_background_raw": grid_background_raw,
        }

    # ------------------------------------------------------------------
    def _load_render_images(
        self, items: list[tuple[str, ParallaxItem]], *, context: str
    ) -> Optional[Dict[str, tuple[ParallaxItem, Image.Image]]]:
        try:
            return {iid: (item, Image.open(item.image_path).convert("RGBA")) for iid, item in items}
        except FileNotFoundError:
            messagebox.showerror(context, "One or more image paths are missing.", parent=self)
            return None
        except Exception as exc:  # pragma: no cover - runtime guard for unexpected image errors
            messagebox.showerror(context, f"Failed to load images: {exc}", parent=self)
            return None

    # ------------------------------------------------------------------
    def _update_frame_selection_label(self) -> None:
        if not hasattr(self, "preview_frame_scale"):
            return
        try:
            value = int(self.preview_frame_var.get())
        except (tk.TclError, ValueError):
            value = 1
        value = min(max(value, 1), max(1, self._total_frames))
        self.preview_frame_var.set(value)
        range_text = f"rendering {self._render_start_frame}-{self._render_end_frame}"
        self.preview_frame_label_var.set(f"Frame: {value}/{self._total_frames} ({range_text})")

    # ------------------------------------------------------------------
    def _sync_frame_controls(self, total_frames: Optional[int], start_frame: int = 1, end_frame: int = 1) -> None:
        if not hasattr(self, "preview_frame_scale"):
            return
        if total_frames is None or total_frames <= 0:
            self._total_frames = 1
            self._render_start_frame = 1
            self._render_end_frame = 1
            self.preview_frame_scale.configure(from_=1, to=1)
            self.preview_frame_var.set(1)
            self.preview_frame_label_var.set("Frame: –")
            return
        self._total_frames = total_frames
        self._render_start_frame = start_frame
        self._render_end_frame = end_frame
        self.preview_frame_scale.configure(from_=1, to=self._total_frames)
        self._update_frame_selection_label()

    # ------------------------------------------------------------------
    def _update_duration(self) -> None:
        distance = self._safe_float(self.distance_var)
        rate = self._safe_float(self.rate_var)
        fps = max(0, int(self.render_settings.fps))

        if distance is None or rate is None:
            self.duration_var.set("Duration: – (enter numeric values)")
            self._sync_frame_controls(None)
            return

        if rate == 0:
            self.duration_var.set("Duration: ∞ (rate must be non-zero)")
            self._sync_frame_controls(None)
            return

        if fps <= 0:
            self.duration_var.set("Duration: – (FPS must be positive)")
            self._sync_frame_controls(None)
            return

        map_travel = math.hypot(
            self.map_camera_end_x - self.map_camera_start_x,
            self.map_camera_end_depth - self.map_camera_start_depth,
        )
        duration = max(map_travel, abs(distance)) / abs(rate)
        total_frames = max(1, math.ceil(duration * fps))
        start_frame = max(1, self.render_settings.start_frame or 1)
        end_frame = min(self.render_settings.end_frame or total_frames, total_frames)
        range_text = ""
        if start_frame != 1 or end_frame != total_frames:
            range_text = f" Rendering frames {start_frame}-{end_frame}."
        self.duration_var.set(
            f"Duration: {duration:.2f}s @ {fps} fps ({total_frames} frames).{range_text}"
        )
        self._sync_frame_controls(total_frames, start_frame, end_frame)

    # ------------------------------------------------------------------
    def _update_preview_distance_label(self) -> None:
        value = self._safe_float(self.preview_distance_var) or 0.0
        self.preview_distance_label.configure(text=f"{value:.2f}")

    # ------------------------------------------------------------------
    def _schedule_preview_refresh(self) -> None:
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.after(120, self._render_preview_background)

    # ------------------------------------------------------------------
    def _preview_geometry(self, width: int, height: int) -> Optional[PerspectiveMath]:
        horizon_height = self._safe_float(self.horizon_height_var)
        horizon_distance = self._safe_float(self.distance_to_horizon_var)
        field_of_view = self._safe_float(self.field_of_view_var)
        foreground_cutoff = self._safe_float(self.foreground_cutoff_var) or 0.0
        if None in {horizon_height, horizon_distance, field_of_view}:
            return None
        assert horizon_height is not None and horizon_distance is not None and field_of_view is not None
        horizon_height = min(1.0, max(0.0, horizon_height))
        horizon_distance = max(0.01, horizon_distance)
        field_of_view = max(1.0, field_of_view)
        return PerspectiveMath(
            width=width,
            height=height,
            horizon_height=horizon_height,
            horizon_distance=horizon_distance,
            field_of_view=field_of_view,
            foreground_cutoff=max(0.0, foreground_cutoff),
        )

    # ------------------------------------------------------------------
    def _render_preview_background(self, include_items: bool = False, record_positions: bool = False) -> None:
        self._preview_after_id = None
        width = max(int(self.preview_canvas.winfo_width()) or 0, 780)
        height = max(int(self.preview_canvas.winfo_height()) or 0, 260)
        geometry = self._preview_geometry(width, height)
        if geometry is None:
            return

        try:
            grid_color_rgb = ImageColor.getrgb(self.grid_color_var.get().strip() or "#0b3d0b")
        except ValueError:
            grid_color_rgb = ImageColor.getrgb("#0b3d0b")
        try:
            background_rgb = ImageColor.getrgb(self.grid_background_color_var.get().strip() or "#050505")
        except ValueError:
            background_rgb = ImageColor.getrgb("#050505")

        self.preview_distance_scale.configure(to=geometry.horizon_distance)
        current_distance = self._safe_float(self.preview_distance_var) or 0.0
        if current_distance > geometry.horizon_distance:
            self.preview_distance_var.set(geometry.horizon_distance)
        self._update_preview_distance_label()

        horizon_y = geometry.horizon_y()
        img = Image.new("RGBA", (width, height), "#0a0a0a")
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([(0, horizon_y), (width, height)], fill=background_rgb)
        grid_alpha_color = (*grid_color_rgb, 200)

        if self.grid_enabled_var.get():
            vanish_x = width / 2
            spacing = max(8.0, self._safe_float(self.grid_vertical_spacing_var) or (width / 10))
            factor = min(0.6, max(0.05, (self._safe_float(self.field_of_view_var) or 60.0) / 200))
            x = 0.0
            while x <= width + 1:
                top_x = vanish_x + (x - vanish_x) * factor
                draw.line([(x, height), (top_x, horizon_y)], fill=grid_alpha_color, width=1)
                x += spacing

            depth_spacing = max(0.5, self._safe_float(self.grid_depth_spacing_var) or 10.0)
            depth = geometry.foreground_cutoff
            while depth <= geometry.horizon_distance:
                y = geometry.depth_to_screen_y(depth)
                if y < horizon_y:
                    break
                draw.line([(0, y), (width, y)], fill=grid_alpha_color, width=1)
                depth += depth_spacing
            draw.line([(0, horizon_y), (width, horizon_y)], fill=grid_alpha_color, width=1)

        if include_items:
            self._overlay_preview_items(img, geometry, grid_color_rgb, record_positions=record_positions)

        self._preview_image = img
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_photo)

    # ------------------------------------------------------------------
    def _overlay_preview_items(
        self, base: Image.Image, geometry: PerspectiveMath, grid_color_rgb: tuple[int, int, int], record_positions: bool = False
    ) -> None:
        items = list(self.items.items())[:5]
        if not items:
            return
        if record_positions:
            self.placed_items = []
        try:
            loaded = [(iid, item, Image.open(item.image_path).convert("RGBA")) for iid, item in items]
        except FileNotFoundError:
            messagebox.showerror("Scene preview", "One or more image paths are missing.", parent=self)
            return
        except Exception as exc:
            messagebox.showerror("Scene preview", f"Failed to load images: {exc}", parent=self)
            return

        distance = max(0.0, min(geometry.horizon_distance, self._safe_float(self.preview_distance_var) or 0.0))
        y_line = geometry.depth_to_screen_y(distance)
        positions = [geometry.width * (idx + 1) / (len(loaded) + 1) for idx in range(len(loaded))]
        landscape_width = geometry.landscape_width_at_depth(distance)

        for x_pos, (_iid, item, image) in zip(positions, loaded):
            scale = geometry.projected_scale(max(distance, 0.01)) * max(item.scale, 0.0)
            target_w = int(max(1, image.width * scale))
            target_h = int(max(1, image.height * scale))
            resized = image.resize((target_w, target_h), RESAMPLE)
            top_left = (int(x_pos - target_w / 2), int(y_line - target_h))
            base.alpha_composite(resized, dest=top_left)
            if record_positions:
                landscape_x = x_pos - geometry.width / 2 + landscape_width / 2
                landscape_fraction = max(0.0, min(1.0, landscape_x / landscape_width))
                self.placed_items.append(
                    PlacedItem(item_id=_iid, landscape_fraction=landscape_fraction, depth=distance)
                )
            ImageDraw.Draw(base, "RGBA").line([(0, y_line), (geometry.width, y_line)], fill=(*grid_color_rgb, 120), width=1)

    # ------------------------------------------------------------------
    def _preview_place_items(self) -> None:
        self._render_preview_background(include_items=True, record_positions=True)

    # ------------------------------------------------------------------
    def _display_scaled_preview(self, frame: Image.Image) -> None:
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None

        canvas_width = max(int(self.preview_canvas.winfo_width()) or 0, 780)
        canvas_height = max(int(self.preview_canvas.winfo_height()) or 0, 260)
        scale = min(canvas_width / max(frame.width, 1), canvas_height / max(frame.height, 1), 1.0)
        target_size = (
            max(1, int(frame.width * scale)),
            max(1, int(frame.height * scale)),
        )
        resized = frame.resize(target_size, RESAMPLE)
        composed = Image.new("RGBA", (canvas_width, canvas_height), "#0a0a0a")
        offset = ((canvas_width - target_size[0]) // 2, (canvas_height - target_size[1]) // 2)
        composed.paste(resized, offset)

        self._preview_image = composed
        self._preview_photo = ImageTk.PhotoImage(composed)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_photo)

    # ------------------------------------------------------------------
    def _draw_frame_preview(self) -> None:  # pragma: no cover - UI callback
        if not self.items:
            messagebox.showwarning("Preview frame", "Add at least one image before previewing.", parent=self)
            return
        if not self.map_points or not self._map_ready():
            messagebox.showwarning(
                "Preview frame",
                "Use the Map tab to lay out items and define a camera path before previewing.",
                parent=self,
            )
            return

        params = self._collect_render_parameters(context="Preview frame")
        if params is None:
            return

        fps = max(1, int(self.render_settings.fps))
        speed = abs(params["rate"])
        map_travel = math.hypot(
            self.map_camera_end_x - self.map_camera_start_x,
            self.map_camera_end_depth - self.map_camera_start_depth,
        )
        duration = max(map_travel, abs(params["distance"])) / max(speed, 1e-6)
        total_frames = max(1, math.ceil(duration * fps))
        start_frame = max(1, self.render_settings.start_frame or 1)
        end_frame = min(self.render_settings.end_frame or total_frames, total_frames)

        self._sync_frame_controls(total_frames, start_frame, end_frame)
        try:
            frame_number = int(self.preview_frame_var.get())
        except (tk.TclError, ValueError):
            frame_number = 1
        frame_number = min(max(frame_number, 1), total_frames)
        self.preview_frame_var.set(frame_number)
        self._update_frame_selection_label()

        geometry = PerspectiveMath(
            width=self.render_settings.width,
            height=self.render_settings.height,
            horizon_height=params["horizon_height"],
            horizon_distance=params["horizon_distance"],
            field_of_view=params["field_of_view"],
            foreground_cutoff=params["foreground_cutoff"],
        )

        loaded_items = self._load_render_images(list(self.items.items()), context="Preview frame")
        if loaded_items is None:
            return

        try:
            grid_color_rgb = ImageColor.getrgb(params["grid_color_raw"])
        except ValueError:
            grid_color_rgb = ImageColor.getrgb("#0b3d0b")
        try:
            background_rgb = ImageColor.getrgb(params["grid_background_raw"])
        except ValueError:
            background_rgb = ImageColor.getrgb("#050505")

        grid_vertical_spacing = max(8.0, self._safe_float(self.grid_vertical_spacing_var) or (self.render_settings.width / 10))
        depth_step = max(0.5, self._safe_float(self.grid_depth_spacing_var) or 10.0)

        frame = self._compose_render_frame(
            frame_number - 1,
            total_frames,
            geometry,
            loaded_items,
            grid_enabled=params["grid_enabled"],
            grid_color_rgb=grid_color_rgb,
            grid_background_rgb=background_rgb,
            grid_vertical_spacing=grid_vertical_spacing,
            depth_step=depth_step,
            fog_amount=params["fog_amount"],
            fog_depth=params["fog_depth"],
            foreground_cutoff=params["foreground_cutoff"],
            field_of_view=params["field_of_view"],
        )

        self._display_scaled_preview(frame)

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
        self._assign_color(iid)
        self._refresh_item_choices()
        self._render_map()

    # ------------------------------------------------------------------
    def _start_edit(self, event: tk.Event) -> None:  # pragma: no cover - UI callback
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        iid = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not iid or column not in {"#2", "#3", "#4", "#5"}:  # min, max, frequency, scale
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

        attr = {"#2": "min_distance", "#3": "max_distance", "#4": "frequency", "#5": "scale"}[column]
        setattr(item, attr, numeric)
        self.tree.set(iid, column, f"{numeric:.2f}")
        self._teardown_editor()

    # ------------------------------------------------------------------
    def _teardown_editor(self) -> None:
        if self._editor is not None:
            self._editor.destroy()
            self._editor = None

    # ------------------------------------------------------------------
    def _open_render_settings(self) -> None:  # pragma: no cover - UI callback
        dialog = RenderSettingsDialog(self, self.render_settings)
        if dialog.result_settings:
            self.render_settings = dialog.result_settings
            messagebox.showinfo(
                "Render settings",
                (
                    f"Updated: {self.render_settings.width}x{self.render_settings.height} "
                    f"@ {self.render_settings.fps}fps, codec {self.render_settings.codec}"
                ),
                parent=self,
            )
            self._update_duration()

    # ------------------------------------------------------------------
    def _compose_render_frame(
        self,
        frame_idx: int,
        total_frames: int,
        geometry: PerspectiveMath,
        loaded_items: Dict[str, tuple[ParallaxItem, Image.Image]],
        *,
        grid_enabled: bool,
        grid_color_rgb: tuple[int, int, int],
        grid_background_rgb: tuple[int, int, int],
        grid_vertical_spacing: float,
        depth_step: float,
        fog_amount: float,
        fog_depth: float,
        foreground_cutoff: float,
        field_of_view: float,
    ) -> Image.Image:
        fog_start = max(0.0, geometry.horizon_distance - fog_depth)
        horizon_y = geometry.horizon_y()
        horizon_x = geometry.width / 2
        grid_alpha_color = (*grid_color_rgb, 140)

        def fog_multiplier(depth: float) -> float:
            if depth <= fog_start:
                return 1.0
            fog_ratio = min(1.0, (depth - fog_start) / max(fog_depth, 1e-3))
            return 1.0 - fog_ratio * fog_amount

        camera_start_x = self.map_camera_start_x or (self.map_width / 2)
        camera_end_x = self.map_camera_end_x or camera_start_x
        camera_start_depth = self.map_camera_start_depth
        camera_end_depth = self.map_camera_end_depth

        progress = frame_idx / max(total_frames - 1, 1)
        camera_x = camera_start_x + (camera_end_x - camera_start_x) * progress
        camera_depth = camera_start_depth + (camera_end_depth - camera_start_depth) * progress

        frame_active: list[ActiveInstance] = []
        for point in self.map_points:
            if point.item_id not in loaded_items:
                continue
            relative_depth = point.depth - camera_depth
            if relative_depth <= foreground_cutoff or relative_depth >= geometry.horizon_distance + fog_depth:
                continue
            visible_width = self._view_width_at_distance(relative_depth, field_of_view)
            x_offset = point.x - camera_x
            normalized = x_offset / max(visible_width / 2, 1e-6)
            screen_x = (geometry.width / 2) + normalized * (geometry.width / 2)
            frame_active.append(
                ActiveInstance(
                    item_id=point.item_id,
                    depth=relative_depth,
                    x=screen_x,
                    lateral_offset=0.0,
                    landscape_fraction=0.5,
                )
            )

        frame = Image.new("RGBA", (geometry.width, geometry.height), "black")
        draw: Optional[ImageDraw.ImageDraw] = None
        if grid_enabled:
            draw = ImageDraw.Draw(frame, "RGBA")
            draw.rectangle([(0, horizon_y), (geometry.width, geometry.height)], fill=grid_background_rgb)
            x = 0.0
            while x <= geometry.width + 1:
                draw.line([(x, geometry.height), (horizon_x, horizon_y)], fill=grid_alpha_color, width=1)
                x += grid_vertical_spacing

            depth = foreground_cutoff
            while depth <= geometry.horizon_distance:
                y = geometry.depth_to_screen_y(depth)
                if y < horizon_y:
                    break
                draw.line([(0, y), (geometry.width, y)], fill=grid_alpha_color, width=1)
                depth += depth_step
            draw.line([(0, horizon_y), (geometry.width, horizon_y)], fill=grid_alpha_color, width=1)

        draw_list = sorted(frame_active, key=lambda inst: inst.depth, reverse=True)
        draw_context: Optional[ImageDraw.ImageDraw] = draw if grid_enabled else None
        for inst in draw_list:
            item, image = loaded_items[inst.item_id]
            scale = geometry.projected_scale(inst.depth) * max(item.scale, 0.0)
            target_w = max(1, int(image.width * scale))
            target_h = max(1, int(image.height * scale))
            ground_y = geometry.depth_to_screen_y(inst.depth) + inst.lateral_offset
            top_y = ground_y - target_h
            top_y = max(-target_h, min(geometry.height + target_h, top_y))
            alpha = fog_multiplier(inst.depth)
            bbox = (
                inst.x - target_w / 2,
                top_y,
                inst.x + target_w / 2,
                top_y + target_h,
            )
            if bbox[2] < 0 or bbox[0] > geometry.width:
                continue
            if bbox[3] < 0 or bbox[1] > geometry.height:
                continue
            if target_w < 4 or target_h < 4:
                if draw_context is None:
                    draw_context = ImageDraw.Draw(frame, "RGBA")
                alpha_int = int(alpha * 255)
                shade = (40, 40, 40, alpha_int)
                draw_context.rectangle(
                    [(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))],
                    fill=shade,
                )
                continue
            scaled = image.resize((target_w, target_h), RESAMPLE)
            if alpha < 1.0:
                alpha_layer = scaled.split()[3].point(lambda a: int(a * alpha))
                scaled.putalpha(alpha_layer)
            frame.alpha_composite(
                scaled,
                (int(inst.x - target_w / 2), int(top_y)),
            )
        return frame

    # ------------------------------------------------------------------
    def _render_animation(self) -> None:  # pragma: no cover - UI callback
        items = list(self.items.items())
        if not items:
            messagebox.showwarning("Render animation", "Add at least one image before rendering.", parent=self)
            return
        if not self.map_points or not self._map_ready():
            messagebox.showwarning(
                "Render animation",
                "Use the Map tab to lay out items and define a camera path before rendering.",
                parent=self,
            )
            return
        params = self._collect_render_parameters(context="Render animation")
        if params is None:
            return

        fps = max(1, int(self.render_settings.fps))
        speed = abs(params["rate"])
        map_travel = math.hypot(
            self.map_camera_end_x - self.map_camera_start_x,
            self.map_camera_end_depth - self.map_camera_start_depth,
        )
        duration = max(map_travel, abs(params["distance"])) / max(speed, 1e-6)
        total_frames = max(1, math.ceil(duration * fps))
        start_frame = self.render_settings.start_frame or 1
        end_frame = self.render_settings.end_frame or total_frames
        end_frame = min(end_frame, total_frames)
        if start_frame > end_frame:
            messagebox.showerror("Render animation", "Start frame must be before end frame.", parent=self)
            return

        geometry = PerspectiveMath(
            width=self.render_settings.width,
            height=self.render_settings.height,
            horizon_height=params["horizon_height"],
            horizon_distance=params["horizon_distance"],
            field_of_view=params["field_of_view"],
            foreground_cutoff=params["foreground_cutoff"],
        )
        try:
            grid_color_rgb = ImageColor.getrgb(params["grid_color_raw"])
        except ValueError:
            grid_color_rgb = ImageColor.getrgb("#0b3d0b")

        loaded_items = self._load_render_images(items, context="Render animation")
        if loaded_items is None:
            return

        output_name = f"parallax_render_{int(time.time())}.mp4"
        output_path = self.project_assets_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None

        try:
            background_rgb = ImageColor.getrgb(params["grid_background_raw"])
        except ValueError:
            background_rgb = ImageColor.getrgb("#050505")

        progress_interval = max(1, total_frames // 20)
        grid_vertical_spacing = max(8.0, self._safe_float(self.grid_vertical_spacing_var) or (self.render_settings.width / 10))
        depth_step = max(0.5, self._safe_float(self.grid_depth_spacing_var) or 10.0)

        for frame_idx in range(total_frames):
            frame = self._compose_render_frame(
                frame_idx,
                total_frames,
                geometry,
                loaded_items,
                grid_enabled=params["grid_enabled"],
                grid_color_rgb=grid_color_rgb,
                grid_background_rgb=background_rgb,
                grid_vertical_spacing=grid_vertical_spacing,
                depth_step=depth_step,
                fog_amount=params["fog_amount"],
                fog_depth=params["fog_depth"],
                foreground_cutoff=params["foreground_cutoff"],
                field_of_view=params["field_of_view"],
            )

            if start_frame - 1 <= frame_idx <= end_frame - 1:
                frame_rgb = frame.convert("RGB")
                imageio_frame = np.array(frame_rgb)
                if writer is None:
                    writer = imageio.get_writer(
                        output_path,
                        fps=fps,
                        codec=self.render_settings.codec,
                        quality=8,
                    )
                writer.append_data(imageio_frame)

            if frame_idx % progress_interval == 0 or frame_idx == total_frames - 1:
                print(f"Rendering frame {frame_idx + 1}/{total_frames}...", flush=True)

        if writer is not None:
            writer.close()
        print(f"Render complete. Saved to {output_path}")
        self.last_render_path = output_path
        messagebox.showinfo("Render animation", f"Saved render to {output_path}", parent=self)

    # ------------------------------------------------------------------
    def _play_last_render(self) -> None:  # pragma: no cover - UI callback
        if self.last_render_path and self.last_render_path.exists():
            messagebox.showinfo(
                "Last render",
                f"Most recent render saved to:\n{self.last_render_path}",
                parent=self,
            )
        else:
            messagebox.showinfo("Last render", "No render has been created yet.", parent=self)

    # ------------------------------------------------------------------
    def _save_to_assets(self) -> None:  # pragma: no cover - UI callback
        messagebox.showinfo("Save to assets", "Saving renders is not yet available in this tool.", parent=self)


class RenderSettingsDialog(simpledialog.Dialog):
    PRESETS = [
        ("Custom", None, None),
        ("HD 1080p (1920x1080)", 1920, 1080),
        ("HD 720p (1280x720)", 1280, 720),
        ("4K UHD (3840x2160)", 3840, 2160),
        ("Square 1080 (1080x1080)", 1080, 1080),
    ]

    CODECS = [
        ("H.264 (libx264)", "libx264"),
        ("H.265 (libx265)", "libx265"),
        ("VP9 (libvpx-vp9)", "libvpx-vp9"),
        ("QuickTime Animation (qtrle)", "qtrle"),
    ]

    def __init__(self, parent, settings: RenderSettings):
        self.settings = settings
        self.result_settings: Optional[RenderSettings] = None
        super().__init__(parent, title="Render settings")

    def body(self, master):
        ttk.Label(master, text="Preset:").grid(row=0, column=0, sticky="w", pady=(0, 4))
        preset_labels = [label for label, _w, _h in self.PRESETS]
        self.preset_var = tk.StringVar(value=self._matching_preset_label())
        preset_combo = ttk.Combobox(master, textvariable=self.preset_var, values=preset_labels, state="readonly")
        preset_combo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 4))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(master, text="Width:").grid(row=1, column=0, sticky="w")
        self.width_var = tk.StringVar(value=str(self.settings.width))
        self.width_entry = ttk.Entry(master, textvariable=self.width_var, width=10)
        self.width_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(master, text="Height:").grid(row=2, column=0, sticky="w")
        self.height_var = tk.StringVar(value=str(self.settings.height))
        ttk.Entry(master, textvariable=self.height_var, width=10).grid(row=2, column=1, sticky="w")

        ttk.Label(master, text="FPS:").grid(row=3, column=0, sticky="w")
        self.fps_var = tk.StringVar(value=str(self.settings.fps))
        ttk.Entry(master, textvariable=self.fps_var, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(master, text="Codec:").grid(row=4, column=0, sticky="w")
        codec_labels = [label for label, _ in self.CODECS]
        self.codec_map = {label: key for label, key in self.CODECS}
        initial_label = next((label for label, key in self.CODECS if key == self.settings.codec), codec_labels[0])
        self.codec_var = tk.StringVar(value=initial_label)
        ttk.Combobox(master, textvariable=self.codec_var, values=codec_labels, state="readonly").grid(
            row=4, column=1, columnspan=2, sticky="ew"
        )

        ttk.Label(master, text="Render from:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.start_frame_var = tk.StringVar(value=str(self.settings.start_frame or ""))
        ttk.Entry(master, textvariable=self.start_frame_var, width=10).grid(row=5, column=1, sticky="w", pady=(8, 0))

        ttk.Label(master, text="Render to:").grid(row=6, column=0, sticky="w")
        self.end_frame_var = tk.StringVar(value=str(self.settings.end_frame or ""))
        ttk.Entry(master, textvariable=self.end_frame_var, width=10).grid(row=6, column=1, sticky="w")

        master.columnconfigure(1, weight=1)
        return self.width_entry

    def _matching_preset_label(self) -> str:
        for label, width, height in self.PRESETS:
            if width is None or height is None:
                continue
            if width == self.settings.width and height == self.settings.height:
                return label
        return "Custom"

    def _on_preset_selected(self, _event=None) -> None:
        label = self.preset_var.get()
        for preset_label, width, height in self.PRESETS:
            if label == preset_label and width and height:
                self.width_var.set(str(width))
                self.height_var.set(str(height))
                break

    def validate(self) -> bool:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = int(self.fps_var.get())
            start = self._parse_optional_int(self.start_frame_var.get())
            end = self._parse_optional_int(self.end_frame_var.get())
        except ValueError:
            messagebox.showerror("Render settings", "Width, height, and FPS must be numbers.", parent=self)
            return False

        if width <= 0 or height <= 0 or fps <= 0:
            messagebox.showerror("Render settings", "Width, height, and FPS must be positive.", parent=self)
            return False
        if start is not None and start <= 0:
            messagebox.showerror("Render settings", "Start frame must be at least 1.", parent=self)
            return False
        if end is not None and end <= 0:
            messagebox.showerror("Render settings", "End frame must be at least 1.", parent=self)
            return False
        if start is not None and end is not None and end < start:
            messagebox.showerror(
                "Render settings",
                "End frame must be greater than or equal to the start frame.",
                parent=self,
            )
            return False
        return True

    def apply(self) -> None:
        codec_label = self.codec_var.get()
        codec = self.codec_map.get(codec_label, "libx264")
        self.result_settings = RenderSettings(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            fps=int(self.fps_var.get()),
            codec=codec,
            start_frame=self._parse_optional_int(self.start_frame_var.get()),
            end_frame=self._parse_optional_int(self.end_frame_var.get()),
        )

    def _parse_optional_int(self, value: str) -> Optional[int]:
        text = str(value).strip()
        if not text:
            return None
        return int(text)


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
