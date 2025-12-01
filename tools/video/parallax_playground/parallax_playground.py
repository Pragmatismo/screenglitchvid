#!/usr/bin/env python3
"""Interactive scene builder for experimenting with parallax layers."""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import unquote, urlparse

import imageio.v2 as imageio
import numpy as np
import tkinter as tk
from PIL import Image
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

        self.direction_var = tk.StringVar(value="right")
        self.distance_var = tk.DoubleVar(value=50.0)
        self.rate_var = tk.DoubleVar(value=10.0)
        self.horizon_height_var = tk.DoubleVar(value=0.5)
        self.distance_to_horizon_var = tk.DoubleVar(value=100.0)
        self.horizon_fog_var = tk.DoubleVar(value=0.25)
        self.horizon_fog_depth_var = tk.DoubleVar(value=50.0)
        self.foreground_cutoff_var = tk.DoubleVar(value=5.0)
        self.duration_var = tk.StringVar()
        self.render_settings = RenderSettings()
        self.last_render_path: Optional[Path] = None

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

        self.distance_var.trace_add("write", lambda *_: self._update_duration())
        self.rate_var.trace_add("write", lambda *_: self._update_duration())

        info_bar = ttk.Frame(container)
        info_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(info_bar, textvariable=self.duration_var).pack(anchor="w")

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
    def _update_duration(self) -> None:
        distance = self._safe_float(self.distance_var)
        rate = self._safe_float(self.rate_var)

        if distance is None or rate is None:
            self.duration_var.set("Duration: – (enter numeric values)")
            return

        if rate == 0:
            self.duration_var.set("Duration: ∞ (rate must be non-zero)")
            return

        duration = distance / rate
        self.duration_var.set(f"Duration: {duration:.2f} time units")

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

    # ------------------------------------------------------------------
    def _render_animation(self) -> None:  # pragma: no cover - UI callback
        items = list(self.items.items())
        if not items:
            messagebox.showwarning("Render animation", "Add at least one image before rendering.", parent=self)
            return

        distance = self._safe_float(self.distance_var)
        rate = self._safe_float(self.rate_var)
        horizon_height = self._safe_float(self.horizon_height_var)
        horizon_distance = self._safe_float(self.distance_to_horizon_var)
        fog_amount = self._safe_float(self.horizon_fog_var)
        fog_depth = self._safe_float(self.horizon_fog_depth_var)
        foreground_cutoff = self._safe_float(self.foreground_cutoff_var)

        if None in {distance, rate, horizon_height, horizon_distance, fog_amount, fog_depth, foreground_cutoff}:
            messagebox.showerror("Render animation", "Please enter numeric animation settings before rendering.", parent=self)
            return
        assert distance is not None and rate is not None
        assert horizon_height is not None and horizon_distance is not None
        assert fog_amount is not None and fog_depth is not None and foreground_cutoff is not None

        if rate == 0:
            messagebox.showerror("Render animation", "Rate of travel must be non-zero.", parent=self)
            return
        if horizon_distance <= 0:
            messagebox.showerror("Render animation", "Distance to horizon must be greater than zero.", parent=self)
            return
        if not 0 <= horizon_height <= 1:
            messagebox.showerror("Render animation", "Horizon height must be between 0 and 1.", parent=self)
            return

        fps = self.render_settings.fps
        duration = abs(distance / rate)
        total_frames = max(1, math.ceil(duration * fps))
        start_frame = self.render_settings.start_frame or 1
        end_frame = self.render_settings.end_frame or total_frames
        end_frame = min(end_frame, total_frames)
        if start_frame > end_frame:
            messagebox.showerror("Render animation", "Start frame must be before end frame.", parent=self)
            return

        horizon_y = self.render_settings.height * horizon_height
        direction = self.direction_var.get()
        horizontal_sign = -1 if direction in {"right", "clock-wise"} else 1
        forward_motion = direction == "up"
        backward_motion = direction == "down"
        sideways_motion = direction in {"left", "right"}
        rotational_motion = direction in {"clock-wise", "counter clock-wise"}

        rng = random.Random()
        try:
            loaded_items: Dict[str, tuple[ParallaxItem, Image.Image]] = {
                iid: (item, Image.open(item.image_path).convert("RGBA")) for iid, item in items
            }
        except FileNotFoundError:
            messagebox.showerror("Render animation", "One or more image paths are missing.", parent=self)
            return
        except Exception as exc:  # pragma: no cover - runtime guard for unexpected image errors
            messagebox.showerror("Render animation", f"Failed to load images: {exc}", parent=self)
            return

        spawn_budget: Dict[str, float] = defaultdict(float)
        active: list[ActiveInstance] = []
        speed = abs(rate)
        delta_depth = speed / fps
        delta_time = 1 / fps
        fog_start = max(0.0, horizon_distance - fog_depth)
        parallax_margin = 80
        output_name = f"parallax_render_{int(time.time())}.mp4"
        output_path = self.project_assets_dir / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None

        def project_scale(depth: float) -> float:
            base = horizon_distance / max(depth, 1e-3) * 0.1
            return max(0.05, min(base, 6.0))

        def depth_progress(depth: float) -> float:
            return max(0.0, min(1.0, 1 - depth / (horizon_distance + 1e-3)))

        def fog_multiplier(depth: float) -> float:
            if depth <= fog_start:
                return 1.0
            fog_ratio = min(1.0, (depth - fog_start) / max(fog_depth, 1e-3))
            return 1.0 - fog_ratio * fog_amount

        def parallax_factor(depth: float) -> float:
            return 0.6 + (horizon_distance / (depth + 1e-3)) * 0.2

        def spawn_instance(item_id: str) -> None:
            item, _image = loaded_items[item_id]
            depth = foreground_cutoff + rng.uniform(max(0.01, item.min_distance), max(0.02, item.max_distance))
            if forward_motion:
                x_pos = rng.uniform(0, self.render_settings.width)
            elif backward_motion:
                x_pos = rng.uniform(0, self.render_settings.width)
            elif sideways_motion or rotational_motion:
                if direction == "right":
                    x_pos = self.render_settings.width + parallax_margin
                elif direction == "left":
                    x_pos = -parallax_margin
                else:  # rotation
                    x_pos = rng.uniform(-parallax_margin, self.render_settings.width + parallax_margin)
            lateral_offset = rng.uniform(-self.render_settings.height * 0.12, self.render_settings.height * 0.12)
            active.append(ActiveInstance(item_id=item_id, depth=depth, x=x_pos, lateral_offset=lateral_offset))

        progress_interval = max(1, total_frames // 20)
        for frame_idx in range(total_frames):
            for iid, (item, _image) in loaded_items.items():
                spawn_budget[iid] += item.frequency * delta_time
                while spawn_budget[iid] >= 1:
                    spawn_instance(iid)
                    spawn_budget[iid] -= 1
                if spawn_budget[iid] > 0 and rng.random() < spawn_budget[iid]:
                    spawn_instance(iid)
                    spawn_budget[iid] = 0

            updated_active: list[ActiveInstance] = []
            for instance in active:
                if forward_motion:
                    instance.depth -= delta_depth
                    if instance.depth <= foreground_cutoff:
                        continue
                elif backward_motion:
                    instance.depth += delta_depth
                    if instance.depth >= horizon_distance + fog_depth:
                        continue
                elif sideways_motion:
                    factor = parallax_factor(instance.depth)
                    instance.x += horizontal_sign * speed * delta_time * factor
                    if instance.x < -parallax_margin or instance.x > self.render_settings.width + parallax_margin:
                        continue
                elif rotational_motion:
                    factor = parallax_factor(instance.depth)
                    instance.x += horizontal_sign * speed * delta_time * factor * 0.6
                updated_active.append(instance)
            active = updated_active

            frame = Image.new("RGBA", (self.render_settings.width, self.render_settings.height), "black")
            draw_list = sorted(active, key=lambda inst: inst.depth, reverse=True)
            for inst in draw_list:
                item, image = loaded_items[inst.item_id]
                scale = project_scale(inst.depth) * max(item.scale, 0.0)
                target_w = int(image.width * scale)
                target_h = int(image.height * scale)
                if target_w < 1 or target_h < 1:
                    continue
                scaled = image.resize((target_w, target_h), RESAMPLE)
                depth_factor = depth_progress(inst.depth)
                y_pos = horizon_y + (self.render_settings.height - horizon_y) * depth_factor + inst.lateral_offset
                y_pos = max(-target_h, min(self.render_settings.height + target_h, y_pos))
                alpha = fog_multiplier(inst.depth)
                if alpha < 1.0:
                    alpha_layer = scaled.split()[3].point(lambda a: int(a * alpha))
                    scaled.putalpha(alpha_layer)
                frame.alpha_composite(
                    scaled,
                    (int(inst.x - target_w / 2), int(y_pos - target_h / 2)),
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
