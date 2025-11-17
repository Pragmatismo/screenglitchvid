#!/usr/bin/env python3
"""GUI tool for pairing timing tracks with simple animation modes."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:  # Pillow < 10 compatibility
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - fallback for older Pillow
    RESAMPLE = Image.LANCZOS

try:  # imageio is optional during import so we can surface nicer errors later.
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - defer error handling until runtime
    imageio = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover - defer error handling until runtime
    sf = None

try:
    import pygame
except Exception:  # pragma: no cover - defer error handling until runtime
    pygame = None

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


DEFAULT_FPS = 30
FRAME_SIZE = (960, 540)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


@dataclass
class TimingEvent:
    time: float
    value: Optional[float] = None
    duration: Optional[float] = None
    label: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TimingEvent":
        return cls(
            time=float(payload.get("time", 0.0)),
            value=(float(payload["value"]) if payload.get("value") is not None else None),
            duration=(float(payload["duration"]) if payload.get("duration") is not None else None),
            label=payload.get("label"),
            data=payload.get("data"),
        )


@dataclass
class TimingTrack:
    name: str
    events: List[TimingEvent] = field(default_factory=list)
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "TimingTrack":
        events = [TimingEvent.from_dict(event) for event in payload.get("events", [])]
        events.sort(key=lambda e: e.time)
        return cls(name=name, events=events, description=payload.get("description"))


@dataclass
class TimingDocument:
    duration: float
    tracks: "OrderedDict[str, TimingTrack]" = field(default_factory=OrderedDict)

    @classmethod
    def from_json(cls, path: Path) -> "TimingDocument":
        payload = json.loads(path.read_text(encoding="utf-8"))
        duration = float(payload.get("duration", 0.0))
        tracks_payload: Dict[str, Any] = payload.get("tracks", {})
        tracks: "OrderedDict[str, TimingTrack]" = OrderedDict()
        for name, track_payload in tracks_payload.items():
            tracks[name] = TimingTrack.from_dict(name, track_payload)
        return cls(duration=duration, tracks=tracks)


# ---------------------------------------------------------------------------
# Rendering primitives
# ---------------------------------------------------------------------------


class VisualEffect:
    def __init__(self, start_time: float, end_time: float) -> None:
        self.start_time = start_time
        self.end_time = end_time

    def is_active(self, time_s: float) -> bool:
        return self.start_time - 0.001 <= time_s <= self.end_time + 0.001

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        raise NotImplementedError


class FireworkEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        options: Dict[str, Any],
    ) -> None:
        lead_frames = max(2, int(options.get("pre_launch_frames", 12)))
        self.lead_time = lead_frames / fps
        intensity = float(event.value) if event.value is not None else float(options.get("intensity", 1.0))
        fade = float(event.duration) if event.duration is not None else float(options.get("fade", 0.6))
        self.event_time = float(event.time)
        start = self.event_time - self.lead_time
        end = self.event_time + fade
        super().__init__(start, end)
        width, height = canvas_size
        self.target_x = rng.uniform(width * 0.1, width * 0.9)
        self.target_y = rng.uniform(height * 0.2, height * 0.8)
        self.intensity = max(0.2, intensity)
        palette = [
            (255, 214, 0),
            (255, 126, 0),
            (255, 46, 99),
            (166, 73, 255),
            (83, 224, 255),
        ]
        self.colour = random.choice(palette)
        self.height = height
        self.width = width

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - requires human inspection
        draw = ImageDraw.Draw(image, "RGBA")
        if time_s < self.event_time:
            progress = 0.0
            if self.lead_time > 0:
                progress = max(0.0, min(1.0, (time_s - self.start_time) / self.lead_time))
            y = self.height - (self.height - self.target_y) * progress
            x = self.target_x
            draw.line([(x, self.height), (x, y)], fill=self.colour + (200,), width=3)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=self.colour + (220,))
            return
        decay = 1.0
        if self.end_time > self.event_time:
            decay = max(0.0, 1.0 - (time_s - self.event_time) / (self.end_time - self.event_time))
        base_radius = 40 * self.intensity
        for layer in range(3):
            ratio = (layer + 1) / 3
            radius = base_radius * ratio * (0.6 + 0.4 * math.sin(min(1.0, decay) * math.pi))
            alpha = int(220 * decay * (1.0 - layer * 0.25))
            colour = self.colour + (alpha,)
            bbox = (
                self.target_x - radius,
                self.target_y - radius,
                self.target_x + radius,
                self.target_y + radius,
            )
            draw.ellipse(bbox, outline=colour, width=3)
        spark_radius = 6 + 8 * self.intensity * decay
        draw.ellipse(
            (
                self.target_x - spark_radius,
                self.target_y - spark_radius,
                self.target_x + spark_radius,
                self.target_y + spark_radius,
            ),
            fill=self.colour + (int(150 * decay),),
        )


class SpritePopEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        sprite: Image.Image,
        options: Dict[str, Any],
    ) -> None:
        self.event_time = float(event.time)
        self.hang_time = max(0.0, float(options.get("hang_time", 0.35)))
        self.scale = max(0.05, float(options.get("scale", 1.0)))
        pre_zoom_frames = max(0, int(options.get("pre_zoom_frames", 5)))
        self.pre_zoom_time = pre_zoom_frames / fps
        self.gravity = max(50.0, float(options.get("gravity", 400.0)))
        width, height = canvas_size
        self.base_x = rng.uniform(width * 0.15, width * 0.85)
        self.base_y = rng.uniform(height * 0.05, height * 0.45)
        self.horizontal_drift = rng.uniform(-40, 40)
        self.sprite = sprite
        self.sprite_size = sprite.size
        fall_limit = height + self.sprite_size[1] * self.scale
        fall_distance = max(0.0, fall_limit - self.base_y)
        fall_duration = math.sqrt(2 * fall_distance / self.gravity)
        start = self.event_time - self.pre_zoom_time
        end = self.event_time + self.hang_time + fall_duration + 0.5
        super().__init__(start, end)
        self.width = width
        self.height = height

    def _build_scaled_sprite(self, scale: float) -> Image.Image:
        target_scale = max(0.05, scale * self.scale)
        w = max(1, int(self.sprite_size[0] * target_scale))
        h = max(1, int(self.sprite_size[1] * target_scale))
        return self.sprite.resize((w, h), RESAMPLE)

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - requires human inspection
        if time_s < self.start_time or time_s > self.end_time:
            return
        stage_start = self.event_time - self.pre_zoom_time
        if time_s < self.event_time:
            progress = 0.0
            if self.pre_zoom_time > 0:
                progress = max(0.0, min(1.0, (time_s - stage_start) / self.pre_zoom_time))
            scale = max(0.01, progress)
            sprite_img = self._build_scaled_sprite(scale)
            x = self.base_x
            y = self.base_y
        elif time_s < self.event_time + self.hang_time:
            sprite_img = self._build_scaled_sprite(1.0)
            x = self.base_x
            y = self.base_y
        else:
            fall_time = time_s - (self.event_time + self.hang_time)
            y = self.base_y + 0.5 * self.gravity * (fall_time**2)
            x = self.base_x + self.horizontal_drift * min(1.0, fall_time)
            sprite_img = self._build_scaled_sprite(1.0)
        if y - sprite_img.size[1] / 2 > self.height:
            return
        top_left = (
            int(x - sprite_img.size[0] / 2),
            int(y - sprite_img.size[1] / 2),
        )
        image.paste(sprite_img, box=top_left, mask=sprite_img)


@dataclass
class AssociationConfig:
    track_name: str
    mode: str
    options: Dict[str, Any]


@dataclass
class AssociationPlan:
    config: AssociationConfig
    effects: List[VisualEffect] = field(default_factory=list)

    def render(self, image: Image.Image, time_s: float) -> None:
        for effect in self.effects:
            if effect.is_active(time_s):
                effect.draw(image, time_s)


@dataclass
class RenderPlan:
    audio_path: Path
    fps: int
    size: tuple[int, int]
    duration: float
    associations: List[AssociationPlan]

    @property
    def total_frames(self) -> int:
        return int(math.ceil(self.duration * self.fps))

    def frame_time(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def generate_frame(self, frame_idx: int) -> np.ndarray:  # pragma: no cover - visual output
        time_s = self.frame_time(frame_idx)
        image = Image.new("RGB", self.size, "black")
        for assoc in self.associations:
            assoc.render(image, time_s)
        return np.array(image, dtype=np.uint8)


# ---------------------------------------------------------------------------
# GUI components
# ---------------------------------------------------------------------------


class AssociationDialog(simpledialog.Dialog):
    def __init__(self, parent, tracks: Iterable[str], config: Optional[AssociationConfig] = None):
        self.track_names = list(tracks)
        self.config = config
        self.result_config: Optional[AssociationConfig] = None
        super().__init__(parent, title="Configure timed action")

    def body(self, master):
        ttk.Label(master, text="Timing track:").grid(row=0, column=0, sticky="w")
        self.track_var = tk.StringVar(value=(self.config.track_name if self.config else (self.track_names[0] if self.track_names else "")))
        self.track_combo = ttk.Combobox(master, values=self.track_names, textvariable=self.track_var, state="readonly")
        self.track_combo.grid(row=0, column=1, sticky="ew")
        master.columnconfigure(1, weight=1)

        ttk.Label(master, text="Mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.mode_var = tk.StringVar(value=(self.config.mode if self.config else "fireworks"))
        self.mode_combo = ttk.Combobox(master, values=["fireworks", "sprite_pop"], textvariable=self.mode_var, state="readonly")
        self.mode_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _evt: self._show_mode_options())

        self.options_frame = ttk.Frame(master)
        self.options_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._build_option_inputs()
        self._show_mode_options()
        return self.track_combo

    def _build_option_inputs(self) -> None:
        self.firework_vars = {
            "pre_launch_frames": tk.StringVar(value=str(self.config.options.get("pre_launch_frames", 12) if self.config and self.config.mode == "fireworks" else 12)),
            "fade": tk.StringVar(value=str(self.config.options.get("fade", 0.6) if self.config and self.config.mode == "fireworks" else 0.6)),
        }
        sprite_defaults = {
            "sprite_path": "",
            "scale": 1.0,
            "hang_time": 0.35,
            "pre_zoom_frames": 5,
        }
        self.sprite_vars = {}
        for key, default in sprite_defaults.items():
            value = default
            if self.config and self.config.mode == "sprite_pop":
                value = self.config.options.get(key, default)
            self.sprite_vars[key] = tk.StringVar(value=str(value))

    def _show_mode_options(self) -> None:
        for child in self.options_frame.winfo_children():
            child.destroy()
        mode = self.mode_var.get()
        if mode == "fireworks":
            ttk.Label(self.options_frame, text="Fireworks options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Pre-launch frames:").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.firework_vars["pre_launch_frames"], width=8).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Fade seconds (fallback when timing has no duration):").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.firework_vars["fade"], width=8).grid(row=2, column=1, sticky="w")
        else:
            ttk.Label(self.options_frame, text="Sprite pop options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=3)
            ttk.Label(self.options_frame, text="Sprite image:").grid(row=1, column=0, sticky="w")
            entry = ttk.Entry(self.options_frame, textvariable=self.sprite_vars["sprite_path"], width=36)
            entry.grid(row=1, column=1, sticky="ew")
            ttk.Button(
                self.options_frame,
                text="Browse",
                command=lambda: self._browse_sprite(entry),
            ).grid(row=1, column=2, sticky="e")
            ttk.Label(self.options_frame, text="Scale factor:").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.sprite_vars["scale"], width=8).grid(row=2, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Hang time (s):").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.sprite_vars["hang_time"], width=8).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Pre-zoom (frames):").grid(row=4, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.sprite_vars["pre_zoom_frames"], width=8).grid(row=4, column=1, sticky="w")

    def _browse_sprite(self, entry: ttk.Entry) -> None:
        initial = Path(self.sprite_vars["sprite_path"].get()).expanduser()
        file_path = filedialog.askopenfilename(title="Select sprite", initialdir=str(initial.parent if initial.exists() else Path.cwd()), filetypes=(("Images", "*.png;*.jpg;*.jpeg"), ("All", "*.*")))
        if file_path:
            self.sprite_vars["sprite_path"].set(file_path)

    def validate(self) -> bool:
        if not self.track_var.get():
            messagebox.showerror("Missing data", "Please choose a timing track.", parent=self)
            return False
        return True

    def apply(self) -> None:
        mode = self.mode_var.get()
        if mode == "fireworks":
            options = {
                "pre_launch_frames": int(self.firework_vars["pre_launch_frames"].get() or 12),
                "fade": float(self.firework_vars["fade"].get() or 0.6),
            }
        else:
            options = {
                "sprite_path": self.sprite_vars["sprite_path"].get().strip(),
                "scale": float(self.sprite_vars["scale"].get() or 1.0),
                "hang_time": float(self.sprite_vars["hang_time"].get() or 0.35),
                "pre_zoom_frames": int(self.sprite_vars["pre_zoom_frames"].get() or 5),
            }
        self.result_config = AssociationConfig(
            track_name=self.track_var.get(),
            mode=mode,
            options=options,
        )


class TimedActionTool(tk.Tk):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.title("Timed Action Mixer")
        self.geometry("980x720")
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(args.project).resolve() if args.project else None
        self.project_name = args.project_name or (self.project_root.name if self.project_root else "Standalone")
        self.output_dir = Path(args.output_dir).resolve() if args.output_dir else self._default_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir = (self.project_root / "assets") if self.project_root else (self.repo_root / "assets")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        if self.project_root:
            self.timing_dir = self.project_root / "internal" / "timing"
        else:
            self.timing_dir = self.repo_root / "assets" / "timing"
        self.timing_dir.mkdir(parents=True, exist_ok=True)
        self.temp_render_path = self.output_dir / "timed_action_preview.mp4"

        self.audio_path = tk.StringVar()
        self.timing_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Select audio and timing files to begin.")

        self.timing_doc: Optional[TimingDocument] = None
        self.associations: List[AssociationConfig] = []
        self.last_plan: Optional[RenderPlan] = None
        self.render_thread: Optional[threading.Thread] = None
        self._playback_state: dict[str, Any] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    def _default_output_dir(self) -> Path:
        if self.project_root:
            return self.project_root / "internal" / "video" / "timed_action_mixer"
        return self.repo_root / "assets" / "timed_action_mixer"

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=20)
        container.pack(fill="both", expand=True)
        style = ttk.Style(self)
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))

        file_frame = ttk.LabelFrame(container, text="Source files")
        file_frame.pack(fill="x", padx=(0, 4), pady=(0, 10))
        audio_types = (
            ("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("All files", "*.*"),
        )
        timing_types = (
            ("Timing JSON", "*.json"),
            ("All files", "*.*"),
        )
        self._build_file_selector(
            file_frame,
            "Audio file:",
            self.audio_path,
            row=0,
            callback=self._guess_audio,
            initial_dir=self.assets_dir,
            filetypes=audio_types,
        )
        self._build_file_selector(
            file_frame,
            "Timing file:",
            self.timing_path,
            row=1,
            callback=self._load_timing,
            initial_dir=self.timing_dir,
            filetypes=timing_types,
        )

        ttk.Label(container, text="Timing tracks", style="Heading.TLabel").pack(anchor="w")
        self.tracks_tree = ttk.Treeview(container, columns=("events", "description"), show="headings", height=5)
        self.tracks_tree.heading("events", text="Events")
        self.tracks_tree.heading("description", text="Description")
        self.tracks_tree.column("events", width=80, anchor="center")
        self.tracks_tree.column("description", width=520)
        self.tracks_tree.pack(fill="x", pady=(4, 12))

        assoc_frame = ttk.LabelFrame(container, text="Track associations")
        assoc_frame.pack(fill="both", expand=True)
        self.assoc_tree = ttk.Treeview(assoc_frame, columns=("track", "mode", "summary"), show="headings", height=6)
        self.assoc_tree.heading("track", text="Track")
        self.assoc_tree.heading("mode", text="Mode")
        self.assoc_tree.heading("summary", text="Options")
        self.assoc_tree.column("track", width=160)
        self.assoc_tree.column("mode", width=100)
        self.assoc_tree.column("summary", width=420)
        self.assoc_tree.pack(fill="both", expand=True, pady=(4, 6))

        buttons = ttk.Frame(assoc_frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Add", command=self.add_association).pack(side="left")
        ttk.Button(buttons, text="Edit", command=self.edit_association).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Remove", command=self.remove_association).pack(side="left", padx=(6, 0))

        action_frame = ttk.Frame(container)
        action_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(action_frame, text="Render animation", command=self.render_video).pack(side="left")
        ttk.Button(action_frame, text="Play last render", command=self.play_render).pack(side="left", padx=(8, 0))
        ttk.Button(action_frame, text="Save to assets", command=self.save_render).pack(side="left", padx=(8, 0))

        ttk.Label(container, textvariable=self.status_var, foreground="#4a4a4a").pack(anchor="w", pady=(12, 0))

    # ------------------------------------------------------------------
    def _build_file_selector(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        callback,
        initial_dir: Optional[Path] = None,
        filetypes: Optional[tuple[tuple[str, str], ...]] = None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(4, 6), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=60)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)
        btn = ttk.Button(
            parent,
            text="Browse",
            command=lambda: self._browse_file(variable, callback, initial_dir, filetypes),
        )
        btn.grid(row=row, column=2, sticky="e", padx=(6, 4))

    # ------------------------------------------------------------------
    def _browse_file(
        self,
        variable: tk.StringVar,
        callback,
        initial_dir: Optional[Path],
        filetypes: Optional[tuple[tuple[str, str], ...]],
    ) -> None:
        current = Path(variable.get()).expanduser()
        if current.is_file():
            start_dir = current.parent
        elif current.exists():
            start_dir = current
        elif initial_dir and initial_dir.exists():
            start_dir = initial_dir
        elif self.project_root:
            start_dir = self.project_root
        else:
            start_dir = self.repo_root
        open_filetypes = filetypes or (("All files", "*.*"),)
        file_path = filedialog.askopenfilename(
            initialdir=str(start_dir),
            title="Select file",
            filetypes=open_filetypes,
        )
        if file_path:
            variable.set(file_path)
            if callback:
                callback()

    # ------------------------------------------------------------------
    def _guess_audio(self) -> None:
        path = Path(self.audio_path.get()).expanduser()
        if path.exists():
            self.status_var.set(f"Audio track selected: {path.name}")
            timing = self._find_matching_timing(path)
            if timing:
                self.timing_path.set(str(timing))
                self._load_timing()

    # ------------------------------------------------------------------
    def _find_matching_timing(self, audio_path: Path) -> Optional[Path]:
        stem = audio_path.stem
        candidates = [
            f"{stem}.timing.json",
            f"{stem}_timing.json",
            f"{stem}.json",
        ]
        search_dirs = [audio_path.parent, self.timing_dir, self.assets_dir]
        for directory in search_dirs:
            if not directory:
                continue
            for candidate in candidates:
                possible = directory / candidate
                if possible.exists():
                    return possible
        return None

    # ------------------------------------------------------------------
    def _load_timing(self) -> None:
        path = Path(self.timing_path.get()).expanduser()
        if not path.exists():
            messagebox.showerror("Timing file", "Selected timing file does not exist.")
            return
        try:
            self.timing_doc = TimingDocument.from_json(path)
        except Exception as exc:
            messagebox.showerror("Timing file", f"Failed to load timing data:\n{exc}")
            return
        for item in self.tracks_tree.get_children():
            self.tracks_tree.delete(item)
        for track in self.timing_doc.tracks.values():
            desc = track.description or "-"
            self.tracks_tree.insert("", "end", iid=track.name, values=(len(track.events), desc))
        self.status_var.set(f"Loaded {len(self.timing_doc.tracks)} tracks from timing file.")

    # ------------------------------------------------------------------
    def add_association(self) -> None:
        if not self.timing_doc or not self.timing_doc.tracks:
            messagebox.showwarning("Timing tracks", "Load a timing file before adding associations.")
            return
        dialog = AssociationDialog(self, self.timing_doc.tracks.keys())
        if dialog.result_config:
            self.associations.append(dialog.result_config)
            self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def _refresh_assoc_tree(self) -> None:
        for item in self.assoc_tree.get_children():
            self.assoc_tree.delete(item)
        for idx, assoc in enumerate(self.associations):
            if assoc.mode == "fireworks":
                summary = f"Lead {assoc.options.get('pre_launch_frames', 12)}f, fade {assoc.options.get('fade', 0.6)}s"
            else:
                summary = (
                    f"Sprite {Path(assoc.options.get('sprite_path', '')).name or 'generated'}, "
                    f"scale {assoc.options.get('scale', 1.0)}, hang {assoc.options.get('hang_time', 0.35)}s, "
                    f"pre-zoom {assoc.options.get('pre_zoom_frames', 5)}f"
                )
            self.assoc_tree.insert("", "end", iid=str(idx), values=(assoc.track_name, assoc.mode, summary))

    # ------------------------------------------------------------------
    def edit_association(self) -> None:
        selection = self.assoc_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        assoc = self.associations[idx]
        dialog = AssociationDialog(self, self.timing_doc.tracks.keys() if self.timing_doc else [], assoc)
        if dialog.result_config:
            self.associations[idx] = dialog.result_config
            self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def remove_association(self) -> None:
        selection = self.assoc_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        del self.associations[idx]
        self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def _build_render_plan(self) -> RenderPlan:
        if not self.associations:
            raise RuntimeError("Add at least one association before rendering.")
        audio_path = Path(self.audio_path.get()).expanduser()
        if not audio_path.exists():
            raise RuntimeError("Select an audio file before rendering.")
        if self.timing_doc is None:
            raise RuntimeError("Load a timing file before rendering.")
        duration = self.timing_doc.duration
        if sf:
            try:
                info = sf.info(str(audio_path))
                duration = max(duration, info.frames / float(info.samplerate))
            except Exception:
                pass
        rng = random.Random(1234)
        associations: List[AssociationPlan] = []
        max_end_time = duration
        for assoc in self.associations:
            track = self.timing_doc.tracks.get(assoc.track_name)
            if not track:
                continue
            effects: List[VisualEffect] = []
            for event in track.events:
                if assoc.mode == "fireworks":
                    effects.append(
                        FireworkEffect(
                            event=event,
                            fps=DEFAULT_FPS,
                            canvas_size=FRAME_SIZE,
                            rng=rng,
                            options=assoc.options,
                        )
                    )
                else:
                    sprite_img = self._load_sprite_image(assoc.options.get("sprite_path"))
                    effects.append(
                        SpritePopEffect(
                            event=event,
                            fps=DEFAULT_FPS,
                            canvas_size=FRAME_SIZE,
                            rng=rng,
                            sprite=sprite_img,
                            options=assoc.options,
                        )
                    )
                if effects:
                    max_end_time = max(max_end_time, effects[-1].end_time)
            associations.append(AssociationPlan(config=assoc, effects=effects))
        duration = max(duration, max_end_time)
        return RenderPlan(
            audio_path=audio_path,
            fps=DEFAULT_FPS,
            size=FRAME_SIZE,
            duration=duration,
            associations=associations,
        )

    # ------------------------------------------------------------------
    def _load_sprite_image(self, sprite_path: Optional[str]) -> Image.Image:
        cache: Dict[str, Image.Image] = getattr(self, "_sprite_cache", {})
        if sprite_path:
            resolved = str(Path(sprite_path).expanduser())
            if resolved in cache:
                return cache[resolved]
            path = Path(resolved)
            if path.exists():
                sprite = Image.open(path).convert("RGBA")
                cache[resolved] = sprite
                self._sprite_cache = cache
                return sprite
        placeholder = cache.get("__placeholder__")
        if placeholder is None:
            placeholder = Image.new("RGBA", (128, 128), (255, 255, 255, 0))
            draw = ImageDraw.Draw(placeholder)
            draw.ellipse((8, 8, 120, 120), fill=(255, 80, 80, 255))
            cache["__placeholder__"] = placeholder
            self._sprite_cache = cache
        return placeholder

    # ------------------------------------------------------------------
    def render_video(self) -> None:
        if imageio is None:
            messagebox.showerror("Missing dependency", "imageio is required for rendering MP4 files. Install imageio and imageio-ffmpeg.")
            return
        if self.render_thread and self.render_thread.is_alive():
            return
        try:
            plan = self._build_render_plan()
        except Exception as exc:
            messagebox.showerror("Render", str(exc))
            return

        self.status_var.set("Rendering… this may take a moment.")
        self.render_thread = threading.Thread(target=self._render_worker, args=(plan,), daemon=True)
        self.render_thread.start()

    # ------------------------------------------------------------------
    def _render_worker(self, plan: RenderPlan) -> None:
        start = time.perf_counter()
        try:
            writer = imageio.get_writer(
                self.temp_render_path,
                fps=plan.fps,
                codec="libx264",
                quality=8,
                format="FFMPEG",
                audio_path=str(plan.audio_path),
                audio_codec="aac",
            )
        except Exception as exc:
            self._set_status_async(f"Unable to start writer: {exc}")
            return

        try:
            for frame_idx in range(plan.total_frames):
                frame = plan.generate_frame(frame_idx)
                writer.append_data(frame)
                if frame_idx % 30 == 0:
                    self._set_status_async(
                        f"Rendering frame {frame_idx + 1}/{plan.total_frames}"
                    )
        finally:
            writer.close()
        elapsed = time.perf_counter() - start
        self.last_plan = plan
        self._set_status_async(
            f"Render complete in {elapsed:.1f}s. Preview saved to {self.temp_render_path}."
        )

    # ------------------------------------------------------------------
    def _set_status_async(self, text: str) -> None:
        self.after(0, self.status_var.set, text)

    # ------------------------------------------------------------------
    def play_render(self) -> None:
        if not self.temp_render_path.exists():
            messagebox.showinfo("Playback", "Render a video first.")
            return
        if pygame is None:
            messagebox.showerror("Playback", "pygame is required to play the audio track.")
            return
        play_window = tk.Toplevel(self)
        play_window.title("Render preview")
        label = ttk.Label(play_window)
        label.pack(fill="both", expand=True)
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(str(self.last_plan.audio_path if self.last_plan else self.audio_path.get()))
        except Exception as exc:
            pygame.mixer.quit()
            messagebox.showerror("Playback", f"Unable to load audio: {exc}")
            play_window.destroy()
            return
        pygame.mixer.music.play()
        try:
            reader = imageio.get_reader(self.temp_render_path, format="FFMPEG")
        except Exception as exc:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            play_window.destroy()
            messagebox.showerror("Playback", f"Unable to read render: {exc}")
            return
        iterator = iter(reader)
        delay_ms = int(1000 / (self.last_plan.fps if self.last_plan else DEFAULT_FPS))

        def show_next_frame() -> None:
            try:
                frame = next(iterator)
            except StopIteration:
                reader.close()
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                play_window.destroy()
                return
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
            play_window.after(delay_ms, show_next_frame)

        show_next_frame()

    # ------------------------------------------------------------------
    def save_render(self) -> None:
        if not self.temp_render_path.exists():
            messagebox.showinfo("Save", "Render a video first.")
            return
        name = simpledialog.askstring("Save render", "File name (without extension):", parent=self)
        if not name:
            return
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_")) or "render"
        target = self.assets_dir / f"{safe_name}.mp4"
        try:
            shutil.copy2(self.temp_render_path, target)
        except Exception as exc:
            messagebox.showerror("Save", f"Unable to copy render: {exc}")
            return
        self.status_var.set(f"Saved render to {target}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Timed Action Mixer")
    parser.add_argument("--project", help="Path to the active project root")
    parser.add_argument("--project-name", help="Display name for the project")
    parser.add_argument("--output-dir", help="Directory for temporary renders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = TimedActionTool(args)
    app.mainloop()


if __name__ == "__main__":
    main()
