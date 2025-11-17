#!/usr/bin/env python3
"""Audio analysis + timeline editor for creating timing tracks."""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


EventValue = Dict[str, float | str | int]


@dataclass
class TimingEvent:
    """Represents a single marker inside a timing track."""

    time: float
    value: Optional[float] = None
    label: Optional[str] = None
    duration: Optional[float] = None
    data: EventValue | None = None

    @classmethod
    def from_dict(cls, payload: MutableMapping) -> "TimingEvent":
        return cls(
            time=float(payload.get("time", 0.0)),
            value=payload.get("value"),
            label=payload.get("label"),
            duration=payload.get("duration"),
            data=payload.get("data"),
        )

    def to_dict(self) -> Dict[str, float | str | EventValue | None]:
        payload: Dict[str, float | str | EventValue | None] = {"time": float(self.time)}
        if self.value is not None:
            payload["value"] = self.value
        if self.label:
            payload["label"] = self.label
        if self.duration is not None:
            payload["duration"] = self.duration
        if self.data:
            payload["data"] = self.data
        return payload


@dataclass
class TimingTrack:
    """Holds a collection of events under a track name."""

    name: str
    events: List[TimingEvent] = field(default_factory=list)
    description: str | None = None

    def sort_events(self) -> None:
        self.events.sort(key=lambda e: e.time)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "events": [event.to_dict() for event in self.events],
        }
        if self.description:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_dict(cls, name: str, payload: MutableMapping) -> "TimingTrack":
        events_data = payload.get("events", [])
        events = [TimingEvent.from_dict(item) for item in events_data]
        track = cls(name=name, events=events, description=payload.get("description"))
        track.sort_events()
        return track


@dataclass
class TimingDocument:
    """Container for all timing tracks associated with one audio file."""

    duration: float
    sample_rate: Optional[int] = None
    tracks: "OrderedDict[str, TimingTrack]" = field(default_factory=OrderedDict)

    version: int = 1

    def ensure_track(self, name: str) -> TimingTrack:
        track = self.tracks.get(name)
        if track is None:
            track = TimingTrack(name=name)
            self.tracks[name] = track
        return track

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "tracks": {name: track.to_dict() for name, track in self.tracks.items()},
        }

    @classmethod
    def from_dict(cls, payload: MutableMapping) -> "TimingDocument":
        duration = float(payload.get("duration", 0.0))
        sample_rate = payload.get("sample_rate")
        doc = cls(duration=duration, sample_rate=sample_rate)
        tracks_payload = payload.get("tracks", {})
        for name, track_payload in tracks_payload.items():
            doc.tracks[name] = TimingTrack.from_dict(name, track_payload)
        return doc


@dataclass
class AnalysisSettings:
    """User-tunable knobs for the librosa analysis pipeline."""

    hop_length: int = 512
    beat_tightness: float = 100.0
    start_bpm: float = 120.0
    rms_percentile: float = 90.0
    pitch_change_percentile: float = 95.0
    onset_backtrack: bool = False
    onset_delta: float = 0.2


# ---------------------------------------------------------------------------
# Audio analysis helpers
# ---------------------------------------------------------------------------


class AudioAnalyzer:
    """Wraps librosa to produce multiple timing tracks from an audio file."""

    def __init__(self, path: Path, settings: Optional[AnalysisSettings] = None):
        self.path = path
        self.duration = 0.0
        self.sample_rate: Optional[int] = None
        self.settings = settings or AnalysisSettings()

    def analyse(self) -> TimingDocument:
        try:
            import librosa
            import numpy as np
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "librosa and numpy are required for analysis."
            ) from exc

        y, sr = librosa.load(self.path, mono=True)
        self.sample_rate = sr
        self.duration = float(librosa.get_duration(y=y, sr=sr))

        tracks: "OrderedDict[str, TimingTrack]" = OrderedDict()

        hop = max(64, int(self.settings.hop_length))
        tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=hop,
            tightness=max(0.0, float(self.settings.beat_tightness)),
            start_bpm=max(1.0, float(self.settings.start_bpm)),
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        beat_events = [
            TimingEvent(time=float(t), value=float(tempo), label="beat")
            for t in beat_times
        ]
        tracks["detected_beats"] = TimingTrack(
            name="detected_beats",
            events=beat_events,
            description="Automatic beat grid (tempo {:.1f} BPM)".format(float(tempo)),
        )

        onset_times = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop,
            units="time",
            backtrack=bool(self.settings.onset_backtrack),
            delta=float(self.settings.onset_delta),
        )
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onset_events = [
            TimingEvent(time=float(t), value=float(onset_strength[min(i, len(onset_strength) - 1)]), label="onset")
            for i, t in enumerate(onset_times)
        ]
        tracks["percussive_onsets"] = TimingTrack(
            name="percussive_onsets",
            events=onset_events,
            description="Detected transient spikes",
        )

        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop)
        norm_rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
        loud_threshold = float(np.percentile(norm_rms, float(self.settings.rms_percentile)))
        loud_sections = self._find_sections(rms_times, norm_rms, loud_threshold)
        loud_events = [
            TimingEvent(time=start, duration=end - start, value=float(level), label="loud")
            for start, end, level in loud_sections
        ]
        tracks["high_energy_sections"] = TimingTrack(
            name="high_energy_sections",
            events=loud_events,
            description="Windows where RMS energy exceeds the 90th percentile",
        )

        try:
            pitch = librosa.yin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
            )
            pitch_times = librosa.frames_to_time(range(len(pitch)), sr=sr)
            pitch_events = self._detect_pitch_changes(
                pitch_times,
                pitch,
                float(self.settings.pitch_change_percentile),
            )
            tracks["pitch_changes"] = TimingTrack(
                name="pitch_changes",
                events=pitch_events,
                description="Moments of large pitch jumps",
            )
        except Exception:
            # yin may fail on very short clips; ignore
            pass

        doc = TimingDocument(duration=self.duration, sample_rate=sr, tracks=tracks)
        return doc

    @staticmethod
    def _find_sections(times: Iterable[float], values: Iterable[float], threshold: float) -> List[Tuple[float, float, float]]:
        sections: List[Tuple[float, float, float]] = []
        active_start: Optional[float] = None
        peak_value = 0.0
        prev_time: Optional[float] = None
        for t, val in zip(times, values):
            if val >= threshold:
                if active_start is None:
                    active_start = float(t)
                    peak_value = float(val)
                else:
                    peak_value = max(peak_value, float(val))
            elif active_start is not None:
                end_time = float(prev_time if prev_time is not None else t)
                sections.append((active_start, end_time, peak_value))
                active_start = None
                peak_value = 0.0
            prev_time = float(t)
        if active_start is not None:
            sections.append((active_start, float(prev_time if prev_time is not None else active_start), peak_value))
        return sections

    @staticmethod
    def _detect_pitch_changes(
        times: Iterable[float],
        pitch: Iterable[float],
        percentile: float,
    ) -> List[TimingEvent]:
        import numpy as np

        pitch_array = np.array(list(pitch))
        diff = np.diff(pitch_array)
        diff = np.insert(diff, 0, 0.0)
        magnitude = np.abs(diff)
        finite_magnitude = magnitude[np.isfinite(magnitude)]
        if finite_magnitude.size:
            percentile = max(0.0, min(100.0, percentile))
            threshold = float(np.percentile(finite_magnitude, percentile))
        else:
            threshold = 0.0
        events: List[TimingEvent] = []
        for idx, mag in enumerate(magnitude):
            if not math.isfinite(mag) or mag < threshold:
                continue
            t = times[idx]
            events.append(
                TimingEvent(time=float(t), value=float(mag), label="pitch_change", data={"hz": float(pitch_array[idx])})
            )
        return events


# ---------------------------------------------------------------------------
# Timeline canvas widget
# ---------------------------------------------------------------------------


class TimelineCanvas(tk.Canvas):
    """Draws stacked tracks with draggable markers and a playhead."""

    def __init__(
        self,
        master: tk.Widget,
        on_select: Callable[[str, int], None],
        on_add_marker: Callable[[str, float], None],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(master, background="#1b1b1b", highlightthickness=0, *args, **kwargs)
        self._pixels_per_second = 80.0
        self._tracks: "OrderedDict[str, TimingTrack]" = OrderedDict()
        self._duration = 0.0
        self._order: List[str] = []
        self._on_select = on_select
        self._on_add_marker = on_add_marker
        self._playhead_item: Optional[int] = None
        self._playhead_time = 0.0
        self._drag_item: Optional[int] = None
        self._drag_track: Optional[str] = None
        self._drag_index: Optional[int] = None
        self._mouse_start_time = 0.0
        self._item_to_event: Dict[int, Tuple[str, int]] = {}
        self._row_height = 60
        self._left_margin = 120
        self._scroll_width = 0.0
        self._scroll_height = 0.0
        self.bind("<Button-1>", self._handle_click)
        self.bind("<B1-Motion>", self._handle_drag)
        self.bind("<ButtonRelease-1>", self._handle_release)
        self.bind("<Double-Button-1>", self._handle_double_click)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.bind("<Button-4>", self._on_scroll_up)
        self.bind("<Button-5>", self._on_scroll_down)

    # public API ------------------------------------------------------------
    def set_tracks(self, tracks: "OrderedDict[str, TimingTrack]") -> None:
        self._tracks = tracks
        self._order = list(tracks.keys())
        self.redraw()

    def set_duration(self, duration: float) -> None:
        self._duration = max(duration, 0.1)
        self.redraw()

    def set_zoom(self, pixels_per_second: float) -> None:
        self._pixels_per_second = max(5.0, pixels_per_second)
        self.redraw()

    def focus_playhead(self, timestamp: float) -> None:
        timestamp = max(0.0, min(self._duration, timestamp))
        self._playhead_time = timestamp
        x = self._time_to_x(timestamp)
        if self._playhead_item is None:
            height = max(1, len(self._order)) * self._row_height + 40
            self._playhead_item = self.create_line(x, 0, x, height, fill="#ffb347", width=2, tags=("playhead",))
        else:
            self.coords(self._playhead_item, x, 0, x, self.winfo_height())
        self._auto_scroll_to_playhead(x)

    @property
    def current_playhead_time(self) -> float:
        return self._playhead_time

    # drawing ---------------------------------------------------------------
    def redraw(self) -> None:
        self.delete("all")
        self._playhead_item = None
        self._item_to_event.clear()
        total_height = max(1, len(self._order)) * self._row_height + 20
        width = self._duration * self._pixels_per_second + self._left_margin + 40
        self.config(scrollregion=(0, 0, width, total_height))
        self._scroll_width = width
        self._scroll_height = total_height

        for idx, track_name in enumerate(self._order):
            y = idx * self._row_height + self._row_height / 2 + 20
            self.create_text(10, y, text=track_name, anchor="w", fill="#f0f0f0")
            x0 = self._left_margin
            x1 = width - 20
            self.create_line(x0, y, x1, y, fill="#444")
            track = self._tracks[track_name]
            for event_index, event in enumerate(track.events):
                x = self._time_to_x(event.time)
                radius = 6
                color = "#4dd0e1" if event.label != "pitch_change" else "#b388ff"
                if event.duration and event.duration > 0:
                    width_px = event.duration * self._pixels_per_second
                    rect = self.create_rectangle(
                        x,
                        y - 8,
                        x + width_px,
                        y + 8,
                        fill="#2e7d32",
                        outline="",
                        stipple="gray25",
                        tags=("marker",),
                    )
                    self._item_to_event[rect] = (track_name, event_index)
                oval = self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline="",
                    tags=("marker",),
                )
                self._item_to_event[oval] = (track_name, event_index)

        if self._playhead_item is not None:
            self.focus_playhead(self._playhead_time)
        elif self._playhead_time:
            self.focus_playhead(self._playhead_time)

    # event handling -------------------------------------------------------
    def _handle_click(self, event: tk.Event) -> None:  # type: ignore[override]
        item = self.find_closest(event.x, event.y)
        if not item:
            return
        tags = self.gettags(item)
        if "marker" in tags:
            track_info = self._item_to_event.get(item[0])
            if track_info:
                self._drag_item = item[0]
                self._drag_track, self._drag_index = track_info
                self._mouse_start_time = self._x_to_time(event.x)
                self._on_select(self._drag_track, self._drag_index)

    def _handle_drag(self, event: tk.Event) -> None:  # type: ignore[override]
        if self._drag_item is None or self._drag_track is None or self._drag_index is None:
            return
        new_time = self._x_to_time(event.x)
        new_time = max(0.0, min(self._duration, new_time))
        track = self._tracks[self._drag_track]
        event_obj = track.events[self._drag_index]
        event_obj.time = new_time
        track.sort_events()
        self._drag_index = track.events.index(event_obj)
        self.redraw()
        self._on_select(self._drag_track, self._drag_index)

    def _handle_release(self, _event: tk.Event) -> None:  # type: ignore[override]
        self._drag_item = None
        self._drag_track = None
        self._drag_index = None

    def _handle_double_click(self, event: tk.Event) -> None:  # type: ignore[override]
        track_idx = int((event.y - 20) // self._row_height)
        if 0 <= track_idx < len(self._order):
            track_name = self._order[track_idx]
            timestamp = self._x_to_time(event.x)
            timestamp = max(0.0, min(self._duration, timestamp))
            self._on_add_marker(track_name, timestamp)

    def _time_to_x(self, timestamp: float) -> float:
        return self._left_margin + timestamp * self._pixels_per_second

    def _x_to_time(self, x: float) -> float:
        return max(0.0, (x - self._left_margin) / self._pixels_per_second)

    def _auto_scroll_to_playhead(self, x: float) -> None:
        total_width = self._scroll_width
        if total_width <= 0:
            return
        self.update_idletasks()
        view_width = max(1, self.winfo_width())
        if total_width <= view_width:
            return
        first, _last = self.xview()
        view_start = first * total_width
        view_end = view_start + view_width
        margin = view_width * 0.2
        if x > view_end - margin:
            new_start = x - (view_width - margin)
            new_start = min(max(0.0, new_start), total_width - view_width)
            self.xview_moveto(new_start / total_width)
        elif x < view_start + margin:
            new_start = max(0.0, x - margin)
            self.xview_moveto(new_start / total_width)

    # scroll helpers --------------------------------------------------------
    def _on_mousewheel(self, event: tk.Event) -> str:
        if getattr(event, "state", 0) & 0x0001:
            return self._on_shift_mousewheel(event)
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.yview_scroll(delta, "units")
        return "break"

    def _on_shift_mousewheel(self, event: tk.Event) -> str:
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.xview_scroll(delta, "units")
        return "break"

    def _on_scroll_up(self, _event: tk.Event) -> None:
        self.yview_scroll(-1, "units")

    def _on_scroll_down(self, _event: tk.Event) -> None:
        self.yview_scroll(1, "units")


# ---------------------------------------------------------------------------
# UI Controller
# ---------------------------------------------------------------------------


class AudioMapTool:
    """Main Tkinter controller for audio analysis and editing."""

    def __init__(self, root: tk.Tk, project: Optional[Path], project_name: Optional[str]):
        self.root = root
        self.project = project
        self.project_name = project_name
        self.assets_dir = self.project / "assets" if self.project else Path.cwd()
        self.timing_dir = self.project / "internal" / "timing" if self.project else Path.cwd()

        self.audio_path: Optional[Path] = None
        self.doc: Optional[TimingDocument] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._playback_job: Optional[str] = None
        self._play_start_time = 0.0
        self._is_playing = False
        self._play_offset = 0.0
        self.analysis_settings = AnalysisSettings()
        self._audio_pcm_array: Optional[Any] = None
        self._audio_samplerate = 0
        self._audio_channels = 0
        self._play_obj: Optional[Any] = None
        self._play_buffer_bytes: Optional[bytes] = None

        self._build_ui()

    # UI construction ------------------------------------------------------
    def _build_ui(self) -> None:
        title = "Create Basic Audio Map"
        if self.project_name:
            title += f" — {self.project_name}"
        self.root.title(title)
        self.root.geometry("1100x720")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.grid(row=0, column=0, sticky="nsew")

        ttk.Label(toolbar, text="Audio file:").grid(row=0, column=0, sticky="w")
        self.audio_var = tk.StringVar()
        entry = ttk.Entry(toolbar, textvariable=self.audio_var, width=60)
        entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        toolbar.columnconfigure(1, weight=1)

        ttk.Button(toolbar, text="Browse…", command=self._choose_audio).grid(row=0, column=2, padx=(0, 6))
        self.analyse_btn = ttk.Button(toolbar, text="Analyse", command=self._run_analysis, state="disabled")
        self.analyse_btn.grid(row=0, column=3, padx=(0, 6))
        ttk.Button(toolbar, text="Analysis settings…", command=self._edit_analysis_settings).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(toolbar, text="Load timing", command=self._load_timing_from_disk).grid(row=0, column=5, padx=(0, 6))
        ttk.Button(toolbar, text="Save timing", command=self._save_timing).grid(row=0, column=6)

        self.status_var = tk.StringVar(value="Select an audio file to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=1, column=0, columnspan=7, sticky="w", pady=(6, 0))

        main = ttk.Frame(self.root)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(2, weight=1)
        main.rowconfigure(1, weight=1)

        # Track list -------------------------------------------------------
        sidebar = ttk.Frame(main, padding=(8, 8, 4, 8))
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")
        ttk.Label(sidebar, text="Timing tracks", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.track_list = tk.Listbox(sidebar, width=28, activestyle="dotbox")
        self.track_list.pack(fill="y", expand=False, pady=(4, 8))
        self.track_list.bind("<<ListboxSelect>>", self._on_track_select)

        ttk.Button(sidebar, text="Add track", command=self._add_track).pack(fill="x", pady=(0, 4))
        ttk.Button(sidebar, text="Rename track", command=self._rename_track).pack(fill="x", pady=(0, 4))
        ttk.Button(sidebar, text="Delete track", command=self._delete_track).pack(fill="x")

        # Timeline ---------------------------------------------------------
        timeline_frame = ttk.Frame(main)
        timeline_frame.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=4, pady=(8, 4))
        timeline_frame.rowconfigure(0, weight=1)
        timeline_frame.columnconfigure(0, weight=1)

        canvas_container = ttk.Frame(timeline_frame)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        yscroll = ttk.Scrollbar(canvas_container, orient="vertical")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(canvas_container, orient="horizontal")
        xscroll.grid(row=1, column=0, sticky="ew")

        self.timeline = TimelineCanvas(
            canvas_container,
            on_select=self._select_event_from_timeline,
            on_add_marker=self._add_event_at_time,
        )
        self.timeline.grid(row=0, column=0, sticky="nsew")
        self.timeline.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        xscroll.config(command=self.timeline.xview)
        yscroll.config(command=self.timeline.yview)

        zoom_frame = ttk.Frame(timeline_frame)
        zoom_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(zoom_frame, text="Zoom").pack(side=tk.LEFT)
        self.zoom_var = tk.DoubleVar(value=80.0)
        zoom_slider = ttk.Scale(zoom_frame, from_=20, to=400, orient="horizontal", variable=self.zoom_var, command=self._on_zoom)
        zoom_slider.pack(fill="x", expand=True, padx=(6, 6))

        play_frame = ttk.Frame(zoom_frame)
        play_frame.pack(side=tk.RIGHT)
        ttk.Button(play_frame, text="Play", command=self._toggle_playback).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(play_frame, text="Stop", command=self._stop_playback).pack(side=tk.LEFT, padx=(6, 0))

        # Event editor -----------------------------------------------------
        editor = ttk.Frame(main, padding=(4, 8, 8, 8))
        editor.grid(row=1, column=1, columnspan=2, sticky="nsew")
        editor.columnconfigure(0, weight=1)
        editor.rowconfigure(1, weight=1)

        ttk.Label(editor, text="Events", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        columns = ("time", "value", "label", "duration")
        self.event_tree = ttk.Treeview(editor, columns=columns, show="headings", height=10)
        for col in columns:
            self.event_tree.heading(col, text=col.title())
            self.event_tree.column(col, width=90, anchor="center")
        self.event_tree.grid(row=1, column=0, sticky="nsew")
        self.event_tree.bind("<<TreeviewSelect>>", self._on_event_select)

        btn_frame = ttk.Frame(editor)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btn_frame, text="Add event", command=self._add_event_dialog).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Edit event", command=self._edit_event_dialog).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(btn_frame, text="Delete event", command=self._delete_event).pack(side=tk.LEFT, padx=(6, 0))

    # UI callbacks --------------------------------------------------------
    def _choose_audio(self) -> None:
        initialdir = self.assets_dir if self.assets_dir.exists() else Path.cwd()
        path = filedialog.askopenfilename(
            title="Choose audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
            initialdir=initialdir,
        )
        if not path:
            return
        self._stop_playback()
        self.audio_path = Path(path)
        self.audio_var.set(str(self.audio_path))
        self.analyse_btn.config(state="normal")
        self._reset_audio_buffer()
        self.doc = None
        self.timeline.set_tracks(OrderedDict())
        self.timeline.set_duration(0.0)
        self._load_timing_from_disk(auto=True)
        self.status_var.set("Ready to analyse.")

    def _on_zoom(self, _value: str) -> None:
        self.timeline.set_zoom(self.zoom_var.get())

    def _toggle_playback(self) -> None:
        if self._is_playing:
            self._stop_playback()
            return
        if not self.doc:
            return
        if not self._ensure_audio_buffer():
            return
        start_time = self.timeline.current_playhead_time
        if start_time >= self.doc.duration:
            start_time = 0.0
            self.timeline.focus_playhead(0.0)
        if not self._start_audio_playback(start_time):
            return
        self._is_playing = True
        self._play_offset = start_time
        self._play_start_time = time.perf_counter()
        self.status_var.set("Playing timeline + audio.")
        self._schedule_playhead()

    def _stop_playback(self) -> None:
        self._is_playing = False
        self._stop_audio_playback()
        if self._playback_job:
            try:
                self.root.after_cancel(self._playback_job)
            except ValueError:
                pass
            self._playback_job = None
        self._play_offset = self.timeline.current_playhead_time
        self.status_var.set("Playback stopped.")

    def _schedule_playhead(self) -> None:
        if not self._is_playing or not self.doc:
            return
        elapsed = time.perf_counter() - self._play_start_time + self._play_offset
        if elapsed >= self.doc.duration:
            self.timeline.focus_playhead(self.doc.duration)
            self._stop_playback()
            return
        self.timeline.focus_playhead(elapsed)
        self._playback_job = self.root.after(33, self._schedule_playhead)

    def _run_analysis(self) -> None:
        if not self.audio_path:
            return
        self.analyse_btn.config(state="disabled")
        self.status_var.set("Analysing audio… This may take a moment.")

        audio_path = self.audio_path
        settings = replace(self.analysis_settings)

        def worker() -> None:
            try:
                analyzer = AudioAnalyzer(audio_path, settings=settings)
                doc = analyzer.analyse()
            except Exception as exc:
                self.root.after(0, lambda: self._analysis_failed(exc))
                return
            self.root.after(0, lambda: self._analysis_done(doc))

        self._analysis_thread = threading.Thread(target=worker, daemon=True)
        self._analysis_thread.start()

    def _analysis_done(self, doc: TimingDocument) -> None:
        self.doc = doc
        self.timeline.set_tracks(doc.tracks)
        self.timeline.set_duration(doc.duration)
        self._refresh_track_list()
        self._refresh_event_tree()
        self.status_var.set("Analysis complete. Review and edit tracks, then save timing file.")
        self.analyse_btn.config(state="normal")

    def _analysis_failed(self, exc: Exception) -> None:
        messagebox.showerror("Analysis failed", f"Could not analyse audio: {exc}")
        self.analyse_btn.config(state="normal")
        self.status_var.set("Analysis failed. Install librosa + numpy and try again.")

    def _default_timing_path(self) -> Optional[Path]:
        if not self.project or not self.audio_path:
            return None
        timing_dir = self.project / "internal" / "timing"
        return timing_dir / f"{self.audio_path.stem}.timing.json"

    def _load_timing_from_disk(self, auto: bool = False) -> None:
        timing_path = self._default_timing_path()
        if not timing_path or not timing_path.exists():
            if not auto:
                messagebox.showinfo("Timing file", "No timing file found for this audio yet.")
            return
        try:
            data = json.loads(timing_path.read_text())
            self.doc = TimingDocument.from_dict(data)
            self.timeline.set_tracks(self.doc.tracks)
            self.timeline.set_duration(self.doc.duration)
            self._refresh_track_list()
            self._refresh_event_tree()
            self.status_var.set(f"Loaded timing from {timing_path.name}.")
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not read timing file: {exc}")

    def _save_timing(self) -> None:
        if not self.doc or not self.audio_path:
            messagebox.showinfo("Save timing", "Analyse or load a timing file first.")
            return
        timing_path = self._default_timing_path()
        if not timing_path:
            path = filedialog.asksaveasfilename(defaultextension=".json", title="Save timing")
            if not path:
                return
            timing_path = Path(path)
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        with timing_path.open("w", encoding="utf-8") as fh:
            json.dump(self.doc.to_dict(), fh, indent=2)
        if self.project:
            try:
                relative_path = timing_path.relative_to(self.project)
            except ValueError:
                relative_path = timing_path
        else:
            relative_path = timing_path
        self.status_var.set(f"Saved timing to {relative_path}.")

    def _reset_audio_buffer(self) -> None:
        self._audio_pcm_array = None
        self._audio_samplerate = 0
        self._audio_channels = 0
        self._play_buffer_bytes = None

    def _edit_analysis_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Analysis settings")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Hop length (samples)").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        hop_var = tk.StringVar(value=str(self.analysis_settings.hop_length))
        ttk.Entry(dialog, textvariable=hop_var).grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Beat tightness").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        beat_var = tk.StringVar(value=str(self.analysis_settings.beat_tightness))
        ttk.Entry(dialog, textvariable=beat_var).grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Start BPM").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        bpm_var = tk.StringVar(value=str(self.analysis_settings.start_bpm))
        ttk.Entry(dialog, textvariable=bpm_var).grid(row=2, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="RMS percentile").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        rms_var = tk.StringVar(value=str(self.analysis_settings.rms_percentile))
        ttk.Entry(dialog, textvariable=rms_var).grid(row=3, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Pitch-change percentile").grid(row=4, column=0, sticky="w", padx=6, pady=4)
        pitch_var = tk.StringVar(value=str(self.analysis_settings.pitch_change_percentile))
        ttk.Entry(dialog, textvariable=pitch_var).grid(row=4, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Onset delta").grid(row=5, column=0, sticky="w", padx=6, pady=4)
        onset_delta_var = tk.StringVar(value=str(self.analysis_settings.onset_delta))
        ttk.Entry(dialog, textvariable=onset_delta_var).grid(row=5, column=1, padx=6, pady=4)

        backtrack_var = tk.BooleanVar(value=self.analysis_settings.onset_backtrack)
        ttk.Checkbutton(dialog, text="Backtrack onsets", variable=backtrack_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 8))

        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(0, 8))

        def on_ok() -> None:
            try:
                hop = max(64, int(hop_var.get()))
            except ValueError:
                messagebox.showerror("Invalid hop length", "Hop length must be an integer.", parent=dialog)
                return
            try:
                beat_tightness = float(beat_var.get())
                start_bpm = max(1.0, float(bpm_var.get()))
                rms_percentile = max(0.0, min(100.0, float(rms_var.get())))
                pitch_percentile = max(0.0, min(100.0, float(pitch_var.get())))
                onset_delta = float(onset_delta_var.get())
            except ValueError:
                messagebox.showerror("Invalid value", "Please enter numeric values for all settings.", parent=dialog)
                return
            self.analysis_settings = AnalysisSettings(
                hop_length=hop,
                beat_tightness=beat_tightness,
                start_bpm=start_bpm,
                rms_percentile=rms_percentile,
                pitch_change_percentile=pitch_percentile,
                onset_backtrack=backtrack_var.get(),
                onset_delta=onset_delta,
            )
            self.status_var.set("Updated analysis settings.")
            dialog.destroy()

        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=6)

        dialog.wait_window()

    def _ensure_audio_buffer(self) -> bool:
        if self._audio_pcm_array is not None:
            return True
        if not self.audio_path:
            messagebox.showinfo("Audio required", "Choose an audio file first.")
            return False
        try:
            import numpy as np
            import soundfile as sf
        except Exception as exc:
            messagebox.showerror(
                "Missing dependency",
                "soundfile + numpy are required to load audio for playback.",
            )
            return False
        try:
            data, sr = sf.read(str(self.audio_path), dtype="float32", always_2d=True)
        except Exception as exc:
            messagebox.showerror("Audio load failed", f"Could not read audio: {exc}")
            return False
        if data.size == 0:
            messagebox.showerror("Audio load failed", "Audio file appears to be empty.")
            return False
        data = np.clip(data, -1.0, 1.0)
        pcm = (data * 32767).astype(np.int16)
        pcm = np.ascontiguousarray(pcm)
        self._audio_pcm_array = pcm
        self._audio_samplerate = int(sr)
        self._audio_channels = pcm.shape[1] if pcm.ndim > 1 else 1
        return True

    def _start_audio_playback(self, offset: float) -> bool:
        if self._audio_pcm_array is None or not self._audio_samplerate:
            return False
        try:
            import simpleaudio as sa
        except Exception:
            messagebox.showerror(
                "Playback unavailable",
                "Install the 'simpleaudio' package to hear the song while previewing the timeline.",
            )
            return False
        samples = len(self._audio_pcm_array)
        start_sample = min(samples, max(0, int(offset * self._audio_samplerate)))
        subset = self._audio_pcm_array[start_sample:]
        if subset.size == 0:
            return False
        audio_bytes = subset.tobytes()
        self._stop_audio_playback()
        self._play_buffer_bytes = audio_bytes
        channels = self._audio_channels or 1
        try:
            self._play_obj = sa.play_buffer(self._play_buffer_bytes, channels, 2, self._audio_samplerate)
        except Exception as exc:
            messagebox.showerror("Playback failed", f"Could not start playback: {exc}")
            self._play_obj = None
            self._play_buffer_bytes = None
            return False
        return True

    def _stop_audio_playback(self) -> None:
        play_obj = self._play_obj
        if play_obj is not None:
            try:
                play_obj.stop()
                # Ensure the playback thread has fully torn down before we
                # release the PCM buffer; otherwise simpleaudio can segfault
                # if the bytes object disappears while PortAudio still has a
                # pointer to it.
                try:
                    play_obj.wait_done()
                except Exception:
                    # ``wait_done`` may not exist on some backends; ignore.
                    pass
            except Exception:
                pass
        self._play_obj = None
        self._play_buffer_bytes = None

    def _refresh_track_list(self) -> None:
        self.track_list.delete(0, tk.END)
        if not self.doc:
            return
        for name in self.doc.tracks:
            self.track_list.insert(tk.END, name)
        if self.doc.tracks:
            self.track_list.select_set(0)
            self._on_track_select()

    def _on_track_select(self, _event: Optional[tk.Event] = None) -> None:
        self._refresh_event_tree()

    def _refresh_event_tree(self) -> None:
        for row in self.event_tree.get_children():
            self.event_tree.delete(row)
        if not self.doc:
            return
        track_name = self._selected_track_name
        if not track_name:
            return
        track = self.doc.tracks.get(track_name)
        if not track:
            return
        for idx, event in enumerate(track.events):
            self.event_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    f"{event.time:.3f}",
                    "" if event.value is None else f"{event.value:.3f}",
                    event.label or "",
                    "" if event.duration is None else f"{event.duration:.3f}",
                ),
            )
        self.timeline.set_tracks(self.doc.tracks)
        self.timeline.set_duration(self.doc.duration)

    @property
    def _selected_track_name(self) -> Optional[str]:
        selection = self.track_list.curselection()
        if not selection or not self.doc:
            return None
        idx = selection[0]
        if idx >= len(self.doc.tracks):
            return None
        return list(self.doc.tracks.keys())[idx]

    def _add_track(self) -> None:
        if not self.doc:
            messagebox.showinfo("Add track", "Load or analyse an audio file first.")
            return
        name = simpledialog.askstring("Track name", "Enter a name for the new track:")
        if not name:
            return
        if name in self.doc.tracks:
            messagebox.showerror("Duplicate", "A track with that name already exists.")
            return
        self.doc.tracks[name] = TimingTrack(name=name)
        self._refresh_track_list()

    def _rename_track(self) -> None:
        track_name = self._selected_track_name
        if not track_name or not self.doc:
            return
        new_name = simpledialog.askstring("Rename track", "New name:", initialvalue=track_name)
        if not new_name or new_name == track_name:
            return
        if new_name in self.doc.tracks:
            messagebox.showerror("Duplicate", "A track with that name already exists.")
            return
        track = self.doc.tracks[track_name]
        track.name = new_name
        original_items = list(self.doc.tracks.items())
        new_tracks: "OrderedDict[str, TimingTrack]" = OrderedDict()
        for name, existing in original_items:
            if name == track_name:
                new_tracks[new_name] = track
            else:
                new_tracks[name] = existing
        self.doc.tracks = new_tracks
        self._refresh_track_list()

    def _delete_track(self) -> None:
        track_name = self._selected_track_name
        if not track_name or not self.doc:
            return
        if messagebox.askyesno("Delete track", f"Delete '{track_name}'?"):
            self.doc.tracks.pop(track_name, None)
            self._refresh_track_list()
            self._refresh_event_tree()

    def _add_event_dialog(self) -> None:
        track_name = self._selected_track_name
        if not track_name or not self.doc:
            return
        event = self._prompt_for_event()
        if not event:
            return
        self.doc.tracks[track_name].events.append(event)
        self.doc.tracks[track_name].sort_events()
        self._refresh_event_tree()

    def _add_event_at_time(self, track_name: str, timestamp: float) -> None:
        if not self.doc:
            return
        event = TimingEvent(time=timestamp, label="manual")
        self.doc.tracks[track_name].events.append(event)
        self.doc.tracks[track_name].sort_events()
        self._refresh_event_tree()

    def _edit_event_dialog(self) -> None:
        track_name = self._selected_track_name
        if not track_name or not self.doc:
            return
        event_idx = self._selected_event_index
        if event_idx is None:
            return
        track = self.doc.tracks[track_name]
        event = track.events[event_idx]
        updated = self._prompt_for_event(event)
        if not updated:
            return
        track.events[event_idx] = updated
        track.sort_events()
        self._refresh_event_tree()

    def _delete_event(self) -> None:
        track_name = self._selected_track_name
        if not track_name or not self.doc:
            return
        idx = self._selected_event_index
        if idx is None:
            return
        if messagebox.askyesno("Delete event", "Remove selected event?"):
            track = self.doc.tracks[track_name]
            track.events.pop(idx)
            self._refresh_event_tree()

    @property
    def _selected_event_index(self) -> Optional[int]:
        selection = self.event_tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def _prompt_for_event(self, existing: Optional[TimingEvent] = None) -> Optional[TimingEvent]:
        dialog = tk.Toplevel(self.root)
        dialog.title("Event details")
        dialog.grab_set()
        ttk.Label(dialog, text="Time (seconds)").grid(row=0, column=0, sticky="w")
        time_var = tk.StringVar(value=f"{existing.time:.3f}" if existing else "0.0")
        ttk.Entry(dialog, textvariable=time_var).grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Value (optional numeric)").grid(row=1, column=0, sticky="w")
        value_var = tk.StringVar(
            value="" if not existing or existing.value is None else str(existing.value)
        )
        ttk.Entry(dialog, textvariable=value_var).grid(row=1, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Label").grid(row=2, column=0, sticky="w")
        label_var = tk.StringVar(value=existing.label if existing else "")
        ttk.Entry(dialog, textvariable=label_var).grid(row=2, column=1, padx=6, pady=4)

        ttk.Label(dialog, text="Duration (seconds)").grid(row=3, column=0, sticky="w")
        duration_var = tk.StringVar(
            value="" if not existing or existing.duration is None else str(existing.duration)
        )
        ttk.Entry(dialog, textvariable=duration_var).grid(row=3, column=1, padx=6, pady=4)

        result: Dict[str, Optional[TimingEvent]] = {"event": None}

        def on_ok() -> None:
            try:
                time_value = float(time_var.get())
            except ValueError:
                messagebox.showerror("Invalid time", "Time must be numeric.", parent=dialog)
                return
            value_field: Optional[float]
            if value_var.get().strip():
                try:
                    value_field = float(value_var.get())
                except ValueError:
                    messagebox.showerror("Invalid value", "Value must be numeric.", parent=dialog)
                    return
            else:
                value_field = None
            duration_field: Optional[float]
            if duration_var.get().strip():
                try:
                    duration_field = float(duration_var.get())
                except ValueError:
                    messagebox.showerror("Invalid duration", "Duration must be numeric.", parent=dialog)
                    return
            else:
                duration_field = None
            result["event"] = TimingEvent(
                time=time_value,
                value=value_field,
                label=label_var.get().strip() or None,
                duration=duration_field,
            )
            dialog.destroy()

        ttk.Button(dialog, text="OK", command=on_ok).grid(row=4, column=0, pady=8)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).grid(row=4, column=1, pady=8)

        dialog.wait_window()
        return result["event"]

    def _select_event_from_timeline(self, track_name: str, index: int) -> None:
        if not self.doc:
            return
        track_names = list(self.doc.tracks.keys())
        if track_name in track_names:
            idx = track_names.index(track_name)
            self.track_list.select_clear(0, tk.END)
            self.track_list.select_set(idx)
            self.track_list.activate(idx)
            self._refresh_event_tree()
            self.event_tree.selection_set(str(index))

    def _on_event_select(self, _event: tk.Event) -> None:  # type: ignore[override]
        track_name = self._selected_track_name
        idx = self._selected_event_index
        if track_name is None or idx is None or not self.doc:
            return
        track = self.doc.tracks[track_name]
        if idx >= len(track.events):
            return
        timestamp = track.events[idx].time
        self.timeline.focus_playhead(timestamp)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio analysis + timing editor")
    parser.add_argument("--project", help="Active project directory", default=None)
    parser.add_argument("--project-name", help="Active project name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve() if args.project else None
    root = tk.Tk()
    tool = AudioMapTool(root=root, project=project, project_name=args.project_name)
    if project and not project.exists():
        messagebox.showwarning("Project missing", f"Project folder {project} does not exist. Timing will be saved next to the chosen audio.")
        tool.project = None
    root.mainloop()


if __name__ == "__main__":
    main()
