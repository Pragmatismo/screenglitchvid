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
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageTk

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
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk


DEFAULT_FPS = 30
FRAME_SIZE = (960, 540)
DEFAULT_CODEC = "libx264"


@dataclass
class RenderSettings:
    width: int = FRAME_SIZE[0]
    height: int = FRAME_SIZE[1]
    fps: int = DEFAULT_FPS
    codec: str = DEFAULT_CODEC
    background: str = "black"
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


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


@dataclass
class FrequencyFrame:
    time: float
    levels: List[float]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrequencyFrame":
        time = float(payload.get("time", 0.0))
        levels = [float(value) for value in payload.get("levels", [])]
        return cls(time=time, levels=levels)


@dataclass
class FrequencyDocument:
    duration: float
    sample_rate: Optional[int]
    bin_edges: List[tuple[float, float]]
    capture_rate: float
    frames: List[FrequencyFrame]

    @classmethod
    def from_json(cls, path: Path) -> "FrequencyDocument":
        payload = json.loads(path.read_text(encoding="utf-8"))
        duration = float(payload.get("duration", 0.0))
        sample_rate = payload.get("sample_rate")
        edges_payload = payload.get("bin_edges", [])
        bin_edges: List[tuple[float, float]] = [
            (float(edge[0]), float(edge[1]))
            for edge in edges_payload
            if isinstance(edge, (list, tuple)) and len(edge) >= 2
        ]
        capture_rate = float(payload.get("capture_rate", 0.0))
        frames_payload = payload.get("frames", [])
        frames = [FrequencyFrame.from_dict(entry) for entry in frames_payload]
        return cls(
            duration=duration,
            sample_rate=sample_rate,
            bin_edges=bin_edges,
            capture_rate=capture_rate,
            frames=frames,
        )

    def level_at(self, timestamp: float, bin_index: int) -> float:
        if not self.frames:
            return 0.0
        times = [frame.time for frame in self.frames]
        idx = bisect_right(times, timestamp) - 1
        idx = max(0, min(idx, len(self.frames) - 1))
        frame = self.frames[idx]
        if bin_index < 0 or bin_index >= len(frame.levels):
            return 0.0
        return float(frame.levels[bin_index])


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
        self.scale = max(0.1, float(options.get("scale", 1.0)))
        self.variance = max(0.0, min(1.0, float(options.get("variance", 0.2))))
        self.scale_variation = 1.0 + rng.uniform(-self.variance, self.variance)
        palette = [
            (255, 214, 0),
            (255, 126, 0),
            (255, 46, 99),
            (166, 73, 255),
            (83, 224, 255),
        ]
        colour_choice = str(options.get("color", "random")).lower()
        colour_lookup = {
            "white": (255, 255, 255),
            "red": (255, 80, 80),
            "green": (120, 255, 120),
            "blue": (120, 180, 255),
        }
        if colour_choice == "random":
            self.colour = rng.choice(palette)
        else:
            self.colour = colour_lookup.get(colour_choice, palette[0])
        self.height = height
        self.width = width
        firework_type = str(options.get("firework_type", "random")).lower()
        self.available_types = ["ring", "star", "burst", "spray"]
        if firework_type not in self.available_types and firework_type != "random":
            firework_type = "random"
        if firework_type == "random":
            firework_type = rng.choice(self.available_types)
        self.firework_type = firework_type
        self.rays: list[tuple[float, float, float]] = []
        if self.firework_type in {"star", "burst"}:
            ray_count = rng.randint(8, 16)
            for _ in range(ray_count):
                angle = rng.uniform(0, math.tau)
                length = rng.uniform(0.6, 1.2)
                width_scale = rng.uniform(1.5, 3.5)
                self.rays.append((angle, length, width_scale))
        self.spray_particles: list[tuple[float, float, float]] = []
        if self.firework_type == "spray":
            particle_count = rng.randint(30, 45)
            for _ in range(particle_count):
                angle = rng.uniform(-math.pi / 2, math.pi / 2)
                distance = rng.uniform(0.4, 1.1)
                size = rng.uniform(2, 4)
                self.spray_particles.append((angle, distance, size))

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
        base_radius = 40 * self.intensity * self.scale * self.scale_variation
        if self.firework_type == "ring":
            self._draw_ring(draw, base_radius, decay)
        elif self.firework_type == "star":
            self._draw_star(draw, base_radius, decay)
        elif self.firework_type == "burst":
            self._draw_burst(draw, base_radius, decay)
        elif self.firework_type == "spray":
            self._draw_spray(draw, base_radius, decay)
        else:
            self._draw_ring(draw, base_radius, decay)
        spark_radius = (6 + 8 * self.intensity * decay) * self.scale * 0.8
        draw.ellipse(
            (
                self.target_x - spark_radius,
                self.target_y - spark_radius,
                self.target_x + spark_radius,
                self.target_y + spark_radius,
            ),
            fill=self.colour + (int(150 * decay),),
        )

    def _draw_ring(self, draw: ImageDraw.ImageDraw, base_radius: float, decay: float) -> None:
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

    def _draw_star(self, draw: ImageDraw.ImageDraw, base_radius: float, decay: float) -> None:
        alpha = int(210 * decay)
        colour = self.colour + (alpha,)
        for angle, length, width_scale in self.rays:
            radius = base_radius * length
            end_x = self.target_x + radius * math.cos(angle)
            end_y = self.target_y + radius * math.sin(angle)
            draw.line(
                [(self.target_x, self.target_y), (end_x, end_y)],
                fill=colour,
                width=max(1, int(2 * width_scale)),
            )

    def _draw_burst(self, draw: ImageDraw.ImageDraw, base_radius: float, decay: float) -> None:
        for angle, length, width_scale in self.rays:
            radius = base_radius * (length + 0.2)
            start_x = self.target_x + base_radius * 0.2 * math.cos(angle)
            start_y = self.target_y + base_radius * 0.2 * math.sin(angle)
            end_x = self.target_x + radius * math.cos(angle)
            end_y = self.target_y + radius * math.sin(angle)
            alpha = int(200 * decay)
            colour = self.colour + (alpha,)
            draw.line(
                [(start_x, start_y), (end_x, end_y)],
                fill=colour,
                width=max(1, int(3 * width_scale)),
            )
            spark_alpha = int(180 * decay)
            spark_radius = max(1.5, 3.5 * (1.0 - decay))
            draw.ellipse(
                (
                    end_x - spark_radius,
                    end_y - spark_radius,
                    end_x + spark_radius,
                    end_y + spark_radius,
                ),
                fill=self.colour + (spark_alpha,),
            )

    def _draw_spray(self, draw: ImageDraw.ImageDraw, base_radius: float, decay: float) -> None:
        for angle, distance, size in self.spray_particles:
            radius = base_radius * distance
            end_x = self.target_x + radius * math.cos(angle)
            end_y = self.target_y - radius * math.sin(angle)
            alpha = int(200 * decay)
            draw.ellipse(
                (
                    end_x - size,
                    end_y - size,
                    end_x + size,
                    end_y + size,
                ),
                fill=self.colour + (alpha,),
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


class PopEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        options: Dict[str, Any],
    ) -> None:
        self.fps = max(1, int(fps))
        self.event_time = float(event.time)
        self.opacity = max(0.0, min(1.0, float(options.get("opacity", 1.0))))
        base_size = max(6.0, float(options.get("size", 36.0)))
        if bool(options.get("use_value_for_size", False)) and event.value is not None:
            base_size *= max(0.1, abs(float(event.value)))
        variance = max(0.0, float(options.get("size_variance", 0.25)))
        self.radius = max(4.0, base_size * (1 + rng.uniform(-variance, variance)))
        self.color = tuple(rng.randint(50, 255) for _ in range(3))
        self.alpha = int(255 * self.opacity)
        start_from_bottom = max(0.0, min(0.95, float(options.get("start_percent_from_bottom", 0.25))))
        width, height = canvas_size
        max_y = height * (1.0 - start_from_bottom)
        self.x = rng.uniform(self.radius + 4, max(self.radius + 4, width - self.radius - 4))
        self.y = rng.uniform(self.radius + 4, max(self.radius + 4, max_y - self.radius - 4))
        self.shards: list[dict[str, Any]] = []
        self.shards_spawned = False
        self.burst_window = 1.0 / float(self.fps)
        self.gravity = 550.0
        shard_lifetime = 1.8
        super().__init__(0.0, self.event_time + shard_lifetime)
        self._last_update_time = self.event_time
        self._rng = rng

    def _spawn_shards(self) -> None:
        if self.shards_spawned:
            return
        self.shards_spawned = True
        shard_count = 10
        for _ in range(shard_count):
            angle = self._rng.uniform(0, 2 * math.pi)
            speed = self._rng.uniform(60.0, 160.0)
            shard = {
                "x": self.x,
                "y": self.y,
                "vx": math.cos(angle) * speed,
                "vy": -abs(math.sin(angle) * speed * 0.6),
                "alpha": self.alpha,
                "radius": self._rng.uniform(self.radius * 0.12, self.radius * 0.24),
            }
            self.shards.append(shard)

    def _update_shards(self, time_s: float) -> None:
        delta = max(0.0, time_s - self._last_update_time)
        if delta <= 0:
            return
        self._last_update_time = time_s
        decay_steps = max(1.0, delta * self.fps)
        decay_factor = 0.88 ** decay_steps
        remaining: list[dict[str, Any]] = []
        for shard in self.shards:
            shard["vy"] += self.gravity * delta
            shard["x"] += shard["vx"] * delta
            shard["y"] += shard["vy"] * delta
            shard["alpha"] = max(0, int(shard["alpha"] * decay_factor))
            if shard["alpha"] > 2:
                remaining.append(shard)
        self.shards = remaining

    def _draw_circle(self, draw: ImageDraw.ImageDraw) -> None:
        box = (
            self.x - self.radius,
            self.y - self.radius,
            self.x + self.radius,
            self.y + self.radius,
        )
        fill = self.color + (self.alpha,)
        draw.ellipse(box, fill=fill, outline=(0, 0, 0, self.alpha), width=2)

    def _draw_star(self, draw: ImageDraw.ImageDraw, points: int = 5) -> None:
        outer_r = self.radius * 1.2
        inner_r = self.radius * 0.5
        coords: list[tuple[float, float]] = []
        for i in range(points * 2):
            angle = math.pi / points * i - math.pi / 2
            radius = outer_r if i % 2 == 0 else inner_r
            coords.append((self.x + radius * math.cos(angle), self.y + radius * math.sin(angle)))
        fill = self.color + (self.alpha,)
        draw.polygon(coords, fill=fill, outline=(0, 0, 0, self.alpha))

    def _draw_shards(self, draw: ImageDraw.ImageDraw) -> None:
        for shard in self.shards:
            alpha = shard["alpha"]
            fill = self.color + (alpha,)
            box = (
                shard["x"] - shard["radius"],
                shard["y"] - shard["radius"],
                shard["x"] + shard["radius"],
                shard["y"] + shard["radius"],
            )
            draw.ellipse(box, fill=fill)

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - requires human inspection
        if time_s < self.start_time or time_s > self.end_time:
            return
        draw = ImageDraw.Draw(image, "RGBA")
        if time_s < self.event_time:
            self._draw_circle(draw)
            return
        if time_s <= self.event_time + self.burst_window:
            self._spawn_shards()
            self._draw_star(draw)
            return
        if not self.shards_spawned:
            self._spawn_shards()
        self._update_shards(time_s)
        self._draw_shards(draw)


class SplashEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        options: Dict[str, Any],
    ) -> None:
        self.fps = max(1, int(fps))
        self.event_time = float(event.time)
        width, height = canvas_size
        self.color = self._parse_color(options.get("color", "150,200,255"))
        base_start_size = max(2.0, float(options.get("start_size", 28.0)))
        base_impact_size = max(2.0, float(options.get("impact_size", 10.0)))
        size_variance = max(0.0, float(options.get("size_variance", 0.25)))
        self.start_size = base_start_size * (1 + rng.uniform(-size_variance, size_variance))
        self.impact_size = base_impact_size * (1 + rng.uniform(-size_variance, size_variance))
        speed_variance = max(0.0, float(options.get("speed_variance", 0.2)))
        base_motion_frames = max(2, int(options.get("motion_frames", 10)))
        varied_frames = int(round(base_motion_frames * (1 + rng.uniform(-speed_variance, speed_variance))))
        self.motion_frames = max(2, varied_frames)
        self.fall_duration = self.motion_frames / self.fps
        self.splash_duration = 0.9
        self.impact_x = rng.uniform(width * 0.15, width * 0.85)
        self.impact_y = rng.uniform(height * 0.35, height * 0.9)
        fall_distance = height * rng.uniform(0.12, 0.28)
        self.start_x = self.impact_x + rng.uniform(-20.0, 20.0)
        self.start_y = max(0.0, self.impact_y - fall_distance)
        self.drift_x = rng.uniform(-18.0, 18.0)
        burst_count = max(3.0, float(options.get("burst_count", 14)))
        burst_count_variance = max(0.0, float(options.get("burst_count_variance", 0.25)))
        self.burst_count = max(3, int(round(burst_count * (1 + rng.uniform(-burst_count_variance, burst_count_variance)))))
        self.burst_distance = max(10.0, float(options.get("burst_distance", 140.0)))
        self.burst_distance_variance = max(0.0, float(options.get("burst_distance_variance", 0.35)))
        start = self.event_time - self.fall_duration
        end = self.event_time + self.splash_duration
        super().__init__(start, end)
        self.particles: list[dict[str, Any]] = []
        self.splash_spawned = False
        self._last_update_time = self.event_time
        self._rng = rng

    def _parse_color(self, value: Any) -> tuple[int, int, int, int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r, g, b = value[:3]
            else:
                parts = str(value).split(",")
                r, g, b = (int(float(parts[i])) for i in range(3))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
        except Exception:
            r, g, b = 150, 200, 255
        return (r, g, b, 230)

    def _spawn_splash(self) -> None:
        if self.splash_spawned:
            return
        self.splash_spawned = True
        distance = self.burst_distance * (1 + self._rng.uniform(-self.burst_distance_variance, self.burst_distance_variance))
        distance = max(6.0, distance)
        fade = max(0.35, self.splash_duration * 0.85)
        for _ in range(self.burst_count):
            angle = self._rng.uniform(0, math.tau)
            speed = distance / fade
            jitter = 0.6 + self._rng.random() * 0.8
            vx = math.cos(angle) * speed * jitter
            vy = math.sin(angle) * speed * jitter
            radius = max(1.0, self.impact_size * (0.2 + self._rng.random() * 0.45))
            self.particles.append(
                {
                    "x": self.impact_x,
                    "y": self.impact_y,
                    "vx": vx,
                    "vy": vy,
                    "alpha": 255,
                    "radius": radius,
                    "age": 0.0,
                    "life": fade,
                }
            )

    def _update_particles(self, time_s: float) -> None:
        delta = max(0.0, time_s - self._last_update_time)
        if delta <= 0:
            return
        self._last_update_time = time_s
        decay_steps = max(1.0, delta * self.fps)
        decay_factor = 0.9 ** decay_steps
        remaining: list[dict[str, Any]] = []
        for particle in self.particles:
            particle["x"] += particle["vx"] * delta
            particle["y"] += particle["vy"] * delta
            particle["vx"] *= decay_factor
            particle["vy"] *= decay_factor
            particle["age"] += delta
            particle["alpha"] = max(0, int(255 * max(0.0, 1.0 - particle["age"] / particle["life"])))
            if particle["alpha"] > 2:
                remaining.append(particle)
        self.particles = remaining

    def _draw_drop(self, draw: ImageDraw.ImageDraw, progress: float) -> None:
        x = self.start_x + (self.impact_x - self.start_x) * progress + self.drift_x * (progress**1.2)
        y = self.start_y + (self.impact_y - self.start_y) * progress
        size = self.start_size + (self.impact_size - self.start_size) * progress
        radius = size / 2.0
        alpha = int(self.color[3] * (0.75 + 0.25 * (1.0 - progress)))
        color = self.color[:3] + (alpha,)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        glint_radius = max(1.0, radius * 0.45)
        draw.ellipse(
            (x - glint_radius, y - glint_radius, x + glint_radius, y + glint_radius),
            fill=self.color[:3] + (min(255, int(alpha * 0.8)),),
        )

    def _draw_splash(self, draw: ImageDraw.ImageDraw) -> None:
        for particle in self.particles:
            radius = particle["radius"] * (0.6 + 0.4 * max(0.0, 1.0 - particle["age"] / particle["life"]))
            color = self.color[:3] + (particle["alpha"],)
            box = (
                particle["x"] - radius,
                particle["y"] - radius,
                particle["x"] + radius,
                particle["y"] + radius,
            )
            draw.ellipse(box, fill=color)

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        if time_s < self.start_time or time_s > self.end_time:
            return
        draw = ImageDraw.Draw(image, "RGBA")
        if time_s < self.event_time:
            if self.fall_duration > 0:
                progress = max(0.0, min(1.0, (time_s - self.start_time) / self.fall_duration))
            else:
                progress = 1.0
            self._draw_drop(draw, progress)
            return
        if not self.splash_spawned:
            self._spawn_splash()
        self._update_particles(time_s)
        if time_s - self.event_time < 0.12:
            draw.ellipse(
                (
                    self.impact_x - self.impact_size,
                    self.impact_y - self.impact_size,
                    self.impact_x + self.impact_size,
                    self.impact_y + self.impact_size,
                ),
                fill=self.color[:3] + (self.color[3],),
            )
        self._draw_splash(draw)


@dataclass
class PlantTarget:
    center: tuple[float, float]
    base_y: float


class PlantEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        options: Dict[str, Any],
        target: PlantTarget,
        duration: float,
        area: tuple[float, float, float, float],
    ) -> None:
        self.fps = max(1, int(fps))
        self.event_time = float(event.time)
        self.target = target
        self.area = area
        self.start_x = (area[0] + area[2]) / 2.0
        self.start_y = area[3]
        lead_frames = max(1, int(options.get("seed_lead_frames", options.get("pre_launch_frames", 12))))
        self.lead_time = lead_frames / self.fps
        base_grow_frames = max(1, int(options.get("grow_frames", 20)))
        grow_variance = max(0.0, float(options.get("grow_variance", options.get("growth_variance", 0.1))))
        grow_frames = max(1, int(round(base_grow_frames * (1 + rng.uniform(-grow_variance, grow_variance)))))
        self.grow_time = grow_frames / self.fps
        base_height = max(8.0, float(options.get("plant_height", options.get("height", 120.0))))
        height_variance = max(0.0, float(options.get("height_variance", options.get("height_randomness", 0.2))))
        self.height = base_height * (1 + rng.uniform(-height_variance, height_variance))
        self.arc_height = max(10.0, (area[3] - area[1]) * 0.2 * (1 + rng.uniform(-0.3, 0.3)))
        plant_type = str(options.get("plant_type", "grass")).lower()
        choices = {"grass", "mushroom", "random"}
        if plant_type not in choices:
            plant_type = "grass"
        if plant_type == "random":
            plant_type = rng.choice(["grass", "mushroom"])
        self.plant_type = plant_type
        start = self.event_time - self.lead_time
        end = max(duration, self.event_time + self.grow_time)
        super().__init__(start, end)
        self.seed_radius = 5.0
        self.seed_color = (180, 140, 90, 220)

    def _progress(self, now: float, start: float, duration: float) -> float:
        if duration <= 0:
            return 1.0
        return max(0.0, min(1.0, (now - start) / duration))

    def _draw_seed(self, draw: ImageDraw.ImageDraw, time_s: float) -> None:
        t = self._progress(time_s, self.event_time - self.lead_time, self.lead_time)
        if t <= 0 or t > 1:
            return
        cx = self.start_x + (self.target.center[0] - self.start_x) * t
        cy = self.start_y + (self.target.center[1] - self.start_y) * t - self.arc_height * 4 * t * (1 - t)
        r = self.seed_radius
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=self.seed_color)

    def _draw_grass(self, draw: ImageDraw.ImageDraw, growth: float) -> None:
        height = self.height * growth
        base_width = max(2.0, height * 0.08)
        lean = (growth - 0.5) * 0.15 * self.height
        x0 = self.target.center[0] - base_width / 2
        x1 = self.target.center[0] + base_width / 2
        y0 = self.target.base_y
        tip = (self.target.center[0] + lean, max(self.area[1], y0 - height))
        points = [(x0, y0), tip, (x1, y0)]
        draw.polygon(points, fill=(60, 150, 60, 255))

    def _draw_mushroom(self, draw: ImageDraw.ImageDraw, growth: float) -> None:
        height = self.height * growth
        stem_height = height * 0.65
        cap_height = max(4.0, height * 0.35)
        stem_width = max(3.0, height * 0.12)
        base_y = self.target.base_y
        stem_top_y = base_y - stem_height
        x0 = self.target.center[0] - stem_width / 2
        x1 = self.target.center[0] + stem_width / 2
        draw.rectangle((x0, stem_top_y, x1, base_y), fill=(232, 224, 200, 255))
        cap_radius_x = max(stem_width * 0.8, stem_width * 1.8)
        cap_radius_y = cap_height
        cap_center = (self.target.center[0], stem_top_y)
        cap_box = (
            cap_center[0] - cap_radius_x,
            cap_center[1] - cap_radius_y,
            cap_center[0] + cap_radius_x,
            cap_center[1] + cap_radius_y,
        )
        draw.pieslice(cap_box, start=0, end=180, fill=(210, 90, 90, 255))

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        if time_s < self.start_time:
            return
        draw = ImageDraw.Draw(image, "RGBA")
        if time_s < self.event_time:
            self._draw_seed(draw, time_s)
            return
        growth = self._progress(time_s, self.event_time, self.grow_time)
        if self.plant_type == "grass":
            self._draw_grass(draw, growth)
        else:
            self._draw_mushroom(draw, growth)


class WallEffect(VisualEffect):
    def __init__(
        self,
        events: Iterable[TimingEvent],
        duration: float,
        canvas_size: tuple[int, int],
        options: Dict[str, Any],
    ) -> None:
        super().__init__(0.0, duration)
        self.events = sorted(events, key=lambda e: e.time)
        self.width, self.height = canvas_size
        self.aspect_ratio = max(1.2, float(options.get("aspect_ratio", 2.0)))
        self.edge_thickness = max(0, int(options.get("edge_thickness", 2)))
        self.brick_color = self._parse_color(options.get("brick_color", "186,84,60"))
        self.edge_color = self._parse_color(options.get("edge_color", "80,40,28"))
        self.bricks: list[tuple[float, float, float, float, float]] = []
        self._plan_wall()

    def _parse_color(self, value: Any) -> tuple[int, int, int, int]:
        if isinstance(value, str):
            parts = value.replace(";", ",").split(",")
            if len(parts) >= 3:
                try:
                    return tuple(
                        max(0, min(255, int(float(part.strip())))) for part in parts[:3]
                    ) + (255,)
                except Exception:
                    pass
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            try:
                return tuple(max(0, min(255, int(v))) for v in value[:3]) + (255,)
            except Exception:
                pass
        return 186, 84, 60, 255

    def _plan_wall(self) -> None:
        total = len(self.events)
        if total == 0:
            return
        ratio = self.aspect_ratio
        columns_guess = max(1, int(math.ceil(math.sqrt(total * (self.width / max(1.0, self.height)) / ratio))))
        rows = max(1, int(math.ceil(total / columns_guess)))
        brick_height = self.height / rows
        brick_width = ratio * brick_height
        columns = max(columns_guess, int(math.ceil(self.width / brick_width)) + 2)
        rows = max(1, int(math.ceil(total / columns)))
        brick_height = self.height / rows
        brick_width = ratio * brick_height
        start_x = -brick_width
        for idx, event in enumerate(self.events):
            row = idx // columns
            col = idx % columns
            x_offset = (brick_width / 2.0) if row % 2 else 0.0
            x0 = start_x + col * brick_width + x_offset
            x1 = x0 + brick_width
            y1 = self.height - row * brick_height
            y0 = y1 - brick_height
            self.bricks.append((x0, y0, x1, y1, float(event.time)))

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        if not self.bricks:
            return
        draw = ImageDraw.Draw(image, "RGBA")
        for x0, y0, x1, y1, event_time in self.bricks:
            if time_s < event_time:
                continue
            draw.rectangle((x0, y0, x1, y1), fill=self.brick_color)
            if self.edge_thickness > 0:
                draw.rectangle((x0, y0, x1, y1), outline=self.edge_color, width=self.edge_thickness)


class ZigZagEffect(VisualEffect):
    def __init__(
        self,
        event: TimingEvent,
        fps: int,
        canvas_size: tuple[int, int],
        rng: random.Random,
        options: Dict[str, Any],
    ) -> None:
        self.event_time = float(event.time)
        fade_frames = max(0, int(options.get("fade", 12)))
        self.fade_time = fade_frames / fps if fps > 0 else 0.0
        start = self.event_time
        end = self.event_time + self.fade_time
        super().__init__(start, end)
        self.width, self.height = canvas_size
        base_color = self._parse_color(options.get("color", (100, 200, 100)))
        color_variance = float(options.get("color_variance", 10))
        base_line_width = float(options.get("line_width", 2.0))
        line_width_variance = float(options.get("line_width_variance", 0.0))
        base_bar_width = float(options.get("bar_width", 25.0))
        bar_width_variance = float(options.get("bar_width_variance", 0.0))
        self.amount = max(1, int(options.get("amount", 1)))
        alignment = str(options.get("alignment", "both")).lower()
        self.alignment_options = {alignment} if alignment in {"horizontal", "vertical"} else {"horizontal", "vertical"}

        self.lines: list[dict[str, Any]] = []
        for _ in range(self.amount):
            orientation = self._choose_orientation(rng)
            zig_width = max(1.0, base_bar_width + rng.uniform(-bar_width_variance, bar_width_variance))
            line_width = max(1.0, base_line_width + rng.uniform(-line_width_variance, line_width_variance))
            color = self._vary_color(base_color, color_variance, rng)
            points = (
                self._build_horizontal_path(rng, zig_width)
                if orientation == "horizontal"
                else self._build_vertical_path(rng, zig_width)
            )
            self.lines.append({"points": points, "color": color, "width": line_width})

    def _parse_color(self, value: Any) -> tuple[int, int, int]:
        if isinstance(value, str):
            parts = value.replace(";", ",").split(",")
            if len(parts) == 3:
                try:
                    return tuple(max(0, min(255, int(float(part.strip())))) for part in parts)  # type: ignore[return-value]
                except Exception:
                    pass
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return tuple(max(0, min(255, int(v))) for v in value)  # type: ignore[return-value]
            except Exception:
                pass
        return 100, 200, 100

    def _vary_color(self, base: tuple[int, int, int], variance: float, rng: random.Random) -> tuple[int, int, int]:
        return tuple(
            max(0, min(255, int(channel + rng.uniform(-variance, variance)))) for channel in base
        )

    def _choose_orientation(self, rng: random.Random) -> str:
        if len(self.alignment_options) == 1:
            return next(iter(self.alignment_options))
        return "horizontal" if rng.random() < 0.5 else "vertical"

    def _build_horizontal_path(self, rng: random.Random, zig_width: float) -> list[tuple[float, float]]:
        y_base = rng.uniform(0, self.height)
        step = max(4.0, zig_width)
        x = 0.0
        points: list[tuple[float, float]] = [(0.0, y_base)]
        direction = 1.0
        while x < self.width:
            x += step + rng.uniform(-step * 0.35, step * 0.35)
            offset = direction * zig_width
            y = min(max(y_base + offset, 0.0), float(max(self.height - 1, 0)))
            points.append((min(x, self.width), y))
            direction *= -1.0
        points.append((float(self.width), y_base))
        return points

    def _build_vertical_path(self, rng: random.Random, zig_width: float) -> list[tuple[float, float]]:
        x_base = rng.uniform(0, self.width)
        step = max(4.0, zig_width)
        y = 0.0
        points: list[tuple[float, float]] = [(x_base, 0.0)]
        direction = 1.0
        while y < self.height:
            y += step + rng.uniform(-step * 0.35, step * 0.35)
            offset = direction * zig_width
            x = min(max(x_base + offset, 0.0), float(max(self.width - 1, 0)))
            points.append((x, min(y, self.height)))
            direction *= -1.0
        points.append((x_base, float(self.height)))
        return points

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - requires human inspection
        if time_s < self.start_time or time_s > self.end_time:
            return
        decay = 1.0
        if self.fade_time > 0:
            decay = max(0.0, 1.0 - (time_s - self.event_time) / self.fade_time)
        alpha = int(255 * decay)
        if alpha <= 0:
            return
        draw = ImageDraw.Draw(image, "RGBA")
        for line in self.lines:
            color = tuple(line["color"]) + (alpha,)
            draw.line(line["points"], fill=color, width=int(line["width"]))


class ConveyorBuildEffect(VisualEffect):
    def __init__(
        self,
        events: list[TimingEvent],
        duration: float,
        canvas_size: tuple[int, int],
        options: Dict[str, Any],
        rng: random.Random,
        sprite_root: Optional[Path] = None,
    ) -> None:
        end_time = duration + float(options.get("bar_length", 1200.0)) / max(
            1.0, float(options.get("bar_speed", 180.0))
        )
        super().__init__(0.0, end_time)
        self.events = sorted(events, key=lambda e: e.time)
        self.width, self.height = canvas_size
        self.rng = rng
        self.bar_speed = max(1.0, float(options.get("bar_speed", 180.0)))
        self.bar_length = max(10.0, float(options.get("bar_length", 1200.0)))
        self.horizon_percent = max(0.05, min(0.95, float(options.get("horizon_percent", 0.22))))
        self.origin_percent = max(0.1, min(0.98, float(options.get("origin_percent", 0.82))))
        self.active_zone = max(5.0, float(options.get("active_zone_length", 420.0)))
        self.spawn_depth = max(0.0, float(options.get("spawn_depth", 0.0)))
        self.show_bar = bool(options.get("show_bar", True))
        self.bar_width_near = max(2.0, float(options.get("bar_width_near", 80.0)))
        self.bar_width_far = max(1.0, float(options.get("bar_width_far", 12.0)))
        self.bar_color = self._parse_color(options.get("bar_color", "200,200,210"))
        self.small_dir = self._resolve_dir(options.get("small_dir"), sprite_root, "small")
        self.mid_dir = self._resolve_dir(options.get("mid_dir"), sprite_root, "mid")
        self.large_dir = self._resolve_dir(options.get("large_dir"), sprite_root, "large")
        self.level_scales = (
            max(0.05, float(options.get("scale_small", 1.0))),
            max(0.05, float(options.get("scale_mid", 2.0))),
            max(0.05, float(options.get("scale_large", 4.0))),
        )
        self.thresholds = (
            max(1, int(options.get("threshold_small", 1))),
            max(1, int(options.get("threshold_mid", 3))),
            max(1, int(options.get("threshold_large", 6))),
        )
        self._sprite_cache: dict[str, list[Image.Image]] = {}
        self.attachments: list[dict[str, Any]] = []
        self._next_id = 1
        self._event_idx = 0
        self._last_update_time = 0.0
        self.side_state: dict[str, dict[str, Any]] = {
            "left": {"counter": 0, "active_id": None},
            "top": {"counter": 0, "active_id": None},
            "right": {"counter": 0, "active_id": None},
        }

    def _parse_color(self, value: Any) -> tuple[int, int, int, int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r, g, b = value[:3]
            else:
                parts = str(value).split(",")
                r, g, b = (int(float(parts[i])) for i in range(3))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
        except Exception:
            r, g, b = 200, 200, 210
        return (r, g, b, 255)

    def _resolve_dir(self, path_value: Any, sprite_root: Optional[Path], fallback: str) -> Path:
        if path_value:
            path = Path(str(path_value)).expanduser()
            return path
        base = sprite_root or Path.cwd()
        return base / fallback

    def _load_sprites(self, directory: Path) -> list[Image.Image]:
        cache_key = str(directory.expanduser().resolve())
        if cache_key in self._sprite_cache:
            return self._sprite_cache[cache_key]
        sprites: list[Image.Image] = []
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                try:
                    sprites.append(Image.open(path).convert("RGBA"))
                except Exception:
                    continue
        if not sprites:
            placeholder = Image.new("RGBA", (96, 96), (255, 140, 140, 255))
            draw = ImageDraw.Draw(placeholder)
            draw.rectangle((10, 10, 86, 86), outline=(0, 0, 0, 255), width=4)
            draw.line((10, 48, 86, 48), fill=(0, 0, 0, 255), width=4)
            sprites.append(placeholder)
        self._sprite_cache[cache_key] = sprites
        return sprites

    def _random_sprite(self, level: int, side: str) -> Image.Image:
        level_idx = max(1, min(3, level))
        directory = [self.small_dir, self.mid_dir, self.large_dir][level_idx - 1]
        candidates = self._load_sprites(directory)
        base = self.rng.choice(candidates)
        if side == "right":
            base = base.transpose(Image.FLIP_LEFT_RIGHT)
        elif side == "top":
            base = base.rotate(90, expand=True)
        return base

    def _add_attachment(self, side: str, level: int, z: float) -> int:
        sprite = self._random_sprite(level, side)
        attachment = {
            "id": self._next_id,
            "side": side,
            "level": level,
            "z": z,
            "sprite": sprite,
            "size": sprite.size,
        }
        self._next_id += 1
        self.attachments.append(attachment)
        return attachment["id"]

    def _find_attachment(self, attachment_id: Optional[int]) -> Optional[dict[str, Any]]:
        if attachment_id is None:
            return None
        for att in self.attachments:
            if att.get("id") == attachment_id:
                return att
        return None

    def _perspective_at(self, z: float) -> tuple[float, float, float]:
        progress = max(0.0, min(1.0, z / self.bar_length))
        origin_y = self.height * self.origin_percent
        horizon_y = self.height * self.horizon_percent
        y = origin_y + (horizon_y - origin_y) * progress
        scale = max(0.05, 1.0 - 0.75 * progress)
        bar_width = self.bar_width_near + (self.bar_width_far - self.bar_width_near) * progress
        return y, scale, bar_width

    def _move_attachments(self, delta: float) -> None:
        if delta <= 0:
            return
        for att in self.attachments:
            att["z"] += self.bar_speed * delta
        self.attachments = [att for att in self.attachments if att["z"] <= self.bar_length]
        for side, state in self.side_state.items():
            att = self._find_attachment(state.get("active_id"))
            if att is None or att["z"] > self.active_zone:
                state["active_id"] = None
                state["counter"] = 0

    def _handle_event(self, event: TimingEvent) -> None:
        side = self.rng.choice(["left", "top", "right"])
        state = self.side_state[side]
        att = self._find_attachment(state.get("active_id"))
        if att is None or att["z"] > self.active_zone:
            new_id = self._add_attachment(side, 1, max(0.0, self.spawn_depth))
            state["active_id"] = new_id
            state["counter"] = 1
            return

        state["counter"] += 1
        threshold_idx = min(att["level"] - 1, 2)
        threshold = self.thresholds[threshold_idx]
        if state["counter"] > threshold and att["level"] < 3:
            att["level"] += 1
            att["sprite"] = self._random_sprite(att["level"], side)
            att["size"] = att["sprite"].size
            state["counter"] = 0

    def _advance(self, target_time: float) -> None:
        if target_time < self._last_update_time:
            self._last_update_time = target_time
        while self._event_idx < len(self.events) and self.events[self._event_idx].time <= target_time:
            event_time = max(self._last_update_time, self.events[self._event_idx].time)
            self._move_attachments(event_time - self._last_update_time)
            self._last_update_time = event_time
            self._handle_event(self.events[self._event_idx])
            self._event_idx += 1
        self._move_attachments(target_time - self._last_update_time)
        self._last_update_time = target_time

    def _draw_bar(self, draw: ImageDraw.ImageDraw) -> None:
        if not self.show_bar:
            return
        origin_y = self.height * self.origin_percent
        horizon_y = self.height * self.horizon_percent
        near_half = self.bar_width_near / 2.0
        far_half = self.bar_width_far / 2.0
        center_x = self.width / 2.0
        points = [
            (center_x - near_half, origin_y),
            (center_x + near_half, origin_y),
            (center_x + far_half, horizon_y),
            (center_x - far_half, horizon_y),
        ]
        draw.polygon(points, fill=self.bar_color)

    def _draw_attachment(self, image: Image.Image, att: dict[str, Any]) -> None:
        y, scale, _bar_width = self._perspective_at(att["z"])
        level_idx = max(1, min(3, att["level"]))
        level_scale = self.level_scales[level_idx - 1]
        sprite = att["sprite"]
        w = max(1, int(sprite.size[0] * scale * level_scale))
        h = max(1, int(sprite.size[1] * scale * level_scale))
        sprite_img = sprite.resize((w, h), RESAMPLE)
        center_x = self.width / 2.0
        if att["side"] == "left":
            top_left = (int(center_x), int(y - h / 2))
        elif att["side"] == "right":
            top_left = (int(center_x - w), int(y - h / 2))
        else:
            top_left = (int(center_x - w / 2), int(y - h))
        image.paste(sprite_img, box=top_left, mask=sprite_img)

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        self._advance(time_s)
        draw = ImageDraw.Draw(image)
        self._draw_bar(draw)
        for att in sorted(self.attachments, key=lambda entry: entry["z"], reverse=True):
            self._draw_attachment(image, att)


class FrequencyEQEffect(VisualEffect):
    def __init__(
        self,
        freq_doc: FrequencyDocument,
        bin_indices: list[int],
        duration: float,
        canvas_size: tuple[int, int],
        fps: int,
        options: Dict[str, Any],
    ) -> None:
        super().__init__(0.0, duration)
        self.freq_doc = freq_doc
        self.bin_indices = [max(0, int(idx)) for idx in bin_indices] or [0]
        self.width, self.height = canvas_size
        self.fps = max(1, int(fps))
        self.opacity = max(0.0, min(1.0, float(options.get("opacity", 1.0))))
        self.gap = max(1, int(options.get("gap", 3)))
        self.bar_width = max(2, int(options.get("bar_width", self.gap * 2)))
        self.fade_frames = max(0, int(options.get("fade_frames", 8)))
        self.jump_threshold = max(0.0, float(options.get("jump_threshold_percent", options.get("jump_percent", 0.0))))
        self.jump_window_frames = max(0, int(round(float(options.get("jump_window_seconds", options.get("jump_time", 0.0))) * self.fps)))
        self.jump_sustain_frames = max(0, int(options.get("jump_sustain_frames", options.get("jump_sustain", 0))))
        self.eq_height = max(0.0, min(1.0, float(options.get("eq_height", 1.0))))
        self.color = self._parse_color(options.get("color", "0,255,0"))
        self.jump_color = self._parse_color(options.get("jump_color", "255,80,80"))
        history_size = max(2, self.jump_window_frames + 2)
        self.bar_states: dict[int, dict[str, Any]] = {
            idx: {
                "last_active_frame": None,
                "last_height": 0,
                "history": deque(maxlen=history_size),
                "jump_until": -1,
            }
            for idx in self.bin_indices
        }

    def _parse_color(self, value: Any) -> tuple[int, int, int, int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r, g, b = value[:3]
            else:
                parts = str(value).split(",")
                r, g, b = (int(float(parts[i])) for i in range(3))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
        except Exception:
            r, g, b = 0, 255, 0
        alpha = int(255 * self.opacity)
        return (r, g, b, alpha)

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        frame_idx = int(round(time_s * self.fps))
        draw = ImageDraw.Draw(image, "RGBA")
        count = max(1, len(self.bin_indices))
        segment_width = self.width / count
        for position, bin_index in enumerate(self.bin_indices):
            level = self.freq_doc.level_at(time_s, bin_index)
            intensity = max(0.0, min(1.0, level / 100.0))
            max_bar_height = self.height * self.eq_height
            bar_height = int(max_bar_height * intensity)
            state = self.bar_states.setdefault(
                bin_index,
                {
                    "last_active_frame": None,
                    "last_height": 0,
                    "history": deque(maxlen=max(2, self.jump_window_frames + 2)),
                    "jump_until": -1,
                },
            )
            history: deque[tuple[int, float]] = state["history"]
            history.append((frame_idx, level))
            if self.jump_window_frames:
                while history and frame_idx - history[0][0] > self.jump_window_frames:
                    history.popleft()
                if (
                    self.jump_threshold > 0
                    and history
                    and level > history[0][1] * (1.0 + self.jump_threshold / 100.0)
                ):
                    state["jump_until"] = frame_idx + self.jump_sustain_frames

            active = bar_height > 0
            alpha: Optional[int] = None
            display_height = bar_height
            if active:
                state["last_active_frame"] = frame_idx
                state["last_height"] = bar_height
                alpha = self.color[3]
            else:
                last_frame = state.get("last_active_frame")
                display_height = state.get("last_height", 0)
                if last_frame is not None and self.fade_frames > 0 and display_height > 0:
                    elapsed = frame_idx - last_frame
                    if elapsed == 1:
                        alpha = int(self.color[3] * 0.5)
                    elif 1 < elapsed <= self.fade_frames:
                        remaining = max(self.fade_frames - 1, 1)
                        decay = max(0.0, 0.5 * (1 - (elapsed - 1) / remaining))
                        alpha = int(self.color[3] * decay)
            if alpha is None or display_height <= 0:
                continue

            color = self.jump_color if frame_idx <= state.get("jump_until", -1) else self.color
            color = color[:3] + (alpha,)
            bar_width = max(2, int(min(self.bar_width, segment_width - self.gap)))
            center = (position + 0.5) * segment_width
            x0 = int(round(center - bar_width / 2))
            x1 = min(self.width, x0 + bar_width)
            y0 = max(0, self.height - display_height)
            draw.rectangle((x0, y0, x1, self.height), fill=color)


@dataclass
class AssociationConfig:
    track_name: str
    mode: str
    options: Dict[str, Any]
    source: str = "timing"
    bin_index: Optional[int] = None
    bin_indices: Optional[list[int]] = None


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
    codec: str
    background: tuple[int, int, int, int]
    use_alpha: bool
    associations: List[AssociationPlan]
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    @property
    def base_total_frames(self) -> int:
        return int(math.ceil(self.duration * self.fps))

    @property
    def start_frame_index(self) -> int:
        base = max(0, self.base_total_frames - 1)
        if self.start_frame is None:
            return 0
        return min(max(0, int(self.start_frame) - 1), base)

    @property
    def end_frame_index(self) -> int:
        base = max(0, self.base_total_frames - 1)
        if self.end_frame is None:
            return max(self.start_frame_index, base)
        return max(self.start_frame_index, min(base, int(self.end_frame) - 1))

    @property
    def total_frames(self) -> int:
        return max(0, self.end_frame_index - self.start_frame_index + 1)

    def frame_time(self, frame_idx: int) -> float:
        return (self.start_frame_index + frame_idx) / self.fps

    def generate_frame(self, frame_idx: int) -> np.ndarray:  # pragma: no cover - visual output
        time_s = self.frame_time(frame_idx)
        mode = "RGBA" if self.use_alpha else "RGB"
        background = self.background if self.use_alpha else self.background[:3]
        image = Image.new(mode, self.size, background)
        for assoc in self.associations:
            assoc.render(image, time_s)
        if not self.use_alpha and image.mode == "RGBA":
            image = image.convert("RGB")
        return np.array(image, dtype=np.uint8)


class FrequencyWavesEffect(VisualEffect):
    def __init__(
        self,
        freq_doc: FrequencyDocument,
        bin_indices: list[int],
        duration: float,
        canvas_size: tuple[int, int],
        fps: int,
        options: Dict[str, Any],
    ) -> None:
        super().__init__(0.0, duration)
        self.freq_doc = freq_doc
        self.bin_indices = [max(0, int(idx)) for idx in bin_indices] or [0]
        self.width, self.height = canvas_size
        self.fps = max(1, int(fps))
        self.opacity = max(0.0, min(1.0, float(options.get("opacity", 1.0))))
        self.future_steps = max(1, int(options.get("future_steps", 20)))
        self.horizon_percent = max(0.05, min(0.95, float(options.get("horizon_percent", 0.5))))
        self.bottom_offset = max(0.0, min(0.9, float(options.get("bottom_offset", 0.0))))
        self.crash_method = str(options.get("crash_method", "vanish")).lower()
        self.base_color = self._parse_color(options.get("color", "80,180,255"))
        self.highlight_color = self._parse_color(options.get("highlight_color", "200,230,255"))
        self.reverse_direction = bool(options.get("reverse_direction", False))
        if self.freq_doc.capture_rate:
            self.time_step = 1.0 / float(self.freq_doc.capture_rate)
        elif self.freq_doc.frames:
            self.time_step = max(1.0 / self.fps, self.freq_doc.duration / max(len(self.freq_doc.frames), 1))
        else:
            self.time_step = 1.0 / self.fps
        self.wave_state: dict[int, dict[str, Any]] = {
            idx: {"crash_particles": []} for idx in self.bin_indices
        }

    def _parse_color(self, value: Any) -> tuple[int, int, int, int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r, g, b = value[:3]
            else:
                parts = str(value).split(",")
                r, g, b = (int(float(parts[i])) for i in range(3))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
        except Exception:
            r, g, b = 80, 180, 255
        alpha = int(255 * self.opacity)
        return (r, g, b, alpha)

    def _sample_level(self, time_s: float, bin_index: int, offset_idx: int) -> float:
        direction = -1.0 if self.reverse_direction else 1.0
        timestamp = time_s + direction * offset_idx * self.time_step
        return self.freq_doc.level_at(timestamp, bin_index)

    def _draw_wave_band(
        self,
        draw: ImageDraw.ImageDraw,
        band: tuple[float, float, float, float],
        color: tuple[int, int, int, int],
        highlight: tuple[int, int, int, int],
    ) -> None:
        x0, x1, crest_y, base_y = band
        thickness = max(2.0, (base_y - crest_y) * 0.35)
        body_color = color[:3] + (color[3],)
        draw.rectangle((x0, crest_y, x1, base_y), fill=body_color)
        draw.line((x0, crest_y, x1, crest_y), fill=highlight, width=int(max(1, thickness)))

    def _spawn_break_particles(self, bin_index: int, x_range: tuple[float, float], y_pos: float, intensity: float) -> None:
        particles = self.wave_state.setdefault(bin_index, {}).setdefault("crash_particles", [])
        count = max(1, int(3 + intensity * 6))
        for _ in range(count):
            particles.append(
                {
                    "x": random.uniform(*x_range),
                    "y": y_pos,
                    "vy": random.uniform(6.0, 14.0) * (0.5 + intensity),
                    "radius": random.uniform(2.0, 4.0),
                    "alpha": int(self.base_color[3] * 0.5),
                }
            )

    def _draw_particles(self, draw: ImageDraw.ImageDraw, bin_index: int) -> None:
        particles = self.wave_state.setdefault(bin_index, {}).setdefault("crash_particles", [])
        remaining = []
        for particle in particles:
            particle["y"] += particle["vy"]
            particle["vy"] *= 0.9
            particle["alpha"] = max(0, int(particle["alpha"] * 0.8))
            if particle["y"] >= self.height or particle["alpha"] <= 2:
                continue
            radius = particle["radius"]
            alpha = particle["alpha"]
            x0 = particle["x"] - radius
            y0 = particle["y"] - radius
            x1 = particle["x"] + radius
            y1 = particle["y"] + radius
            color = self.base_color[:3] + (alpha,)
            draw.ellipse((x0, y0, x1, y1), fill=color)
            remaining.append(particle)
        self.wave_state[bin_index]["crash_particles"] = remaining

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        draw = ImageDraw.Draw(image, "RGBA")
        count = max(1, len(self.bin_indices))
        segment_width = self.width / count
        horizon_y = self.height * (1.0 - self.horizon_percent)
        beach_y = self.height * (1.0 - self.bottom_offset)
        beach_y = min(self.height, max(horizon_y + 1, beach_y))
        depth_range = max(1.0, beach_y - horizon_y)
        for position, bin_index in enumerate(self.bin_indices):
            offset_iterable = range(self.future_steps + 1)
            if not self.reverse_direction:
                offset_iterable = reversed(range(self.future_steps + 1))
            x_start = position * segment_width
            x_end = x_start + segment_width
            for offset in offset_iterable:
                depth_ratio = offset / float(self.future_steps)
                if self.reverse_direction:
                    y_pos = horizon_y + depth_range * depth_ratio
                else:
                    y_pos = horizon_y + depth_range * (1 - depth_ratio)
                level = self._sample_level(time_s, bin_index, offset)
                intensity = max(0.0, min(1.0, level / 100.0))
                crest_height = self.height * 0.28 * (0.4 + 0.6 * (1 - depth_ratio)) * intensity
                crest_y = y_pos - crest_height
                band = (x_start + 2, x_end - 2, crest_y, y_pos)
                color = self.base_color
                highlight = self.highlight_color
                if intensity > 0.85:
                    boost = min(1.0, (intensity - 0.85) / 0.15)
                    color = tuple(min(255, int(c + (255 - c) * 0.35 * boost)) for c in color[:3]) + (color[3],)
                    highlight = tuple(min(255, int(h + (255 - h) * 0.5 * boost)) for h in highlight[:3]) + (highlight[3],)
                self._draw_wave_band(draw, band, color, highlight)
                if offset == 0 and intensity > 0.02:
                    if self.crash_method == "break":
                        self._spawn_break_particles(bin_index, (x_start, x_end), y_pos - crest_height * 0.4, intensity)
                    elif self.crash_method == "fall":
                        fall_height = crest_height * 0.8
                        shade = color[:3] + (int(color[3] * 0.5),)
                        draw.rectangle((x_start + 4, y_pos, x_end - 4, min(self.height, y_pos + fall_height)), fill=shade)
            if self.crash_method == "break":
                self._draw_particles(draw, bin_index)


class FrequencyRainEffect(VisualEffect):
    def __init__(
        self,
        freq_doc: FrequencyDocument,
        bin_indices: list[int],
        duration: float,
        canvas_size: tuple[int, int],
        fps: int,
        options: Dict[str, Any],
    ) -> None:
        super().__init__(0.0, duration)
        self.freq_doc = freq_doc
        self.bin_indices = [max(0, int(idx)) for idx in bin_indices] or [0]
        self.width, self.height = canvas_size
        self.fps = max(1, int(fps))
        self.opacity = max(0.0, min(1.0, float(options.get("opacity", 0.85))))
        self.base_density = max(0.0, float(options.get("base_density", 2.0)))
        self.density_multiplier = max(0.0, float(options.get("density_multiplier", 2.0)))
        self.min_transit_frames = max(1, int(options.get("rain_transit_min", 12)))
        self.max_transit_frames = max(self.min_transit_frames, int(options.get("rain_transit_max", 36)))
        self.drop_size = max(0.5, float(options.get("drop_size", 3.0)))
        self.drop_variance = max(1.0, float(options.get("drop_variance", 1.6)))
        self.color_variance = max(0.0, float(options.get("color_variance", 0.0)))
        self.base_color = self._parse_color(options.get("color", "140,200,255"))
        self.rng = random.Random(42)
        self.bin_colors: dict[int, tuple[int, int, int]] = {}
        seed_color = self._vary_color(self.base_color, max(self.color_variance, 8.0))
        for position, idx in enumerate(self.bin_indices):
            factor = max(0.35, 1.0 - position * 0.12)
            tinted = tuple(max(0, min(255, int(channel * factor))) for channel in seed_color)
            self.bin_colors[idx] = tinted
        self.drops: list[dict[str, Any]] = []

    def _parse_color(self, value: Any) -> tuple[int, int, int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                r, g, b = value[:3]
            else:
                parts = str(value).split(",")
                r, g, b = (int(float(parts[i])) for i in range(3))
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
        except Exception:
            r, g, b = 140, 200, 255
        return r, g, b

    def _vary_color(self, base: tuple[int, int, int], variance: float) -> tuple[int, int, int]:
        return tuple(
            max(0, min(255, int(channel + self.rng.uniform(-variance, variance)))) for channel in base
        )

    def _transit_frames(self, intensity: float) -> int:
        if intensity <= 0.0:
            return self.max_transit_frames
        clamped = max(0.01, min(1.0, intensity))
        span = max(0, self.max_transit_frames - self.min_transit_frames)
        if span == 0:
            return self.min_transit_frames
        return int(round(self.max_transit_frames - span * ((clamped - 0.01) / 0.99)))

    def _spawn_for_bin(self, time_s: float, bin_index: int, base_color: tuple[int, int, int]) -> None:
        level = self.freq_doc.level_at(time_s, bin_index)
        intensity = max(0.0, min(1.0, level / 100.0))
        if intensity <= 0.0:
            return
        transit_frames = self._transit_frames(intensity)
        speed = self.height / float(transit_frames)
        count = self.base_density * (1.0 + (self.density_multiplier - 1.0) * intensity)
        drop_count = int(count)
        if self.rng.random() < (count - drop_count):
            drop_count += 1
        stretch_ratio = 0.0
        if self.max_transit_frames != self.min_transit_frames:
            stretch_ratio = (self.max_transit_frames - transit_frames) / (self.max_transit_frames - self.min_transit_frames)
        for _ in range(drop_count):
            size = self.drop_size * self.rng.uniform(1.0, self.drop_variance)
            length = size * (1.0 + stretch_ratio * 5.0)
            alpha = int(255 * self.opacity * (0.5 + 0.5 * intensity))
            color = self._vary_color(base_color, self.color_variance) + (alpha,)
            self.drops.append(
                {
                    "x": self.rng.uniform(0, self.width),
                    "y": self.rng.uniform(-length, 0.0),
                    "vy": speed,
                    "size": size,
                    "length": length,
                    "color": color,
                }
            )

    def draw(self, image: Image.Image, time_s: float) -> None:  # pragma: no cover - visual output
        for bin_index in self.bin_indices:
            base_color = self.bin_colors.get(bin_index, self.base_color)
            self._spawn_for_bin(time_s, bin_index, base_color)

        draw = ImageDraw.Draw(image, "RGBA")
        remaining: list[dict[str, Any]] = []
        for drop in self.drops:
            drop["y"] += drop["vy"]
            if drop["y"] - drop["length"] > self.height:
                continue
            x = drop["x"]
            y = drop["y"]
            size = drop["size"]
            length = drop["length"]
            color = drop["color"]
            if length <= size * 1.2:
                radius = size / 2.0
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            else:
                width = max(1, int(size * 0.7))
                draw.line((x, y - length, x, y), fill=color, width=width)
                radius = size / 2.0
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            remaining.append(drop)
        self.drops = remaining


# ---------------------------------------------------------------------------
# GUI components
# ---------------------------------------------------------------------------


class AssociationDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent,
        tracks: Iterable[str],
        config: Optional[AssociationConfig] = None,
        selected_track: Optional[str] = None,
        default_sprite_root: Optional[Path] = None,
    ):
        self.track_names = list(tracks)
        self.config = config
        self.selected_track = selected_track
        self.default_sprite_root = default_sprite_root
        self.result_config: Optional[AssociationConfig] = None
        super().__init__(parent, title="Configure timed action")

    def body(self, master):
        ttk.Label(master, text="Timing track:").grid(row=0, column=0, sticky="w")
        default_track = ""
        if self.config:
            default_track = self.config.track_name
        elif self.selected_track and self.selected_track in self.track_names:
            default_track = self.selected_track
        elif self.track_names:
            default_track = self.track_names[0]
        self.track_var = tk.StringVar(value=default_track)
        self.track_combo = ttk.Combobox(master, values=self.track_names, textvariable=self.track_var, state="readonly")
        self.track_combo.grid(row=0, column=1, sticky="ew")
        master.columnconfigure(1, weight=1)

        ttk.Label(master, text="Mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.mode_var = tk.StringVar(value=(self.config.mode if self.config else "fireworks"))
        self.mode_combo = ttk.Combobox(
            master,
            values=["fireworks", "sprite_pop", "pop", "splash", "plant", "wall", "zigzag", "conveyor_bar"],
            textvariable=self.mode_var,
            state="readonly",
        )
        self.mode_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _evt: self._show_mode_options())

        self.options_frame = ttk.Frame(master)
        self.options_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self._build_option_inputs()
        self._show_mode_options()
        return self.track_combo

    def _build_option_inputs(self) -> None:
        firework_defaults = {
            "pre_launch_frames": 12,
            "fade": 0.6,
            "scale": 1.0,
            "variance": 0.2,
            "color": "random",
            "firework_type": "random",
        }
        self.firework_vars: dict[str, tk.StringVar] = {}
        for key, default in firework_defaults.items():
            value = default
            if self.config and self.config.mode == "fireworks":
                value = self.config.options.get(key, default)
            self.firework_vars[key] = tk.StringVar(value=str(value))
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
        pop_defaults = {
            "start_percent_from_bottom": 0.25,
            "size": 36.0,
            "size_variance": 0.25,
            "use_value_for_size": False,
            "opacity": 1.0,
        }
        self.pop_vars: dict[str, tk.StringVar] = {}
        for key, default in pop_defaults.items():
            value = default
            if self.config and self.config.mode == "pop":
                value = self.config.options.get(key, default)
            self.pop_vars[key] = tk.StringVar(value=str(value))
        splash_defaults = {
            "start_size": 28.0,
            "impact_size": 10.0,
            "size_variance": 0.25,
            "speed_variance": 0.2,
            "motion_frames": 10,
            "color": "150,200,255",
            "burst_count": 14,
            "burst_count_variance": 0.25,
            "burst_distance": 140.0,
            "burst_distance_variance": 0.35,
        }
        self.splash_vars: dict[str, tk.StringVar] = {}
        for key, default in splash_defaults.items():
            value = default
            if self.config and self.config.mode == "splash":
                value = self.config.options.get(key, default)
            self.splash_vars[key] = tk.StringVar(value=str(value))
        plant_defaults = {
            "seed_lead_frames": 10,
            "grow_frames": 18,
            "plant_height": 120.0,
            "height_variance": 0.2,
            "grow_variance": 0.1,
            "top_margin": 0.05,
            "bottom_margin": 0.05,
            "left_margin": 0.1,
            "right_margin": 0.1,
            "plant_type": "random",
        }
        self.plant_vars: dict[str, tk.StringVar] = {}
        for key, default in plant_defaults.items():
            value = default
            if self.config and self.config.mode == "plant":
                value = self.config.options.get(key, default)
            self.plant_vars[key] = tk.StringVar(value=str(value))
        wall_defaults = {
            "brick_color": "186,84,60",
            "edge_color": "80,40,28",
            "edge_thickness": 2,
        }
        self.wall_vars: dict[str, tk.StringVar] = {}
        for key, default in wall_defaults.items():
            value = default
            if self.config and self.config.mode == "wall":
                value = self.config.options.get(key, default)
            self.wall_vars[key] = tk.StringVar(value=str(value))
        zigzag_defaults = {
            "line_width": 2,
            "line_width_variance": 0,
            "bar_width": 25,
            "bar_width_variance": 0,
            "color": "100,200,100",
            "color_variance": 10,
            "amount": 1,
            "alignment": "both",
            "fade": 12,
        }
        self.zigzag_vars: dict[str, tk.StringVar] = {}
        for key, default in zigzag_defaults.items():
            value = default
            if self.config and self.config.mode == "zigzag":
                value = self.config.options.get(key, default)
            self.zigzag_vars[key] = tk.StringVar(value=str(value))
        sprite_root = self.default_sprite_root
        conveyor_defaults = {
            "small_dir": str((sprite_root / "small") if sprite_root else ""),
            "mid_dir": str((sprite_root / "mid") if sprite_root else ""),
            "large_dir": str((sprite_root / "large") if sprite_root else ""),
            "scale_small": 1.0,
            "scale_mid": 2.0,
            "scale_large": 4.0,
            "threshold_small": 1,
            "threshold_mid": 3,
            "threshold_large": 6,
            "bar_speed": 180.0,
            "bar_length": 1200.0,
            "horizon_percent": 0.22,
            "origin_percent": 0.82,
            "active_zone_length": 420.0,
            "spawn_depth": 0.0,
            "show_bar": True,
            "bar_width_near": 80.0,
            "bar_width_far": 12.0,
            "bar_color": "200,200,210",
        }
        self.conveyor_vars: dict[str, tk.Variable] = {}
        for key, default in conveyor_defaults.items():
            value = default
            if self.config and self.config.mode == "conveyor_bar":
                value = self.config.options.get(key, default)
            if key == "show_bar":
                self.conveyor_vars[key] = tk.BooleanVar(value=bool(value))
            else:
                self.conveyor_vars[key] = tk.StringVar(value=str(value))

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
            ttk.Label(self.options_frame, text="Scale factor:").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.firework_vars["scale"], width=8).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Size variance (0-1):").grid(row=4, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.firework_vars["variance"], width=8).grid(row=4, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Color:").grid(row=5, column=0, sticky="w")
            ttk.Combobox(
                self.options_frame,
                values=["random", "white", "red", "green", "blue"],
                textvariable=self.firework_vars["color"],
                state="readonly",
                width=10,
            ).grid(row=5, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Firework type:").grid(row=6, column=0, sticky="w")
            ttk.Combobox(
                self.options_frame,
                values=["random", "ring", "star", "burst", "spray"],
                textvariable=self.firework_vars["firework_type"],
                state="readonly",
                width=10,
            ).grid(row=6, column=1, sticky="w")
        elif mode == "sprite_pop":
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
        elif mode == "pop":
            ttk.Label(self.options_frame, text="Pop options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Start offset from bottom (0-1):").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.pop_vars["start_percent_from_bottom"], width=10).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Base size (px):").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.pop_vars["size"], width=10).grid(row=2, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Size randomness (0-1):").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.pop_vars["size_variance"], width=10).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Use event value for size:").grid(row=4, column=0, sticky="w")
            ttk.Checkbutton(self.options_frame, variable=self.pop_vars["use_value_for_size"], onvalue="True", offvalue="False").grid(row=4, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Opacity (0-1):").grid(row=5, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.pop_vars["opacity"], width=10).grid(row=5, column=1, sticky="w")
        elif mode == "splash":
            ttk.Label(self.options_frame, text="Splash options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Start size (px):").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["start_size"], width=10).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Impact size (px):").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["impact_size"], width=10).grid(row=2, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Size randomness (0-1):").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["size_variance"], width=10).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Speed randomness (0-1):").grid(row=4, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["speed_variance"], width=10).grid(row=4, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Frames before impact:").grid(row=5, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["motion_frames"], width=10).grid(row=5, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Drop color (R,G,B):").grid(row=6, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["color"], width=12).grid(row=6, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Burst drops (approx):").grid(row=7, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["burst_count"], width=12).grid(row=7, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Burst count randomness (0-1):").grid(row=8, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["burst_count_variance"], width=12).grid(row=8, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Max travel distance (px):").grid(row=9, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["burst_distance"], width=12).grid(row=9, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Distance randomness (0-1):").grid(row=10, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.splash_vars["burst_distance_variance"], width=12).grid(row=10, column=1, sticky="w")
        elif mode == "plant":
            ttk.Label(self.options_frame, text="Plant options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Seed lead time (frames):").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.plant_vars["seed_lead_frames"], width=12).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Grow duration (frames):").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.plant_vars["grow_frames"], width=12).grid(row=2, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Plant height (px):").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.plant_vars["plant_height"], width=12).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Height randomness (0-1):").grid(row=4, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.plant_vars["height_variance"], width=12).grid(row=4, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Growth randomness (0-1):").grid(row=5, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.plant_vars["grow_variance"], width=12).grid(row=5, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Top/Bottom margin (0-0.45):").grid(row=6, column=0, sticky="w")
            margin_row = ttk.Frame(self.options_frame)
            margin_row.grid(row=6, column=1, sticky="w")
            ttk.Entry(margin_row, textvariable=self.plant_vars["top_margin"], width=6).pack(side="left")
            ttk.Entry(margin_row, textvariable=self.plant_vars["bottom_margin"], width=6).pack(side="left", padx=(6, 0))
            ttk.Label(self.options_frame, text="Left/Right margin (0-0.45):").grid(row=7, column=0, sticky="w")
            margin_row_lr = ttk.Frame(self.options_frame)
            margin_row_lr.grid(row=7, column=1, sticky="w")
            ttk.Entry(margin_row_lr, textvariable=self.plant_vars["left_margin"], width=6).pack(side="left")
            ttk.Entry(margin_row_lr, textvariable=self.plant_vars["right_margin"], width=6).pack(side="left", padx=(6, 0))
            ttk.Label(self.options_frame, text="Plant type:").grid(row=8, column=0, sticky="w")
            ttk.Combobox(
                self.options_frame,
                values=["random", "grass", "mushroom"],
                textvariable=self.plant_vars["plant_type"],
                state="readonly",
                width=12,
            ).grid(row=8, column=1, sticky="w")
        elif mode == "wall":
            ttk.Label(self.options_frame, text="Wall options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Brick color (R,G,B):").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.wall_vars["brick_color"], width=14).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Edge color (R,G,B):").grid(row=2, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.wall_vars["edge_color"], width=14).grid(row=2, column=1, sticky="w", pady=(6, 0))
            ttk.Label(self.options_frame, text="Edge thickness (px):").grid(row=3, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.wall_vars["edge_thickness"], width=10).grid(row=3, column=1, sticky="w", pady=(6, 0))
        elif mode == "conveyor_bar":
            ttk.Label(self.options_frame, text="Conveyor build options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=3)
            ttk.Label(self.options_frame, text="Small sprites:").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["small_dir"], width=32).grid(row=1, column=1, sticky="ew")
            ttk.Button(self.options_frame, text="Browse", command=lambda: self._browse_directory(self.conveyor_vars["small_dir"])).grid(row=1, column=2, padx=(6, 0))
            ttk.Label(self.options_frame, text="Mid sprites:").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["mid_dir"], width=32).grid(row=2, column=1, sticky="ew")
            ttk.Button(self.options_frame, text="Browse", command=lambda: self._browse_directory(self.conveyor_vars["mid_dir"])).grid(row=2, column=2, padx=(6, 0))
            ttk.Label(self.options_frame, text="Large sprites:").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["large_dir"], width=32).grid(row=3, column=1, sticky="ew")
            ttk.Button(self.options_frame, text="Browse", command=lambda: self._browse_directory(self.conveyor_vars["large_dir"])).grid(row=3, column=2, padx=(6, 0))
            ttk.Label(self.options_frame, text="Scale (small/mid/large):").grid(row=4, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["scale_small"], width=8).grid(row=4, column=1, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["scale_mid"], width=8).grid(row=4, column=1, sticky="", padx=(70, 0), pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["scale_large"], width=8).grid(row=4, column=1, sticky="e", pady=(6, 0))
            ttk.Label(self.options_frame, text="Upgrade thresholds:").grid(row=5, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["threshold_small"], width=8).grid(row=5, column=1, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["threshold_mid"], width=8).grid(row=5, column=1, sticky="", padx=(70, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["threshold_large"], width=8).grid(row=5, column=1, sticky="e")
            ttk.Label(self.options_frame, text="Bar speed (units/s):").grid(row=6, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["bar_speed"], width=12).grid(row=6, column=1, sticky="w", pady=(6, 0))
            ttk.Label(self.options_frame, text="Bar length (units):").grid(row=7, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["bar_length"], width=12).grid(row=7, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Horizon percent:").grid(row=8, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["horizon_percent"], width=12).grid(row=8, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Origin percent:").grid(row=9, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["origin_percent"], width=12).grid(row=9, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Active zone length:").grid(row=10, column=0, sticky="w", pady=(6, 0))
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["active_zone_length"], width=12).grid(row=10, column=1, sticky="w", pady=(6, 0))
            ttk.Label(self.options_frame, text="Spawn depth:").grid(row=11, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["spawn_depth"], width=12).grid(row=11, column=1, sticky="w")
            ttk.Checkbutton(self.options_frame, text="Show bar", variable=self.conveyor_vars["show_bar"], onvalue=True, offvalue=False).grid(row=12, column=0, sticky="w", pady=(6, 0))
            ttk.Label(self.options_frame, text="Bar width (near/far):").grid(row=13, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["bar_width_near"], width=10).grid(row=13, column=1, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["bar_width_far"], width=10).grid(row=13, column=1, sticky="e")
            ttk.Label(self.options_frame, text="Bar color (R,G,B):").grid(row=14, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.conveyor_vars["bar_color"], width=14).grid(row=14, column=1, sticky="w")
            self.options_frame.columnconfigure(1, weight=1)
        else:
            ttk.Label(self.options_frame, text="Zigzag options:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", columnspan=2)
            ttk.Label(self.options_frame, text="Line width:").grid(row=1, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["line_width"], width=10).grid(row=1, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Line width variance:").grid(row=2, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["line_width_variance"], width=10).grid(row=2, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Bar width:").grid(row=3, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["bar_width"], width=10).grid(row=3, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Bar width variance:").grid(row=4, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["bar_width_variance"], width=10).grid(row=4, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Color (R,G,B):").grid(row=5, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["color"], width=10).grid(row=5, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Color variance:").grid(row=6, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["color_variance"], width=10).grid(row=6, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Amount:").grid(row=7, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["amount"], width=10).grid(row=7, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Alignment:").grid(row=8, column=0, sticky="w")
            ttk.Combobox(
                self.options_frame,
                values=["horizontal", "vertical", "both"],
                textvariable=self.zigzag_vars["alignment"],
                state="readonly",
                width=10,
            ).grid(row=8, column=1, sticky="w")
            ttk.Label(self.options_frame, text="Fade (frames):").grid(row=9, column=0, sticky="w")
            ttk.Entry(self.options_frame, textvariable=self.zigzag_vars["fade"], width=10).grid(row=9, column=1, sticky="w")

    def _browse_sprite(self, entry: ttk.Entry) -> None:
        initial = Path(self.sprite_vars["sprite_path"].get()).expanduser()
        file_path = filedialog.askopenfilename(title="Select sprite", initialdir=str(initial.parent if initial.exists() else Path.cwd()), filetypes=(("Images", "*.png;*.jpg;*.jpeg"), ("All", "*.*")))
        if file_path:
            self.sprite_vars["sprite_path"].set(file_path)

    def _browse_directory(self, var: tk.Variable) -> None:
        initial = Path(str(var.get() or "")).expanduser()
        directory = filedialog.askdirectory(title="Select sprite folder", initialdir=str(initial if initial.exists() else Path.cwd()))
        if directory:
            var.set(directory)

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
                "scale": float(self.firework_vars["scale"].get() or 1.0),
                "variance": float(self.firework_vars["variance"].get() or 0.2),
                "color": self.firework_vars["color"].get() or "random",
                "firework_type": self.firework_vars["firework_type"].get() or "random",
            }
        elif mode == "sprite_pop":
            options = {
                "sprite_path": self.sprite_vars["sprite_path"].get().strip(),
                "scale": float(self.sprite_vars["scale"].get() or 1.0),
                "hang_time": float(self.sprite_vars["hang_time"].get() or 0.35),
                "pre_zoom_frames": int(self.sprite_vars["pre_zoom_frames"].get() or 5),
            }
        elif mode == "pop":
            options = {
                "start_percent_from_bottom": float(self.pop_vars["start_percent_from_bottom"].get() or 0.25),
                "size": float(self.pop_vars["size"].get() or 36.0),
                "size_variance": float(self.pop_vars["size_variance"].get() or 0.25),
                "use_value_for_size": str(self.pop_vars["use_value_for_size"].get()).lower() == "true",
                "opacity": float(self.pop_vars["opacity"].get() or 1.0),
            }
        elif mode == "splash":
            options = {
                "start_size": float(self.splash_vars["start_size"].get() or 28.0),
                "impact_size": float(self.splash_vars["impact_size"].get() or 10.0),
                "size_variance": float(self.splash_vars["size_variance"].get() or 0.25),
                "speed_variance": float(self.splash_vars["speed_variance"].get() or 0.2),
                "motion_frames": int(self.splash_vars["motion_frames"].get() or 10),
                "color": self.splash_vars["color"].get() or "150,200,255",
                "burst_count": float(self.splash_vars["burst_count"].get() or 14),
                "burst_count_variance": float(self.splash_vars["burst_count_variance"].get() or 0.25),
                "burst_distance": float(self.splash_vars["burst_distance"].get() or 140.0),
                "burst_distance_variance": float(self.splash_vars["burst_distance_variance"].get() or 0.35),
            }
        elif mode == "plant":
            options = {
                "seed_lead_frames": int(self.plant_vars["seed_lead_frames"].get() or 10),
                "grow_frames": int(self.plant_vars["grow_frames"].get() or 18),
                "plant_height": float(self.plant_vars["plant_height"].get() or 120.0),
                "height_variance": float(self.plant_vars["height_variance"].get() or 0.2),
                "grow_variance": float(self.plant_vars["grow_variance"].get() or 0.1),
                "top_margin": float(self.plant_vars["top_margin"].get() or 0.05),
                "bottom_margin": float(self.plant_vars["bottom_margin"].get() or 0.05),
                "left_margin": float(self.plant_vars["left_margin"].get() or 0.1),
                "right_margin": float(self.plant_vars["right_margin"].get() or 0.1),
                "plant_type": self.plant_vars["plant_type"].get() or "random",
            }
        elif mode == "wall":
            options = {
                "brick_color": self.wall_vars["brick_color"].get() or "186,84,60",
                "edge_color": self.wall_vars["edge_color"].get() or "80,40,28",
                "edge_thickness": int(self.wall_vars["edge_thickness"].get() or 2),
            }
        elif mode == "conveyor_bar":
            options = {
                "small_dir": self.conveyor_vars["small_dir"].get(),
                "mid_dir": self.conveyor_vars["mid_dir"].get(),
                "large_dir": self.conveyor_vars["large_dir"].get(),
                "scale_small": float(self.conveyor_vars["scale_small"].get() or 1.0),
                "scale_mid": float(self.conveyor_vars["scale_mid"].get() or 2.0),
                "scale_large": float(self.conveyor_vars["scale_large"].get() or 4.0),
                "threshold_small": int(self.conveyor_vars["threshold_small"].get() or 1),
                "threshold_mid": int(self.conveyor_vars["threshold_mid"].get() or 3),
                "threshold_large": int(self.conveyor_vars["threshold_large"].get() or 6),
                "bar_speed": float(self.conveyor_vars["bar_speed"].get() or 180.0),
                "bar_length": float(self.conveyor_vars["bar_length"].get() or 1200.0),
                "horizon_percent": float(self.conveyor_vars["horizon_percent"].get() or 0.22),
                "origin_percent": float(self.conveyor_vars["origin_percent"].get() or 0.82),
                "active_zone_length": float(self.conveyor_vars["active_zone_length"].get() or 420.0),
                "spawn_depth": float(self.conveyor_vars["spawn_depth"].get() or 0.0),
                "show_bar": bool(self.conveyor_vars["show_bar"].get()),
                "bar_width_near": float(self.conveyor_vars["bar_width_near"].get() or 80.0),
                "bar_width_far": float(self.conveyor_vars["bar_width_far"].get() or 12.0),
                "bar_color": self.conveyor_vars["bar_color"].get() or "200,200,210",
            }
        else:
            options = {
                "line_width": float(self.zigzag_vars["line_width"].get() or 2),
                "line_width_variance": float(self.zigzag_vars["line_width_variance"].get() or 0),
                "bar_width": float(self.zigzag_vars["bar_width"].get() or 25),
                "bar_width_variance": float(self.zigzag_vars["bar_width_variance"].get() or 0),
                "color": self.zigzag_vars["color"].get() or "100,200,100",
                "color_variance": float(self.zigzag_vars["color_variance"].get() or 0),
                "amount": int(self.zigzag_vars["amount"].get() or 1),
                "alignment": self.zigzag_vars["alignment"].get() or "both",
                "fade": int(self.zigzag_vars["fade"].get() or 12),
            }
        self.result_config = AssociationConfig(
            track_name=self.track_var.get(),
            mode=mode,
            options=options,
        )


class FrequencyAssociationDialog(simpledialog.Dialog):
    def __init__(
        self,
        parent,
        freq_doc: FrequencyDocument,
        config: Optional[AssociationConfig] = None,
    ):
        self.freq_doc = freq_doc
        self.config = config
        self.result_config: Optional[AssociationConfig] = None
        super().__init__(parent, title="Configure frequency effect")

    def _bin_labels(self) -> list[str]:
        labels: list[str] = []
        for idx, edge in enumerate(self.freq_doc.bin_edges):
            start, end = edge
            labels.append(f"Bin {idx + 1} ({start:.0f}-{end:.0f} Hz)")
        if not labels:
            labels.append("Bin 1")
        return labels

    def _pick_color(self, target_var: tk.StringVar) -> None:
        rgb, _ = colorchooser.askcolor(title="Choose EQ color")
        if rgb:
            r, g, b = map(int, rgb)
            target_var.set(f"{r},{g},{b}")

    def _on_mode_change(self, _event=None) -> None:
        mode = self.mode_var.get().lower()
        self.eq_frame.grid_remove()
        self.waves_frame.grid_remove()
        self.rain_frame.grid_remove()
        if mode == "waves":
            self.waves_frame.grid()
        elif mode == "rain":
            self.rain_frame.grid()
        else:
            self.eq_frame.grid()

    def body(self, master):
        bin_labels = self._bin_labels()
        defaults = []
        if self.config:
            if self.config.bin_indices:
                defaults = self.config.bin_indices
            elif self.config.bin_index is not None:
                defaults = [self.config.bin_index]
        if not defaults:
            defaults = list(range(len(bin_labels))) or [0]
        ttk.Label(master, text="Frequency bins:").grid(row=0, column=0, sticky="nw")
        bins_frame = ttk.Frame(master)
        bins_frame.grid(row=0, column=1, sticky="w")
        self.bin_vars: list[tk.IntVar] = []
        for idx, label in enumerate(bin_labels):
            var = tk.IntVar(value=1 if idx in defaults else 0)
            self.bin_vars.append(var)
            ttk.Checkbutton(bins_frame, text=label, variable=var).grid(row=idx // 2, column=idx % 2, sticky="w")
        master.columnconfigure(1, weight=1)

        ttk.Label(master, text="Mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        default_mode = (self.config.mode.upper() if self.config else "EQ")
        self.mode_var = tk.StringVar(value=default_mode)
        mode_combo = ttk.Combobox(master, values=["EQ", "WAVES", "RAIN"], textvariable=self.mode_var, state="readonly")
        mode_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        options = self.config.options if self.config else {}
        self.opacity_var = tk.StringVar(value=str(options.get("opacity", 1.0)))
        self.color_var = tk.StringVar(value=str(options.get("color", "0,255,0")))
        self.rain_color_var = tk.StringVar(value=str(options.get("color", "140,200,255")))
        self.rain_color_variance_var = tk.StringVar(value=str(options.get("color_variance", 0)))

        self.eq_frame = ttk.LabelFrame(master, text="EQ options")
        self.eq_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(self.eq_frame, text="Opacity (0-1):").grid(row=0, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(self.eq_frame, textvariable=self.opacity_var, width=12).grid(row=0, column=1, sticky="w", pady=(2, 0))

        ttk.Label(self.eq_frame, text="Gap (px):").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.gap_var = tk.StringVar(value=str(options.get("gap", 3)))
        ttk.Entry(self.eq_frame, textvariable=self.gap_var, width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Bar width (px):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.bar_width_var = tk.StringVar(value=str(options.get("bar_width", int(self.gap_var.get() or 3) * 2)))
        ttk.Entry(self.eq_frame, textvariable=self.bar_width_var, width=12).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="EQ height (0-1):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.eq_height_var = tk.StringVar(value=str(options.get("eq_height", 1.0)))
        ttk.Entry(self.eq_frame, textvariable=self.eq_height_var, width=12).grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Color (R,G,B):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        color_row = ttk.Frame(self.eq_frame)
        color_row.grid(row=4, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(color_row, textvariable=self.color_var, width=14).pack(side="left")
        ttk.Button(color_row, text="Pick", command=lambda: self._pick_color(self.color_var)).pack(side="left", padx=(6, 0))

        ttk.Label(self.eq_frame, text="Fade frames:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.fade_frames_var = tk.StringVar(value=str(options.get("fade_frames", 8)))
        ttk.Entry(self.eq_frame, textvariable=self.fade_frames_var, width=12).grid(row=5, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Jump threshold (%):").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.jump_threshold_var = tk.StringVar(
            value=str(options.get("jump_threshold_percent", options.get("jump_percent", 0.0)))
        )
        ttk.Entry(self.eq_frame, textvariable=self.jump_threshold_var, width=12).grid(row=6, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Jump window (s):").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.jump_window_var = tk.StringVar(value=str(options.get("jump_window_seconds", options.get("jump_time", 0.0))))
        ttk.Entry(self.eq_frame, textvariable=self.jump_window_var, width=12).grid(row=7, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Jump sustain (frames):").grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.jump_sustain_var = tk.StringVar(value=str(options.get("jump_sustain_frames", options.get("jump_sustain", 0))))
        ttk.Entry(self.eq_frame, textvariable=self.jump_sustain_var, width=12).grid(row=8, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.eq_frame, text="Jump color (R,G,B):").grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.jump_color_var = tk.StringVar(value=str(options.get("jump_color", "255,80,80")))
        jump_color_row = ttk.Frame(self.eq_frame)
        jump_color_row.grid(row=9, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(jump_color_row, textvariable=self.jump_color_var, width=14).pack(side="left")
        ttk.Button(jump_color_row, text="Pick", command=lambda: self._pick_color(self.jump_color_var)).pack(side="left", padx=(6, 0))

        self.waves_frame = ttk.LabelFrame(master, text="Waves options")
        self.waves_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(self.waves_frame, text="Opacity (0-1):").grid(row=0, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(self.waves_frame, textvariable=self.opacity_var, width=12).grid(row=0, column=1, sticky="w", pady=(2, 0))

        ttk.Label(self.waves_frame, text="Future steps:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.wave_depth_var = tk.StringVar(value=str(options.get("future_steps", 20)))
        ttk.Entry(self.waves_frame, textvariable=self.wave_depth_var, width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.waves_frame, text="Horizon (% from bottom):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.wave_horizon_var = tk.StringVar(value=str(options.get("horizon_percent", 0.5)))
        ttk.Entry(self.waves_frame, textvariable=self.wave_horizon_var, width=12).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.waves_frame, text="Bottom offset (0-1):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.wave_bottom_offset_var = tk.StringVar(value=str(options.get("bottom_offset", 0.0)))
        ttk.Entry(self.waves_frame, textvariable=self.wave_bottom_offset_var, width=12).grid(row=3, column=1, sticky="w", pady=(6, 0))

        self.reverse_wave_var = tk.BooleanVar(value=bool(options.get("reverse_direction", False)))
        ttk.Checkbutton(
            self.waves_frame,
            text="Reverse direction (toward horizon)",
            variable=self.reverse_wave_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(self.waves_frame, text="Crash method:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.crash_method_var = tk.StringVar(value=str(options.get("crash_method", "vanish")).lower())
        ttk.Combobox(
            self.waves_frame,
            values=["vanish", "fall", "break"],
            textvariable=self.crash_method_var,
            state="readonly",
            width=12,
        ).grid(row=5, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.waves_frame, text="Wave color (R,G,B):").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.wave_color_var = tk.StringVar(value=str(options.get("color", "80,180,255")))
        ttk.Entry(self.waves_frame, textvariable=self.wave_color_var, width=14).grid(row=6, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.waves_frame, text="Highlight color (R,G,B):").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.wave_highlight_var = tk.StringVar(value=str(options.get("highlight_color", "200,230,255")))
        ttk.Entry(self.waves_frame, textvariable=self.wave_highlight_var, width=14).grid(row=7, column=1, sticky="w", pady=(6, 0))

        self.rain_frame = ttk.LabelFrame(master, text="Rain options")
        self.rain_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(self.rain_frame, text="Opacity (0-1):").grid(row=0, column=0, sticky="w", pady=(2, 0))
        ttk.Entry(self.rain_frame, textvariable=self.opacity_var, width=12).grid(row=0, column=1, sticky="w", pady=(2, 0))

        ttk.Label(self.rain_frame, text="Base rain density:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.base_density_var = tk.StringVar(value=str(options.get("base_density", 2.0)))
        ttk.Entry(self.rain_frame, textvariable=self.base_density_var, width=12).grid(row=1, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Density multiplier:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.density_multiplier_var = tk.StringVar(value=str(options.get("density_multiplier", 2.0)))
        ttk.Entry(self.rain_frame, textvariable=self.density_multiplier_var, width=12).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Rain transit min (frames):").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.rain_min_var = tk.StringVar(value=str(options.get("rain_transit_min", 12)))
        ttk.Entry(self.rain_frame, textvariable=self.rain_min_var, width=12).grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Rain transit max (frames):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.rain_max_var = tk.StringVar(value=str(options.get("rain_transit_max", 36)))
        ttk.Entry(self.rain_frame, textvariable=self.rain_max_var, width=12).grid(row=4, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Drop size (px):").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.drop_size_var = tk.StringVar(value=str(options.get("drop_size", 3.0)))
        ttk.Entry(self.rain_frame, textvariable=self.drop_size_var, width=12).grid(row=5, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Drop variance (x):").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.drop_variance_var = tk.StringVar(value=str(options.get("drop_variance", 1.6)))
        ttk.Entry(self.rain_frame, textvariable=self.drop_variance_var, width=12).grid(row=6, column=1, sticky="w", pady=(6, 0))

        ttk.Label(self.rain_frame, text="Color (R,G,B):").grid(row=7, column=0, sticky="w", pady=(6, 0))
        rain_color_row = ttk.Frame(self.rain_frame)
        rain_color_row.grid(row=7, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(rain_color_row, textvariable=self.rain_color_var, width=14).pack(side="left")
        ttk.Button(rain_color_row, text="Pick", command=lambda: self._pick_color(self.rain_color_var)).pack(side="left", padx=(6, 0))

        ttk.Label(self.rain_frame, text="Color variance:").grid(row=8, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.rain_frame, textvariable=self.rain_color_variance_var, width=12).grid(row=8, column=1, sticky="w", pady=(6, 0))

        self._on_mode_change()
        return bins_frame

    def validate(self) -> bool:
        if not self.freq_doc or not self.freq_doc.bin_edges:
            messagebox.showerror("Frequency data", "No frequency bins available.")
            return False
        if not any(var.get() for var in getattr(self, "bin_vars", [])):
            messagebox.showerror("Frequency data", "Select at least one frequency bin.")
            return False
        try:
            float(self.opacity_var.get() or 1.0)
            mode = self.mode_var.get().lower()
            if mode == "waves":
                int(self.wave_depth_var.get() or 1)
                float(self.wave_horizon_var.get() or 0.5)
                float(self.wave_bottom_offset_var.get() or 0.0)
            elif mode == "rain":
                float(self.base_density_var.get() or 0.0)
                float(self.density_multiplier_var.get() or 1.0)
                int(self.rain_min_var.get() or 1)
                int(self.rain_max_var.get() or 1)
                float(self.drop_size_var.get() or 1.0)
                float(self.drop_variance_var.get() or 1.0)
                float(self.rain_color_variance_var.get() or 0.0)
            else:
                int(self.gap_var.get() or 3)
                int(self.bar_width_var.get() or 2)
                float(self.eq_height_var.get() or 1.0)
                int(self.fade_frames_var.get() or 0)
                float(self.jump_threshold_var.get() or 0.0)
                float(self.jump_window_var.get() or 0.0)
                int(self.jump_sustain_var.get() or 0)
        except Exception:
            messagebox.showerror("Frequency data", "Please enter valid numeric values.")
            return False
        return True

    def apply(self) -> None:
        selected_bins = [idx for idx, var in enumerate(self.bin_vars) if var.get()]
        bin_index = selected_bins[0] if len(selected_bins) == 1 else None
        mode = self.mode_var.get().lower()
        if mode == "waves":
            options = {
                "opacity": float(self.opacity_var.get() or 1.0),
                "bin_index": bin_index,
                "bin_indices": selected_bins,
                "future_steps": int(self.wave_depth_var.get() or 20),
                "horizon_percent": float(self.wave_horizon_var.get() or 0.5),
                "bottom_offset": float(self.wave_bottom_offset_var.get() or 0.0),
                "crash_method": self.crash_method_var.get() or "vanish",
                "color": self.wave_color_var.get() or "80,180,255",
                "highlight_color": self.wave_highlight_var.get() or "200,230,255",
                "reverse_direction": bool(self.reverse_wave_var.get()),
            }
        elif mode == "rain":
            options = {
                "opacity": float(self.opacity_var.get() or 1.0),
                "bin_index": bin_index,
                "bin_indices": selected_bins,
                "base_density": float(self.base_density_var.get() or 0.0),
                "density_multiplier": float(self.density_multiplier_var.get() or 1.0),
                "rain_transit_min": int(self.rain_min_var.get() or 1),
                "rain_transit_max": int(self.rain_max_var.get() or 1),
                "drop_size": float(self.drop_size_var.get() or 1.0),
                "drop_variance": float(self.drop_variance_var.get() or 1.0),
                "color": self.rain_color_var.get() or "140,200,255",
                "color_variance": float(self.rain_color_variance_var.get() or 0.0),
            }
        else:
            options = {
                "opacity": float(self.opacity_var.get() or 1.0),
                "gap": int(self.gap_var.get() or 3),
                "color": self.color_var.get() or "0,255,0",
                "bar_width": int(self.bar_width_var.get() or (int(self.gap_var.get() or 3) * 2)),
                "eq_height": float(self.eq_height_var.get() or 1.0),
                "bin_index": bin_index,
                "bin_indices": selected_bins,
                "fade_frames": int(self.fade_frames_var.get() or 8),
                "jump_threshold_percent": float(self.jump_threshold_var.get() or 0.0),
                "jump_window_seconds": float(self.jump_window_var.get() or 0.0),
                "jump_sustain_frames": int(self.jump_sustain_var.get() or 0),
                "jump_color": self.jump_color_var.get() or "255,80,80",
            }
        labels = self._bin_labels()
        label_parts = [labels[idx] for idx in selected_bins if idx < len(labels)]
        label = ", ".join(label_parts) if label_parts else labels[0]
        self.result_config = AssociationConfig(
            track_name=label,
            mode=mode,
            options=options,
            source="frequency",
            bin_index=bin_index,
            bin_indices=selected_bins,
        )


class RenderSettingsDialog(simpledialog.Dialog):
    PRESETS = [
        ("4K UHD (3840x2160, 16:9)", 3840, 2160),
        ("QHD (2560x1440, 16:9)", 2560, 1440),
        ("Full HD (1920x1080, 16:9)", 1920, 1080),
        ("HD (1280x720, 16:9)", 1280, 720),
        ("Square HD (1080x1080, 1:1)", 1080, 1080),
        ("Vertical HD (1080x1920, 9:16)", 1080, 1920),
        ("Custom", None, None),
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
        preset_values = [label for label, _w, _h in self.PRESETS]
        self.preset_var = tk.StringVar(value=self._matching_preset_label())
        preset_combo = ttk.Combobox(master, textvariable=self.preset_var, values=preset_values, state="readonly")
        preset_combo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 4))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(master, text="Width:").grid(row=1, column=0, sticky="w")
        self.width_var = tk.StringVar(value=str(self.settings.width))
        width_entry = ttk.Entry(master, textvariable=self.width_var, width=10)
        width_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(master, text="Height:").grid(row=2, column=0, sticky="w")
        self.height_var = tk.StringVar(value=str(self.settings.height))
        height_entry = ttk.Entry(master, textvariable=self.height_var, width=10)
        height_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(master, text="FPS:").grid(row=3, column=0, sticky="w")
        self.fps_var = tk.StringVar(value=str(self.settings.fps))
        fps_entry = ttk.Entry(master, textvariable=self.fps_var, width=10)
        fps_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(master, text="Codec:").grid(row=4, column=0, sticky="w")
        codec_values = [label for label, _ in self.CODECS]
        self.codec_map = {label: key for label, key in self.CODECS}
        initial_codec_label = next((label for label, key in self.CODECS if key == self.settings.codec), codec_values[0])
        self.codec_var = tk.StringVar(value=initial_codec_label)
        codec_combo = ttk.Combobox(master, textvariable=self.codec_var, values=codec_values, state="readonly")
        codec_combo.grid(row=4, column=1, columnspan=2, sticky="ew")

        self.transparent_var = tk.BooleanVar(value=str(self.settings.background).lower() == "transparent")
        self.background_var = tk.StringVar(value=str(self.settings.background))
        ttk.Checkbutton(master, text="Transparent (if supported)", variable=self.transparent_var, command=self._update_background_state).grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 2))

        ttk.Label(master, text="Background:").grid(row=6, column=0, sticky="w")
        self.background_entry = ttk.Entry(master, textvariable=self.background_var, width=14)
        self.background_entry.grid(row=6, column=1, sticky="w")
        ttk.Button(master, text="Pick", command=self._pick_color).grid(row=6, column=2, sticky="e")

        ttk.Label(master, text="First frame:").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.start_frame_var = tk.StringVar(value=str(self.settings.start_frame or ""))
        ttk.Entry(master, textvariable=self.start_frame_var, width=10).grid(row=7, column=1, sticky="w", pady=(8, 0))

        ttk.Label(master, text="Last frame:").grid(row=8, column=0, sticky="w")
        self.end_frame_var = tk.StringVar(value=str(self.settings.end_frame or ""))
        ttk.Entry(master, textvariable=self.end_frame_var, width=10).grid(row=8, column=1, sticky="w")
        ttk.Button(master, text="Reset to full", command=self._reset_frame_window).grid(row=8, column=2, sticky="e")

        master.columnconfigure(1, weight=1)
        self._update_background_state()
        return width_entry

    def _matching_preset_label(self) -> str:
        for label, width, height in self.PRESETS:
            if width is None or height is None:
                continue
            if width == self.settings.width and height == self.settings.height:
                return label
        return "Custom"

    def _on_preset_selected(self, _event=None):
        label = self.preset_var.get()
        for preset_label, width, height in self.PRESETS:
            if preset_label == label and width and height:
                self.width_var.set(str(width))
                self.height_var.set(str(height))
                break

    def _pick_color(self) -> None:
        color = colorchooser.askcolor(title="Choose background", initialcolor=self.background_var.get())
        if color and color[1]:
            self.background_var.set(color[1])

    def _update_background_state(self) -> None:
        state = "disabled" if self.transparent_var.get() else "normal"
        self.background_entry.configure(state=state)

    def validate(self) -> bool:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            fps = int(self.fps_var.get())
            start_frame = self._parse_frame_input(self.start_frame_var.get())
            end_frame = self._parse_frame_input(self.end_frame_var.get())
        except ValueError:
            messagebox.showerror("Render settings", "Width, height, and FPS must be numbers.", parent=self)
            return False
        if width <= 0 or height <= 0 or fps <= 0:
            messagebox.showerror("Render settings", "Width, height, and FPS must be positive.", parent=self)
            return False
        if start_frame is not None and start_frame <= 0:
            messagebox.showerror("Render settings", "First frame must be at least 1.", parent=self)
            return False
        if end_frame is not None and end_frame <= 0:
            messagebox.showerror("Render settings", "Last frame must be at least 1.", parent=self)
            return False
        if start_frame is not None and end_frame is not None and end_frame < start_frame:
            messagebox.showerror("Render settings", "Last frame must be greater than or equal to the first frame.", parent=self)
            return False
        return True

    def apply(self) -> None:
        label = self.codec_var.get()
        codec = self.codec_map.get(label, DEFAULT_CODEC)
        background = "transparent" if self.transparent_var.get() else (self.background_var.get() or "black")
        self.result_settings = RenderSettings(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            fps=int(self.fps_var.get()),
            codec=codec,
            background=background,
            start_frame=self._parse_frame_input(self.start_frame_var.get()),
            end_frame=self._parse_frame_input(self.end_frame_var.get()),
        )

    def _parse_frame_input(self, value: str) -> Optional[int]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(text)

    def _reset_frame_window(self) -> None:
        self.start_frame_var.set("")
        self.end_frame_var.set("")


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
        self.render_settings_path = self.output_dir / "render_settings.json"
        self.assets_dir = (self.project_root / "assets") if self.project_root else (self.repo_root / "assets")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        if self.project_root:
            self.timing_dir = self.project_root / "internal" / "timing"
        else:
            self.timing_dir = self.repo_root / "assets" / "timing"
        self.timing_dir.mkdir(parents=True, exist_ok=True)
        self.render_settings: RenderSettings = self._load_render_settings()
        self.temp_render_path = self._render_output_path(self.render_settings.codec)

        self.audio_path = tk.StringVar()
        self.timing_path = tk.StringVar()
        self.status_var = tk.StringVar(value="Select audio and timing files to begin.")

        self.timing_doc: Optional[TimingDocument] = None
        self.frequency_doc: Optional[FrequencyDocument] = None
        self.frequency_path: Optional[Path] = None
        self.associations: List[AssociationConfig] = []
        self.last_plan: Optional[RenderPlan] = None
        self.render_thread: Optional[threading.Thread] = None
        self._playback_state: dict[str, Any] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    def configure_render(self) -> None:
        dialog = RenderSettingsDialog(self, self.render_settings)
        if dialog.result_settings:
            self.render_settings = dialog.result_settings
            self.temp_render_path = self._render_output_path(self.render_settings.codec)
            self._save_render_settings()
            self.status_var.set(
                f"Render settings updated: {self.render_settings.width}x{self.render_settings.height} @ {self.render_settings.fps}fps"
            )

    # ------------------------------------------------------------------
    def _default_output_dir(self) -> Path:
        if self.project_root:
            return self.project_root / "internal" / "video" / "timed_action_mixer"
        return self.repo_root / "assets" / "timed_action_mixer"

    # ------------------------------------------------------------------
    def _default_sprite_root(self) -> Path:
        base = self.project_root or self.repo_root
        return base / "internal" / "timed_action_mixer" / "sprites"

    # ------------------------------------------------------------------
    def _render_suffix(self, codec: str) -> str:
        lookup = {
            "libx264": "mp4",
            "libx265": "mp4",
            "libvpx-vp9": "webm",
            "qtrle": "mov",
        }
        return lookup.get(codec.lower(), "mp4")

    # ------------------------------------------------------------------
    def _render_output_path(self, codec: str) -> Path:
        return self.output_dir / f"timed_action_preview.{self._render_suffix(codec)}"

    # ------------------------------------------------------------------
    def _load_render_settings(self) -> RenderSettings:
        if self.render_settings_path.exists():
            try:
                payload = json.loads(self.render_settings_path.read_text(encoding="utf-8"))
                return self._parse_render_settings(payload)
            except Exception:
                pass
        return RenderSettings()

    # ------------------------------------------------------------------
    def _parse_render_settings(self, payload: Dict[str, Any]) -> RenderSettings:
        width = int(payload.get("width", FRAME_SIZE[0]) or FRAME_SIZE[0])
        height = int(payload.get("height", FRAME_SIZE[1]) or FRAME_SIZE[1])
        fps = int(payload.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
        codec = str(payload.get("codec", DEFAULT_CODEC) or DEFAULT_CODEC)
        background = str(payload.get("background", "black") or "black")
        start_frame = self._parse_optional_frame(payload.get("start_frame"))
        end_frame = self._parse_optional_frame(payload.get("end_frame"))
        return RenderSettings(
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            background=background,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def _parse_optional_frame(self, value: Any) -> Optional[int]:
        try:
            frame = int(value)
        except Exception:
            return None
        if frame <= 0:
            return None
        return frame

    # ------------------------------------------------------------------
    def _save_render_settings(self) -> None:
        payload = asdict(self.render_settings)
        try:
            self.render_settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _validated_render_settings(self) -> RenderSettings:
        settings = self.render_settings
        width = max(1, int(settings.width or FRAME_SIZE[0]))
        height = max(1, int(settings.height or FRAME_SIZE[1]))
        fps = max(1, int(settings.fps or DEFAULT_FPS))
        codec = settings.codec or DEFAULT_CODEC
        background = settings.background or "black"
        start_frame = settings.start_frame if settings.start_frame is None else max(1, int(settings.start_frame))
        end_frame = settings.end_frame if settings.end_frame is None else max(1, int(settings.end_frame))
        if start_frame is not None and end_frame is not None and end_frame < start_frame:
            end_frame = start_frame
        self.render_settings = RenderSettings(
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            background=background,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self._save_render_settings()
        return self.render_settings

    # ------------------------------------------------------------------
    def _supports_alpha(self, codec: str) -> bool:
        return codec.lower() in {"qtrle"}

    # ------------------------------------------------------------------
    def _resolve_background(self, settings: RenderSettings) -> tuple[tuple[int, int, int, int], bool]:
        choice = str(settings.background or "black")
        codec = settings.codec or DEFAULT_CODEC
        if choice.lower() == "transparent":
            if self._supports_alpha(codec):
                return (0, 0, 0, 0), True
            self.status_var.set("Selected codec does not support transparency; using opaque background.")
            choice = "black"
        try:
            rgb = ImageColor.getrgb(choice)
        except Exception:
            rgb = ImageColor.getrgb("black")
        return (rgb[0], rgb[1], rgb[2], 255), False

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
        self.tracks_tree = ttk.Treeview(
            container,
            columns=("name", "events", "description"),
            show="headings",
            height=5,
        )
        self.tracks_tree.heading("name", text="Track")
        self.tracks_tree.heading("events", text="Events")
        self.tracks_tree.heading("description", text="Description")
        self.tracks_tree.column("name", width=180)
        self.tracks_tree.column("events", width=80, anchor="center")
        self.tracks_tree.column("description", width=420)
        self.tracks_tree.pack(fill="x", pady=(4, 12))
        self.tracks_tree.bind("<Double-1>", self._on_track_double_click)

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
        ttk.Button(buttons, text="Add Trigger", command=self.add_association).pack(side="left")
        ttk.Button(buttons, text="Add Frequency", command=self.add_frequency_association).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Edit", command=self.edit_association).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Duplicate", command=self.duplicate_association).pack(side="left", padx=(6, 0))
        ttk.Button(buttons, text="Remove", command=self.remove_association).pack(side="left", padx=(6, 0))
        self.assoc_tree.bind("<Double-1>", lambda _evt: self.edit_association())

        action_frame = ttk.Frame(container)
        action_frame.pack(fill="x", pady=(12, 0))
        ttk.Button(action_frame, text="Render settings", command=self.configure_render).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="Render animation", command=self.render_video).pack(side="left")
        ttk.Button(action_frame, text="Play last render", command=self.play_render).pack(side="left", padx=(8, 0))
        ttk.Button(action_frame, text="Save to assets", command=self.save_render).pack(side="left", padx=(8, 0))
        ttk.Button(action_frame, text="Load settings", command=self.load_settings).pack(side="right", padx=(0, 8))
        ttk.Button(action_frame, text="Save settings", command=self.save_settings).pack(side="right")

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
        current_value = variable.get().strip()
        current = Path(current_value).expanduser() if current_value else None
        if current and current.is_file():
            start_dir = current.parent
        elif current and current.exists():
            start_dir = current
        elif initial_dir and initial_dir.exists():
            start_dir = initial_dir
        elif self.assets_dir and self.assets_dir.exists():
            start_dir = self.assets_dir
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
    def _find_matching_frequency(self, timing_path: Path) -> Optional[Path]:
        base = timing_path.stem
        audio_stem = base[:-7] if base.endswith(".timing") else base
        candidates = [
            timing_path.with_name(f"{audio_stem}.frequency.json"),
            timing_path.with_name(f"{base}.frequency.json"),
            timing_path.with_suffix(".frequency.json"),
        ]
        if self.audio_path.get():
            audio_base = Path(self.audio_path.get()).expanduser()
            candidates.insert(0, audio_base.with_suffix(".frequency.json"))
        for candidate in candidates:
            if candidate.exists():
                return candidate
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
            self.tracks_tree.insert(
                "",
                "end",
                iid=track.name,
                values=(track.name, len(track.events), desc),
            )
        self._load_frequency(path)
        self.status_var.set(f"Loaded {len(self.timing_doc.tracks)} tracks from timing file.")

    # ------------------------------------------------------------------
    def _load_frequency(self, timing_path: Path) -> None:
        frequency_path = self._find_matching_frequency(timing_path)
        self.frequency_doc = None
        self.frequency_path = None
        if not frequency_path:
            return
        try:
            self.frequency_doc = FrequencyDocument.from_json(frequency_path)
            self.frequency_path = frequency_path
        except Exception:
            messagebox.showwarning(
                "Frequency data",
                "Could not read the matching frequency file. Regenerate it with the audio map tool.",
            )

    # ------------------------------------------------------------------
    def _on_track_double_click(self, event) -> None:
        track_id = self.tracks_tree.identify_row(event.y)
        if not track_id:
            return
        self.tracks_tree.selection_set(track_id)
        self.add_association(track_name=track_id)

    # ------------------------------------------------------------------
    def add_association(self, track_name: Optional[str] = None) -> None:
        if not self.timing_doc or not self.timing_doc.tracks:
            messagebox.showwarning("Timing tracks", "Load a timing file before adding associations.")
            return
        dialog = AssociationDialog(
            self,
            self.timing_doc.tracks.keys(),
            selected_track=track_name,
            default_sprite_root=self._default_sprite_root(),
        )
        if dialog.result_config:
            self.associations.append(dialog.result_config)
            self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def add_frequency_association(self) -> None:
        if not self.frequency_doc:
            messagebox.showwarning(
                "Frequency data",
                "Load a timing file that has matching frequency data from the basic audio map tool.",
            )
            return
        dialog = FrequencyAssociationDialog(self, self.frequency_doc)
        if dialog.result_config:
            self.associations.append(dialog.result_config)
            self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def _refresh_assoc_tree(self) -> None:
        for item in self.assoc_tree.get_children():
            self.assoc_tree.delete(item)
        for idx, assoc in enumerate(self.associations):
            if assoc.source == "frequency":
                bin_indices = assoc.bin_indices or assoc.options.get("bin_indices")
                if not bin_indices and assoc.bin_index is not None:
                    bin_indices = [assoc.bin_index]
                if bin_indices:
                    labels = [self._frequency_bin_label(idx) for idx in bin_indices]
                    label = ", ".join(labels)
                else:
                    label = assoc.track_name or self._frequency_bin_label(assoc.bin_index or 0)
                opacity = assoc.options.get("opacity", 1.0)
                if assoc.mode == "waves":
                    future_steps = assoc.options.get("future_steps", 20)
                    horizon = assoc.options.get("horizon_percent", 0.5)
                    color = assoc.options.get("color", "80,180,255")
                    crash_method = assoc.options.get("crash_method", "vanish")
                    summary = (
                        f"{label}, opacity {opacity}, steps {future_steps}, horizon {horizon}, color {color}, crash {crash_method}"
                    )
                elif assoc.mode == "rain":
                    base_density = assoc.options.get("base_density", 0.0)
                    multiplier = assoc.options.get("density_multiplier", 1.0)
                    transit_min = assoc.options.get("rain_transit_min", 1)
                    transit_max = assoc.options.get("rain_transit_max", 1)
                    drop_size = assoc.options.get("drop_size", 1.0)
                    drop_variance = assoc.options.get("drop_variance", 1.0)
                    color = assoc.options.get("color", "140,200,255")
                    summary = (
                        f"{label}, opacity {opacity}, base {base_density}, x{multiplier}, speed {transit_min}-{transit_max}f, "
                        f"size {drop_size}±{drop_variance}x, color {color}"
                    )
                else:
                    gap = assoc.options.get("gap", 3)
                    color = assoc.options.get("color", "0,255,0")
                    fade_frames = assoc.options.get("fade_frames", 8)
                    jump_threshold = assoc.options.get("jump_threshold_percent", assoc.options.get("jump_percent", 0.0))
                    jump_window = assoc.options.get("jump_window_seconds", assoc.options.get("jump_time", 0.0))
                    jump_sustain = assoc.options.get("jump_sustain_frames", assoc.options.get("jump_sustain", 0))
                    summary = (
                        f"{label}, opacity {opacity}, gap {gap}, color {color}, fade {fade_frames}f, "
                        f"jump {jump_threshold}%/{jump_window}s x{jump_sustain}f"
                    )
            elif assoc.mode == "fireworks":
                summary = (
                    f"Lead {assoc.options.get('pre_launch_frames', 12)}f, "
                    f"fade {assoc.options.get('fade', 0.6)}s, "
                    f"scale {assoc.options.get('scale', 1.0)}, "
                    f"variance {assoc.options.get('variance', 0.2)}, "
                    f"color {assoc.options.get('color', 'random')}, "
                    f"type {assoc.options.get('firework_type', 'random')}"
                )
            elif assoc.mode == "sprite_pop":
                summary = (
                    f"Sprite {Path(assoc.options.get('sprite_path', '')).name or 'generated'}, "
                    f"scale {assoc.options.get('scale', 1.0)}, hang {assoc.options.get('hang_time', 0.35)}s, "
                    f"pre-zoom {assoc.options.get('pre_zoom_frames', 5)}f"
                )
            elif assoc.mode == "pop":
                summary = (
                    f"Pop start {assoc.options.get('start_percent_from_bottom', 0.25)}, "
                    f"size {assoc.options.get('size', 36.0)}, "
                    f"variance {assoc.options.get('size_variance', 0.25)}, "
                    f"value->size {assoc.options.get('use_value_for_size', False)}, "
                    f"opacity {assoc.options.get('opacity', 1.0)}"
                )
            elif assoc.mode == "splash":
                summary = (
                    f"Start {assoc.options.get('start_size', 28.0)}px -> {assoc.options.get('impact_size', 10.0)}px, "
                    f"frames {assoc.options.get('motion_frames', 10)}, color {assoc.options.get('color', '150,200,255')}, "
                    f"burst {assoc.options.get('burst_count', 14)}±{assoc.options.get('burst_count_variance', 0.25)}, "
                    f"range {assoc.options.get('burst_distance', 140.0)}±{assoc.options.get('burst_distance_variance', 0.35)}"
                )
            elif assoc.mode == "plant":
                summary = (
                    f"Seed lead {assoc.options.get('seed_lead_frames', 10)}f, "
                    f"grow {assoc.options.get('grow_frames', 18)}f, "
                    f"height {assoc.options.get('plant_height', 120.0)}, "
                    f"type {assoc.options.get('plant_type', 'random')}"
                )
            elif assoc.mode == "conveyor_bar":
                speed = assoc.options.get("bar_speed", 180.0)
                length = assoc.options.get("bar_length", 1200.0)
                active = assoc.options.get("active_zone_length", 420.0)
                scales = (
                    assoc.options.get("scale_small", 1.0),
                    assoc.options.get("scale_mid", 2.0),
                    assoc.options.get("scale_large", 4.0),
                )
                thresholds = (
                    assoc.options.get("threshold_small", 1),
                    assoc.options.get("threshold_mid", 3),
                    assoc.options.get("threshold_large", 6),
                )
                show_bar = assoc.options.get("show_bar", True)
                summary = (
                    f"Speed {speed}u/s, length {length}, active {active}, "
                    f"scales {scales[0]}/{scales[1]}/{scales[2]}, "
                    f"thresholds {thresholds[0]}/{thresholds[1]}/{thresholds[2]}, "
                    f"bar {'on' if show_bar else 'off'}"
                )
            else:
                summary = (
                    f"Lines {assoc.options.get('amount', 1)}, "
                    f"align {assoc.options.get('alignment', 'both')}, "
                    f"width {assoc.options.get('line_width', 2)}, "
                    f"bars {assoc.options.get('bar_width', 25)}, "
                    f"fade {assoc.options.get('fade', 12)}f"
                )
            self.assoc_tree.insert("", "end", iid=str(idx), values=(assoc.track_name, assoc.mode, summary))

    # ------------------------------------------------------------------
    def _frequency_bin_label(self, bin_index: int) -> str:
        if not self.frequency_doc or not self.frequency_doc.bin_edges:
            return f"Bin {bin_index + 1}"
        clamped = max(0, min(bin_index, len(self.frequency_doc.bin_edges) - 1))
        start, end = self.frequency_doc.bin_edges[clamped]
        return f"Bin {clamped + 1} ({start:.0f}-{end:.0f} Hz)"

    # ------------------------------------------------------------------
    def edit_association(self) -> None:
        selection = self.assoc_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        assoc = self.associations[idx]
        if assoc.source == "frequency":
            if not self.frequency_doc:
                messagebox.showwarning(
                    "Frequency data",
                    "Load a timing file with matching frequency data to edit this association.",
                )
                return
            dialog = FrequencyAssociationDialog(self, self.frequency_doc, assoc)
        else:
            dialog = AssociationDialog(
                self,
                self.timing_doc.tracks.keys() if self.timing_doc else [],
                assoc,
                default_sprite_root=self._default_sprite_root(),
            )
        if dialog.result_config:
            self.associations[idx] = dialog.result_config
            self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def duplicate_association(self) -> None:
        selection = self.assoc_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        assoc = self.associations[idx]
        duplicate = AssociationConfig(
            track_name=assoc.track_name,
            mode=assoc.mode,
            options=dict(assoc.options),
            source=assoc.source,
            bin_index=assoc.bin_index,
            bin_indices=list(assoc.bin_indices) if assoc.bin_indices else None,
        )
        self.associations.insert(idx + 1, duplicate)
        self._refresh_assoc_tree()
        self.assoc_tree.selection_set(str(idx + 1))

    # ------------------------------------------------------------------
    def remove_association(self) -> None:
        selection = self.assoc_tree.selection()
        if not selection:
            return
        idx = int(selection[0])
        del self.associations[idx]
        self._refresh_assoc_tree()

    # ------------------------------------------------------------------
    def _plant_area(self, options: Dict[str, Any], canvas_size: tuple[int, int]) -> tuple[float, float, float, float]:
        width, height = canvas_size
        def _pct(key: str, default: float) -> float:
            try:
                value = float(options.get(key, default))
            except Exception:
                value = default
            return max(0.0, min(0.45, value))

        top = _pct("top_margin", 0.05)
        bottom = _pct("bottom_margin", 0.05)
        left = _pct("left_margin", 0.1)
        right = _pct("right_margin", 0.1)
        x0 = width * left
        x1 = width * (1.0 - right)
        y0 = height * top
        y1 = height * (1.0 - bottom)
        if x1 - x0 < width * 0.1:
            area_width = width * 0.1
            x0 = (width - area_width) / 2.0
            x1 = x0 + area_width
        if y1 - y0 < height * 0.1:
            area_height = height * 0.1
            y0 = (height - area_height) / 2.0
            y1 = y0 + area_height
        return (x0, y0, x1, y1)

    def _plan_plant_targets(
        self,
        events: list[TimingEvent],
        canvas_size: tuple[int, int],
        options: Dict[str, Any],
        rng: random.Random,
    ) -> tuple[list[PlantTarget], tuple[float, float, float, float]]:
        area = self._plant_area(options, canvas_size)
        area_width = area[2] - area[0]
        area_height = area[3] - area[1]
        cells = max(1, len(events) * 2)
        cols = max(1, int(math.sqrt(cells)))
        rows = math.ceil(cells / cols)
        cell_width = area_width / cols
        cell_height = area_height / rows
        available_cells = [(r, c) for r in range(rows) for c in range(cols)]
        rng.shuffle(available_cells)
        targets: list[PlantTarget] = []
        for event in events:
            if available_cells:
                row, col = available_cells.pop()
            else:
                row = rng.randrange(rows)
                col = rng.randrange(cols)
            center_x = area[0] + (col + 0.5) * cell_width
            center_y = area[1] + (row + 0.5) * cell_height
            base_y = min(area[3], area[1] + (row + 1) * cell_height)
            targets.append(PlantTarget(center=(center_x, center_y), base_y=base_y))
        return targets, area

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
        if self.frequency_doc:
            duration = max(duration, self.frequency_doc.duration)
        if sf:
            try:
                info = sf.info(str(audio_path))
                duration = max(duration, info.frames / float(info.samplerate))
            except Exception:
                pass
        rng = random.Random(1234)
        associations: List[AssociationPlan] = []
        max_end_time = duration
        render_settings = self._validated_render_settings()
        self.temp_render_path = self._render_output_path(render_settings.codec)
        for assoc in self.associations:
            if assoc.source == "frequency":
                if not self.frequency_doc:
                    raise RuntimeError("Load matching frequency data before rendering frequency effects.")
                bin_indices = assoc.bin_indices or assoc.options.get("bin_indices")
                if not bin_indices:
                    bin_index = assoc.bin_index
                    if bin_index is None:
                        bin_index = int(assoc.options.get("bin_index", 0))
                    bin_indices = [bin_index]
                if assoc.mode == "waves":
                    effect: VisualEffect = FrequencyWavesEffect(
                        freq_doc=self.frequency_doc,
                        bin_indices=list(bin_indices),
                        duration=duration,
                        canvas_size=(render_settings.width, render_settings.height),
                        fps=render_settings.fps,
                        options=assoc.options,
                    )
                elif assoc.mode == "rain":
                    effect = FrequencyRainEffect(
                        freq_doc=self.frequency_doc,
                        bin_indices=list(bin_indices),
                        duration=duration,
                        canvas_size=(render_settings.width, render_settings.height),
                        fps=render_settings.fps,
                        options=assoc.options,
                    )
                else:
                    effect = FrequencyEQEffect(
                        freq_doc=self.frequency_doc,
                        bin_indices=list(bin_indices),
                        duration=duration,
                        canvas_size=(render_settings.width, render_settings.height),
                        fps=render_settings.fps,
                        options=assoc.options,
                    )
                associations.append(AssociationPlan(config=assoc, effects=[effect]))
                continue

            track = self.timing_doc.tracks.get(assoc.track_name)
            if not track:
                continue
            effects: List[VisualEffect] = []
            event_iterable: Iterable[TimingEvent] = track.events
            if assoc.mode == "pop":
                event_iterable = reversed(track.events)
            plant_targets: list[PlantTarget] = []
            plant_area: Optional[tuple[float, float, float, float]] = None
            if assoc.mode == "plant":
                plant_targets, plant_area = self._plan_plant_targets(
                    track.events,
                    (render_settings.width, render_settings.height),
                    assoc.options,
                    rng,
                )
            if assoc.mode == "wall":
                effect = WallEffect(
                    events=track.events,
                    duration=duration,
                    canvas_size=(render_settings.width, render_settings.height),
                    options=assoc.options,
                )
                effects.append(effect)
                max_end_time = max(max_end_time, effect.end_time)
                associations.append(AssociationPlan(config=assoc, effects=effects))
                continue
            if assoc.mode == "conveyor_bar":
                effect = ConveyorBuildEffect(
                    events=track.events,
                    duration=duration,
                    canvas_size=(render_settings.width, render_settings.height),
                    options=assoc.options,
                    rng=rng,
                    sprite_root=self._default_sprite_root(),
                )
                effects.append(effect)
                max_end_time = max(max_end_time, effect.end_time)
                associations.append(AssociationPlan(config=assoc, effects=effects))
                continue
            for event in event_iterable:
                if assoc.mode == "fireworks":
                    effects.append(
                        FireworkEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            options=assoc.options,
                        )
                    )
                elif assoc.mode == "sprite_pop":
                    sprite_img = self._load_sprite_image(assoc.options.get("sprite_path"))
                    effects.append(
                        SpritePopEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            sprite=sprite_img,
                            options=assoc.options,
                        )
                    )
                elif assoc.mode == "pop":
                    effects.append(
                        PopEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            options=assoc.options,
                        )
                    )
                elif assoc.mode == "splash":
                    effects.append(
                        SplashEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            options=assoc.options,
                        )
                    )
                elif assoc.mode == "plant":
                    if not plant_targets:
                        plant_targets, plant_area = self._plan_plant_targets(
                            track.events,
                            (render_settings.width, render_settings.height),
                            assoc.options,
                            rng,
                        )
                    target = plant_targets.pop(0)
                    assert plant_area is not None
                    effects.append(
                        PlantEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            options=assoc.options,
                            target=target,
                            duration=duration,
                            area=plant_area,
                        )
                    )
                else:
                    effects.append(
                        ZigZagEffect(
                            event=event,
                            fps=render_settings.fps,
                            canvas_size=(render_settings.width, render_settings.height),
                            rng=rng,
                            options=assoc.options,
                        )
                    )
                if effects:
                    max_end_time = max(max_end_time, effects[-1].end_time)
            associations.append(AssociationPlan(config=assoc, effects=effects))
        duration = max(duration, max_end_time)
        background, use_alpha = self._resolve_background(render_settings)
        return RenderPlan(
            audio_path=audio_path,
            fps=render_settings.fps,
            size=(render_settings.width, render_settings.height),
            duration=duration,
            codec=render_settings.codec,
            background=background,
            use_alpha=use_alpha,
            associations=associations,
            start_frame=render_settings.start_frame,
            end_frame=render_settings.end_frame,
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
        writer_kwargs: Dict[str, Any] = {
            "fps": plan.fps,
            "codec": plan.codec,
            "format": "FFMPEG",
            "macro_block_size": 1,
            "audio_path": str(plan.audio_path),
            "audio_codec": "aac",
        }
        if plan.codec.startswith("libx264"):
            writer_kwargs["quality"] = 8
        if plan.use_alpha:
            writer_kwargs["ffmpeg_params"] = ["-pix_fmt", "rgba"]
        try:
            writer = imageio.get_writer(self.temp_render_path, **writer_kwargs)
        except Exception as exc:
            self._set_status_async(f"Unable to start writer: {exc}")
            return

        if plan.total_frames <= 0:
            writer.close()
            self._set_status_async("Nothing to render for the selected frame range.")
            return

        try:
            for relative_idx in range(plan.total_frames):
                frame_idx = plan.start_frame_index + relative_idx
                frame = plan.generate_frame(relative_idx)
                writer.append_data(frame)
                if relative_idx % 30 == 0:
                    self._set_status_async(
                        f"Rendering frame {frame_idx + 1}/{plan.base_total_frames}"
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
        self._cleanup_playback()
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
        state = self._playback_state
        state.clear()
        state.update({"window": play_window, "timer": None})
        play_window.protocol("WM_DELETE_WINDOW", self._cleanup_playback)
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(str(self.last_plan.audio_path if self.last_plan else self.audio_path.get()))
        except Exception as exc:
            self._cleanup_playback()
            messagebox.showerror("Playback", f"Unable to load audio: {exc}")
            return
        pygame.mixer.music.play()
        try:
            reader = imageio.get_reader(self.temp_render_path, format="FFMPEG")
        except Exception as exc:
            self._cleanup_playback()
            messagebox.showerror("Playback", f"Unable to read render: {exc}")
            return
        state["reader"] = reader
        iterator = iter(reader)
        delay_ms = int(1000 / (self.last_plan.fps if self.last_plan else DEFAULT_FPS))

        def show_next_frame() -> None:
            if not state:
                return
            try:
                frame = next(iterator)
            except StopIteration:
                self._cleanup_playback()
                return
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self._cleanup_playback()
                messagebox.showerror("Playback", f"Error during preview: {exc}")
                return
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
            if not play_window.winfo_exists():
                self._cleanup_playback()
                return
            timer_id = play_window.after(delay_ms, show_next_frame)
            state["timer"] = timer_id

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
        target = self.assets_dir / f"{safe_name}{self.temp_render_path.suffix}"
        try:
            shutil.copy2(self.temp_render_path, target)
        except Exception as exc:
            messagebox.showerror("Save", f"Unable to copy render: {exc}")
            return
        self.status_var.set(f"Saved render to {target}")

    # ------------------------------------------------------------------
    def save_settings(self) -> None:
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save mixer settings",
            defaultextension=".json",
            initialdir=str(self.output_dir),
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not file_path:
            return
        payload = self._serialize_settings()
        try:
            Path(file_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Save settings", f"Unable to save settings:\n{exc}")
            return
        self.status_var.set(f"Settings saved to {Path(file_path).name}.")

    # ------------------------------------------------------------------
    def load_settings(self) -> None:
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Load mixer settings",
            initialdir=str(self.output_dir),
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not file_path:
            return
        try:
            payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Load settings", f"Unable to read settings:\n{exc}")
            return
        self._apply_settings(payload)
        self.status_var.set(f"Settings loaded from {Path(file_path).name}.")

    # ------------------------------------------------------------------
    def _serialize_settings(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path.get(),
            "timing_path": self.timing_path.get(),
            "frequency_path": str(self.frequency_path) if self.frequency_path else "",
            "associations": [
                {
                    "track_name": assoc.track_name,
                    "mode": assoc.mode,
                    "options": dict(assoc.options),
                    "source": assoc.source,
                    "bin_index": assoc.bin_index,
                    "bin_indices": assoc.bin_indices,
                }
                for assoc in self.associations
            ],
            "render_settings": asdict(self.render_settings),
        }

    # ------------------------------------------------------------------
    def _apply_settings(self, payload: Dict[str, Any]) -> None:
        audio_path = str(payload.get("audio_path", ""))
        timing_path = str(payload.get("timing_path", ""))
        frequency_path_value = str(payload.get("frequency_path", ""))
        self.audio_path.set(audio_path)
        self.timing_path.set(timing_path)
        self.frequency_doc = None
        self.frequency_path = None
        self.associations = []
        assoc_payloads = payload.get("associations", [])
        if isinstance(assoc_payloads, list):
            for entry in assoc_payloads:
                if not isinstance(entry, dict):
                    continue
                track_name = entry.get("track_name")
                mode = entry.get("mode")
                options = entry.get("options", {})
                source = str(entry.get("source", "timing"))
                bin_index = entry.get("bin_index")
                bin_indices = entry.get("bin_indices")
                if not track_name or not mode or not isinstance(options, dict):
                    continue
                try:
                    parsed_bin_index = int(bin_index) if bin_index is not None else None
                except Exception:
                    parsed_bin_index = None
                parsed_bin_indices: Optional[list[int]] = None
                if isinstance(bin_indices, list):
                    try:
                        parsed_bin_indices = [int(idx) for idx in bin_indices]
                    except Exception:
                        parsed_bin_indices = None
                self.associations.append(
                    AssociationConfig(
                        track_name=str(track_name),
                        mode=str(mode),
                        options=dict(options),
                        source=source,
                        bin_index=parsed_bin_index,
                        bin_indices=parsed_bin_indices,
                    )
                )
        render_settings_payload = payload.get("render_settings")
        if isinstance(render_settings_payload, dict):
            self.render_settings = self._parse_render_settings(render_settings_payload)
            self.temp_render_path = self._render_output_path(self.render_settings.codec)
            self._save_render_settings()
        if frequency_path_value:
            candidate = Path(frequency_path_value).expanduser()
            if candidate.exists():
                try:
                    self.frequency_doc = FrequencyDocument.from_json(candidate)
                    self.frequency_path = candidate
                except Exception:
                    messagebox.showwarning("Frequency data", "Unable to load saved frequency file.")
        self._refresh_assoc_tree()
        if timing_path:
            self._load_timing()

    # ------------------------------------------------------------------
    def _cleanup_playback(self) -> None:
        state = self._playback_state
        if not state:
            return
        window = state.get("window")
        timer = state.get("timer")
        reader = state.get("reader")
        if timer and window and window.winfo_exists():
            try:
                window.after_cancel(timer)
            except Exception:
                pass
        if reader:
            try:
                reader.close()
            except Exception:
                pass
        if pygame and pygame.mixer.get_init():
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            pygame.mixer.quit()
        if window and window.winfo_exists():
            window.destroy()
        state.clear()


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
