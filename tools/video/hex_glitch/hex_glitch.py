#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hex Propagation Animator — consolidated build (wash fix)
-------------------------------------------------------
Adds F2 help overlay, F9 mouse-target toggle, and restores Y/U color wash step.
"""

import argparse
import os
import math
import random
import time
import subprocess
import json
import copy
from typing import Tuple, Dict, List, Optional, Iterable, Set

import pygame
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --------------------------- CONFIG ---------------------------
DEFAULT_CONFIG: Dict = {
    # Canvas / grid
    "width": 2280,
    "height": 1720,
    "cols": 100,
    "rows": 100,
    "hex_radius": None,  # None = auto-fit

    # Colors
    "base_color": (6, 8, 12),
    "palette": [
        (255, 99, 72),
        (255, 196, 0),
        (80, 220, 100),
        (72, 159, 255),
        (196, 72, 255),
        (255, 128, 176),
        (0, 230, 210),
    ],

    # Propagation
    "max_jump": 15,
    "mid_factor": 0.2,
    "steps_per_frame": 1500,
    "origin_retries": 18,

    # Random "color drops"
    "drop_interval_ms": 900,
    "drop_jitter_ms": 500,
    "drops_per_event": (2, 5),

    # Edge shots
    "edge_shot_interval_ms": 500,
    "edge_shot_jitter_ms": 1500,
    "edge_shot_length": (8, 40),
    "edge_shot_hop": (1, 4),

    # Image init / overlays
    "image_path": ["nav1.png", "nav2.png"],
    "image_gamma": 1.0,

    # Text mode
    "word_interval_ms": 1000,
    "text_area_margin": 60,
    "font_name": None,
    "text_bold": True,
    "max_font_size_cap": 820,

    # Motion (waves/quake)
    "wave_divisions": (3, 3),
    "motion_alpha": 0.45,

    # Erupt controls
    "erupt_band_thickness": 2.5,
    "erupt_blend": 0.6,
    "erupt_speed_cells_per_sec": 120.0,

    # Jump/Kick (sweeps)
    "jump_blend": 0.75,
    "jumpwave_duration_ms": (700, 1400),

    "kick_blend": 0.6,
    "kick_base_offset": 1,
    "kick_speed_multiplier_max": 4.0,
    "kick_bulge_half_height_range": (5, 15),
    "kickwave_duration_ms": (800, 1600),

    # Leap
    "leap_band_height_range": (4, 20),
    "leap_gap_range": (2, 12),
    "leap_copies_range": (3, 6),
    "leap_start_opacity": 0.75,
    "leap_end_opacity": 0.25,

    # Grow (G)
    "grow_seed_count": (1, 3),
    "grow_branch_prob": 0.55,
    "grow_blend": 0.7,
    "grow_duration_ms": (1200, 2500),

    # Halo v2 (H)
    "halo_target_cells_range": (10, 50),
    "halo_speed_cells_per_sec": 160.0,
    "halo_band_thickness": 2.0,
    "halo_power": 0.55,
    "halo_style": "tint",
    "halo_tint_palette": True,

    # Divide/Fuse
    "divide_duration_ms": (900, 1600),
    "divide_band_thickness": 1.5,
    "divide_blend": 0.6,

    "fuse_duration_ms": (900, 1600),
    "fuse_band_thickness": 1.5,
    "fuse_blend": 0.6,

    # Align (A)
    "align_duration_ms": (1200, 2500),
    "align_override_center": False,
    "align_skip_if_few_neighbours": True,
    "align_stride_cols": 1,

    # Scatter (S)
    "scatter_seed_count": (20, 80),
    "scatter_distance_range": (2, 8),
    "scatter_fade": 0.6,
    "scatter_duration_ms": (500, 1200),
    "scatter_sparks_per_seed_per_frame": (1, 3),
    "scatter_deviation_threshold": 35.0,

    # Recording
    "record_dir": "frames_out",
    "record_scale": 1.0,
    "ffmpeg_path": "ffmpeg",
    "ffmpeg_crf": 18,
    "ffmpeg_preset": "medium",
    "ffmpeg_fps": 60,

    # Display / UX
    "fps": 60,
    "show_grid_lines": False,
    "antialias": False,
    "debug_overlay": True,
    "save_frames_dir": "frames",
    "auto_reseed_threshold": 0.985,
    "initial_seed_cells": 400,

    # Fading
    "fade_revert_rate": 0.2,
    "words_file": "vidtext.txt",
}
# --------------------------------------------------------------


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DEFAULT_WORDS_FILE = "vidtext.txt"
WORDS_FILE_PATH = os.path.join(SCRIPT_DIR, DEFAULT_WORDS_FILE)
CONFIG: Dict = copy.deepcopy(DEFAULT_CONFIG)


def _coerce_config_value(value, default):
    """Best-effort coercion of JSON-loaded values to match defaults."""

    if isinstance(default, tuple):
        if isinstance(value, list):
            return tuple(value)
        return tuple(value) if not isinstance(value, tuple) else value
    if isinstance(default, list):
        if isinstance(value, list):
            if default and isinstance(default[0], tuple):
                return [tuple(item) if isinstance(item, list) else item for item in value]
            return value
        return default
    if isinstance(default, dict) and isinstance(value, dict):
        coerced = default.copy()
        for key, sub_value in value.items():
            if key in default:
                coerced[key] = _coerce_config_value(sub_value, default[key])
            else:
                coerced[key] = sub_value
        return coerced
    return value


def load_config(path: str) -> Dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        for key, value in user_config.items():
            if key in config:
                config[key] = _coerce_config_value(value, config[key])
            else:
                config[key] = value
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        print(f"Warning: Failed to parse config file {path}: {exc}")
    return config


def resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if path_value in (None, ""):
        return path_value
    path_str = str(path_value)
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(base_dir, path_str))


def prepare_runtime_config(
    config_path: Optional[str] = None,
    output_root: Optional[str] = None,
    words_file_override: Optional[str] = None,
) -> None:
    """Load config/paths based on CLI args or defaults."""

    global CONFIG, WORDS_FILE_PATH

    config_path = config_path or DEFAULT_CONFIG_PATH
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else SCRIPT_DIR
    output_dir = os.path.abspath(output_root) if output_root else config_dir

    CONFIG = load_config(config_path)

    image_paths = []
    for path in get_image_list_from_config(CONFIG.get("image_path")):
        resolved = resolve_path(path, config_dir)
        if resolved:
            image_paths.append(resolved)
    CONFIG["image_path"] = image_paths

    record_dir = resolve_path(CONFIG.get("record_dir", "frames_out"), output_dir)
    save_dir = resolve_path(CONFIG.get("save_frames_dir", "frames"), output_dir)
    CONFIG["record_dir"] = record_dir or os.path.join(output_dir, "frames_out")
    CONFIG["save_frames_dir"] = save_dir or os.path.join(output_dir, "frames")

    words_source = words_file_override or CONFIG.get("words_file") or DEFAULT_WORDS_FILE
    WORDS_FILE_PATH = resolve_path(words_source, config_dir) or os.path.join(config_dir, DEFAULT_WORDS_FILE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hex Propagation Animator")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a JSON config file")
    parser.add_argument(
        "--output-dir",
        help="Base directory for rendered frames/recordings (overrides record/save paths)",
    )
    parser.add_argument("--words-file", help="Optional override for vidtext/words file")
    return parser.parse_args()

HEX_DIRS = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

# --------------------- Mouse-target toggle --------------------
MOUSE_TARGET_MODE = True  # F9 toggles this at runtime

# --------------------- Fade mode toggle -----------------------
FADE_MODES = ("none", "blank", "image")


def describe_fade_mode(mode: str) -> str:
    if mode == "blank":
        return "Blank"
    if mode == "image":
        return "Image"
    return "Off"

def pick_anchor_axial(cols: int, rows: int, hex_radius: float, origin, use_mouse: bool):
    """Return (q,r) from mouse if enabled & inside window; else random."""
    import pygame
    if use_mouse and pygame.mouse.get_focused():
        mx, my = pygame.mouse.get_pos()
        if 0 <= mx < CONFIG["width"] and 0 <= my < CONFIG["height"]:
            q, r = pixel_to_axial(mx, my, hex_radius, origin, cols, rows)
            return q, r
    return (random.randrange(cols), random.randrange(rows))

# ---------------------- Help overlay --------------------------
def _draw_help_overlay(screen, width, height, font_name=None):
    """Draw an in-app help window listing hotkeys and features. F2 toggles."""
    import pygame
    pad = 16
    max_w = min(900, int(width * 0.8))
    max_h = min(700, int(height * 0.8))
    surf = pygame.Surface((max_w, max_h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 200))
    try:
        title_font = pygame.font.SysFont(font_name, 28, bold=True)
        item_font = pygame.font.SysFont("monospace", 18)
    except Exception:
        title_font = pygame.font.Font(None, 28)
        item_font = pygame.font.Font(None, 18)

    y = pad
    surf.blit(title_font.render("Hex Animator — Help (F2 to close)", True, (230, 230, 235)), (pad, y))
    y += 36

    sections = [
        ("Core", [
            ("Space", "Pause / Resume"),
            ("R", "Reset + reseed"),
            ("F3", "Toggle debug overlay"),
            ("F4", "Screenshot to 'frames/'"),
            ("F5", "PNG sequence on/off"),
            ("F6", "Start/stop ffmpeg writer"),
            ("F9", "Toggle mouse-target mode ON/OFF"),
            ("F10", "Cycle fade mode (Off → Blank → Image)"),
        ]),
        ("Images / Text", [
            ("I", "Overlay image 100%"),
            ("O", "Overlay image 50%"),
            ("P", "Next image in list"),
            ("T", "Stamp words from 'vidtext.txt'"),
        ]),
        ("Washes / Shots", [
            ("Y", "Color wash from edge"),
            ("U", "Half-opacity wash"),
            (";", "Random edge shot"),
        ]),
        ("Motions", [
            ("Q", "Quake (one-way smear)"),
            ("W", "Waves (radial smear)"),
            ("E", "Erupt (expanding ripple) [mouse-targeted]"),
            ("H", "Halo v2 (expanding ring) [mouse-targeted]"),
            ("G", "Grow (branch tint) [mouse-targeted]"),
            ("J", "JumpWave (upward copy sweep)"),
            ("K", "KickWave (side bulge sweep)"),
            ("L", "Leap (band echo copies)"),
            ("D", "Divide (push outward) [mouse-targeted line]"),
            ("F", "Fuse (push inward) [mouse-targeted line]"),
            ("A", "Align (neighbour unify to most-different)"),
            ("S", "Scatter (deviation-based sparks)"),
        ]),
        ("Mouse", [
            ("LMB", "Paint random palette colour at cursor")
        ]),
        ("Exit", [
            ("Esc", "Quit")
        ])
    ]

    for section, items in sections:
        surf.blit(title_font.render(section, True, (210, 210, 220)), (pad, y))
        y += 28
        for key, desc in items:
            text = f"{key:>6}  —  {desc}"
            surf.blit(item_font.render(text, True, (235, 235, 240)), (pad, y))
            y += 22
        y += 10

    dst = screen.get_rect()
    x = dst.centerx - max_w // 2
    y = dst.centery - max_h // 2
    screen.blit(surf, (x, y))

# --------------------------- Utils ----------------------------
def clamp(x, a, b):
    return a if x < a else (b if x > b else x)

def lerp_color(c1, c2, t: float) -> Tuple[int, int, int]:
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return (clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255))

def brighten(c: Tuple[int,int,int], amt: float) -> Tuple[int,int,int]:
    r = clamp(int(c[0] + (255 - c[0]) * amt), 0, 255) if amt >= 0 else clamp(int(c[0] * (1 + amt)), 0, 255)
    g = clamp(int(c[1] + (255 - c[1]) * amt), 0, 255) if amt >= 0 else clamp(int(c[1] * (1 + amt)), 0, 255)
    b = clamp(int(c[2] + (255 - c[2]) * amt), 0, 255) if amt >= 0 else clamp(int(c[2] * (1 + amt)), 0, 255)
    return (r, g, b)

def color_distance(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    dr = a[0]-b[0]; dg = a[1]-b[1]; db = a[2]-b[2]
    return math.sqrt(dr*dr + dg*dg + db*db)

def average_color(colors: List[Tuple[int,int,int]]) -> Tuple[float,float,float]:
    if not colors: return (0.0,0.0,0.0)
    r = sum(c[0] for c in colors) / len(colors)
    g = sum(c[1] for c in colors) / len(colors)
    b = sum(c[2] for c in colors) / len(colors)
    return (r,g,b)

def axial_to_pixel(q: int, r: int, hex_radius: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    size = hex_radius
    x = (math.sqrt(3) * (q + 0.5 * (r & 1))) * size + origin[0]
    y = (1.5 * r) * size + origin[1]
    return (x, y)

def pixel_to_axial(px: float, py: float, hex_radius: float, origin: Tuple[float, float], cols: int, rows: int):
    size = hex_radius
    y_rel = py - origin[1]
    r_est = int(round(y_rel / (1.5 * size)))
    r_est = max(0, min(rows - 1, r_est))
    x_rel = px - origin[0]
    qf = (x_rel / (math.sqrt(3) * size)) - 0.5 * (r_est & 1)
    q_est = int(round(qf))
    q_est = max(0, min(cols - 1, q_est))
    return (q_est, r_est)

def polygon_hex(center: Tuple[float, float], size: float):
    cx, cy = center
    pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        x = cx + size * math.cos(angle)
        y = cy + size * math.sin(angle)
        pts.append((int(x), int(y)))
    return pts

def axial_to_cube(q, r):
    x = q
    z = r
    y = -x - z
    return (x, y, z)

def cube_distance(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2]))

def nearest_hex_dir(dq, dr):
    best = None
    best_score = -1e18
    for (x, y) in HEX_DIRS:
        score = x * dq + y * dr
        if score > best_score:
            best_score = score; best = (x, y)
    return best

def dir_index(v: Tuple[int,int]) -> int:
    return HEX_DIRS.index(v)

def dot_axial(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return a[0]*b[0] + a[1]*b[1]

def perpendicular_dirs(v: Tuple[int,int]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    i = dir_index(v)
    n1 = HEX_DIRS[(i + 2) % 6]
    n2 = HEX_DIRS[(i - 2) % 6]
    return n1, n2

# ------------------------ Image I/O ---------------------------
def load_image_resized(path: str, cols: int, rows: int, gamma: float):
    if not PIL_AVAILABLE or not path or not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize((cols, rows), Image.LANCZOS)
    if gamma != 1.0:
        inv = 1.0 / gamma
        lut = [int((i/255.0)**inv * 255.0 + 0.5) for i in range(256)]
        img = img.point(lut * 3)
    return img

def get_image_list_from_config(image_path_conf) -> List[str]:
    if image_path_conf is None: return []
    if isinstance(image_path_conf, str): return [image_path_conf]
    if isinstance(image_path_conf, (list, tuple)): return list(image_path_conf)
    return []

def overlay_image_on_grid(grid: "HexGrid", img_path: str, alpha: float, gamma: float = 1.0):
    img = load_image_resized(img_path, grid.cols, grid.rows, gamma)
    if img is None:
        print(f"[WARN] Cannot overlay (missing?): {img_path}"); return
    px = img.load()
    for r in range(grid.rows):
        for q in range(grid.cols):
            target = px[q, r]
            current = grid.get(q, r)
            blended = lerp_color(current, target, alpha)
            grid.set(q, r, blended)

# ------------------------- Grid -------------------------------
class HexGrid:
    def __init__(self, cols: int, rows: int, base_color: Tuple[int,int,int]):
        self.cols = cols; self.rows = rows; self.base = base_color
        self.grid: Dict[Tuple[int,int], Tuple[int,int,int]] = {}
        self.base_count = cols * rows
        for r in range(rows):
            for q in range(cols):
                self.grid[(q, r)] = base_color

    def wrap(self, q: int, r: int) -> Tuple[int,int]:
        return (q % self.cols, r % self.rows)

    def get(self, q: int, r: int) -> Tuple[int,int,int]:
        return self.grid[(q, r)]

    def set(self, q: int, r: int, color: Tuple[int,int,int]):
        prev = self.grid[(q, r)]
        if prev == self.base and color != self.base: self.base_count -= 1
        elif prev != self.base and color == self.base: self.base_count += 1
        self.grid[(q, r)] = color

    def fill_from_image(self, img_path: str, gamma: float = 1.0):
        img = load_image_resized(img_path, self.cols, self.rows, gamma)
        if img is None: return
        px = img.load()
        self.base_count = 0
        for r in range(self.rows):
            for q in range(self.cols):
                c = px[q, r]; self.grid[(q, r)] = c
                if c == self.base: self.base_count += 1

# ------------------------- Text -------------------------------
def load_words_from_file(path: str) -> List[str]:
    if not os.path.exists(path): return []
    words: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks = [t for t in line.strip().split() if t]; words.extend(toks)
    return words

def find_max_font_size(text: str, area_w: int, area_h: int, font_name: Optional[str], bold: bool, cap: int) -> int:
    if not text: return 1
    lo, hi = 1, cap; best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        try: font = pygame.font.SysFont(font_name, mid, bold=bold)
        except Exception: font = pygame.font.Font(None, mid)
        tw, th = font.size(text)
        if tw <= area_w and th <= area_h: best = mid; lo = mid + 1
        else: hi = mid - 1
    return max(1, best)

def paint_word_to_grid(word: str, color: Tuple[int,int,int], screen_w: int, screen_h: int,
                       origin: Tuple[float,float], hex_radius: float,
                       cols: int, rows: int, grid: HexGrid):
    margin = CONFIG["text_area_margin"]
    area_x = area_y = margin
    area_w = screen_w - 2 * margin; area_h = screen_h - 2 * margin
    size = find_max_font_size(word, area_w, area_h, CONFIG["font_name"], CONFIG["text_bold"], CONFIG["max_font_size_cap"])
    try: font = pygame.font.SysFont(CONFIG["font_name"], size, bold=CONFIG["text_bold"])
    except Exception: font = pygame.font.Font(None, size)
    text_surface = font.render(word, True, (255, 255, 255))
    tw, th = text_surface.get_size()
    tx = area_x + (area_w - tw) // 2; ty = area_y + (area_h - th) // 2
    text_mask = pygame.mask.from_surface(text_surface)
    for r in range(rows):
        for q in range(cols):
            cx, cy = axial_to_pixel(q, r, hex_radius, origin)
            ix = int(cx) - tx; iy = int(cy) - ty
            if 0 <= ix < tw and 0 <= iy < th and text_mask.get_at((ix, iy)):
                grid.set(q, r, color)

# ------------------------ Wash (BFS) --------------------------
def neighbors(q: int, r: int) -> Iterable[Tuple[int,int]]:
    for dq, dr in HEX_DIRS: yield (q + dq, r + dr)

def start_color_wash(cols: int, rows: int) -> Tuple[Set[Tuple[int,int]], Set[Tuple[int,int]], Tuple[int,int,int]]:
    edge = random.choice(["top","bottom","left","right"])
    color = random.choice(CONFIG["palette"])
    frontier: Set[Tuple[int,int]] = set(); seen: Set[Tuple[int,int]] = set()
    if edge == "top":
        for q in range(cols): frontier.add((q, 0))
    elif edge == "bottom":
        for q in range(cols): frontier.add((q, rows-1))
    elif edge == "left":
        for r in range(rows): frontier.add((0, r))
    else:
        for r in range(rows): frontier.add((cols-1, r))
    seen |= frontier; return frontier, seen, color

def advance_wash(frontier: Set[Tuple[int,int]], seen: Set[Tuple[int,int]], cols: int, rows: int) -> Set[Tuple[int,int]]:
    nxt: Set[Tuple[int,int]] = set()
    for (q, r) in frontier:
        for (nq, nr) in neighbors(q, r):
            if 0 <= nq < cols and 0 <= nr < rows and (nq, nr) not in seen: nxt.add((nq, nr))
    seen |= nxt; return nxt

# --------------------- Motion effects -------------------------
class MotionState:
    def __init__(self):
        self.active = False
        self.kind: Optional[str] = None
        self.dir = (1, 0)
        self.t0 = 0
        self.duration = 1000
        self.centers: List[Tuple[int,int]] = []  # waves
        # erupt
        self.center: Tuple[int,int] = (0, 0)
        self.erupt_maxdist: float = 1.0
        # kick bulge
        self.kick_center_row: int = 0
        self.kick_half_h: int = 8
        # grow
        self.grow_frontier: Set[Tuple[int,int]] = set()
        self.grow_seen: Set[Tuple[int,int]] = set()
        self.grow_color: Tuple[int,int,int] = (255,255,255)
        # halo
        self.halo_center: Tuple[int,int] = (0,0)
        self.halo_speed = 160.0
        self.halo_band = 2.0
        self.halo_power = 0.55
        self.halo_target = 30.0
        self.halo_fade = 15.0
        self.halo_style = "tint"
        self.halo_color: Tuple[int,int,int] = (255,255,255)
        # divide/fuse geometry
        self.line_anchor: Tuple[int,int] = (0,0)
        self.line_dir: Tuple[int,int] = (1,0)
        self.norm_dirs: Tuple[Tuple[int,int], Tuple[int,int]] = ((0,1),(0,-1))
        self.cross_extent: int = 0

        # align/scatter
        self.scatter_seeds: List[Tuple[int,int,Tuple[int,int,int]]] = []

        # geometry (injected)
        self._hex_radius: float = 10.0
        self._origin: Tuple[float,float] = (0.0, 0.0)

    def set_geometry(self, hex_radius: float, origin: Tuple[float,float]):
        self._hex_radius = float(hex_radius); self._origin = origin

    def start_quake(self):
        self.active = True; self.kind = 'quake'
        self.dir = random.choice(HEX_DIRS); self.t0 = pygame.time.get_ticks()
        self.duration = int(random.uniform(500, 1500))

    def start_waves(self, cols: int, rows: int):
        self.active = True; self.kind = 'waves'
        self.t0 = pygame.time.get_ticks(); self.duration = int(random.uniform(700, 1600))
        cxn, cyn = CONFIG["wave_divisions"]; self.centers = []
        for iy in range(cyn):
            r = int(round((iy + 0.5) * rows / cyn)); r = max(0, min(rows - 1, r))
            for ix in range(cxn):
                q = int(round((ix + 0.5) * cols / cxn)); q = max(0, min(cols - 1, q))
                self.centers.append((q, r))

    def start_erupt(self, cols: int, rows: int, mouse_target_mode: bool):
        self.active = True; self.kind = 'erupt'; self.t0 = pygame.time.get_ticks()
        cq, cr = pick_anchor_axial(cols, rows, self._hex_radius, self._origin, mouse_target_mode)
        self.center = (cq, cr)
        corners = [(0, 0), (cols-1, 0), (0, rows-1), (cols-1, rows-1)]
        c_cube = axial_to_cube(cq, cr)
        self.erupt_maxdist = max(float(cube_distance(c_cube, axial_to_cube(x, y))) for (x, y) in corners)
        speed = max(1e-3, float(CONFIG.get("erupt_speed_cells_per_sec", 120.0)))
        self.duration = int((self.erupt_maxdist / speed) * 1000.0)
        if self.duration < 200: self.duration = 200

    def start_jumpwave(self, rows: int):
        self.active = True; self.kind = 'jumpwave'
        self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["jumpwave_duration_ms"])

    def start_kickwave(self, rows: int, cols: int):
        self.active = True; self.kind = 'kickwave'
        self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["kickwave_duration_ms"])
        self.dir = random.choice([(1,0), (-1,0)])
        self.kick_center_row = random.randrange(rows)
        self.kick_half_h = random.randint(*CONFIG["kick_bulge_half_height_range"])

    def start_grow(self, cols: int, rows: int, mouse_target_mode: bool):
        self.active = True; self.kind = 'grow'; self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["grow_duration_ms"])
        seeds = []
        count = random.randint(*CONFIG["grow_seed_count"])
        for _ in range(count):
            q, r = pick_anchor_axial(cols, rows, self._hex_radius, self._origin, mouse_target_mode)
            seeds.append((q, r))
        self.grow_frontier = set(seeds); self.grow_seen = set(seeds)
        self.grow_color = random.choice(CONFIG["palette"])

    def start_halo(self, cols: int, rows: int, mouse_target_mode: bool):
        self.active = True; self.kind = 'halo'; self.t0 = pygame.time.get_ticks()
        self.halo_center = pick_anchor_axial(cols, rows, self._hex_radius, self._origin, mouse_target_mode)
        lo, hi = CONFIG["halo_target_cells_range"]
        self.halo_target = float(random.randint(lo, hi))
        self.halo_fade = max(1.0, self.halo_target * 0.5)
        self.halo_speed = float(CONFIG["halo_speed_cells_per_sec"])
        self.halo_band = float(CONFIG["halo_band_thickness"])
        self.halo_power = float(CONFIG["halo_power"])
        self.halo_style = CONFIG["halo_style"]
        self.halo_color = random.choice(CONFIG["palette"]) if CONFIG["halo_tint_palette"] else (255,255,255)
        total_cells = self.halo_target + self.halo_fade
        self.duration = int((total_cells / max(self.halo_speed,1e-3)) * 1000.0)

    def _prepare_line(self, cols: int, rows: int, mouse_target_mode: bool):
        aq, ar = pick_anchor_axial(cols, rows, self._hex_radius, self._origin, mouse_target_mode)
        v = random.choice(HEX_DIRS)
        n1, n2 = perpendicular_dirs(v)
        corners = [(0,0),(cols-1,0),(0,rows-1),(cols-1,rows-1)]
        cross_vals = []
        for (x,y) in corners:
            dq, dr = x - aq, y - ar
            cross_vals.append(abs(dot_axial((dq,dr), n1)))
            cross_vals.append(abs(dot_axial((dq,dr), n2)))
        self.line_anchor = (aq, ar)
        self.line_dir = v
        self.norm_dirs = (n1, n2)
        self.cross_extent = max(cross_vals) + 2

    def start_divide(self, cols: int, rows: int, mouse_target_mode: bool):
        self.active = True; self.kind = 'divide'; self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["divide_duration_ms"])
        self._prepare_line(cols, rows, mouse_target_mode)

    def start_fuse(self, cols: int, rows: int, mouse_target_mode: bool):
        self.active = True; self.kind = 'fuse'; self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["fuse_duration_ms"])
        self._prepare_line(cols, rows, mouse_target_mode)

    def start_align(self, cols: int, rows: int):
        self.active = True; self.kind = 'align'; self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["align_duration_ms"])

    def start_scatter(self, cols: int, rows: int, grid: "HexGrid"):
        self.active = True; self.kind = 'scatter'; self.t0 = pygame.time.get_ticks()
        self.duration = random.randint(*CONFIG["scatter_duration_ms"])
        seeds_n = random.randint(*CONFIG["scatter_seed_count"])
        self.scatter_seeds = []
        for _ in range(seeds_n):
            q = random.randrange(cols); r = random.randrange(rows)
            neigh = []
            for (dq, dr) in HEX_DIRS:
                nq, nr = q + dq, r + dr
                if 0 <= nq < cols and 0 <= nr < rows:
                    neigh.append((nq, nr))
            neigh_cols = [grid.get(nq, nr) for (nq,nr) in neigh] if neigh else []
            if neigh_cols:
                avg = average_color(neigh_cols)
                maxd, bestc = -1.0, None
                for (nq,nr) in neigh:
                    c = grid.get(nq, nr)
                    d = color_distance(c, (avg[0],avg[1],avg[2]))
                    if d > maxd: maxd, bestc = d, c
                if maxd >= CONFIG["scatter_deviation_threshold"]:
                    seed_color = bestc
                else:
                    seed_color = random.choice(CONFIG["palette"])
            else:
                seed_color = random.choice(CONFIG["palette"])
            self.scatter_seeds.append((q, r, seed_color))

    def phase(self) -> float:
        if not self.active: return 0.0
        t = pygame.time.get_ticks() - self.t0
        p = t / max(1, self.duration)
        if p >= 1.0:
            self.active = False; return 1.0
        return p

# helpers
def smear_direction(grid: 'HexGrid', dir_axial: Tuple[int,int], alpha: float):
    cols, rows = grid.cols, grid.rows
    dq, dr = dir_axial
    orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
    for r in range(rows):
        for q in range(cols):
            nq, nr = grid.wrap(q + dq, r + dr)
            c0 = orig[(q, r)]; c1 = orig[(nq, nr)]
            grid.set(q, r, lerp_color(c0, c1, alpha))

def smear_radial_from_centers(grid: 'HexGrid', centers: List[Tuple[int,int]], alpha: float, outward: bool):
    cols, rows = grid.cols, grid.rows
    orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
    for r in range(rows):
        for q in range(cols):
            a = axial_to_cube(q, r)
            best_c = None; best_d = 1e18
            for (cq, cr) in centers:
                d = cube_distance(a, axial_to_cube(cq, cr))
                if d < best_d: best_d = d; best_c = (cq, cr)
            dq = q - best_c[0]; dr = r - best_c[1]
            dir_ax = nearest_hex_dir(dq, dr)
            if dir_ax is None: continue
            if not outward: dir_ax = (-dir_ax[0], -dir_ax[1])
            nq, nr = grid.wrap(q + dir_ax[0], r + dir_ax[1])
            c0 = orig[(q, r)]; c1 = orig[(nq, nr)]
            grid.set(q, r, lerp_color(c0, c1, alpha))

def erupt_ripple(grid: 'HexGrid', center: Tuple[int,int], phase: float, maxdist: float, band_thickness: float = 1.0):
    cols, rows = grid.cols, grid.rows
    cq, cr = center; target_r = phase * maxdist
    blend = float(CONFIG.get("erupt_blend", 0.6)); bt = float(CONFIG.get("erupt_band_thickness", band_thickness))
    orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
    c_cube = axial_to_cube(cq, cr)
    for r in range(rows):
        for q in range(cols):
            a_cube = axial_to_cube(q, r)
            d = cube_distance(a_cube, c_cube)
            if abs(d - target_r) <= bt:
                dq = cq - q; dr = cr - r
                dir_ax = nearest_hex_dir(dq, dr)
                if dir_ax is None: continue
                nq, nr = grid.wrap(q + dir_ax[0], r + dir_ax[1])
                c0 = orig[(q, r)]; c1 = orig[(nq, nr)]
                grid.set(q, r, lerp_color(c0, c1, blend))

# ---------------------- Leap (L) ------------------------------
def op_leap(grid: 'HexGrid'):
    rows, cols = grid.rows, grid.cols
    band_h = random.randint(*CONFIG["leap_band_height_range"])
    copies = random.randint(*CONFIG["leap_copies_range"])
    gap = random.randint(*CONFIG["leap_gap_range"])
    start_op = float(CONFIG["leap_start_opacity"]); end_op = float(CONFIG["leap_end_opacity"])
    base_r = random.randrange(rows)
    orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
    if copies <= 1: alphas = [start_op]
    else: alphas = [start_op + (end_op - start_op) * (i / (copies - 1)) for i in range(copies)]
    cumulative = 0
    for i in range(copies):
        cumulative += band_h + gap * (i + 1); alpha = alphas[i]
        for dh in range(band_h):
            src_r = (base_r + dh) % rows; dst_r = (src_r - cumulative) % rows
            for q in range(cols):
                src = orig[(q, src_r)]; dst = grid.get(q, dst_r)
                grid.set(q, dst_r, lerp_color(dst, src, alpha))

# -------------------------- Main ------------------------------
def main():
    pygame.init()
    flags = pygame.DOUBLEBUF
    screen = pygame.display.set_mode((CONFIG["width"], CONFIG["height"]), flags)
    pygame.display.set_caption("Hex Propagation Animator — consolidated (wash fix)")
    clock = pygame.time.Clock()

    cols, rows = CONFIG["cols"], CONFIG["rows"]
    if CONFIG["hex_radius"] is None:
        size_w = CONFIG["width"] / (math.sqrt(3) * (cols + 0.5))
        size_h = CONFIG["height"] / (1.6 * rows + 0.6)
        hex_radius = max(min(size_w, size_h), 3.0)
    else:
        hex_radius = float(CONFIG["hex_radius"])

    grid_w = math.sqrt(3) * hex_radius * (cols + 0.5)
    grid_h = (1.5 * (rows - 1) + 2.0) * hex_radius
    origin = ((CONFIG["width"] - grid_w) / 2.0, (CONFIG["height"] - grid_h) / 2.0)

    os.makedirs(CONFIG["save_frames_dir"], exist_ok=True)

    grid = HexGrid(cols, rows, CONFIG["base_color"])

    image_list = get_image_list_from_config(CONFIG["image_path"])
    image_idx = 0
    if image_list: grid.fill_from_image(image_list[0], CONFIG["image_gamma"])

    image_cache: Dict[str, Optional[List[List[Tuple[int, int, int]]]]] = {}

    def get_image_target(path: str) -> Optional[List[List[Tuple[int, int, int]]]]:
        if not path:
            return None
        if path not in image_cache:
            img = load_image_resized(path, cols, rows, CONFIG["image_gamma"])
            if img is None:
                image_cache[path] = None
            else:
                px = img.load()
                data: List[List[Tuple[int, int, int]]] = []
                for r in range(rows):
                    row_vals: List[Tuple[int, int, int]] = []
                    for q in range(cols):
                        row_vals.append(tuple(px[q, r]))
                    data.append(row_vals)
                image_cache[path] = data
        return image_cache[path]

    paused = False
    debug = CONFIG["debug_overlay"]
    show_grid = CONFIG["show_grid_lines"]
    help_visible = False
    mouse_target_mode = True   # F9 toggles this
    fade_mode_index = 0        # F10 cycles this

    text_mode = False; words: List[str] = []; word_index = 0; next_word_at = 0

    wash_active = False; wash_half = False
    wash_frontier: Set[Tuple[int,int]] = set()
    wash_seen: Set[Tuple[int,int]] = set()
    wash_color: Tuple[int,int,int] = CONFIG["base_color"]

    motion = MotionState(); motion.set_geometry(hex_radius, origin)

    # --- Recording state ---
    record_png = False; ffmpeg_proc = None

    def start_ffmpeg_writer():
        nonlocal ffmpeg_proc
        if ffmpeg_proc is not None: return
        w = int(CONFIG["width"] * CONFIG["record_scale"]); h = int(CONFIG["height"] * CONFIG["record_scale"])
        out_name = time.strftime("render_%Y%m%d_%H%M%S.mp4")
        cmd = [CONFIG["ffmpeg_path"], "-y", "-f","rawvideo", "-pix_fmt","rgb24", "-s", f"{w}x{h}",
               "-r", str(CONFIG["ffmpeg_fps"]), "-i","-", "-an", "-c:v","libx264", "-crf", str(CONFIG["ffmpeg_crf"]),
               "-preset", CONFIG["ffmpeg_preset"], "-pix_fmt","yuv420p", out_name]
        try: ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError: print("[REC] ffmpeg not found.")

    def stop_ffmpeg_writer():
        nonlocal ffmpeg_proc
        if ffmpeg_proc:
            try: ffmpeg_proc.stdin.close()
            except Exception: pass
            try: ffmpeg_proc.wait()
            except Exception: pass
            ffmpeg_proc = None

    def jittered(mean_ms, jitter_ms):
        return max(100, int(random.gauss(mean_ms, jitter_ms)))

    next_drop_at = pygame.time.get_ticks() + jittered(CONFIG["drop_interval_ms"], CONFIG["drop_jitter_ms"])
    next_edge_shot_at = pygame.time.get_ticks() + jittered(CONFIG["edge_shot_interval_ms"], CONFIG["edge_shot_jitter_ms"])

    def do_random_drops(n=None):
        if n is None: n = random.randint(CONFIG["drops_per_event"][0], CONFIG["drops_per_event"][1])
        for _ in range(n):
            q = random.randrange(cols); r = random.randrange(rows)
            color = random.choice(CONFIG["palette"]); grid.set(q, r, color)

    def fire_random_edge_shot():
        color = random.choice(CONFIG["palette"]); edge = random.choice(["top","bottom","left","right"])
        if edge == "top":
            r0 = 0; q0 = random.randrange(cols); direction = random.choice([(0,1),(-1,1),(1,0)])
        elif edge == "bottom":
            r0 = rows-1; q0 = random.randrange(cols); direction = random.choice([(0,-1),(1,-1),(-1,0)])
        elif edge == "left":
            q0 = 0; r0 = random.randrange(rows); direction = random.choice([(1,0),(1,-1),(0,1)])
        else:
            q0 = cols-1; r0 = random.randrange(rows); direction = random.choice([(-1,0),(-1,1),(0,-1)])
        length = random.randint(*CONFIG["edge_shot_length"]); hop_min, hop_max = CONFIG["edge_shot_hop"]
        q, r = q0, r0
        for _ in range(length):
            hop = random.randint(hop_min, hop_max); dq, dr = direction
            q += dq * hop; r += dr * hop; q, r = grid.wrap(q, r); grid.set(q, r, color)

    do_random_drops(CONFIG["initial_seed_cells"])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_F2:    # Help toggle
                    help_visible = not help_visible
                elif event.key == pygame.K_F3:    # debug toggle
                    debug = not debug
                elif event.key == pygame.K_F4:    # screenshot
                    timestamp = int(time.time() * 1000)
                    path = os.path.join(CONFIG["save_frames_dir"], f"hex_{timestamp}.png")
                    pygame.image.save(screen, path); print(f"[INFO] Saved screenshot: {path}")
                elif event.key == pygame.K_F5:
                    record_png = not record_png; os.makedirs(CONFIG["record_dir"], exist_ok=True)
                    print(f"[REC] PNG frames: {'ON' if record_png else 'OFF'} -> {CONFIG['record_dir']}")
                elif event.key == pygame.K_F6:
                    if ffmpeg_proc is None:
                        start_ffmpeg_writer()
                        if ffmpeg_proc is not None: print("[REC] ffmpeg: ON (writing MP4)")
                    else:
                        stop_ffmpeg_writer(); print("[REC] ffmpeg: OFF (file finalized)")
                elif event.key == pygame.K_F9:
                    mouse_target_mode = not mouse_target_mode
                    print(f"[UI] Mouse-target mode: {'ON' if mouse_target_mode else 'OFF'}")
                elif event.key == pygame.K_F10:
                    for _ in range(len(FADE_MODES)):
                        fade_mode_index = (fade_mode_index + 1) % len(FADE_MODES)
                        candidate = FADE_MODES[fade_mode_index]
                        if candidate != "image" or image_list:
                            break
                    mode = FADE_MODES[fade_mode_index]
                    if mode == "image" and not get_image_target(image_list[image_idx]):
                        print("[UI] Fade mode: Image (unavailable - missing image data)")
                    else:
                        print(f"[UI] Fade mode: {describe_fade_mode(mode)}")

                elif event.key == pygame.K_r:
                    for rr in range(rows):
                        for qq in range(cols): grid.set(qq, rr, CONFIG["base_color"])
                    do_random_drops(CONFIG["initial_seed_cells"])

                # Images / text
                elif event.key == pygame.K_i:
                    if image_list: overlay_image_on_grid(grid, image_list[image_idx], 1.0, CONFIG["image_gamma"])
                elif event.key == pygame.K_o:
                    if image_list: overlay_image_on_grid(grid, image_list[image_idx], 0.5, CONFIG["image_gamma"])
                elif event.key == pygame.K_p:
                    if image_list: image_idx = (image_idx + 1) % len(image_list)
                elif event.key == pygame.K_t:
                    if not text_mode:
                        words = load_words_from_file(WORDS_FILE_PATH); word_index = 0
                        if not words: print(f"[WARN] Words file not found or empty: {WORDS_FILE_PATH}")
                        else: print(f"[INFO] Loaded {len(words)} words from {WORDS_FILE_PATH}")
                        text_mode = True; next_word_at = pygame.time.get_ticks()
                    else:
                        text_mode = False

                # Washes / shots
                elif event.key == pygame.K_SEMICOLON: fire_random_edge_shot()
                elif event.key == pygame.K_y:
                    wash_frontier, wash_seen, wash_color = start_color_wash(cols, rows)
                    wash_active, wash_half = True, False
                elif event.key == pygame.K_u:
                    wash_frontier, wash_seen, wash_color = start_color_wash(cols, rows)
                    wash_active, wash_half = True, True

                # Motions
                elif event.key == pygame.K_q: motion.start_quake()
                elif event.key == pygame.K_w: motion.start_waves(cols, rows)
                elif event.key == pygame.K_e: motion.start_erupt(cols, rows, mouse_target_mode)

                elif event.key == pygame.K_g: motion.start_grow(cols, rows, mouse_target_mode)
                elif event.key == pygame.K_h: motion.start_halo(cols, rows, mouse_target_mode)

                elif event.key == pygame.K_d: motion.start_divide(cols, rows, mouse_target_mode)
                elif event.key == pygame.K_f: motion.start_fuse(cols, rows, mouse_target_mode)

                elif event.key == pygame.K_a: motion.start_align(cols, rows)
                elif event.key == pygame.K_s: motion.start_scatter(cols, rows, grid)

                elif event.key == pygame.K_j: motion.start_jumpwave(rows)
                elif event.key == pygame.K_k: motion.start_kickwave(rows, cols)
                elif event.key == pygame.K_l: op_leap(grid)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # paint random palette colour at cursor
                q, r = pixel_to_axial(*pygame.mouse.get_pos(), hex_radius, origin, cols, rows)
                grid.set(q, r, random.choice(CONFIG["palette"]))

        now = pygame.time.get_ticks()
        if now >= next_drop_at:
            do_random_drops(); next_drop_at = now + jittered(CONFIG["drop_interval_ms"], CONFIG["drop_jitter_ms"])
        if now >= next_edge_shot_at:
            fire_random_edge_shot()
            next_edge_shot_at = now + jittered(CONFIG["edge_shot_interval_ms"], CONFIG["edge_shot_jitter_ms"])

        total = cols * rows
        if grid.base_count / total >= CONFIG["auto_reseed_threshold"]:
            do_random_drops(CONFIG["initial_seed_cells"] // 2)

        # --- RESTORED: color wash step (Y/U) ---
        if not paused and wash_active:
            if not wash_frontier:
                wash_active = False
            else:
                for (q, r) in list(wash_frontier):
                    current = grid.get(q, r)
                    grid.set(q, r, lerp_color(current, wash_color, 0.5 if wash_half else 1.0))
                wash_frontier = advance_wash(wash_frontier, wash_seen, cols, rows)

        # Text stamping
        if not paused and text_mode and words:
            if now >= next_word_at and word_index < len(words):
                word = words[word_index]; word_color = random.choice(CONFIG["palette"])
                paint_word_to_grid(word, word_color, CONFIG["width"], CONFIG["height"], origin, hex_radius, cols, rows, grid)
                word_index += 1; next_word_at = now + CONFIG["word_interval_ms"]
            elif word_index >= len(words): text_mode = False

        # Propagation steps
        if not paused:
            for _ in range(CONFIG["steps_per_frame"]):
                for _try in range(CONFIG["origin_retries"]):
                    oq = random.randrange(cols); orr = random.randrange(rows); oc = grid.get(oq, orr)
                    if oc != CONFIG["base_color"]:
                        break
                else:
                    continue
                dq, dr = random.choice(HEX_DIRS); dist = random.randint(1, CONFIG["max_jump"])
                tq, tr = oq + dq * dist, orr + dr * dist; tq, tr = grid.wrap(tq, tr)
                tc = grid.get(tq, tr); mid = lerp_color(oc, tc, CONFIG["mid_factor"]); grid.set(tq, tr, mid)

        # Motion pass
        if not paused and getattr(motion, "active", False):
            p = motion.phase()
            if motion.kind == 'quake':
                smear_direction(grid, motion.dir, CONFIG["motion_alpha"])
            elif motion.kind == 'waves':
                smear_radial_from_centers(grid, motion.centers, CONFIG["motion_alpha"], outward=(p < 0.5))
            elif motion.kind == 'erupt':
                erupt_ripple(grid, motion.center, p, motion.erupt_maxdist, band_thickness=1.0)
            elif motion.kind == 'jumpwave':
                front = int(p * (rows - 1))
                orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
                alpha = float(CONFIG["jump_blend"])
                for r in range(front + 1):
                    dest_r = (r - 1) % rows
                    for q in range(cols):
                        src = orig[(q, r)]; dst = grid.get(q, dest_r)
                        grid.set(q, dest_r, lerp_color(dst, src, alpha))
            elif motion.kind == 'kickwave':
                front = int(p * (cols - 1))
                dx = 1 if motion.dir == (1,0) else -1
                base_off = int(CONFIG["kick_base_offset"])
                mult_max = float(CONFIG["kick_speed_multiplier_max"])
                half_h = int(motion.kick_half_h)
                center_r = int(motion.kick_center_row)
                alpha = float(CONFIG["kick_blend"])
                orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
                if dx == 1: c_range = range(0, front + 1)
                else: c_range = range(cols - 1, cols - 1 - (front + 1), -1)
                for c in c_range:
                    for r in range(rows):
                        dist_row = abs(r - center_r)
                        if dist_row <= half_h and half_h > 0:
                            trow = 1.0 - (dist_row / half_h)
                            mult = 1.0 + (mult_max - 1.0) * trow
                        else:
                            mult = 1.0
                        offset = max(1, int(round(base_off * mult)))
                        src = orig[(c, r)]; dest_c = (c + dx * offset) % cols
                        dst = grid.get(dest_c, r)
                        grid.set(dest_c, r, lerp_color(dst, src, alpha))
            elif motion.kind == 'grow':
                grow_prob = float(CONFIG["grow_branch_prob"]); blend = float(CONFIG["grow_blend"])
                new_front = set()
                for (q, r) in list(motion.grow_frontier):
                    current = grid.get(q, r)
                    grid.set(q, r, lerp_color(current, motion.grow_color, blend))
                    for dq, dr in HEX_DIRS:
                        if random.random() < grow_prob:
                            nq, nr = q + dq, r + dr
                            if 0 <= nq < cols and 0 <= nr < rows and (nq, nr) not in motion.grow_seen:
                                motion.grow_seen.add((nq, nr)); new_front.add((nq, nr))
                motion.grow_frontier = new_front
                if not motion.grow_frontier: motion.active = False
            elif motion.kind == 'halo':
                cq, cr = motion.halo_center; c_cube = axial_to_cube(cq, cr)
                elapsed_ms = pygame.time.get_ticks() - motion.t0
                radius = (elapsed_ms / 1000.0) * motion.halo_speed
                if radius <= motion.halo_target: amp = 1.0
                elif radius <= motion.halo_target + motion.halo_fade:
                    amp = 1.0 - (radius - motion.halo_target) / motion.halo_fade
                else:
                    motion.active = False; amp = 0.0
                if amp > 0.0:
                    band = motion.halo_band; power = motion.halo_power * amp
                    tint = motion.halo_color; style = motion.halo_style
                    orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
                    for r in range(rows):
                        for q in range(cols):
                            d = cube_distance(axial_to_cube(q, r), c_cube)
                            if abs(d - radius) <= band:
                                base = orig[(q, r)]
                                if style == 'tint': grid.set(q, r, lerp_color(base, tint, power))
                                else: grid.set(q, r, brighten(base, power))
            elif motion.kind in ('divide', 'fuse'):
                aq, ar = motion.line_anchor
                n1, _n2 = motion.norm_dirs
                if motion.kind == 'divide':
                    band = float(CONFIG["divide_band_thickness"]); blend = float(CONFIG["divide_blend"]); ppos = p
                else:
                    band = float(CONFIG["fuse_band_thickness"]); blend = float(CONFIG["fuse_blend"]); ppos = 1.0 - p
                extent = float(motion.cross_extent); pos_center = ppos * extent; neg_center = -pos_center
                orig = {(q, r): grid.get(q, r) for r in range(rows) for q in range(cols)}
                for r in range(rows):
                    for q in range(cols):
                        dq = q - aq; dr = r - ar
                        cpos1 = dot_axial((dq, dr), n1)
                        near_pos = abs(cpos1 - pos_center) <= band
                        near_neg = abs(cpos1 - neg_center) <= band
                        if not (near_pos or near_neg): continue
                        sign = 1 if cpos1 >= 0 else -1
                        if motion.kind == 'fuse': sign *= -1
                        step = n1 if sign > 0 else (-n1[0], -n1[1])
                        dst_q, dst_r = grid.wrap(q + step[0], r + step[1])
                        src = orig[(q, r)]; dst = grid.get(dst_q, dst_r)
                        grid.set(dst_q, dst_r, lerp_color(dst, src, blend))
            elif motion.kind == 'align':
                rows_local = rows
                front = int(p * (rows_local - 1))
                stride = max(1, int(CONFIG["align_stride_cols"]))
                override_center = bool(CONFIG["align_override_center"])
                skip_if_few = bool(CONFIG["align_skip_if_few_neighbours"])
                orig = {(q, r): grid.get(q, r) for r in range(rows_local) for q in range(cols)}
                for r in range(front + 1):
                    for q in range(0, cols, stride):
                        neigh = []
                        for (dq, dr) in HEX_DIRS:
                            nq, nr = q + dq, r + dr
                            if 0 <= nq < cols and 0 <= nr < rows_local:
                                neigh.append((nq, nr))
                        if skip_if_few and len(neigh) < 4: continue
                        neigh_cols = [orig[(nq, nr)] for (nq, nr) in neigh]
                        if not neigh_cols: continue
                        avg = average_color(neigh_cols)
                        maxd, best_idx = -1.0, -1
                        for idx, (nq, nr) in enumerate(neigh):
                            c = orig[(nq, nr)]
                            d = color_distance(c, (avg[0], avg[1], avg[2]))
                            if d > maxd: maxd, best_idx = d, idx
                        if best_idx >= 0:
                            target_col = orig[neigh[best_idx]]
                            for (nq, nr) in neigh: grid.set(nq, nr, target_col)
                            if override_center: grid.set(q, r, target_col)
            elif motion.kind == 'scatter':
                sparks_lo, sparks_hi = CONFIG["scatter_sparks_per_seed_per_frame"]
                dmin, dmax = CONFIG["scatter_distance_range"]; fade = float(CONFIG["scatter_fade"])
                for (sq, sr, scol) in motion.scatter_seeds:
                    sparks = random.randint(sparks_lo, sparks_hi)
                    for _ in range(sparks):
                        direction = random.choice(HEX_DIRS)
                        dist = random.randint(dmin, dmax)
                        tq, tr = sq + direction[0]*dist, sr + direction[1]*dist
                        if 0 <= tq < cols and 0 <= tr < rows:
                            base = grid.get(tq, tr); grid.set(tq, tr, lerp_color(base, scol, fade))

        if not paused:
            fade_mode = FADE_MODES[fade_mode_index]
            fade_rate = clamp(float(CONFIG.get("fade_revert_rate", 0.2)), 0.0, 1.0)
            if fade_rate > 0.0 and fade_mode != "none":
                if fade_mode == "blank":
                    target = CONFIG["base_color"]
                    for r in range(rows):
                        for q in range(cols):
                            current = grid.get(q, r)
                            if current != target:
                                grid.set(q, r, lerp_color(current, target, fade_rate))
                elif fade_mode == "image" and image_list:
                    target_data = get_image_target(image_list[image_idx])
                    if target_data is not None:
                        for r in range(rows):
                            row_target = target_data[r]
                            for q in range(cols):
                                target = row_target[q]
                                current = grid.get(q, r)
                                if current != target:
                                    grid.set(q, r, lerp_color(current, target, fade_rate))

        # Draw
        screen.fill((0, 0, 0))
        side = hex_radius - 0.7
        for r in range(rows):
            for q in range(cols):
                cx, cy = axial_to_pixel(q, r, hex_radius, origin)
                color = grid.get(q, r)
                pts = polygon_hex((cx, cy), side)
                pygame.draw.polygon(screen, color, pts)
                if show_grid: pygame.draw.polygon(screen, (28, 28, 30), pts, 1)

        # Draw overlays
        if help_visible:
            _draw_help_overlay(screen, CONFIG["width"], CONFIG["height"], CONFIG.get("font_name"))

        # HUD / Recording IO
        frame_surf = pygame.display.get_surface()
        scale = CONFIG["record_scale"]
        surf_for_io = frame_surf if abs(scale - 1.0) < 1e-6 else pygame.transform.smoothscale(
            frame_surf, (int(CONFIG["width"] * scale), int(CONFIG["height"] * scale))
        )
        if record_png:
            os.makedirs(CONFIG["record_dir"], exist_ok=True)
            ts = int(time.time() * 1000)
            out_path = os.path.join(CONFIG["record_dir"], f"f_{ts}.png")
            pygame.image.save(surf_for_io, out_path)
        if ffmpeg_proc is not None:
            try:
                frame_bytes = pygame.image.tostring(surf_for_io, "RGB")
                ffmpeg_proc.stdin.write(frame_bytes)
            except BrokenPipeError:
                try: ffmpeg_proc.stdin.close()
                except Exception: pass
                try: ffmpeg_proc.wait()
                except Exception: pass
                ffmpeg_proc = None

        if debug:
            fps = clock.get_fps()
            rec = []
            if record_png: rec.append("PNG")
            if ffmpeg_proc is not None: rec.append("FFMPEG")
            fade_mode = FADE_MODES[fade_mode_index]
            fade_desc = describe_fade_mode(fade_mode)
            if fade_mode == "image" and image_list and get_image_target(image_list[image_idx]) is None:
                fade_desc += "*"
            txt = f"FPS:{fps:5.1f}  MT:{'ON' if mouse_target_mode else 'OFF'}  FD:{fade_desc}"
            if rec: txt += " REC[" + "+".join(rec) + "]"
            pygame.display.get_surface().blit(pygame.font.SysFont("monospace", 14).render(txt, True, (230, 230, 235)), (12, 10))

        pygame.display.flip(); clock.tick(CONFIG["fps"])

    if ffmpeg_proc is not None:
        try: ffmpeg_proc.stdin.close()
        except Exception: pass
        try: ffmpeg_proc.wait()
        except Exception: pass
    pygame.quit()


# Load default config immediately so imported helpers can rely on CONFIG.
prepare_runtime_config()

if __name__ == "__main__":
    args = parse_args()
    prepare_runtime_config(args.config, args.output_dir, args.words_file)
    main()
