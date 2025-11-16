# 🎛️ ScreenGlitchVid Toolbox

ScreenGlitchVid is evolving from a single-purpose glitch visualizer into a toolbox of creative helpers that share
projects, assets, and settings.  The new launcher provides a home base for organising work, spinning up placeholder
analysis tools, and jumping into production-ready video utilities like Hex Glitch.

---

## 🚀 Quick start

```bash
# 1. Clone the repository
git clone https://github.com/Pragmatismo/screenglitchvid.git
cd screenglitchvid

# 2. Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install pygame pillow numpy soundfile librosa
```

### Launch the toolbox menu

```bash
python main_menu.py
```

From here you can select/create projects, review their asset folders, and launch individual tools with a single click.
Each tool exposes a “Use project settings” toggle: when enabled the launcher injects project-specific config files and
output directories so clips, timing charts, and derived assets stay sandboxed.

### Run tools directly (optional)
All tools remain runnable as standalone scripts.  For example:

```bash
python tools/video/hex_glitch/hex_glitch.py --config tools/video/hex_glitch/config.json
```

Supply `--output-dir` to override recording/screenshot folders when working outside the menu.

---

## 🗂️ Project structure

```
screenglitchvid/
├── assets/
│   └── projects/
│       └── <project>/
│           ├── assets/          # user-imported media (audio, images, clips, etc.)
│           └── internal/
│               ├── timing/      # analysis outputs (BPM maps, marker JSON, ...)
│               └── video/
│                   └── hex_glitch/   # per-project configs + renders for Hex Glitch
├── data/
│   └── projects.json           # registered projects + last selection
├── tools/
│   ├── analysis/...
│   └── video/...
└── main_menu.py
```

The launcher persists project metadata in `data/projects.json` and scaffolds folders on demand.  Shared helper code lives
in `app_core/` (project management + future settings helpers).

---

## 🛠️ Available tools

### Analysis — Create Basic Audio Map
*Path:* `tools/analysis/create_basic_audio_map/tool.py`

The Create Basic Audio Map tool ingests any audio file, runs a multi-track analysis pass (librosa-based beat/onset/pitch
detectors plus high-energy windows), and then drops you into an interactive editor.  Each track is rendered on a stacked
timeline with draggable markers, zoom controls, playhead preview, and a detail grid for precise edits.  Add/remove tracks
for user annotations, double-click to create new events, or rename tracks so other programs can reference them directly.

Timing data is stored as JSON inside each project’s `internal/timing/` folder using the following schema:

```jsonc
{
  "version": 1,
  "duration": 123.4,
  "tracks": {
    "track_name": {
      "description": "optional",
      "events": [
        { "time": 0.0, "value": 1.0, "label": "beat", "duration": 0.0 }
      ]
    }
  }
}
```

Every track is functionally identical (a list of `{time, value?, label?, duration?}` events) so downstream tools can simply
pick the track that best matches their needs, whether it originated from automatic analysis, MIDI taps, or manual notes.
See `assets/projects/playground/internal/timing/example_showcase.timing.json` for a concrete reference file.

### Video — Hex Glitch
*Path:* `tools/video/hex_glitch/hex_glitch.py`

Hex Glitch is the original generative glitch-visual tool by **Jumble Sale of Stimuli**.  It paints and mutates colour
fields across a hexagon grid using dozens of propagation modes, overlays, and recording workflows.

When launched from the menu with project settings enabled the tool automatically:
- Loads the project-specific config file in `internal/video/hex_glitch/config.json` (created on demand).
- Redirects PNG/FFmpeg output to the same folder so renders stay grouped with the project.
- Resolves relative asset paths (images, `vidtext` word lists, etc.) relative to the chosen config file.

You can still run it directly for experiments by choosing a config and optional output directory via CLI flags.

#### Controls recap

| Key | Action |
|-----|--------|
| **Space** | Pause / resume animation |
| **R** | Reset grid and reseed random colours |
| **F2** | Show / hide help overlay |
| **F3** | Toggle debug info (FPS, record status) |
| **F4** | Save screenshot (`save_frames_dir`) |
| **F5** | Toggle PNG sequence capture (`record_dir`) |
| **F6** | Start / stop live FFmpeg MP4 recording |
| **F9** | Toggle mouse-target mode |
| **I / O / P** | Overlay current / half-opacity / next image |
| **T** | Stamp words from the configured words file |
| **Y / U** | Full / half-opacity colour wash from random edge |
| **;** | Fire a dashed “edge shot” line |
| **Q / W / E** | Quake • Waves • Erupt |
| **G / H** | Grow • Halo |
| **A / S** | Align • Scatter |
| **D / F** | Divide • Fuse |
| **J / K / L** | Jump • Kick • Leap |
| **Mouse click** | Paint random colour at cursor |
| **Esc** | Quit program |

#### Recording workflows
- **PNG sequence** — press **F5** to dump frames to the configured `record_dir`.
- **FFmpeg MP4** — press **F6** to stream frames into FFmpeg with the quality/fps options specified in your config.
  Playback is typically faster than the live preview, so retime clips in your editor to taste.

#### Creative uses
Use renders as overlays in sci-fi music videos, projection mapping, VJ sets, or motion-graphics HUD inserts.  Combine
with AI-generated elements for even richer compositions.

---

## 🧠 Technical notes
- Python 3.10+
- Pygame for rendering, Pillow for image manipulation
- Tkinter for the launcher + placeholder tools
- All code is open-source and designed for experimentation; copy the configs into each project to craft bespoke looks.

Have fun glitching!  Contributions and new tool ideas are always welcome.
