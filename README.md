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
timeline with zoom controls, scrollbars with auto-pan as the playhead approaches the edge, one-second/ten-second tick
marks, and a detail grid for precise edits.  Click anywhere (without modifiers) to move the playhead instantly.  Hold
**Shift** or enable the **Edit mode** toggle to manipulate markers: drag them along the timeline, Shift+double-click to
add events, and Shift+right-click to remove the clicked marker.  Duration events are drawn as full-width rectangles
spanning their entire time range so you can see overlaps at a glance.  A Play button streams the song (via
`pygame.mixer`) so you can preview the markers in sync while the playhead keeps itself in view, and the Analysis Settings
dialog lets you tweak hop length, tempo tightness, RMS/pitch thresholds, and onset behaviour before re-running the
detector.  Selecting a marker also exposes inline fields for editing its time/value/duration without leaving the event
list.

Timing data is stored as JSON inside each project’s `internal/timing/` folder.  Every timing file is a single document
that can bundle any number of interchangeable tracks—automatic analysis passes, MIDI taps, manual annotations, or other
sources.  Each track is simply a list of timestamped events so downstream tools can select whichever track they require
without worrying about its origin.  The on-disk structure is intentionally small and friendly to other tools:

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

**Key integration notes**

- `time` and `duration` are always expressed in seconds relative to the audio’s start.  Tools that operate in beats can
  convert using their own BPM maps while keeping the same timestamps for cross-sync.
- `value` is optional and can represent any scalar (intensity, confidence, velocity…).  Consumers that do not care about
  it can simply ignore the field.
- `label` is a free-form string that callers can use for grouping (e.g., `impact`, `verse`, `drop`).
- `data` can hold additional per-event metadata (IDs, colour hints, etc.) without changing the schema.  Unknown keys
  should be preserved if a tool rewrites the file so collaborative workflows remain intact.

Because every track is represented with the exact same schema, you can mix and match sources freely: copy MIDI-derived
beats into the same file as an automatically-detected onset track, add a “pyro_cues” list recorded live, or maintain
multiple variations for experimentation.  Downstream programs only need to prompt the user for a timing file, enumerate
`tracks`, and read the `events` list for the chosen entry.  See
`assets/projects/playground/internal/timing/example_showcase.timing.json` for a concrete reference file that demonstrates
beats, one-shot FX triggers, and long-form section markers side by side.

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
