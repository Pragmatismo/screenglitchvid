# 🌀 Hex Glitch
### Generative glitch-visual tool for music videos and sci-fi overlays
**by [Jumble Sale of Stimuli](https://www.youtube.com/@JumbleSaleOfStimuli)**  
**Repository:** [Pragmatismo/screenglitchvid](https://github.com/Pragmatismo/screenglitchvid)  
**Created with ChatGPT-5 · Open Source**

---

## ✨ Overview

**Hex Glitch** (`hex_glitch.py`) is a real-time **generative art** program that paints and mutates color fields across a hexagonal grid, creating continuously evolving glitch patterns.

It’s designed for producing **short video clips or live overlays** — ideal for use in **sci-fi music videos**, **projection art**, or **VJ loops**.

Features include:

- Real-time animation in `pygame`
- Dozens of reactive visual effects (waves, quakes, divides, halos, etc.)
- Word and image compositing
- PNG sequence or live FFmpeg recording to MP4
- On-screen help (F2) and mouse-target toggle (F9)
- Fully open-source and editable

---

## 🧰 Installation (Linux)

You’ll need **Python 3.10+** and `venv` (included with Python).

```bash
# 1. Clone the repository
git clone https://github.com/Pragmatismo/screenglitchvid.git
cd screenglitchvid

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install pygame pillow
```

---

## ▶️ Running

```bash
python3 hex_glitch.py
```

A window will open showing the animated hex grid.  
Press **F2** for an on-screen list of all key controls.

---

## 🎹 Key Controls

| Key | Action |
|-----|--------|
| **Space** | Pause / resume animation |
| **R** | Reset grid and reseed random colors |
| **F2** | Show / hide help overlay |
| **F3** | Toggle debug info (FPS, record status) |
| **F4** | Save screenshot (`frames/`) |
| **F5** | Toggle PNG sequence capture (`frames_out/`) |
| **F6** | Start / stop live FFmpeg MP4 recording |
| **F9** | Toggle *Mouse-target mode* (affects Grow, Halo, Erupt, Divide, Fuse) |
| **I / O / P** | Overlay current / half-opacity / next image |
| **T** | Read and stamp words from `vidtext.txt` |
| **Y / U** | Full / half-opacity color wash from random edge |
| **;** | Fire a dashed “edge shot” line |
| **Q / W / E** | Quake • Waves • Erupt |
| **G / H** | Grow • Halo |
| **A / S** | Align • Scatter |
| **D / F** | Divide • Fuse |
| **J / K / L** | Jump • Kick • Leap |
| **Mouse click** | Paint random color at cursor |
| **Esc** | Quit program |

---

## 🎬 Recording and Output

There are two recording modes:

- **PNG Sequence:**  
  Press **F5** — each frame is saved as a numbered PNG in `frames_out/`.

- **FFmpeg MP4:**  
  Press **F6** — live stream frames to FFmpeg, producing a compressed video file.

### ⚡ Playback Speed
Recorded videos will likely **play faster than the live view** — this is expected.  
Rendering runs at a fixed frame rate (default 60 FPS) while real-time preview speed may vary.  
In post-production, **adjust the playback speed** to match your song’s rhythm or desired pacing.

---

## 🧠 Creative Use

Use the generated clips as **overlays** or **backgrounds** in:
- Sci-fi control panels and HUD screens  
- Music video scenes  
- VJ loops or projection mapping installations  

Combine with AI-generated overlays such as the *Retro Spacecraft HUD* Sora prompt for enhanced digital depth.

"
    Black background — futuristic spacecraft HUD (heads-up display) rendered in retro digital style.
Bright, saturated neon vector lines (cyan, amber, magenta, lime, and white) arranged in a blocky, old-computer layout. simple thick lines.

text labels “NAV” and “THRUST” in a blocky mono font.

Include rectangular data panels, radar grid arcs, vector crosshairs, numeric telemetry rows, rotating target markers, and ASCII-style bar graphs.

Interface should look glowing and crisp, with slight CRT scanline texture.
Layout symmetrical, but cluttered with readouts and segmented frames like a 1980s starship command terminal.

No background scene — pure black canvas
Style keywords: digital vector art, luminous edges, sci-fi instrumentation, minimal color palette, pixel precision, high contrast.
"

---

## 💡 Technical Notes

- Written in **Python 3** using **Pygame** (display) and **Pillow** (image I/O)  
- Linux-focused but works cross-platform with SDL2  
- Open-source, hackable, and community-driven  
- Developed collaboratively with **ChatGPT-5**

---

## 📽️ Project Context

**Hex Glitch** is part of the *Jumble Sale of Stimuli* open-source art series —  
a creative experiment exploring how humans and AI can **co-create generative tools** for music, visuals, and storytelling.

---

## 🪐 License

Released under a permissive open-source license.  
You are free to modify, remix, and use the output in your own creative projects.  
Credit to **Jumble Sale of Stimuli** is appreciated.

---

> 💬 *“This code isn’t about perfection — it’s about play.  
> Every glitch, ripple, and pulse is an invitation to create.”*  
> — *Jumble Sale of Stimuli*

There will be a music video using this tool on the youtube page soon! 
