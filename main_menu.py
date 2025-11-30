#!/usr/bin/env python3
"""Graphical launcher for the ScreenGlitchVid toolbox."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from app_core.projects import ProjectManager
from app_core.settings import load_settings


BUTTON_IMAGE_SIZE = (220, 120)


class VerticalScrollFrame(ttk.Frame):
    """Scrollable container that provides vertical scrolling for its child frame."""

    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._resize_inner)

    def _update_scrollregion(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_inner(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)


class ToolLauncherFrame(ttk.Frame):
    """Reusable widget that renders a launch button and settings panel."""

    def __init__(self, master, app: "ToolboxApp", tool_info: dict):
        super().__init__(master, padding=(0, 6, 0, 6))
        self.app = app
        self.tool_info = tool_info
        self.use_project_var = tk.BooleanVar(value=True)
        self.config_path = tk.StringVar(value=str(tool_info.get("default_config", "")))
        self.status_var = tk.StringVar(value="Using project settings")
        self._build()
        self.update_state()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.button_image = self.app.get_tool_button_image(self.tool_info)
        self.launch_btn = ttk.Button(
            self,
            text="",
            image=self.button_image,
            compound="center",
            command=self.launch_tool,
            width=0,
        )
        self.launch_btn.grid(row=0, column=0, rowspan=6, padx=(0, 12), sticky="n")

        info = ttk.Frame(self)
        info.grid(row=0, column=1, sticky="nsew")
        info.columnconfigure(0, weight=1)

        ttk.Label(info, text=self.tool_info["name"], style="ToolTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(info, text=self.tool_info.get("description", ""), wraplength=520, justify="left").grid(
            row=1, column=0, sticky="w", pady=(2, 6)
        )

        if self.tool_info.get("supports_project_settings", False):
            ttk.Checkbutton(
                info,
                text="Use project settings",
                variable=self.use_project_var,
                command=self.update_state,
            ).grid(row=2, column=0, sticky="w")

        ttk.Label(info, textvariable=self.status_var, foreground="#4a4a4a").grid(
            row=3, column=0, sticky="w", pady=(2, 4)
        )

        if self.tool_info.get("requires_config", False):
            self.config_frame = ttk.Frame(info)
            self.config_frame.grid(row=4, column=0, sticky="ew", pady=(4, 0))
            self.config_frame.columnconfigure(1, weight=1)
            ttk.Label(self.config_frame, text="Config file:").grid(row=0, column=0, sticky="w")
            self.config_entry = ttk.Entry(self.config_frame, textvariable=self.config_path, width=50)
            self.config_entry.grid(row=0, column=1, sticky="ew", padx=(6, 6))
            self.browse_btn = ttk.Button(self.config_frame, text="Browse", command=self.browse_config)
            self.browse_btn.grid(row=0, column=2, sticky="e")
        else:
            self.config_frame = None
            self.config_entry = None
            self.browse_btn = None

    # ------------------------------------------------------------------
    def browse_config(self) -> None:
        initial_dir = self.app.current_project.root if self.app.current_project else self.app.repo_root
        file_path = filedialog.askopenfilename(
            title="Select config file",
            initialdir=str(initial_dir),
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if file_path:
            self.config_path.set(file_path)

    # ------------------------------------------------------------------
    def update_state(self) -> None:
        project = self.app.current_project
        allow_project = self.tool_info.get("supports_project_settings", False) and project is not None
        if not allow_project:
            self.use_project_var.set(False)
        use_project = self.use_project_var.get() and allow_project

        if self.tool_info.get("requires_config", False) and self.config_entry and self.browse_btn:
            if use_project:
                config_path = self.get_project_config_path()
                self.config_path.set(str(config_path))
                state = "disabled"
            else:
                if not self.config_path.get():
                    self.config_path.set(str(self.tool_info.get("default_config", "")))
                state = "normal"
            self.config_entry.configure(state=state)
            self.browse_btn.configure(state=state)
            if self.config_frame:
                if use_project:
                    self.config_frame.grid_remove()
                else:
                    self.config_frame.grid()

        if use_project:
            self.status_var.set("Using selected project context")
        elif project is None:
            self.status_var.set("No project selected — standalone mode")
        else:
            self.status_var.set("Standalone mode (custom config)")

    # ------------------------------------------------------------------
    def get_project_config_path(self) -> Path:
        project = self.app.current_project
        internal_parts = self.tool_info.get("project_internal_path", [])
        config_name = self.tool_info.get("project_config_name", "config.json")
        if not project:
            return Path(self.config_path.get())
        path = project.internal_dir
        for part in internal_parts:
            path /= part
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / config_name
        config_path.parent.mkdir(parents=True, exist_ok=True)
        return config_path

    # ------------------------------------------------------------------
    def get_project_output_dir(self) -> Path | None:
        project = self.app.current_project
        if not project:
            return None
        internal_parts = self.tool_info.get("project_internal_path", [])
        path = project.internal_dir
        for part in internal_parts:
            path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    def launch_tool(self) -> None:
        script = Path(self.tool_info["script"])
        if not script.exists():
            messagebox.showerror("Missing tool", f"Could not find tool script: {script}")
            return

        use_project = self.use_project_var.get() and self.app.current_project is not None
        project = self.app.current_project if use_project else None

        cmd = [sys.executable, str(script)]
        if self.tool_info.get("requires_config", False):
            config_path = Path(self.config_path.get()).expanduser()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--config", str(config_path)])

        if project and self.tool_info.get("uses_output_dir"):
            output_dir = self.get_project_output_dir()
            if output_dir:
                cmd.extend(["--output-dir", str(output_dir)])

        if project and self.tool_info.get("passes_project", False):
            cmd.extend(["--project", str(project.root)])
            cmd.extend(["--project-name", project.name])

        try:
            subprocess.Popen(cmd, cwd=self.app.repo_root)
        except OSError as exc:
            messagebox.showerror("Launch failed", f"Unable to start tool:\n{exc}")

    # ------------------------------------------------------------------
    def refresh_project(self) -> None:
        self.update_state()


class ToolboxApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ScreenGlitchVid Toolbox")
        self.geometry("1080x780")
        self.repo_root = Path(__file__).resolve().parent
        self.settings = load_settings()
        self.project_manager = ProjectManager(self.repo_root, self.settings)
        self.current_project = self.project_manager.get_last_selected_project()
        self.tool_frames: list[ToolLauncherFrame] = []
        self.button_image_cache: dict[str, tk.PhotoImage] = {}
        self._blank_button_image: tk.PhotoImage | None = None

        style = ttk.Style(self)
        style.configure("ToolTitle.TLabel", font=("Segoe UI", 12, "bold"))

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=20)
        container.pack(fill="both", expand=True)

        self._build_project_picker(container)
        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=12)
        self._build_summary(container)
        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=12)
        self._build_tools(container)

    # ------------------------------------------------------------------
    def _build_project_picker(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 8))

        ttk.Label(frame, text="Project:").grid(row=0, column=0, sticky="w")
        self.project_var = tk.StringVar()
        self.project_combo = ttk.Combobox(frame, textvariable=self.project_var, state="readonly")
        self.project_combo.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        frame.columnconfigure(1, weight=1)
        self.project_combo.bind("<<ComboboxSelected>>", self.on_project_selected)

        ttk.Button(frame, text="New project", command=self.create_project).grid(row=0, column=2)
        self.refresh_project_list()

    # ------------------------------------------------------------------
    def refresh_project_list(self) -> None:
        projects = self.project_manager.list_projects()
        self.project_display = []
        self.display_to_slug: dict[str, str] = {}
        for project in projects:
            label = f"{project.name} ({project.slug})"
            self.project_display.append(label)
            self.display_to_slug[label] = project.slug
        self.project_combo.configure(values=self.project_display)
        if self.current_project:
            match = next(
                (label for label, slug in self.display_to_slug.items() if slug == self.current_project.slug),
                None,
            )
            if match:
                self.project_var.set(match)
        elif self.project_display:
            self.project_var.set(self.project_display[0])
            self.on_project_selected()

    # ------------------------------------------------------------------
    def on_project_selected(self, *_args) -> None:
        label = self.project_var.get()
        slug = self.display_to_slug.get(label)
        if not slug:
            return
        project = self.project_manager.get_project(slug)
        if not project:
            messagebox.showerror("Missing project", f"Project '{slug}' could not be loaded")
            return
        self.current_project = project
        self.project_manager.set_last_selected(project.slug)
        self.update_summary()
        for frame in self.tool_frames:
            frame.refresh_project()

    # ------------------------------------------------------------------
    def create_project(self) -> None:
        name = simpledialog.askstring("Create project", "Project name:", parent=self)
        if not name:
            return
        try:
            project = self.project_manager.create_project(name)
        except ValueError as exc:
            messagebox.showerror("Invalid name", str(exc))
            return
        self.current_project = project
        self.refresh_project_list()
        self.update_summary()
        for frame in self.tool_frames:
            frame.refresh_project()

    # ------------------------------------------------------------------
    def _build_summary(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="")
        frame.pack(fill="x")
        labels = [("assets", "Assets")]
        self.summary_vars: dict[str, tk.StringVar] = {}
        for row, (key, title) in enumerate(labels):
            ttk.Label(frame, text=f"{title}:").grid(row=row, column=0, sticky="w", padx=(6, 6), pady=2)
            var = tk.StringVar(value="-")
            self.summary_vars[key] = var
            ttk.Label(frame, textvariable=var).grid(row=row, column=1, sticky="w", pady=2)

        self.update_summary()

    # ------------------------------------------------------------------
    def update_summary(self) -> None:
        if not self.current_project:
            for var in self.summary_vars.values():
                var.set("No project selected")
            return
        summary = self.project_manager.summarize_project(self.current_project)
        for key, var in self.summary_vars.items():
            var.set(summary.get(key, "-"))

    # ------------------------------------------------------------------
    def _build_tools(self, parent: ttk.Frame) -> None:
        tools_frame = ttk.Frame(parent)
        tools_frame.pack(fill="both", expand=True)

        scroll = VerticalScrollFrame(tools_frame)
        scroll.pack(fill="both", expand=True)
        scroll.inner.columnconfigure(0, weight=1)

        sections = [
            ("Analysis", self.analysis_tools()),
            ("Video", self.video_tools()),
        ]
        for idx, (title, tools) in enumerate(sections):
            section = ttk.LabelFrame(scroll.inner, text=title, padding=(12, 8))
            pady = (0, 0) if idx == len(sections) - 1 else (0, 12)
            section.grid(row=idx, column=0, sticky="ew", pady=pady)
            section.columnconfigure(0, weight=1)
            for tool_info in tools:
                frame = ToolLauncherFrame(section, self, tool_info)
                frame.pack(fill="x", padx=4, pady=4)
                self.tool_frames.append(frame)

    # ------------------------------------------------------------------
    def analysis_tools(self) -> list[dict]:
        return [
            {
                "key": "audio_map",
                "name": "Create Basic Audio Map",
                "button_label": "Launch audio map tool",
                "description": "Load a song and build tempo/timing markers.",
                "script": self.repo_root / "tools/analysis/create_basic_audio_map/tool.py",
                "button_image": self.repo_root / "assets/ui/button_images/audio_map.png",
                "supports_project_settings": True,
                "requires_config": False,
                "passes_project": True,
            }
        ]

    # ------------------------------------------------------------------
    def video_tools(self) -> list[dict]:
        return [
            {
                "key": "hex_glitch",
                "name": "Hex Glitch",
                "button_label": "Launch Hex Glitch",
                "description": "Realtime hex-grid glitch animator with capture + overlays.",
                "script": self.repo_root / "tools/video/hex_glitch/hex_glitch.py",
                "button_image": self.repo_root / "assets/ui/button_images/hex_glitch.png",
                "supports_project_settings": True,
                "requires_config": True,
                "default_config": self.repo_root / "tools/video/hex_glitch/config.json",
                "project_internal_path": ("video", "hex_glitch"),
                "project_config_name": "config.json",
                "uses_output_dir": True,
            },
            {
                "key": "parallax_playground",
                "name": "Parallax Playground",
                "button_label": "Launch parallax playground",
                "description": "Create background scenes with parallax and perspective motion effects.",
                "script": self.repo_root / "tools/video/parallax_playground/parallax_playground.py",
                "button_image": self.repo_root / "assets/ui/button_images/paralax.png",
                "supports_project_settings": True,
                "requires_config": False,
            },
            {
                "key": "timed_action_mixer",
                "name": "Timed Action Mixer",
                "button_label": "Launch timed action mixer",
                "description": "Associate timing tracks with fireworks + sprite pop overlays and render video clips.",
                "script": self.repo_root / "tools/video/timed_action_mixer/timed_action_mixer.py",
                "button_image": self.repo_root / "assets/ui/button_images/timed_action_mixer.png",
                "supports_project_settings": True,
                "requires_config": False,
                "project_internal_path": ("video", "timed_action_mixer"),
                "uses_output_dir": True,
                "passes_project": True,
            }
        ]

    # ------------------------------------------------------------------
    def get_tool_button_image(self, tool_info: dict) -> tk.PhotoImage:
        key = tool_info["key"]
        if key in self.button_image_cache:
            return self.button_image_cache[key]

        path = Path(tool_info.get("button_image", ""))
        photo: tk.PhotoImage
        if path.exists():
            try:
                photo = tk.PhotoImage(file=str(path))
            except tk.TclError:
                photo = self._get_blank_button_image()
        else:
            default_path = self.repo_root / "assets/ui/button_images/default.png"
            if default_path.exists():
                try:
                    photo = tk.PhotoImage(file=str(default_path))
                except tk.TclError:
                    photo = self._get_blank_button_image()
            else:
                photo = self._get_blank_button_image()

        self.button_image_cache[key] = photo
        return photo

    def _get_blank_button_image(self) -> tk.PhotoImage:
        if self._blank_button_image is not None:
            return self._blank_button_image
        width, height = BUTTON_IMAGE_SIZE
        img = tk.PhotoImage(width=width, height=height)
        bg_color = "#1f1f1f"
        border_color = "#3a3a3a"
        img.put(bg_color, to=(0, 0, width, height))
        img.put(border_color, to=(0, 0, width, 1))
        img.put(border_color, to=(0, height - 1, width, height))
        img.put(border_color, to=(0, 0, 1, height))
        img.put(border_color, to=(width - 1, 0, width, height))
        self._blank_button_image = img
        return img
    def run(self) -> None:
        self.mainloop()


if __name__ == "__main__":
    app = ToolboxApp()
    app.run()
