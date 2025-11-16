#!/usr/bin/env python3
"""Placeholder UI for the "Create Basic Audio Map" analysis tool."""
from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import ttk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder audio analysis tool")
    parser.add_argument("--project", help="Active project directory", default=None)
    parser.add_argument("--project-name", help="Active project name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    title = "Create Basic Audio Map"
    if args.project_name:
        title += f" — {args.project_name}"
    root.title(title)
    root.geometry("420x200")

    frm = ttk.Frame(root, padding=20)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Work in progress", font=("Segoe UI", 18, "bold")).pack(pady=(10, 6))
    ttk.Label(
        frm,
        text=(
            "This placeholder will eventually load a song, analyse BPM, and "
            "export timing markers for other tools."
        ),
        wraplength=360,
        justify="center",
    ).pack(pady=(0, 12))

    if args.project:
        ttk.Label(frm, text=f"Project folder:\n{args.project}", justify="center").pack()

    ttk.Button(frm, text="Close", command=root.destroy).pack(pady=(18, 0))

    root.mainloop()


if __name__ == "__main__":
    main()
