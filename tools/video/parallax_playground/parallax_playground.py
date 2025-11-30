#!/usr/bin/env python3
"""Placeholder launcher for the Parallax Playground tool."""
from __future__ import annotations

import tkinter as tk


class ParallaxPlaygroundApp(tk.Tk):
    """Simple placeholder UI for the parallax playground."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Parallax Playground")
        self.geometry("520x320")
        self._build_ui()

    def _build_ui(self) -> None:
        container = tk.Frame(self, padx=24, pady=24)
        container.pack(fill="both", expand=True)

        tk.Label(container, text="Parallax Playground", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", pady=(0, 12)
        )
        tk.Label(
            container,
            text=(
                "Create background scenes with perspective and parallax motion.\n"
                "Feature development is underway — check back soon!"
            ),
            justify="left",
            anchor="w",
        ).pack(anchor="w")

        spacer = tk.Frame(container)
        spacer.pack(fill="both", expand=True)

        tk.Label(container, text="Coming soon", font=("Segoe UI", 12, "italic"), fg="#666").pack(
            anchor="center", pady=(8, 0)
        )


if __name__ == "__main__":
    app = ParallaxPlaygroundApp()
    app.mainloop()
