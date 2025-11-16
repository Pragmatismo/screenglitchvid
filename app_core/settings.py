"""Helpers for loading and working with global toolbox settings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_SETTINGS: Dict[str, Any] = {
    "data_dir": "data",
    "projects_dir": "assets/projects",
    "project_index": "data/projects.json",
}


def load_settings(settings_path: Path | None = None) -> Dict[str, Any]:
    """Load the settings file if it exists, otherwise return defaults."""
    data: Dict[str, Any] = dict(DEFAULT_SETTINGS)
    if settings_path and settings_path.exists():
        try:
            loaded = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse settings file {settings_path}: {exc}") from exc
        data.update(loaded)
    return data


__all__ = ["DEFAULT_SETTINGS", "load_settings"]
