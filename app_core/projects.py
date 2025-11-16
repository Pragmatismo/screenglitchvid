"""Project management utilities for the toolbox menu."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProjectInfo:
    name: str
    slug: str
    created: str
    root: Path
    assets_dir: Path
    internal_dir: Path

    def ensure_folders(self) -> None:
        """Ensure that all required directories exist for the project."""
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.internal_dir.mkdir(parents=True, exist_ok=True)
        # Known sub-areas used by early tools
        (self.internal_dir / "timing").mkdir(parents=True, exist_ok=True)
        (self.internal_dir / "video").mkdir(parents=True, exist_ok=True)

    def tool_subdir(self, *parts: str) -> Path:
        """Return a path inside the internal folder for a specific tool."""
        path = self.internal_dir
        for part in parts:
            path /= part
        path.mkdir(parents=True, exist_ok=True)
        return path


class ProjectManager:
    """Persist and expose information about toolbox projects."""

    def __init__(self, repo_root: Path, settings: Optional[Dict[str, str]] = None) -> None:
        self.repo_root = repo_root
        self.settings = settings or {}
        projects_dir = self.settings.get("projects_dir", "assets/projects")
        data_dir = self.settings.get("data_dir", "data")
        index_file = self.settings.get("project_index", "data/projects.json")

        self.projects_root = (repo_root / projects_dir).resolve()
        self.data_dir = (repo_root / data_dir).resolve()
        self.projects_file = (repo_root / index_file).resolve()

        self.projects_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._meta = self._load_metadata()
        if not self._meta["projects"]:
            default = self.create_project("Playground")
            self.set_last_selected(default.slug)

    # ------------------------------------------------------------------
    def _load_metadata(self) -> Dict:
        if self.projects_file.exists():
            try:
                return json.loads(self.projects_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {"projects": [], "last_selected": None}

    def _save_metadata(self) -> None:
        payload = json.dumps(self._meta, indent=2)
        self.projects_file.write_text(payload, encoding="utf-8")

    def _build_info(self, entry: Dict[str, str]) -> ProjectInfo:
        slug = entry["slug"]
        root = self.projects_root / slug
        info = ProjectInfo(
            name=entry["name"],
            slug=slug,
            created=entry.get("created", datetime.utcnow().isoformat()),
            root=root,
            assets_dir=root / "assets",
            internal_dir=root / "internal",
        )
        info.ensure_folders()
        return info

    def list_projects(self) -> List[ProjectInfo]:
        return [self._build_info(p) for p in self._meta["projects"]]

    def get_project(self, slug: Optional[str]) -> Optional[ProjectInfo]:
        if not slug:
            return None
        for entry in self._meta["projects"]:
            if entry["slug"] == slug:
                return self._build_info(entry)
        return None

    def get_last_selected_project(self) -> Optional[ProjectInfo]:
        return self.get_project(self._meta.get("last_selected"))

    def set_last_selected(self, slug: Optional[str]) -> None:
        self._meta["last_selected"] = slug
        self._save_metadata()

    def create_project(self, name: str) -> ProjectInfo:
        name = name.strip()
        if not name:
            raise ValueError("Project name cannot be empty")
        slug = self._slugify(name)
        base = slug
        idx = 1
        existing_slugs = {p["slug"] for p in self._meta["projects"]}
        while slug in existing_slugs:
            slug = f"{base}_{idx}"
            idx += 1
        entry = {"name": name, "slug": slug, "created": datetime.utcnow().isoformat()}
        self._meta["projects"].append(entry)
        self._save_metadata()
        info = self._build_info(entry)
        info.ensure_folders()
        return info

    # ------------------------------------------------------------------
    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9-_]+", "_", value.lower()).strip("_")
        return slug or "project"

    # ------------------------------------------------------------------
    def summarize_project(self, project: ProjectInfo) -> Dict[str, str]:
        assets_count = sum(1 for _ in project.assets_dir.glob("**/*") if _.is_file())
        timing_dir = project.internal_dir / "timing"
        timing_files = list(timing_dir.glob("*.json"))
        timing_summary = "Timing file available" if timing_files else "No timing file yet"
        return {
            "assets": f"{assets_count} asset file(s)",
            "assets_path": str(project.assets_dir),
            "internal_path": str(project.internal_dir),
            "timing": timing_summary,
        }

    def project_exists(self, slug: str) -> bool:
        return any(entry["slug"] == slug for entry in self._meta["projects"])


__all__ = ["ProjectInfo", "ProjectManager"]
