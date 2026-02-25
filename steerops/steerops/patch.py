"""
Patch loader, validator, and merger.

A SteerOps patch JSON file has this structure:
{
  "metadata": { "name": "...", "model": "...", "version": "1.0", ... },
  "interventions": [
    { "layer": 5, "strength": 3.0, "direction_vector": [...] | null, "notes": "..." },
    ...
  ],
  "validation": { ... },
  "deployment_instructions": "..."
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path BBB  

from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("steerops")


@dataclass
class Intervention:
    """A single layer intervention from a patch."""
    layer: int
    strength: float
    direction_vector: Optional[List[float]] = None
    notes: str = ""

    def __repr__(self) -> str:
        sign = "+" if self.strength > 0 else ""
        vec = "custom" if self.direction_vector else "auto"
        return f"Intervention(L{self.layer}, {sign}{self.strength}, {vec})"


@dataclass
class Patch:
    """
    Parsed and validated SteerOps patch.

    Load from file:
        patch = Patch.from_file("fix_helpfulness.json")

    Load from dict:
        patch = Patch.from_dict(data)

    Load from URL:
        patch = Patch.from_url("https://example.com/patch.json")

    Merge patches:
        combined = Patch.merge([patch1, patch2], name="combined")
    """
    name: str = ""
    model: str = ""
    version: str = "1.0"
    description: str = ""
    interventions: List[Intervention] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    # â”€â”€ Factory methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Patch":
        """Load a patch from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Patch file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded patch from: {path}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Patch":
        """Parse a patch from a dictionary."""
        meta = data.get("metadata", {})
        interventions = [
            Intervention(
                layer=i["layer"],
                strength=i["strength"],
                direction_vector=i.get("direction_vector"),
                notes=i.get("notes", ""),
            )
            for i in data.get("interventions", [])
        ]

        if not interventions:
            raise ValueError("Patch has no interventions defined")

        return cls(
            name=meta.get("name", "unnamed"),
            model=meta.get("model", ""),
            version=meta.get("version", "1.0"),
            description=meta.get("description", ""),
            interventions=interventions,
            raw=data,
        )

    @classmethod
    def from_url(cls, url: str) -> "Patch":
        """Load a patch from a remote URL (https only)."""
        import urllib.request

        # Security: restrict to HTTPS to prevent SSRF (file://, ftp://, internal IPs)
        if not url.startswith("https://"):
            raise ValueError(
                f"Only https:// URLs are allowed for security. Got: {url[:50]}"
            )

        logger.info(f"Downloading patch from: {url}")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            return cls.from_dict(data)
        except Exception as e:
            raise ConnectionError(
                f"Failed to download patch from {url}: {e}"
            ) from e

    @classmethod
    def merge(
        cls,
        patches: List["Patch"],
        name: str = "merged",
        strategy: str = "average",
    ) -> "Patch":
        """
        Merge multiple patches into one.

        Parameters
        ----------
        patches : List[Patch]
            Patches to merge.
        name : str
            Name for the merged patch.
        strategy : str
            'average' - average strengths for overlapping layers
            'sum' - sum strengths for overlapping layers
            'first' - keep first encountered for each layer

        Returns
        -------
        A new merged Patch.
        """
