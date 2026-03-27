"""
Patch loader, validator, and merger.

A SteerOps patch JSON file:
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
from pathlib import Path
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
    """Parsed and validated SteerOps patch."""

    name: str = ""
    model: str = ""
    version: str = "1.0"
    description: str = ""
    interventions: List[Intervention] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

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

        # Restrict to HTTPS to prevent SSRF via file://, ftp://, internal IPs
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

        Strategies for overlapping layers:
        - 'average': average strengths
        - 'sum': sum strengths
        - 'first': keep first encountered
        """
        if not patches:
            raise ValueError("Cannot merge empty list of patches")

        if len(patches) == 1:
            return patches[0]

        layer_interventions: Dict[int, List[Intervention]] = {}
        for patch in patches:
            for iv in patch.interventions:
                if iv.layer not in layer_interventions:
                    layer_interventions[iv.layer] = []
                layer_interventions[iv.layer].append(iv)

        merged_interventions = []
        for layer_idx in sorted(layer_interventions.keys()):
            ivs = layer_interventions[layer_idx]

            if strategy == "first":
                merged_interventions.append(ivs[0])
            elif strategy == "sum":
                total_strength = sum(iv.strength for iv in ivs)
                vec = next(
                    (iv.direction_vector for iv in ivs if iv.direction_vector),
                    None,
                )
                merged_interventions.append(
                    Intervention(
                        layer=layer_idx,
                        strength=total_strength,
                        direction_vector=vec,
                        notes=f"Merged ({len(ivs)} sources, sum)",
                    )
                )
            else:  # average
                avg_strength = sum(iv.strength for iv in ivs) / len(ivs)
                vec = next(
                    (iv.direction_vector for iv in ivs if iv.direction_vector),
                    None,
                )
                merged_interventions.append(
                    Intervention(
                        layer=layer_idx,
                        strength=round(avg_strength, 4),
                        direction_vector=vec,
                        notes=f"Merged ({len(ivs)} sources, avg)",
                    )
                )

        models = list({p.model for p in patches if p.model})

        logger.info(
            f"Merged {len(patches)} patches -> {len(merged_interventions)} "
            f"interventions ({strategy})"
        )

        return cls(
            name=name,
            model=models[0] if len(models) == 1 else "",
            version="1.0",
            description=f"Merged from: {', '.join(p.name for p in patches)}",
            interventions=merged_interventions,
            raw={},
        )

    def validate_for_model(self, num_layers: int) -> List[str]:
        """Return warnings if the patch may be incompatible with a model of this depth."""
        warnings = []
        for iv in self.interventions:
            if iv.layer >= num_layers:
                warnings.append(
                    f"Layer {iv.layer} exceeds model layer count ({num_layers})"
                )
            if abs(iv.strength) > 10:
                warnings.append(
                    f"Layer {iv.layer} strength {iv.strength} is unusually high"
                )
        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Export the patch as a serializable dictionary."""
        return {
            "metadata": {
                "name": self.name,
                "model": self.model,
                "version": self.version,
                "description": self.description,
            },
            "interventions": [
                {
                    "layer": iv.layer,
                    "strength": iv.strength,
                    "direction_vector": iv.direction_vector,
                    "notes": iv.notes,
                }
                for iv in self.interventions
            ],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save the patch to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved patch to: {path}")
        return path

    def summary(self) -> str:
        lines = [
            f"Patch: {self.name}",
            f"Model: {self.model or 'any'}",
            f"Interventions: {len(self.interventions)}",
        ]
        for iv in self.interventions:
            lines.append(f"  {iv}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Patch(name={self.name!r}, interventions={len(self.interventions)})"
