"""
PCA Feature Extraction Pipeline â€” Offline Feature Dictionary.

Replaces static weight scanning (scanner.py) and CAA (vector_calculator.py)
with mathematically rigorous, data-driven feature discovery.

Pipeline:
  1. Run diverse prompts through the model
  2. Collect residual stream activations at each layer
  3. Run PCA to find top-K principal components per layer
  4. Auto-label top components by amplifying them and observing output
  5. Store everything in SQLite as a reusable Feature Dictionary

Usage:
  # From CLI
  python -m app.core.feature_extractor --model HuggingFaceTB/SmolLM2-135M

  # Programmatic
  extractor = FeatureExtractor()
  extractor.extract(model, tokenizer, model_name="SmolLM2-135M")
  dict = FeatureDictionary.load(model_name="SmolLM2-135M")
  feature = dict.get("L14_PC0")
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from app.core.feature_dataset import (
    BEHAVIORAL_KEYWORDS,
    get_diverse_prompts,
    get_labeling_prompts,
)

# â”€â”€ Storage paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_DATA_DIR = Path(__file__).parent.parent / "data"
_FEATURES_DB = _DATA_DIR / "features.db"
_VECTORS_DIR = _DATA_DIR / "feature_vectors"


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  DATA CLASSES                                                â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


@dataclass
class Feature:
    """A single extracted feature (PCA component)."""
    feature_id: str          # e.g., "L14_PC3"
    layer_idx: int           # transformer layer index
    component_idx: int       # PCA rank (0 = highest variance)
    vector: np.ndarray       # shape: (hidden_dim,)
    label: str               # auto-generated or user-provided label
    variance_explained: float  # fraction of variance this PC captures
    model_name: str          # which model this was extracted from

    def to_torch(self, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
        """Convert vector to PyTorch tensor."""
        return torch.from_numpy(self.vector).to(device=device, dtype=dtype)


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  FEATURE DICTIONARY â€” Runtime Lookup                         â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class FeatureDictionary:
    """
    O(1) runtime lookup for extracted features.

    Loads from SQLite + numpy files on startup.
    Stays in memory for the lifetime of the server.
    """

    def __init__(self) -> None:
        self._features: Dict[str, Feature] = {}
        self._by_layer: Dict[int, List[Feature]] = {}
        self._by_label: Dict[str, List[Feature]] = {}

    @classmethod
    def load(cls, model_name: str, db_path: Path = _FEATURES_DB) -> "FeatureDictionary":
        """Load all features for a model from SQLite."""
        fd = cls()

        if not db_path.exists():
            logger.warning(f"Feature database not found: {db_path}")
            return fd

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT feature_id, layer_idx, component_idx, label, "
            "variance_explained, vector_path FROM features WHERE model_name = ?",
            (model_name,),
        )

        for row in cursor:
            fid, layer, comp, label, var_exp, vec_path = row
            vec_file = Path(vec_path)
            if not vec_file.exists():
                logger.warning(f"Missing vector file: {vec_file}")
                continue

            vector = np.load(str(vec_file))
            feature = Feature(
                feature_id=fid,
                layer_idx=layer,
                component_idx=comp,
                vector=vector,
                label=label,
                variance_explained=var_exp,
                model_name=model_name,
            )
            fd._features[fid] = feature
            fd._by_layer.setdefault(layer, []).append(feature)
            if label:
                fd._by_label.setdefault(label.lower(), []).append(feature)

        conn.close()
        logger.info(
            f"Loaded {len(fd._features)} features for {model_name} "
            f"across {len(fd._by_layer)} layers"
        )
        return fd

    def get(self, feature_id: str) -> Optional[Feature]:
        """O(1) lookup by feature ID."""
        return self._features.get(feature_id)

    def get_by_layer(self, layer_idx: int) -> List[Feature]:
        """Get all features for a specific layer."""
        return self._by_layer.get(layer_idx, [])

    def get_labeled(self) -> List[Feature]:
        """Get all features that have labels (not just L{n}_PC{m})."""
        return [
            f for f in self._features.values()
            if f.label and not f.label.startswith("L")
        ]

    def get_by_label(self, label: str) -> List[Feature]:
        """Get features matching a label."""
        return self._by_label.get(label.lower(), [])

    def search(self, query: str) -> List[Feature]:
        """Simple substring search across labels."""
        query_lower = query.lower()
        return [
            f for f in self._features.values()
            if query_lower in f.label.lower()
        ]

    @property
    def all_features(self) -> List[Feature]:
        return list(self._features.values())

    @property
    def all_labels(self) -> List[str]:
        return list(self._by_label.keys())

    @property
    def layer_count(self) -> int:
        return len(self._by_layer)

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serializable list of all features (without vectors)."""
        return [
            {
                "feature_id": f.feature_id,
                "layer_idx": f.layer_idx,
                "component_idx": f.component_idx,
                "label": f.label,
                "variance_explained": round(f.variance_explained, 6),
            }
            for f in self._features.values()
        ]


# â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
# â•‘  FEATURE EXTRACTOR â€” Offline Pipeline                        â•‘
# â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class FeatureExtractor:
    """
    Offline PCA feature extraction pipeline.

    Steps:
      1. Collect activations from diverse prompts
      2. Run PCA on each layer's activations
      3. Auto-label top components
      4. Store in SQLite + numpy files
    """

    def __init__(
        self,
        top_k: int = 20,
        auto_label_top_n: int = 5,
        labeling_strength: float = 5.0,
        labeling_max_tokens: int = 30,
        min_variance: float = 0.001,  # drop components < 0.1%
    ) -> None:
        self.top_k = top_k
        self.auto_label_top_n = auto_label_top_n
        self.labeling_strength = labeling_strength
        self.labeling_max_tokens = labeling_max_tokens
        self.min_variance = min_variance
        self._embedder = None

    def _ensure_embedder(self) -> None:
        """Lazy-load sentence transformer for auto-labeling."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded sentence embedder for auto-labeling")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available â€” "
                    "auto-labeling will be skipped"
                )

    # â”€â”€ Step 1: Activation Collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @torch.no_grad()
    def _collect_activations(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        layer_modules: List[torch.nn.Module],
        prompts: List[str],
        device: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Run prompts through the model and collect residual stream
        activations at each layer.

        Returns: dict mapping layer_idx â†’ tensor(n_prompts, hidden_dim)
        """
        n_layers = len(layer_modules)
        layer_activations: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(n_layers)
        }
        handles = []

        # Register hooks on all layers
        for idx in range(n_layers):
            def make_hook(layer_idx: int):
                def _capture(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    # Mean pool across sequence dimension â†’ (batch, hidden_dim)
                    pooled = h.float().mean(dim=1)
                    layer_activations[layer_idx].append(pooled.cpu())
                return _capture
            handle = layer_modules[idx].register_forward_hook(make_hook(idx))
            handles.append(handle)

        # Run all prompts
        total = len(prompts)
        batch_size = 8
        for start in range(0, total, batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            model(**inputs)

            if (start // batch_size) % 10 == 0:
                logger.info(
                    f"Collecting activations: {min(start + batch_size, total)}/{total}"
                )

        # Remove hooks
        for h in handles:
            h.remove()

        # Concatenate
        result = {}
        for idx in range(n_layers):
            if layer_activations[idx]:
                result[idx] = torch.cat(layer_activations[idx], dim=0)
                # result[idx] shape: (n_prompts, hidden_dim)

        logger.info(
            f"Collected activations: {len(result)} layers, "
            f"{result[0].shape[0]} samples each"
        )
        return result

    # â”€â”€ Step 2: PCA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _run_pca(
        self,
        activations: Dict[int, torch.Tensor],
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Run PCA on each layer's activations.

        Returns: dict mapping layer_idx â†’ (components, explained_variance)
          components: (top_k, hidden_dim)
          explained_variance: (top_k,) â€” fraction of variance
        """
        results = {}

        for layer_idx, acts in activations.items():
            # Center the data
            acts_np = acts.numpy()
            mean = acts_np.mean(axis=0)
            centered = acts_np - mean

            # SVD-based PCA (more numerically stable than covariance)
            # centered: (n_samples, hidden_dim)
            try:
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.warning(f"SVD failed for layer {layer_idx}, skipping")
                continue

            # Top-K components
            k = min(self.top_k, len(S))
            components = Vt[:k]  # (k, hidden_dim)

            # Explained variance ratio
            total_variance = (S ** 2).sum()
            explained = (S[:k] ** 2) / total_variance

            # Min-variance filter: drop noise components (< 0.1%)
            keep_mask = explained >= self.min_variance
            components = components[keep_mask]
            explained = explained[keep_mask]

            if len(components) == 0:
                logger.warning(f"Layer {layer_idx}: all components below min_variance")
                continue

            results[layer_idx] = (components, explained)
            logger.debug(
                f"Layer {layer_idx}: {len(components)} PCs (filtered from {k}) explain "
                f"{explained.sum()*100:.1f}% variance"
            )

        logger.info(f"PCA complete: {len(results)} layers processed")
        return results

    # â”€â”€ Step 3: Auto-Labeling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @torch.no_grad()
    def _auto_label_component(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        layer_module: torch.nn.Module,
        component: np.ndarray,
        device: str,
    ) -> str:
        """
        Label a PCA component using contrastive generation.

        Contrastive approach (CAA-style, doubles signal):
          1. Generate with component AMPLIFIED (+strength)
          2. Generate with component SUPPRESSED (-strength)
          3. delta = embed(amplified) - embed(suppressed)
          4. Match delta to behavioral keywords
        """
        if self._embedder is None:
            return ""

        prompts = get_labeling_prompts()[:5]
        v = torch.from_numpy(component).to(device=device, dtype=torch.float32)
        v = F.normalize(v, dim=-1)

        def generate_texts(steer_strength: float) -> List[str]:
            """Generate text with given steering strength (0 = no steering)."""
            handle = None
            if steer_strength != 0:
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple):
                        x = out[0]
                        rest = out[1:]
                    else:
                        x = out
                        rest = None
                    x = x + steer_strength * v
                    # Norm preservation
                    orig_norm = (out[0] if isinstance(out, tuple) else out).norm(dim=-1, keepdim=True)
                    new_norm = x.norm(dim=-1, keepdim=True)
                    scale = (orig_norm / (new_norm + 1e-8)).clamp(0.95, 1.05)
                    x = x * scale
                    if rest is not None:
                        return (x,) + rest
                    return x
                handle = layer_module.register_forward_hook(hook_fn)

            texts = []
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.labeling_max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_ids = outputs[0][inputs["input_ids"].shape[1]:]
                texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))

            if handle is not None:
                handle.remove()
            return texts

        # Contrastive generation: amplify vs suppress (CAA-style)
        amplified_texts = generate_texts(+self.labeling_strength)
        suppressed_texts = generate_texts(-self.labeling_strength)

        # Embed both sets
        amplified_emb = self._embedder.encode(amplified_texts)
        suppressed_emb = self._embedder.encode(suppressed_texts)

        # Contrastive delta: 2x signal strength vs single-sided
        delta = (amplified_emb - suppressed_emb).mean(axis=0)
        delta_norm = np.linalg.norm(delta)

        if delta_norm < 0.01:
            return ""  # No meaningful change

        # Compare delta to behavioral keywords
        keyword_emb = self._embedder.encode(BEHAVIORAL_KEYWORDS)
        delta_normalized = delta / (delta_norm + 1e-8)

        # Cosine similarity
        sims = (keyword_emb * delta_normalized).sum(axis=1)
        best_idx = sims.argmax()
        best_sim = sims[best_idx]

        if best_sim < 0.15:
            return ""  # No confident match

        label = BEHAVIORAL_KEYWORDS[best_idx]
        logger.debug(f"Auto-label: sim={best_sim:.3f} â†’ '{label}'")
        return label

    # â”€â”€ Step 4: Storage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _init_db(self, db_path: Path) -> sqlite3.Connection:
        """Initialize SQLite database for feature storage."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                feature_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                component_idx INTEGER NOT NULL,
                label TEXT DEFAULT '',
                variance_explained REAL NOT NULL,
                vector_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_model
            ON features(model_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_layer
            ON features(model_name, layer_idx)
        """)
        conn.commit()
        return conn

    def _save_feature(
        self,
        conn: sqlite3.Connection,
        feature: Feature,
        vectors_dir: Path,
    ) -> None:
        """Save a single feature to DB + numpy file."""
        vectors_dir.mkdir(parents=True, exist_ok=True)
        vec_path = vectors_dir / f"{feature.feature_id}.npy"
        np.save(str(vec_path), feature.vector)

        conn.execute(
            "INSERT OR REPLACE INTO features "
            "(feature_id, model_name, layer_idx, component_idx, "
            "label, variance_explained, vector_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                feature.feature_id,
                feature.model_name,
                feature.layer_idx,
                feature.component_idx,
                feature.label,
                feature.variance_explained,
                str(vec_path),
            ),
        )

    # â”€â”€ Main Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

