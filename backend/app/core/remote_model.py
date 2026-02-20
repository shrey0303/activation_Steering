"""
Remote Model Manager — uses HuggingFace Hub + Inference API.

No local model download required. Fetches model config for scanning
and uses the HF Inference API for text generation.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from loguru import logger


# ── HuggingFace API endpoints ──────────────────────────────────
HF_HUB_API = "https://huggingface.co/api/models"
HF_INFERENCE_API = "https://router.huggingface.co/hf-inference/models"

# ── Import categories from scanner (single source of truth) ────
from app.core.scanner import CATEGORIES, BEHAVIORAL_ROLES


def _position_to_category(relative_pos: float) -> str:
    """Map relative position [0,1] to a research-backed category."""
    if relative_pos < 0.05:
        return "token_embedding"
    elif relative_pos < 0.12:
        return "positional_morphological"
    elif relative_pos < 0.25:
        return "syntactic_processing"
    elif relative_pos < 0.40:
        return "entity_semantic"
    elif relative_pos < 0.55:
        return "knowledge_retrieval"
    elif relative_pos < 0.70:
        return "reasoning_planning"
    elif relative_pos < 0.78:
        return "safety_alignment"
    elif relative_pos < 0.88:
        return "information_integration"
    elif relative_pos < 0.95:
        return "style_personality"
    else:
        return "output_distribution"


class RemoteModelManager:
    """
    Manages a remote HuggingFace model without downloading weights.
    Uses the Hub API for config and the Inference API for generation.
    """

    _instance: Optional["RemoteModelManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RemoteModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.model_name: Optional[str] = None
        self.config: Optional[Dict[str, Any]] = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.num_layers: int = 0
        self.hidden_dim: int = 0
        self.architecture: str = "unknown"
        self.loaded: bool = False
        self.hf_token: Optional[str] = None

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # ── Connect to a remote model ──────────────────────────────

    def connect(self, model_name: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch model config from HuggingFace Hub.
        No weights are downloaded — only config.json and model card metadata.
        """
        self.model_name = model_name
        if hf_token:
            self.hf_token = hf_token

        logger.info(f"🌐 Connecting to remote model: {model_name}")

        # Fetch model info from Hub API
        try:
            resp = requests.get(
                f"{HF_HUB_API}/{model_name}",
                headers=self._headers,
                timeout=15,
            )
            resp.raise_for_status()
            self.model_info = resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model '{model_name}' not found on HuggingFace Hub")
            elif e.response.status_code == 403:
                raise ValueError(
                    f"Access denied for '{model_name}'. This model may require authentication. "
                    f"Provide an HF token with access to this model."
                )
            raise ValueError(f"HuggingFace Hub error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to connect to HuggingFace Hub: {e}")

        # Fetch config.json
        try:
            config_resp = requests.get(
                f"https://huggingface.co/{model_name}/resolve/main/config.json",
                headers=self._headers,
                timeout=15,
            )
            config_resp.raise_for_status()
            self.config = config_resp.json()
        except Exception as e:
            raise ValueError(f"Failed to fetch config.json for '{model_name}': {e}")

        # Extract architecture info from config
        self._extract_architecture()

        self.loaded = True
        logger.info(
            f"✅ Remote model connected: {model_name} | "
            f"{self.num_layers} layers | hidden_dim={self.hidden_dim} | "
            f"arch={self.architecture}"
        )

        return self.get_info()

    def _extract_architecture(self) -> None:
        """Extract layer count, hidden dim, and architecture from config."""
        if not self.config:
            return

        cfg = self.config

        # Architecture type
        arch_list = cfg.get("architectures", [])
        self.architecture = arch_list[0] if arch_list else cfg.get("model_type", "unknown")

        # Number of layers — different models use different keys
        self.num_layers = (
            cfg.get("num_hidden_layers")
            or cfg.get("n_layer")
            or cfg.get("num_layers")
            or cfg.get("n_layers")
            or 0
        )

        # Hidden dimension
        self.hidden_dim = (
            cfg.get("hidden_size")
            or cfg.get("n_embd")
            or cfg.get("d_model")
            or 0
        )

    # ── Remote scan (position-based heuristics) ────────────────

    def remote_scan(self) -> Dict[str, Any]:
        """
        Generate layer profiles using config metadata + position heuristics.
        No weights are analysed — classification is based on layer position.
        """
        if not self.loaded:
            raise RuntimeError("No remote model connected")

        t0 = time.perf_counter()
        n = self.num_layers

        if n == 0:
            raise RuntimeError(
                f"Could not determine layer count from config for {self.model_name}"
            )

        profiles = []
        for idx in range(n):
            pos = idx / max(n - 1, 1)
            category = _position_to_category(pos)

            # Confidence is lower for remote scans (no weight analysis)
            base_confidence = 0.65
            # Layers at clear position extremes get higher confidence
            if pos < 0.05 or pos > 0.95:
                confidence = base_confidence + 0.15
            elif 0.35 < pos < 0.65:
                confidence = base_confidence + 0.10
            else:
                confidence = base_confidence + 0.05

            profiles.append({
                "layer_index": idx,
                "category": category,
                "confidence": round(min(confidence, 0.85), 3),
                "behavioral_role": BEHAVIORAL_ROLES.get(category, ""),
                "weight_stats": {},  # No weight data in remote mode
                "description": (
                    f"Layer {idx} ({category}) — position-based classification "
                    f"(remote scan, no weight analysis)"
                ),
            })

        elapsed = time.perf_counter() - t0
        logger.info(f"✅ Remote scan complete: {n} layers in {elapsed:.3f}s")

        return {
            "num_layers": n,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "layer_profiles": profiles,
            "scan_time_ms": round(elapsed * 1000, 1),
            "from_cache": False,
            "remote": True,
        }

    # ── Remote generation via Inference API ────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate text using HuggingFace Inference API."""
        if not self.loaded:
            raise RuntimeError("No remote model connected")

        t0 = time.perf_counter()

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "do_sample": temperature > 0,
            },
        }

        try:
            resp = requests.post(
                f"{HF_INFERENCE_API}/{self.model_name}",
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            body = e.response.json() if e.response.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = body.get("error", str(e))

            if status == 503:
                # Model is loading on HF servers
                estimated_time = body.get("estimated_time", 60)
                raise RuntimeError(
                    f"Model is loading on HuggingFace servers. "
                    f"Estimated wait: ~{int(estimated_time)}s. Try again shortly."
                )
            elif status == 429:
                raise RuntimeError("Rate limited by HuggingFace. Wait a moment and try again.")
            else:
                raise RuntimeError(f"HF Inference API error: {error_msg}")
        except requests.exceptions.Timeout:
            raise RuntimeError("HF Inference API timed out. The model may be loading — try again.")

        elapsed = time.perf_counter() - t0

        # Parse response
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            text = result.get("generated_text", "")
        else:
            text = str(result)

        tokens_approx = len(text.split())

        return {
            "text": text,
            "tokens_generated": tokens_approx,
            "latency_ms": round(elapsed * 1000, 1),
            "tokens_per_sec": round(tokens_approx / max(elapsed, 0.001), 1),
            "steering_applied": False,
            "remote": True,
        }

    # ── Simulated activations ──────────────────────────────────

    def simulate_activations(self, prompt: str) -> Dict[str, Any]:
        """
        Generate ESTIMATED activation magnitudes based on layer position
        and prompt characteristics. These are heuristic estimates, NOT
        real activations. The 'estimated' flag MUST be passed to the UI.
        """
        import hashlib
        import math

        if not self.loaded or self.num_layers == 0:
            return {}

        # Use prompt hash for deterministic but varied activations
        prompt_hash = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        prompt_len = len(prompt.split())

        activations = {}
        for idx in range(self.num_layers):
            pos = idx / max(self.num_layers - 1, 1)

            # Base activation curve: higher in middle layers (semantic/reasoning)
            base = 0.3 + 0.5 * math.exp(-8 * (pos - 0.45) ** 2)

            # Add prompt-dependent variation
            seed = (prompt_hash + idx * 7919) % 10000
            noise = (seed / 10000.0 - 0.5) * 0.3

            # Longer prompts → slightly higher activations in reasoning layers
            length_boost = min(prompt_len / 50.0, 1.0) * 0.1 * (1 if 0.4 < pos < 0.7 else 0)

            activation = max(0.05, min(1.0, base + noise + length_boost))
            activations[idx] = round(activation, 4)

        return activations

    # ── Info ───────────────────────────────────────────────────

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "id": self.model_name or "",
            "name": self.model_name or "",
            "loaded": self.loaded,
            "parameters": "",
            "quantized": False,
            "quantization_bits": 0,
            "memory_mb": 0.0,
            "device": "remote (HF API)",
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "scanned": False,
            "remote": True,
        }

    def disconnect(self) -> None:
        """Clear remote model state."""
        self.model_name = None
        self.config = None
        self.model_info = None
        self.num_layers = 0
        self.hidden_dim = 0
        self.architecture = "unknown"
        self.loaded = False
        logger.info("🌐 Remote model disconnected")
