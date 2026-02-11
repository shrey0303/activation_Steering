"""
Singleton Model Loader.

Loads any HuggingFace transformer model with optional 4-bit quantization.
Auto-detects CUDA / MPS / CPU and keeps a single instance throughout
the application lifecycle.
"""

from __future__ import annotations

import gc
import threading
import time
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger


class ModelManager:
    """
    Thread-safe singleton that holds the loaded model + tokenizer.

    Usage:
        mgr = ModelManager.get_instance()
        mgr.load("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model, tokenizer = mgr.model, mgr.tokenizer
    """

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    # ── Singleton accessor ────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Release the loaded model and reset singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.unload()
            cls._instance = None

    # ── Init ──────────────────────────────────────────────────
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_name: str = ""
        self.device: torch.device = torch.device("cpu")
        self.device_name: str = "cpu"
        self.loaded: bool = False
        self.num_layers: int = 0
        self.hidden_dim: int = 0
        self.architecture: str = ""
        self.quantized: bool = False
        self.quantization_bits: int = 0
        self.memory_mb: float = 0.0
        self._load_lock = threading.Lock()

    # ── Device Detection ──────────────────────────────────────
    @staticmethod
    def detect_device(preference: str = "auto") -> torch.device:
        """Pick the best available device."""
        if preference != "auto":
            return torch.device(preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ── Main Load ─────────────────────────────────────────────
    def load(
        self,
        model_name: str,
        device_preference: str = "auto",
        quantize: bool = True,
        quantization_bits: int = 4,
    ) -> None:
        """
        Download (if needed) and load a HuggingFace model + tokenizer.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        device_preference : str
            "auto" | "cuda" | "mps" | "cpu"
        quantize : bool
            Whether to apply 4-bit quantisation via bitsandbytes.
        quantization_bits : int
            Quantisation bit depth (4 or 8).
        """
        with self._load_lock:
            if self.loaded and self.model_name == model_name:
                logger.info(f"Model {model_name} already loaded – skipping")
                return

            if self.loaded:
                logger.info(f"Unloading current model {self.model_name}...")
                self.unload()

            t0 = time.perf_counter()
            self.device = self.detect_device(device_preference)
            self.device_name = str(self.device)
            logger.info(f"🔄 Loading {model_name} on {self.device_name}...")

            # Lazy-import heavy dependencies
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=False
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if quantize and self.device.type == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=(quantization_bits == 4),
                        load_in_8bit=(quantization_bits == 8),
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    load_kwargs["quantization_config"] = bnb_config
                    load_kwargs["device_map"] = "auto"
                    self.quantized = True
                    self.quantization_bits = quantization_bits
                    logger.info(
                        f"   Quantisation: {quantization_bits}-bit NF4 enabled"
                    )
                except ImportError:
                    logger.warning(
                        "bitsandbytes not available – loading in full precision"
                    )
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
            elif self.device.type == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            elif self.device.type == "mps":
                load_kwargs["torch_dtype"] = torch.float16
            # else: CPU – default float32

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **load_kwargs
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory — falling back to CPU")
                torch.cuda.empty_cache()
                self.device = torch.device("cpu")
                self.device_name = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Falling back to CPU with float32...")
                self.device = torch.device("cpu")
                self.device_name = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )

            # Move to device if not already mapped
            if not hasattr(self.model, "hf_device_map"):
                self.model = self.model.to(self.device)

            self.model.eval()
            self.model_name = model_name

            # Extract architecture info
            self._extract_model_info()

            elapsed = time.perf_counter() - t0
            self.loaded = True
            self.memory_mb = self._get_memory_usage()

            logger.info(
                f"✅ Model loaded in {elapsed:.1f}s | "
                f"Layers={self.num_layers} | Hidden={self.hidden_dim} | "
                f"Device={self.device_name} | Memory={self.memory_mb:.0f}MB"
            )

    # ── Architecture Extraction ───────────────────────────────
    def _extract_model_info(self) -> None:
        """Pull layer count, hidden dim, and architecture type from the model."""
        config = self.model.config

        # Number of layers
        self.num_layers = getattr(
            config, "num_hidden_layers",
            getattr(config, "n_layer",
                    getattr(config, "num_layers", 0))
        )

        # Hidden dimension
        self.hidden_dim = getattr(
            config, "hidden_size",
            getattr(config, "n_embd",
                    getattr(config, "d_model", 0))
        )

        # Architecture name
        self.architecture = getattr(config, "model_type", "unknown")

    # ── Helpers ───────────────────────────────────────────────
    def get_layer_modules(self):
        """
        Return the list of transformer layer modules.

        Handles different HF model architectures (LLaMA, GPT-2,
        Mistral, Phi, etc.).
        """
        if not self.loaded:
            return []

        model = self.model

        # Try common attribute chains
        for attr_path in [
            "model.layers",       # LLaMA, Mistral, Phi
            "transformer.h",     # GPT-2, GPT-Neo
            "gpt_neox.layers",   # GPT-NeoX, Pythia
            "model.decoder.layers",  # OPT
        ]:
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue

        logger.warning("Could not auto-detect layer modules")
        return []

    def _get_memory_usage(self) -> float:
        """Estimate model memory in MB."""
        if self.model is None:
            return 0.0
        try:
            param_bytes = sum(
                p.nelement() * p.element_size() for p in self.model.parameters()
            )
            return param_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def get_info(self) -> Dict[str, Any]:
        """Serialisable metadata dictionary."""
        return {
            "model_name": self.model_name,
            "loaded": self.loaded,
            "device": self.device_name,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "architecture": self.architecture,
            "quantized": self.quantized,
            "quantization_bits": self.quantization_bits,
            "memory_mb": round(self.memory_mb, 1),
        }

    def unload(self) -> None:
        """Free GPU/CPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and memory released")
