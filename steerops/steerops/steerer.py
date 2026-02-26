"""
Core Steerer class â€” the main public API for steerops.

Usage:
    from steerops import Steerer

    # Load a patch and apply to a model
    steerer = Steerer.from_patch("patch_helpfulness.json")
    model = steerer.apply(model)

    # Generate with steering active
    output = model.generate(input_ids, max_new_tokens=100)

    # Remove steering hooks
    steerer.remove()

    # One-shot convenience
    text = Steerer.run(
        patch_path="patch.json",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        prompt="Hello, how are you?",
    )

    # Compare steered vs unsteered output
    result = steerer.compare(model, tokenizer, "Tell me about climate")

    # Batch generate
    results = steerer.batch_generate(model, tokenizer, ["prompt1", "prompt2"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from steerops.hooks import HookManager
from steerops.patch import Patch

logger = logging.getLogger("steerops")


class Steerer:
    """
    Apply a SteerOps activation steering patch to any HuggingFace model.

    The patch defines which layers to intervene on, the steering strength,
    and optional direction vectors.  Steerer registers PyTorch forward hooks
    that modify hidden states during generation.
    """

    def __init__(self, patch: Patch) -> None:
        self.patch = patch
        self._hooks = HookManager()
        self._applied = False
        self._model = None

    # â”€â”€ Factory methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @classmethod
    def from_patch(cls, path: Union[str, Path]) -> "Steerer":
        """Load a Steerer from a patch JSON file."""
        patch = Patch.from_file(path)
        return cls(patch)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Steerer":
        """Load a Steerer from a patch dictionary."""
        patch = Patch.from_dict(data)
        return cls(patch)

    @classmethod
    def from_url(cls, url: str) -> "Steerer":
        """Load a Steerer from a remote patch URL."""
        patch = Patch.from_url(url)
        return cls(patch)

    # â”€â”€ Core API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def apply(self, model: Any) -> Any:
        """
        Apply steering hooks to the model.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Any HuggingFace causal LM model.

        Returns
        -------
        The same model (with hooks registered).
        """
        if self._applied:
            self.remove()

        layers = self._get_layer_modules(model)
        num_layers = len(layers)

        # Validate
        warnings = self.patch.validate_for_model(num_layers)
        for w in warnings:
            logger.warning(w)

        # Register hooks
        for intervention in self.patch.interventions:
            if intervention.layer >= num_layers:
                logger.warning(
                    f"Skipping layer {intervention.layer} "
                    f"(model has {num_layers})"
                )
                continue

            direction = None
            if intervention.direction_vector:
                direction = torch.tensor(
                    intervention.direction_vector, dtype=torch.float32
                )
                # Validate dimension
                hidden_dim = self._get_hidden_dim(model)
                if hidden_dim and direction.shape[0] != hidden_dim:
                    logger.warning(
                        f"Direction vector dim ({direction.shape[0]}) "
                        f"!= model hidden_dim ({hidden_dim}) at layer "
                        f"{intervention.layer}. Steering may fail."
                    )

            self._hooks.add_hook(
                layer_module=layers[intervention.layer],
                layer_idx=intervention.layer,
                strength=intervention.strength,
                direction_vector=direction,
            )

        self._applied = True
        self._model = model
        logger.info(
            f"Applied {self._hooks.active_hooks} steering hooks "
            f"from patch '{self.patch.name}'"
        )
        return model

    def remove(self) -> None:
        """Remove all steering hooks from the model."""
        count = self._hooks.remove_all()
        self._applied = False
        self._model = None
        if count:
            logger.info(f"Removed {count} steering hooks")

    def is_active(self) -> bool:
        """Whether steering hooks are currently registered."""
        return self._applied and self._hooks.active_hooks > 0

    def status(self) -> str:
        """Human-readable status."""
        lines = [
            f"Patch: {self.patch.name}",
            f"Active: {self.is_active()}",
            self._hooks.summary(),
        ]
        return "\n".join(lines)

    # â”€â”€ Convenience: one-shot generate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        One-shot: apply patch â†’ generate â†’ remove â†’ return text.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        tokenizer : transformers.PreTrainedTokenizer
        prompt : str
        max_new_tokens : int
        temperature : float

        Returns
        -------
        Generated text string.
        """
        self.apply(model)
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id
                    or tokenizer.eos_token_id,
                    **kwargs,
                )
            text = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return text
        finally:
            self.remove()

    def batch_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs,
    ) -> List[str]:
        """
        Generate steered outputs for multiple prompts.

        Parameters
        ----------
        model : transformers.PreTrainedModel
        tokenizer : transformers.PreTrainedTokenizer
        prompts : list of str
        max_new_tokens : int
        temperature : float

        Returns
        -------
        List of generated text strings.
        """
        self.apply(model)
        try:
            results = []
            for prompt in prompts:
                inputs = tokenizer(
                    prompt, return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                        **kwargs,
                    )
                text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                results.append(text)
            return results
        finally:
            self.remove()

    def compare(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Dict[str, str]:
        """
        Compare steered vs unsteered output for the same prompt.

        Returns
        -------
        Dict with 'baseline' and 'steered' keys.
        """
        # Baseline (no steering)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            base_ids = model.generate(
