"""
steerops — Apply SteerOps patches to any HuggingFace model.

Usage:
    from steerops import Steerer, Patch

    # Load and apply a patch
    steerer = Steerer.from_patch("patch_helpfulness.json")
    model = steerer.apply(model)
    output = model.generate(...)
    steerer.remove()

    # Compare steered vs unsteered
    result = steerer.compare(model, tokenizer, "prompt")

    # Batch generate
    results = steerer.batch_generate(model, tokenizer, ["p1", "p2"])

    # Load from URL
    steerer = Steerer.from_url("https://example.com/patch.json")

    # Merge patches
    combined = Patch.merge([patch1, patch2])
"""

import logging

from steerops.steerer import Steerer
from steerops.patch import Patch, Intervention
from steerops.hooks import HookManager

__version__ = "0.2.0"
__all__ = ["Steerer", "Patch", "Intervention", "HookManager"]

# Configure library logging
logging.getLogger("steerops").addHandler(logging.NullHandler())
