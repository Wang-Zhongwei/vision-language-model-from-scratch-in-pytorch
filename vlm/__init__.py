"""Real-track LLaVA-style VLM: frozen SigLIP + our MLP projector + a modern LM.

This package wraps the from-scratch components in `model.py` (specifically the
2-layer GELU projector design) with production backbones so the model can be
trained on real image-text data at scale.
"""

from .modeling import LlavaConfig, LlavaVLM, MLPProjector

__all__ = ["LlavaConfig", "LlavaVLM", "MLPProjector"]
