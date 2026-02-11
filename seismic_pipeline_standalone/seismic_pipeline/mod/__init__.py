"""
Target-aware components for the seismic pipeline.

The `mod` subpackage contains scikit-learn-compatible components that preserve and
propagate targets (y) alongside features (X) through transformers and pipelines.

Why this matters
---------------
Some preprocessing steps in this project *expand* or otherwise transform the sample
set demonstrated by a list of events (X), and must therefore transform the matching
labels (y) in lockstep. Standard scikit-learn `Pipeline` assumes a fixed y, so this
subpackage provides Yt variants that pass (X, y) through the whole chain.

If you want feature-only (X-only) transformers, use `seismic_pipeline.seismo`.
"""

from .threading_config import configure_threading
from .sklearnbaseyt import TransformerMixinYt
from .pipelineyt import PipelineYt, FeatureUnionYt, make_pipeline_yt, make_union_yt
from .grid_searchyt import GridSearchCVYt, RandomizedSearchCVYt
from .scoreryt import yt_accuracy_scorer
from .utilsyt import save_step_data

__all__ = [
    # Threading (import before numpy for proper env setup)
    "configure_threading",

    # Base mixin
    "TransformerMixinYt",

    # Pipeline primitives
    "PipelineYt",
    "FeatureUnionYt",
    "make_pipeline_yt",
    "make_union_yt",

    # Hyperparameter search
    "GridSearchCVYt",
    "RandomizedSearchCVYt",

    # Scoring + utilities
    "yt_accuracy_scorer",
    "save_step_data",
]