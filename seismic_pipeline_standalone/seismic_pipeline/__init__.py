"""
Minimal mod package for seismic pipeline standalone version.

Subpackages:
- mod: Target-aware sklearn pipeline components (Yt suffix)
- seismo: Seismic/REM transformers and cache managers
- visualization: Reports and plotting
- bayesian: Changepoint PyMC models and search (optional PyMC deps)
- features: REM chunk feature engineering for changepoint studies
- config: Defaults, S3 env config, path settings
- common: Shared infrastructure (logging)
"""

# Initialize logging system (single setup site)
from .common.logging import ModLoggerYt, get_mod_logger

ModLoggerYt.setup_logging()

# Import target-aware components
from .mod import (
    configure_threading,
    TransformerMixinYt,
    PipelineYt,
    GridSearchCVYt,
    yt_accuracy_scorer,
    save_step_data
)

# Import seismic components
from .seismo import (
    EventLabelGeneratorYt,
    CustomEventLabelGeneratorYt,
    REMProfileCalculatorYt,
    REMProfileCombinerYt,
    REMProfileCleanerYt,
    REMProfileAdvancedCleanerYt,
    REMDailyExtractorYt,
    REMDailyMultiStatExtractorYt,
    REMProfileMaxMinExtractorYt,
    REMProfileSummaryExtractorYt,
    MetadataAdderYt,
    HypnogramCacheManagerYt,
    HypnoCalculatorYt,
    DatFileCacheManagerYt,
)

# Import visualization tools (optional - can be imported directly)
try:
    from .visualization import visualize_hyperparameter_grid_slices, plot_score_dynamics
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    visualize_hyperparameter_grid_slices = None
    plot_score_dynamics = None

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Threading (import before numpy for proper env setup)
    'configure_threading',

    # Target-aware components
    'TransformerMixinYt',
    'PipelineYt',
    'GridSearchCVYt',
    'yt_accuracy_scorer',
    'save_step_data',
    
    # Seismic components
    'EventLabelGeneratorYt',
    'CustomEventLabelGeneratorYt',
    'REMProfileCalculatorYt',
    'REMProfileCombinerYt',
    'REMProfileCleanerYt',
    'REMProfileAdvancedCleanerYt',
    'REMDailyExtractorYt',
    'REMDailyMultiStatExtractorYt',
    'REMProfileMaxMinExtractorYt',
    'REMProfileSummaryExtractorYt',
    'MetadataAdderYt',
    'HypnogramCacheManagerYt',
    'HypnoCalculatorYt',
    'DatFileCacheManagerYt',
    
    # Visualization tools (if available)
    'visualize_hyperparameter_grid_slices',
    'plot_score_dynamics',
    'VISUALIZATION_AVAILABLE'
]