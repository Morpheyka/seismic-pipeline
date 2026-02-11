"""
Minimal mod package for seismic pipeline standalone version.

This package contains only the classes and functions needed for
full_seismic_pipeline_example.py to work independently.

The package is organized into three subpackages:
- mod: Target-aware components that preserve target variables (y) alongside features (X)
- seismo: Seismic-specific components that work with features (X) only
- visualization: Visualization and reporting tools for hyperparameter tuning and experiments
"""

# Initialize logging system
from .seismo.logging_config import ModLoggerYt, get_mod_logger

# Set up logging for the package
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
    HypnogramCacheManagerYt
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
    
    # Visualization tools (if available)
    'visualize_hyperparameter_grid_slices',
    'plot_score_dynamics',
    'VISUALIZATION_AVAILABLE'
]