"""
Visualization tools for seismic pipeline analysis.

This subpackage provides visualization and reporting tools for hyperparameter tuning
and experiment results.
"""

from .hyperparameter_grid_visualizer import (
    visualize_hyperparameter_grid_slices,
    build_slices,
    autoslice,
    build_imshow,
    plot_score_dynamics
)

from .report_generator import ReportGenerator

__all__ = [
    'visualize_hyperparameter_grid_slices',
    'build_slices',
    'autoslice',
    'build_imshow',
    'plot_score_dynamics',
    'ReportGenerator'
]

