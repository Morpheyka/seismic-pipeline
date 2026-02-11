#!/usr/bin/env python3
"""
Full Seismic Event Pipeline - Scikit-learn Style Interface

This example demonstrates the complete scikit-learn-like interface for the seismic event pipeline
with all 12 events including 2024 events that use S3 fallback.
"""

# Limit parallel computation to MAX_CORES across all libraries
# Set these BEFORE importing numpy or any other libraries that use threading
import os
import sys

MAX_CORES = 60  # Number of parallel jobs for GridSearchCV
THREADS_PER_JOB = 1  # Each job uses 1 thread

# Add the seismic_pipeline package to the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
DEFAULT_QUALITY_MODEL_PATH = os.path.join(
    _workspace_root,
    "Анализ моделей оценки качества сигналов и метод оценки из литературы",
    "Сохраненный модели pickle",
    "mlp_class_new.pickle",
)
DEFAULT_QUALITY_MODEL_MODULE_DIR = os.path.join(
    _workspace_root,
    "Анализ моделей оценки качества сигналов и метод оценки из литературы",
    "скрипты",
)

# Configure threading BEFORE importing numpy (load threading_config directly to avoid loading full package)
import importlib.util
_threading_spec = importlib.util.spec_from_file_location(
    "threading_config",
    os.path.join(_script_dir, "seismic_pipeline", "mod", "threading_config.py"))
_threading_mod = importlib.util.module_from_spec(_threading_spec)
_threading_spec.loader.exec_module(_threading_mod)
_threading_mod.configure_threading(MAX_CORES, THREADS_PER_JOB)

from math import gamma
import time
import numpy as np
import logging
import argparse
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score, make_scorer, accuracy_score
import re
import hashlib
import gc
from joblib import dump, parallel_backend

# Try to import threadpoolctl for better thread control
try:
    import threadpoolctl
    THREADPOOLCTL_AVAILABLE = True
except ImportError:
    THREADPOOLCTL_AVAILABLE = False
    print("Warning: threadpoolctl not available. Thread limits may not be fully enforced in worker processes.")

from seismic_pipeline import (
    PipelineYt,
    EventLabelGeneratorYt,
    CustomEventLabelGeneratorYt,
    REMProfileCalculatorYt,
    REMProfileCombinerYt,
    REMProfileMaxMinExtractorYt,
    REMProfileSummaryExtractorYt,
    REMProfileCleanerYt,
    REMProfileAdvancedCleanerYt,
    REMDailyMultiStatExtractorYt,
    MetadataAdderYt,
    HypnogramCacheManagerYt,
    HypnoCalculatorYt,
    save_step_data
)
from seismic_pipeline.mod.scoreryt import yt_accuracy_scorer
from sklearn.preprocessing import StandardScaler
#from seismic_pipeline.mod.pcayt import PCAYt
from seismic_pipeline.mod.grid_searchyt import GridSearchCVYt
# Scaler and passthrough for param grid (with/without scaling)
from seismic_pipeline.mod.scaleryt import StandardScalerYt, PassthroughYt
from seismic_pipeline.mod.cross_validationyt import cross_val_predict_yt, StratifiedKFoldYt

# Import report generation
from seismic_pipeline.visualization import ReportGenerator, visualize_hyperparameter_grid_slices, plot_score_dynamics


def main():
    """Run the seismic event pipeline with all 12 events including S3 fallback."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the seismic event pipeline')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed logging information')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Hide all warnings and info messages')
    parser.add_argument('--output-prefix', default='processed', 
                       help='Prefix for output CSV files (default: processed)')
    parser.add_argument('--output-dir', default='.', 
                       help='Directory to save output files (default: current directory)')
    parser.add_argument('--steps-range', action='store_true',
                       help='Use range-based pipeline with 2 samples per event (immediate before + control before)')
    parser.add_argument('--window-days', type=int, default=6,
                       help='Number of days in each window (default: 6)')
    parser.add_argument('--window-step-days', type=int, default=3,
                       help='Number of days to step back for Window 0 (label 0) - Window 0 starts window_step_days before Window 1. Can be negative (e.g., -4 means immediate window is 1-3 days before event) (default: 3)')
    parser.add_argument('--use-fixed-control-window', action='store_true', default=False,
                       help='Use fixed 3-day control window at 5-7 days before event (label 0)')
    parser.add_argument('--fixed-control-start-days', type=int, default=7,
                       help='Start day for fixed control window (default: 7, creates 5-7 days before event window)')
    parser.add_argument('--dump-label-generator-output', action='store_true', default=False,
                       help='Dump CustomEventLabelGeneratorYt output to CSV for debugging')
    parser.add_argument('--label-generator-output-csv', default='label_generator_output.csv',
                       help='Filename for dumped label generator output (relative to --output-dir)')
    parser.add_argument('--save-steps', action='store_true', default=False,
                       help='Save dataset at every pipeline step for verification')
    parser.add_argument('--visualize-grid', action='store_true', default=True,
                       help='Create hyperparameter grid visualizations (default: True)')
    parser.add_argument('--inference-window', choices=['both', 'immediate', 'control'], default='immediate',
                       help='Window mode to use at inference: both, immediate, or control (default: both)')
    parser.add_argument('--compile-pdf', action='store_true', default=False,
                       help='Compile markdown report to PDF (requires pandoc)')
    parser.add_argument('--skip-missing-prompt', action='store_true', default=False,
                       help='Automatically continue even if some hypnograms are missing (non-interactive mode)')
    parser.add_argument('--auto-hypnogram', action='store_true', default=False,
                       help='Compute missing hypnograms from raw EEG (with DAT pre-cache); if quality is insufficient, fallback to existing local/S3 hypnograms')
    parser.add_argument('--quality-model-path', default=DEFAULT_QUALITY_MODEL_PATH,
                       help='Path to channel-quality model pickle/joblib used by auto hypnogram')
    parser.add_argument('--quality-model-module-path', action='append', default=[],
                       help='Extra module directory for unpickling quality model (can be passed multiple times)')
    parser.add_argument('--quality-good-classes', default='4,5',
                       help='Comma-separated quality classes considered high quality (default: 4,5)')
    parser.add_argument('--quality-fallback-all-channels', action='store_true', default=False,
                       help='If quality prediction fails, use all channels for compute instead of immediate fallback to existing local/S3 hypnograms')
    args = parser.parse_args()

    quality_model_module_paths = args.quality_model_module_path[:] if args.quality_model_module_path else []
    if DEFAULT_QUALITY_MODEL_MODULE_DIR not in quality_model_module_paths:
        quality_model_module_paths.append(DEFAULT_QUALITY_MODEL_MODULE_DIR)
    quality_good_classes = tuple(
        int(v.strip()) for v in args.quality_good_classes.split(',') if v.strip()
    )
    
    # Configure logging based on arguments
    if args.quiet:
        logging.getLogger('mod.hypnogram_cache_manager').setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger('mod.hypnogram_cache_manager').setLevel(logging.INFO)
    else:
        # Default: show warnings once (due to our "show once" mechanism)
        logging.getLogger('mod.hypnogram_cache_manager').setLevel(logging.WARNING)
    
    events = [
    {'rat_id': 'R2', 'date': '2022-11-07'},
    {'rat_id': 'R2', 'date': '2022-11-18'},
    {'rat_id': 'R2', 'date': '2023-04-03'},
    {'rat_id': 'R2', 'date': '2023-04-11'},
    {'rat_id': 'R2', 'date': '2023-04-18'},
    {'rat_id': 'R2', 'date': '2023-04-21'},
    {'rat_id': 'R2', 'date': '2023-05-03'},
    {'rat_id': 'R2', 'date': '2023-05-09'},
    {'rat_id': 'R2', 'date': '2024-09-30'},
    {'rat_id': 'R2', 'date': '2024-10-29'},
    {'rat_id': 'R3', 'date': '2025-01-23'},
    {'rat_id': 'R3', 'date': '2025-03-14'},
    {'rat_id': 'R1', 'date': '2025-07-02'},
    {'rat_id': 'R2', 'date': '2025-07-02'},
    {'rat_id': 'R3', 'date': '2025-07-02'},
    {'rat_id': 'R4', 'date': '2025-07-02'},
    {'rat_id': 'R1', 'date': '2025-07-20'},
    {'rat_id': 'R2', 'date': '2025-07-20'},
    {'rat_id': 'R3', 'date': '2025-07-20'},
    {'rat_id': 'R4', 'date': '2025-07-20'},      
    ]

    

    # Convert to numpy array as expected by the pipeline
    X = np.array(events)
    y = None  # No initial labels

    if args.dump_label_generator_output:
        print("=== Dumping CustomEventLabelGeneratorYt output ===")
        debug_label_kwargs = {
            'window_days': args.window_days,
            'window_step_days': args.window_step_days,
            'date_format': '%Y-%m-%d',
            'use_fixed_control_window': args.use_fixed_control_window,
            'fixed_control_start_days': args.fixed_control_start_days
        }
        label_generator_debug = CustomEventLabelGeneratorYt(**debug_label_kwargs)
        label_generator_debug.fit(X, y)
        debug_X, debug_y = label_generator_debug.transform(X, y)

        debug_rows = []
        for sample, label in zip(debug_X, debug_y):
            window_dates = sample.get('window_dates', [])
            debug_rows.append({
                'rat_id': sample.get('rat_id'),
                'original_event_date': sample.get('original_event_date'),
                'window_type': sample.get('window_type'),
                'label': int(label),
                'window_dates': ";".join(window_dates)
            })

        debug_df = pd.DataFrame(debug_rows)
        os.makedirs(args.output_dir, exist_ok=True)
        label_generator_csv_path = os.path.join(args.output_dir, args.label_generator_output_csv)
        debug_df.to_csv(label_generator_csv_path, index=False)
        print(f"Label generator output saved to: {label_generator_csv_path}")
    
    # if args.steps_range:
    #     print("=== Full Seismic Event Pipeline - Range-based Configuration ===")
    #     print(f"Processing {len(events)} events")
    #     print(f"Window configuration:")
    #     print(f"  - Window 1: {args.window_days} days immediately before event day (label 1)")
    #     print(f"  - Window 0: {args.window_days} days window 3 days before event (label 0)")
    #     print(f"  - Expected samples: {len(events) * 2} (2 per event)")
    #     print(f"  - Feature per day: max_min_diff REM range")
    #     print(f"  - Total features per sample: {args.window_days}")
    # else:
    #     print("=== Full Seismic Event Pipeline - Scikit-learn Style ===")
    #     print(f"Processing {len(events)} events")
    # print()
    
    # Save initial data if requested
    # if args.save_steps:
    #     print("=== Saving Initial Data ===")
    #     save_step_data(X, y, 'initial', args.output_dir, args.output_prefix, 0)
    #     print()
    
    # 2. Create cache manager with S3 fallback
    local_data_root = '/home/ponomattik/mnt/wd/rat'
    cache_manager = HypnogramCacheManagerYt(
        local_cache_dir='./hypnogram_cache',
        local_data_root=local_data_root,
        s3_config={
            'service_name': 's3',
            'endpoint_url': 'http://10.132.230.2:7770',
            'aws_access_key_id': 'quantum',
            'aws_secret_access_key': 's3password',
        },
        s3_rat_bucket='rat',
        s3_temp_bucket='temp'
    )

    auto_hypnogram_step = (
        'hypno_calculator',
        HypnoCalculatorYt(
            cache_manager=cache_manager,
            use_s3_dat=False,
            local_data_root=local_data_root,
            s3_config=cache_manager.s3_config,
            epoch_length_sec=5,
            threshold='GMM',
            quality_model_path=args.quality_model_path,
            quality_model_module_paths=quality_model_module_paths,
            quality_good_classes=quality_good_classes,
            quality_fallback_to_all_channels=args.quality_fallback_all_channels,
        ),
    )
    
    # 3. Create report generator and prepare output directories
    report = ReportGenerator()
    reports_dir = os.path.join(args.output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_filename = "experiment.json"
    
    # Create events DataFrame for head section
    events_df = pd.DataFrame(events)
    os.makedirs(args.output_dir, exist_ok=True)
    events_csv_path = os.path.join(args.output_dir, "events.csv")
    events_df.to_csv(events_csv_path, index=False)
    
    # 4. Add head section to report
    print("\n" + "="*80)
    print("=== Adding Head Section to Report ===")
    print("="*80)
    
    # Create a template pipeline for head section (will be updated per experiment)
    template_steps = [
        ('label_generator', CustomEventLabelGeneratorYt(
            window_days=6,  # Both immediate and control windows use 6 days
            window_step_days=-5,
            date_format='%Y-%m-%d',
            use_fixed_control_window=True,
            fixed_control_start_days=9  # Creates 6-day control window: days 9, 8, 7, 6, 5, 4
        )),
    ]
    if args.auto_hypnogram:
        template_steps.append(auto_hypnogram_step)
    template_steps.extend([
        ('rem_calculator', REMProfileCalculatorYt(
            cache_manager=cache_manager,
            window_size_hours=6,
            step_size_hours=1,
            rem_stage=2,
            epoch_length_sec=5,
            sampling_rate=250,
            fail_on_missing_data=False  # Set to False for testing with missing data
        )),
        ('feature_extractor', REMFeatureExtra(
            daily_statistics=['mean'],
            window_days=12,
            handle_empty_days='zero'
        )),
        ('scaler', StandardScalerYt(regression=False)),

        ('classifier', LogisticRegression(max_iter=1000, penalty='l1'))
    ])
    template_pipe = PipelineYt(template_steps)
    
    # Define parameter grid for linear classifiers
    # base_params = {
    #     'rem_calculator__window_size_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #     'rem_calculator__step_size_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    #     'feature_extractor__window_days': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Match label_generator window_days
    #     #'feature_extractor__daily_statistic': ['max_min_diff', 'mean', 'max'],  # Different statistics to extract
    #     #'feature_extractor__handle_empty_days': ['zero', 'mean'],  # How to handle missing days
    # }
    


    # Define parameter grid for linear classifiers
    base_params = {
        'rem_calculator__window_size_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'rem_calculator__step_size_hours': [1, 2, 3, 4, 5],
        #'pca__n_components': [2, 3, 4, 6],
        #'feature_extractor__window_days': [3, 6, 12, 18],  # Match label_generator window_days
        #'scaler': [StandardScalerYt(regression=False), PassthroughYt()],  # Test with and without scaling
        #'feature_extractor__daily_statistic': ['max_min_diff', 'mean', 'max'],  # Different statistics to extract
        #'feature_extractor__handle_empty_days': ['zero', 'mean'],  # How to handle missing days
    }

    # LogisticRegression L2 grid
    lr_params_l1 = base_params.copy()
    lr_params_l1.update({
        #'classifier': [],
        'classifier__C': [0.01, 0.1, ],
        'classifier__solver': ['liblinear', 'lbfgs'],
        'classifier__tol': [1e-4, 1e-5],
        #'classifier__fit_intercept': [True, False],
        #'classifier__max_iter': [1000],
    })
    
    # # LogisticRegression L1 grid
    # lr_params_l1 = base_params.copy()
    # lr_params_l1.update({
    #     'classifier': [LogisticRegression(max_iter=1000, penalty='l1')],
    #     'classifier__C': [0.1, 1, 10],
    #     'classifier__solver': ['liblinear'],
    #     'classifier__tol': [1e-4, 1e-5],
    #     'classifier__fit_intercept': [True, False],
    #     #'classifier__max_iter': [1000],
    # })
    
    # SVC grid
    svm_params = base_params.copy()
    svm_params.update({
        'classifier': [SVC(probability=True, cache_size = 500, max_iter=1000)],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'classifier__tol': [1e-4, 1e-5],
        #'classifier__max_iter': [1000],
    })
    
    # Combine parameter grids
    #param_grid = [lr_params_l2, lr_params_l1, svm_params]

    param_grid = [lr_params_l2]#, lr_params_l1, svm_params]
    
    
    # Add head section
    report.head_title("Анализ разделимости скользящих окон шириной 6 дней")
    report.head_samples(events_df, description="Список событий", metadata=False)
    report.head_pipeline_structure(template_pipe, description="Структура пайплайна")
    report.head_param_grid(param_grid, description="Сетка гиперпараметров")
    report.head_add_text("В данном эксперименте в качестве контрольного (нормального) окна шириной 6 дней используются дни с 9 по 4 до сейсмического события, положение же целевого (аномального) окна в каждом эксперименте сдвигается на сутки вперед: от окна, совпадающего с контрольным, до окна с 3 по 8 день после события.\n\nВ кажом отчете представленны результаты 4-fold кросс-валидаци с проекциями пространства гиперпараметров в точке оптимума точности (Accuracy).\n\nВ конце отчета находится сводный график метрик разделимости данных подобранных моделей для разных окон со средними и отклонением по 4-м фолдам кросс-валидации.\n \nВ качестве признака используется размахи профиля на некотором разбиении вектора быстроволнового сна. Диаметр разбиения, а также ширина и шаг скользаящего окна для вычисления самого вектора являются гиперпараметрами и в каждом эксперементе различаются.\n\nДля классификации используются простейшие модели: Логистическая регрессия и Метод опорных векторов с разными ядрами, решателями и прочими гиперпараметрами.")
    report.create_json(reports_dir, report_filename, write="rewrite")
    print("Head section added to report.")
    
    # 5. Pre-cache all hypnograms before experiments
    window_positions = list(range(4, -9, -1))  # Window positions from 4 to -8
    cache_results = cache_manager.precache_for_experiment(
        events,
        window_positions,
        window_days=6,
        fixed_control_start_days=9,
        progress_callback=(lambda m: print(m)) if not args.quiet else None
    )
    
    if cache_results['missing'] > 0:
        print("WARNING: Some hypnograms are missing. Experiments may fail for those dates.")
        if args.skip_missing_prompt:
            print("Auto-continuing (--skip-missing-prompt flag set)...")
        else:
            response = input("Continue with experiments anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # 6. Run experiments for each window position (4 to -8, skipping 5 to avoid day 10)
    print(f"\n{'='*80}")
    print(f"=== Running {len(window_positions)} Experiments ===")
    print(f"Window positions: {window_positions}")
    print(f"{'='*80}\n")
    
    # Store results for score dynamics plots
    all_results = {
        'window_positions': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'precision_class_0_mean': [],
        'precision_class_0_std': [],
        'precision_class_1_mean': [],
        'precision_class_1_std': [],
        'recall_mean': [],
        'recall_std': [],
        'roc_auc_mean': [],
        'roc_auc_std': []
    }
    
    
    # Experiment loop
    for window_pos in window_positions:
        print(f"\n{'='*80}")
        print(f"=== Experiment: Window Position {window_pos} ===")
        print(f"{'='*80}\n")
        
        # Calculate window range for immediate window
        # For window_days=6: window is 6 consecutive days ending at abs_step days before event
        # For position p: window ends at p days before event, starts at p+5 days before event
        # So for position p: window is days [p+5, p+4, p+3, p+2, p+1, p] before event (6 days total)
        # (p+5 is furthest from event, p is closest)
        # For negative positions, use absolute value; for positive positions (including 0), use as-is
        abs_step = abs(window_pos) if window_pos < 0 else window_pos  # Window ends at this many days before event
        window_start = window_pos + 5  # Start day (furthest from event, 6 days before end)
        window_end = window_pos  # End day (closest to event)
        
        # Control window: 6 days starting at fixed_control_start_days (days 9, 8, 7, 6, 5, 4 before event)
        control_window_start = 9
        control_window_end = 4  # 6 days: 9, 8, 7, 6, 5, 4
        
        # Format window range string (handle negative numbers)
        immediate_range = f"{window_start} to {window_end}" if window_end >= 0 else f"{window_start} to {window_end}"
        print(f"Immediate window: days {immediate_range} before event ({window_start - window_end + 1} days)")
        print(f"Control window: days {control_window_start} to {control_window_end} before event ({control_window_start - control_window_end + 1} days)\n")
        
        # Create pipeline steps for this experiment
        # For all positions, use window_step_days=-abs_step to position the window correctly
        # Position 0: abs_step=0, window_step_days=0 -> window ends at event-1 (offset 1)
        # Position -1: abs_step=1, window_step_days=-1 -> window ends at event-2 (offset 2) 
        # Position -2: abs_step=2, window_step_days=-2 -> window ends at event-3 (offset 3)
        # Position 4: abs_step=4, window_step_days=-4 -> window ends at event-5 (offset 5)
        # This ensures each position gets a different window with different data
        window_step_days_for_gen = -abs_step  # Negative: window ends at abs_step days before event
        
        steps = [
            ('label_generator', CustomEventLabelGeneratorYt(
                window_days=6,
                window_step_days=window_step_days_for_gen,
                date_format='%Y-%m-%d',
                use_fixed_control_window=True,
                fixed_control_start_days=9,  # Creates 6-day window: days 9, 8, 7, 6, 5, 4 (fixed_control_window_days = window_days = 6)
                original_position=window_pos  # Pass original position to distinguish positive from negative
            )),
        ]
        if args.auto_hypnogram:
            steps.append(auto_hypnogram_step)
        steps.extend([
            ('rem_calculator', REMProfileCalculatorYt(
                cache_manager=cache_manager,
                window_size_hours=6,
                step_size_hours=1,
                rem_stage=2,
                epoch_length_sec=5,
                sampling_rate=250,
                fail_on_missing_data=False
            )),
            ('feature_extractor', REMDailyMultiStatExtractorYt(
                daily_statistics=['max_min_diff', 'mean'],
                window_days=6,
                handle_empty_days='zero'
            )),
            ('scaler', StandardScalerYt(regression=False)),
  
            ('classifier', LogisticRegression())  # Placeholder, will be replaced by grid search
        ])
        
        # Create pipeline
        pipe = PipelineYt(steps)
        
        # Create grid search
        # Note: Environment variables are set at module level, so they should be inherited
        # by joblib worker processes. Each worker will use THREADS_PER_JOB threads.
        grid_search = GridSearchCVYt(
            estimator=pipe,
            param_grid=param_grid,
            cv=4,
            scoring=yt_accuracy_scorer,
            n_jobs=MAX_CORES,  # 13 parallel jobs
            verbose=1,
            error_score=0.0
        )
        
        # Fit grid search
        print("Running grid search...")
        print(f"Using {MAX_CORES} parallel jobs, {THREADS_PER_JOB} thread(s) per job")
        grid_search_start_time = time.time()
        grid_search.fit(X, y)
        grid_search_elapsed_time = time.time() - grid_search_start_time
        
        # Calculate and print grid search time
        hours = int(grid_search_elapsed_time // 3600)
        minutes = int((grid_search_elapsed_time % 3600) // 60)
        seconds = int(grid_search_elapsed_time % 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        print(f"Grid search completed in: {time_str} ({grid_search_elapsed_time:.2f} seconds)\n")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
        print(f"Best classifier: {type(grid_search.best_estimator_.named_steps['classifier']).__name__}\n")
        
        # Get best estimator
        best_estimator = grid_search.best_estimator_
        
        # Extract transformed X and y for metrics calculation
        # We need to get y from the label generator step
        preprocessing_steps = best_estimator.steps[:-1]  # All steps except classifier
        preprocessing_pipeline = PipelineYt(preprocessing_steps)
        X_transformed, y_transformed = preprocessing_pipeline.transform(X, y)
        
        # Check if y_transformed is valid
        if y_transformed is None:
            raise ValueError("y_transformed is None - labels were not generated properly")
        if not hasattr(y_transformed, '__len__') or len(y_transformed) == 0:
            raise ValueError(f"y_transformed is empty or invalid: {y_transformed}")
        
        # Convert to numpy array if needed
        if not isinstance(y_transformed, np.ndarray):
            y_transformed = np.array(y_transformed)
        if y_transformed.ndim == 0:
            raise ValueError(f"y_transformed is a scalar, not an array: {y_transformed}")
        
        # Show preprocessed dataset info
        print(f"\n{'='*80}")
        print("=== Preprocessed Dataset Info ===")
        print(f"{'='*80}")
        print(f"X_transformed shape: {X_transformed.shape}")
        print(f"X_transformed dtype: {X_transformed.dtype}")
        print(f"X_transformed min: {np.min(X_transformed):.4f}, max: {np.max(X_transformed):.4f}")
        print(f"X_transformed mean: {np.mean(X_transformed):.4f}, std: {np.std(X_transformed):.4f}")
        print(f"X_transformed first row: {X_transformed[0] if len(X_transformed) > 0 else 'empty'}")
        print(f"X_transformed hash (first 1000 values): {hash(str(X_transformed[:min(1000, len(X_transformed))].flatten()))}")
        print(f"y_transformed shape: {y_transformed.shape}")
        print(f"y_transformed dtype: {y_transformed.dtype}")
        print(f"y_transformed unique values: {np.unique(y_transformed)}")
        print(f"y_transformed class distribution: {np.bincount(y_transformed.astype(int))}")
        print(f"{'='*80}\n")
        
        
        
        # Calculate comprehensive metrics using GridSearchCVYt's built-in method
        print("Calculating CV metrics for best parameters...")
        # Use the grid search's method to calculate all metrics across all CV folds
        cv_metrics = grid_search.calculate_best_params_metrics(X, y)
        
        # Use best_score_ directly for accuracy to ensure consistency between report and figures
        # This is the authoritative value from grid search
        accuracy_mean = grid_search.best_score_
        accuracy_std = cv_metrics['accuracy_std']  # Use std from cv_metrics calculation
        
        # Store results for plotting
        # Use the same accuracy value as the report table (best_score_) for consistency
        all_results['window_positions'].append(window_pos)
        all_results['accuracy_mean'].append(accuracy_mean)  # Same as report table
        all_results['accuracy_std'].append(accuracy_std)  # Same as report table
        all_results['precision_class_0_mean'].append(cv_metrics['precision_class_0_mean'])
        all_results['precision_class_0_std'].append(cv_metrics['precision_class_0_std'])
        all_results['precision_class_1_mean'].append(cv_metrics['precision_class_1_mean'])
        all_results['precision_class_1_std'].append(cv_metrics['precision_class_1_std'])
        all_results['recall_mean'].append(cv_metrics['recall_mean'])
        all_results['recall_std'].append(cv_metrics['recall_std'])
        all_results['roc_auc_mean'].append(cv_metrics['roc_auc_mean'])
        all_results['roc_auc_std'].append(cv_metrics['roc_auc_std'])
        
        # Add report section for this experiment
        print("Adding report section...")
        immediate_range_str = f"{window_start} to {window_end}"
        control_range_str = f"{control_window_start} to {control_window_end}"
        report.title(f"Позиция сдвига относительно события {window_pos} (аномальное окно: дни {immediate_range_str}; контрольное окно: дни {control_range_str})")
        
        # Hyperparameter grid visualization
        grid_viz_path = os.path.join(args.output_dir, f"Hyperparameter_Grid_Search_pos_{window_pos}.png")
        cv_param_names = [
            name for name in grid_search.cv_results_.keys()
            if name.startswith('param_')
        ]
        param_pairs_for_viz = []
        if len(cv_param_names) >= 2:
            param_pairs_for_viz = list(combinations(cv_param_names, 2))
        report.hyperparameter_grid_visualization(
            grid_viz_path,
            description="Визуализация сетки гиперпараметров",
            recreate=True,
            grid_search=grid_search,
            param_pairs=param_pairs_for_viz if param_pairs_for_viz else None
        )
        
        # Best parameters
        report.best_params(
            grid_search.best_params_,
            description="Лучшие параметры"
        )
        
        # Scores with all metrics
        # Use the same accuracy_mean and accuracy_std defined above for consistency
        scores_dict = {
            "Accuracy": f"{accuracy_mean:.4f} ± {accuracy_std:.4f}",
            "Precision Class 0": f"{cv_metrics['precision_class_0_mean']:.4f} ± {cv_metrics['precision_class_0_std']:.4f}",
            "Precision Class 1": f"{cv_metrics['precision_class_1_mean']:.4f} ± {cv_metrics['precision_class_1_std']:.4f}",
            "Recall": f"{cv_metrics['recall_mean']:.4f} ± {cv_metrics['recall_std']:.4f}",
            "ROC-AUC": f"{cv_metrics['roc_auc_mean']:.4f} ± {cv_metrics['roc_auc_std']:.4f}"
        }
        report.scores(scores_dict, description="Результаты кросс-валидации")
        
        # Save to JSON
        report.create_json(reports_dir, report_filename, write="append")
        print("Report section added.\n")

    
    # 7. Create score dynamics plots after all experiments
    if not args.quiet:
        print(f"\n{'='*80}")
        print("=== Creating Score Dynamics Plots ===")
        print(f"{'='*80}\n")
    score_dynamics_path = os.path.join(args.output_dir, "Score_Dynamics.png")
    plot_score_dynamics(all_results, score_dynamics_path, invert_x=True)
    if not args.quiet:
        print(f"Score dynamics plot saved to: {score_dynamics_path}")
    
    # Add score dynamics plot to report
    report.title("Динамика разделимости по метрикам")
    report.add_fig(score_dynamics_path, description="Динамика метрик по позициям окна")
    report.create_json(reports_dir, report_filename, write="append")
    
    # 8. Compile final report
    print(f"\n{'='*80}")
    print("=== Compiling Final Report ===")
    print(f"{'='*80}\n")
    
    report_output_dir = os.path.join(args.output_dir, "run report")
    print(f"Compiling report to: {report_output_dir}")
    report.compile(reports_dir, report_output_dir, report_filename, compile_to_pdf=args.compile_pdf)
    
    print(f"Report saved to: {os.path.join(report_output_dir, 'experiment.md')}")
    if args.compile_pdf:
        print(f"PDF report saved to: {os.path.join(report_output_dir, 'experiment.pdf')}")
    print()
    
    print("All experiments completed!")

    



if __name__ == "__main__":
    main()


