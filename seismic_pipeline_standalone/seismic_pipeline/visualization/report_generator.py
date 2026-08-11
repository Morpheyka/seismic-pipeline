#!/usr/bin/env python3
"""
Report Generator for Seismic Pipeline Experiments

This module provides a ReportGenerator class for building structured reports,
storing data in JSON format, and compiling to markdown with figures.

Usage:
    from seismic_pipeline.visualization import ReportGenerator
    
    report = ReportGenerator()
    
    # Regular section
    report.title("12 событий")
    report.samples(df.head(10), description="First 10 lines")
    report.pipeline_structure(pipe, description="Pipeline Structure")
    report.param_grid(param_grid_dict, description="Parameter grid")
    report.best_params(best_params, description="Best parameters")
    report.hyperparameter_grid_visualization("./grid.png", recreate=True, grid_search=gs)
    report.scores({"accuracy": "0.8"}, description="CV scores")
    report.error_matrix(cm, description="Confusion matrix")
    report.create_json("./reports", "experiment.json", write="append")
    
    # Head section (only added once)
    report.head_title("Experiment Overview")
    report.head_best_params(best_params, description="Best parameters")
    report.head_scores({"accuracy": "0.8"}, description="Overall scores")
    report.create_json("./reports", "experiment.json", write="append")
    
    # Additional content methods
    report.title("Text Section")
    report.add_text("Some text content", description="Description")
    report.create_json("./reports", "experiment.json", write="append")
    
    report.title("Table Section")
    report.add_table(df, description="Data table")
    report.create_json("./reports", "experiment.json", write="append")
    
    report.title("Figure Section")
    report.add_fig("./plot.png", description="Plot")
    report.create_json("./reports", "experiment.json", write="append")
    
    # Later, compile to markdown
    report.compile("./reports", "./run report", "run_summary.md")
    
    # Or compile to markdown and PDF
    report.compile("./reports", "./run report", "run_summary.md", compile_to_pdf=True)
"""

import json
import os
import re
import shutil
import pprint
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Import visualization function
from .hyperparameter_grid_visualizer import visualize_hyperparameter_grid_slices
from .report_pdf import compile_markdown_to_pdf, detect_pdf_engine


class ReportGenerator:
    """
    Report generator for creating structured experiment reports.
    
    This class provides a fluent API for collecting experiment data,
    storing it in JSON format, and compiling to markdown reports with figures.
    """
    
    def __init__(self):
        """Initialize the report generator with empty current section."""
        self._current_section = {}
        self._matplotlib_fig_counter = 0  # Counter for matplotlib figure naming
        self._reset_section()
    
    def _reset_section(self):
        """Reset the current section to empty state."""
        self._current_section = {
            "title": None,
            "head_title_text": None,
            "samples": None,
            "pipeline_structure": None,
            "best_params": None,
            "param_grid": None,
            "hyperparameter_grid_visualization": None,
            "scores": None,
            "error_matrix": None,
            "text": None,
            "table": None,
            "figure": None
        }
    
    def title(self, text: str):
        """
        Set the section title.
        
        Args:
            text: Title text for the section
        """
        self._current_section["title"] = text
        return self
    
    def samples(self, df: pd.DataFrame, description: Optional[str] = None, 
                n_rows: Optional[int] = None, metadata: bool = False):
        """
        Store DataFrame samples (head or full) as JSON.
        
        This method is now a wrapper around add_table() for consistency.
        
        Args:
            df: pandas DataFrame to store
            description: Optional description text
            n_rows: Number of rows to store (if None, stores full DataFrame; default: None)
            metadata: If True, include shape, dtypes, column names
        """
        if df is None:
            return self
        
        # Get the data to store
        if n_rows is not None and n_rows > 0:
            data_df = df.head(n_rows)
        else:
            data_df = df
        
        # Use add_table internally
        table_data = {
            "description": description,
            "is_file": False
        }
        
        # Convert to dict
        try:
            data_dict = data_df.to_dict('records')
            table_data["data"] = data_dict
        except Exception:
            # Fallback to index-based dict
            data_dict = data_df.to_dict('index')
            table_data["data"] = data_dict
        
        if metadata:
            table_data["metadata"] = {
                "shape": list(df.shape),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "columns": list(df.columns)
            }
        
        self._current_section["table"] = table_data
        # Also keep in samples for backward compatibility
        samples_data = {
            "data": data_dict,
            "description": description
        }
        if metadata:
            samples_data["metadata"] = table_data["metadata"]
        self._current_section["samples"] = samples_data
        return self
    
    def pipeline_structure(self, pipe: Any, description: Optional[str] = None):
        """
        Extract and store pipeline structure.
        
        Args:
            pipe: PipelineYt or sklearn Pipeline object
            description: Optional description text
        """
        if pipe is None:
            return self
        
        steps = []
        
        # Handle PipelineYt or sklearn Pipeline
        if hasattr(pipe, 'steps'):
            pipeline_steps = pipe.steps
        elif hasattr(pipe, 'named_steps'):
            pipeline_steps = [(name, step) for name, step in pipe.named_steps.items()]
        else:
            pipeline_steps = []
        
        for name, step in pipeline_steps:
            step_info = {
                "name": name,
                "class_name": type(step).__name__
            }
            
            # Try to get parameters
            try:
                if hasattr(step, 'get_params'):
                    params = step.get_params(deep=False)
                    # Convert numpy types to Python types for JSON serialization
                    step_info["parameters"] = self._convert_to_json_serializable(params)
                else:
                    step_info["parameters"] = {}
            except Exception:
                step_info["parameters"] = {}
            
            steps.append(step_info)
        
        self._current_section["pipeline_structure"] = {
            "steps": steps,
            "description": description
        }
        return self
    
    def best_params(self, param_dict: Dict[str, Any], description: Optional[str] = None):
        """
        Store best hyperparameters dictionary.
        
        This method is now a wrapper around add_table() but preserves special formatting
        (two-column table with "Параметр | Значение") during compilation for backward compatibility.
        
        Args:
            param_dict: Dictionary of best hyperparameters
            description: Optional description text
        """
        if param_dict is None:
            return self
        
        # Convert to JSON-serializable format
        serializable_params = self._convert_to_json_serializable(param_dict)
        
        # Store using add_table internally, but mark it as best_params for special formatting
        table_data = {
            "description": description,
            "data": serializable_params,
            "is_file": False,
            "is_best_params": True  # Flag for special formatting during compilation
        }
        self._current_section["table"] = table_data
        # Also keep in best_params for backward compatibility
        best_params_data = {
            "params": serializable_params,
            "description": description
        }
        self._current_section["best_params"] = best_params_data
        return self
    
    def param_grid(self, param_grid_dict: Dict[str, Any], description: Optional[str] = None):
        """
        Store parameter grid dictionary (all tested values).
        
        This method is now a wrapper around add_table() but preserves special formatting
        (two-column table with "Параметр | Возможные значения") during compilation for backward compatibility.
        
        Args:
            param_grid_dict: Dictionary of parameter grid (all tested values)
            description: Optional description text
        """
        if param_grid_dict is None:
            return self
        
        # Convert to JSON-serializable format
        serializable_grid = self._convert_to_json_serializable(param_grid_dict)
        
        # Store using add_table internally, but mark it as param_grid for special formatting
        table_data = {
            "description": description,
            "data": serializable_grid,
            "is_file": False,
            "is_param_grid": True  # Flag for special formatting during compilation
        }
        self._current_section["table"] = table_data
        # Also keep in param_grid for backward compatibility
        param_grid_data = {
            "param_grid": serializable_grid,
            "description": description
        }
        self._current_section["param_grid"] = param_grid_data
        return self
    
    def hyperparameter_grid_visualization(self, path: str, description: Optional[str] = None,
                                         recreate: bool = False, grid_search: Optional[Any] = None,
                                         param_pairs: Optional[List[tuple]] = None,
                                         focus_param_prefixes: Optional[List[str]] = None):
        """
        Store hyperparameter grid visualization path, optionally recreate.
        
        Args:
            path: Path to visualization file
            description: Optional description text
            recreate: If True and grid_search provided, recreate visualization
            grid_search: GridSearchCV object for recreating visualization
            param_pairs: Optional list of tuples specifying which parameter pairs to visualize.
                        Each tuple should contain two parameter names (with 'param_' prefix).
                        If None, all possible pairs will be visualized.
            focus_param_prefixes: Optional list of parameter prefixes to auto-select when
                                  param_pairs is not provided. Defaults to pipeline preprocessing
                                  components (label generator, REM calculator, feature extractor).
        """
        if recreate and grid_search is not None:
            # Recreate the visualization
            output_dir = os.path.dirname(path) if os.path.dirname(path) else '.'
            # Extract feature name from path (without extension)
            feature_name = os.path.splitext(os.path.basename(path))[0]
            try:
                visualize_hyperparameter_grid_slices(
                    grid_search,
                    output_dir,
                    param_pairs=param_pairs,
                    focus_param_prefixes=focus_param_prefixes,
                    feature_name=feature_name
                )
                # Update path to the generated file (with .png extension)
                generated_path = os.path.join(output_dir, f"{feature_name}.png")
                if os.path.exists(generated_path):
                    path = generated_path
            except Exception as e:
                print(f"Warning: Failed to recreate visualization: {e}")
        
        self._current_section["hyperparameter_grid_visualization"] = {
            "path": path,
            "description": description
        }
        return self
    
    def scores(self, score_dict: Dict[str, Union[str, float]], description: Optional[str] = None):
        """
        Store score dictionary.
        
        This method is now a wrapper around add_table() but preserves special formatting
        (shell code block) during compilation for backward compatibility.
        
        Args:
            score_dict: Dictionary of scores (e.g., {"accuracy": "0.8", ...})
            description: Optional description text
        """
        if score_dict is None:
            return self
        
        # Convert values to strings for consistency
        serializable_scores = {k: str(v) for k, v in score_dict.items()}
        
        # Store using add_table internally, but mark it as scores for special formatting
        table_data = {
            "description": description,
            "data": serializable_scores,
            "is_file": False,
            "is_scores": True  # Flag for special formatting during compilation
        }
        self._current_section["table"] = table_data
        # Also keep in scores for backward compatibility
        self._current_section["scores"] = {
            "scores": serializable_scores,
            "description": description
        }
        return self
    
    def error_matrix(self, matrix: np.ndarray, description: Optional[str] = None):
        """
        Store confusion/error matrix (numpy array) as nested list.
        
        This method is now a wrapper around add_table() but preserves special formatting
        (table with "Class 0", "Class 1" headers) during compilation for backward compatibility.
        
        Args:
            matrix: Numpy array representing confusion/error matrix
            description: Optional description text
        """
        if matrix is None:
            return self
        
        # Convert numpy array to nested list
        matrix_list = matrix.tolist()
        
        # Store using add_table internally, but mark it as error_matrix for special formatting
        table_data = {
            "description": description,
            "data": matrix_list,
            "is_file": False,
            "is_error_matrix": True  # Flag for special formatting during compilation
        }
        self._current_section["table"] = table_data
        # Also keep in error_matrix for backward compatibility
        self._current_section["error_matrix"] = {
            "matrix": matrix_list,
            "description": description
        }
        return self
    
    def _has_head_section(self, json_path: str) -> bool:
        """
        Check if JSON file already contains a head section (title == "__HEAD__").
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            True if head section exists, False otherwise
        """
        if not os.path.exists(json_path):
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)
            if not isinstance(sections, list):
                return False
            
            # Check if any section has title "__HEAD__"
            for section in sections:
                if section.get("title") == "__HEAD__":
                    return True
            return False
        except Exception:
            return False
    
    def head_title(self, text: str):
        """
        Set the head section title (only if head section doesn't exist).
        
        Args:
            text: Title text for the head section
        """
        # Set title to "__HEAD__" to mark as head section
        self._current_section["title"] = "__HEAD__"
        # Store the actual title text for display
        self._current_section["head_title_text"] = text
        return self
    
    def head_samples(self, df: pd.DataFrame, description: Optional[str] = None, 
                    n_rows: Optional[int] = None, metadata: bool = False):
        """
        Store DataFrame samples in head section (only if head section doesn't exist).
        
        Args:
            df: pandas DataFrame to store
            description: Optional description text
            n_rows: Number of rows to store (if None, stores full DataFrame; default: None)
            metadata: If True, include shape, dtypes, column names
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        # Use the regular samples method logic
        if df is None:
            return self
        
        if n_rows is not None and n_rows > 0:
            data_df = df.head(n_rows)
        else:
            data_df = df
        
        try:
            data_dict = data_df.to_dict('records')
        except Exception:
            data_dict = data_df.to_dict('index')
        
        samples_data = {
            "data": data_dict,
            "description": description
        }
        
        if metadata:
            samples_data["metadata"] = {
                "shape": list(df.shape),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
                "columns": list(df.columns)
            }
        
        self._current_section["samples"] = samples_data
        return self
    
    def head_pipeline_structure(self, pipe: Any, description: Optional[str] = None):
        """
        Extract and store pipeline structure in head section (only if head section doesn't exist).
        
        Args:
            pipe: PipelineYt or sklearn Pipeline object
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        # Use the regular pipeline_structure method logic
        if pipe is None:
            return self
        
        steps = []
        
        if hasattr(pipe, 'steps'):
            pipeline_steps = pipe.steps
        elif hasattr(pipe, 'named_steps'):
            pipeline_steps = [(name, step) for name, step in pipe.named_steps.items()]
        else:
            pipeline_steps = []
        
        for name, step in pipeline_steps:
            step_info = {
                "name": name,
                "class_name": type(step).__name__
            }
            
            try:
                if hasattr(step, 'get_params'):
                    params = step.get_params(deep=False)
                    step_info["parameters"] = self._convert_to_json_serializable(params)
                else:
                    step_info["parameters"] = {}
            except Exception:
                step_info["parameters"] = {}
            
            steps.append(step_info)
        
        self._current_section["pipeline_structure"] = {
            "steps": steps,
            "description": description
        }
        return self
    
    def head_best_params(self, param_dict: Dict[str, Any], description: Optional[str] = None):
        """
        Store best hyperparameters in head section (only if head section doesn't exist).
        
        Args:
            param_dict: Dictionary of best hyperparameters
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if param_dict is None:
            return self
        
        serializable_params = self._convert_to_json_serializable(param_dict)
        
        best_params_data = {
            "params": serializable_params,
            "description": description
        }
        
        self._current_section["best_params"] = best_params_data
        return self
    
    def head_param_grid(self, param_grid_dict: Dict[str, Any], description: Optional[str] = None):
        """
        Store parameter grid in head section (only if head section doesn't exist).
        
        Args:
            param_grid_dict: Dictionary of parameter grid (all tested values)
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if param_grid_dict is None:
            return self
        
        serializable_grid = self._convert_to_json_serializable(param_grid_dict)
        
        param_grid_data = {
            "param_grid": serializable_grid,
            "description": description
        }
        
        self._current_section["param_grid"] = param_grid_data
        return self
    
    def head_hyperparameter_grid_visualization(self, path: str, description: Optional[str] = None,
                                               recreate: bool = False, grid_search: Optional[Any] = None,
                                               param_pairs: Optional[List[tuple]] = None,
                                               focus_param_prefixes: Optional[List[str]] = None):
        """
        Store hyperparameter grid visualization in head section (only if head section doesn't exist).
        
        Args:
            path: Path to visualization file
            description: Optional description text
            recreate: If True and grid_search provided, recreate visualization
            grid_search: GridSearchCV object for recreating visualization
            param_pairs: Optional list of tuples specifying which parameter pairs to visualize
            focus_param_prefixes: Optional list of parameter prefixes to auto-select
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if recreate and grid_search is not None:
            output_dir = os.path.dirname(path) if os.path.dirname(path) else '.'
            try:
                visualize_hyperparameter_grid_slices(
                    grid_search,
                    output_dir,
                    param_pairs=param_pairs,
                    focus_param_prefixes=focus_param_prefixes
                )
                generated_path = os.path.join(output_dir, "Hyperparameter_Grid_Search.png")
                if os.path.exists(generated_path):
                    path = generated_path
            except Exception as e:
                print(f"Warning: Failed to recreate visualization: {e}")
        
        self._current_section["hyperparameter_grid_visualization"] = {
            "path": path,
            "description": description
        }
        return self
    
    def head_scores(self, score_dict: Dict[str, Union[str, float]], description: Optional[str] = None):
        """
        Store score dictionary in head section (only if head section doesn't exist).
        
        Args:
            score_dict: Dictionary of scores (e.g., {"accuracy": "0.8", ...})
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if score_dict is None:
            return self
        
        serializable_scores = {k: str(v) for k, v in score_dict.items()}
        
        self._current_section["scores"] = {
            "scores": serializable_scores,
            "description": description
        }
        return self
    
    def head_error_matrix(self, matrix: np.ndarray, description: Optional[str] = None):
        """
        Store confusion/error matrix in head section (only if head section doesn't exist).
        
        Args:
            matrix: Numpy array representing confusion/error matrix
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if matrix is None:
            return self
        
        matrix_list = matrix.tolist()
        
        self._current_section["error_matrix"] = {
            "matrix": matrix_list,
            "description": description
        }
        return self
    
    def head_add_text(self, content: Union[str, Path], description: Optional[str] = None):
        """
        Add text content to head section (only if head section doesn't exist).
        
        Args:
            content: String text or path to text file (.txt, .log)
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if content is None:
            return self
        
        text_data = {
            "description": description
        }
        
        if self._is_file_path(content):
            # Store file path reference
            text_data["file_path"] = str(content)
            text_data["is_file"] = True
        else:
            # Store text content directly
            text_data["content"] = str(content)
            text_data["is_file"] = False
        
        self._current_section["text"] = text_data
        return self
    
    def head_add_table(self, data: Union[pd.DataFrame, List[Dict], Dict, List[List], Path], 
                      description: Optional[str] = None):
        """
        Add table data to head section (only if head section doesn't exist).
        
        Args:
            data: Table data (DataFrame, list of dicts, dict, list of lists) or file path
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if data is None:
            return self
        
        table_data = {
            "description": description
        }
        
        if self._is_file_path(data):
            # Store file path reference
            table_data["file_path"] = str(data)
            table_data["is_file"] = True
        else:
            # Convert data to JSON-serializable format
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of dicts
                try:
                    table_data["data"] = data.to_dict('records')
                    table_data["metadata"] = {
                        "shape": list(data.shape),
                        "columns": list(data.columns),
                        "dtypes": {str(k): str(v) for k, v in data.dtypes.items()}
                    }
                except Exception:
                    table_data["data"] = data.to_dict('index')
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    # List of dicts
                    table_data["data"] = self._convert_to_json_serializable(data)
                else:
                    # List of lists or other list types
                    table_data["data"] = self._convert_to_json_serializable(data)
            elif isinstance(data, dict):
                # Dictionary - convert to list of key-value pairs or keep as dict
                table_data["data"] = self._convert_to_json_serializable(data)
            else:
                # Fallback: convert to string representation
                table_data["data"] = str(data)
            
            table_data["is_file"] = False
        
        self._current_section["table"] = table_data
        return self
    
    def head_add_fig(self, source: Union[str, Path, Any], description: Optional[str] = None):
        """
        Add figure to head section (only if head section doesn't exist).
        
        Args:
            source: File path or matplotlib figure object
            description: Optional description text
        """
        # Ensure title is set to "__HEAD__"
        if self._current_section.get("title") != "__HEAD__":
            self._current_section["title"] = "__HEAD__"
        
        if source is None:
            return self
        
        figure_data = {
            "description": description
        }
        
        # Check if it's a matplotlib figure
        try:
            import matplotlib.figure
            if isinstance(source, matplotlib.figure.Figure):
                # Save matplotlib figure to a predictable location
                # Use counter and timestamp for unique but readable names
                import tempfile
                import uuid
                
                # Create a session-specific temp directory for this report generator instance
                if not hasattr(self, '_temp_fig_dir'):
                    temp_base = tempfile.gettempdir()
                    session_id = str(uuid.uuid4())[:8]
                    self._temp_fig_dir = os.path.join(temp_base, f"reportgen_figs_{session_id}")
                    os.makedirs(self._temp_fig_dir, exist_ok=True)
                
                self._matplotlib_fig_counter += 1
                temp_path = os.path.join(self._temp_fig_dir, f"fig_{self._matplotlib_fig_counter:03d}.png")
                source.savefig(temp_path, dpi=150, bbox_inches='tight')
                figure_data["path"] = temp_path
                figure_data["is_file"] = True
                figure_data["is_matplotlib"] = True
                self._current_section["figure"] = figure_data
                return self
        except (ImportError, AttributeError):
            pass
        
        # Check if it's a file path
        if self._is_file_path(source):
            figure_data["path"] = str(source)
            figure_data["is_file"] = True
            figure_data["is_matplotlib"] = False
        else:
            # Try to treat as file path anyway
            source_str = str(source)
            if os.path.exists(source_str):
                figure_data["path"] = source_str
                figure_data["is_file"] = True
                figure_data["is_matplotlib"] = False
            else:
                # Invalid input
                print(f"Warning: head_add_fig received invalid input: {type(source)}")
                return self
        
        self._current_section["figure"] = figure_data
        return self
    
    def _is_file_path(self, obj: Any) -> bool:
        """
        Determine if the object is a file path (string or Path object).
        
        Args:
            obj: Object to check
            
        Returns:
            True if object appears to be a file path, False otherwise
        """
        if isinstance(obj, Path):
            return True
        
        if isinstance(obj, str):
            path_str = obj
            # If string contains newlines, it's definitely not a file path
            if '\n' in path_str or '\r' in path_str:
                return False
            
            # Check if it's actually an existing file
            if os.path.isfile(path_str):
                return True
            
            # Check if parent directory exists (suggests it's a path)
            parent_dir = os.path.dirname(path_str)
            if parent_dir and os.path.isdir(parent_dir):
                # Check for common file extensions
                ext = os.path.splitext(path_str)[1].lower()
                if ext in ['.txt', '.log', '.csv', '.png', '.jpeg', '.jpg']:
                    return True
            
            # Check for common file extensions even without existing parent
            # but only if the string is short and looks like a path
            ext = os.path.splitext(path_str)[1].lower()
            if ext in ['.txt', '.log', '.csv', '.png', '.jpeg', '.jpg']:
                # Only treat as file if it's a reasonable path (not too long, no spaces in middle)
                if len(path_str) < 500 and (os.sep in path_str or path_str.startswith('.')):
                    return True
        
        return False
    
    def add_text(self, content: Union[str, Path], description: Optional[str] = None):
        """
        Add text content to the current section.
        
        Can accept either a string variable or a file path (.txt, .log).
        Each call creates a new section when create_json() is called.
        
        Args:
            content: String text or path to text file (.txt, .log)
            description: Optional description text
        """
        if content is None:
            return self
        
        text_data = {
            "description": description
        }
        
        if self._is_file_path(content):
            # Store file path reference
            text_data["file_path"] = str(content)
            text_data["is_file"] = True
        else:
            # Store text content directly
            text_data["content"] = str(content)
            text_data["is_file"] = False
        
        self._current_section["text"] = text_data
        return self
    
    def add_table(self, data: Union[pd.DataFrame, List[Dict], Dict, List[List], Path], 
                  description: Optional[str] = None):
        """
        Add table data to the current section.
        
        Can accept DataFrame, list of dicts, dict, list of lists, or file path (.csv, .txt).
        Each call creates a new section when create_json() is called.
        
        Args:
            data: Table data (DataFrame, list of dicts, dict, list of lists) or file path
            description: Optional description text
        """
        if data is None:
            return self
        
        table_data = {
            "description": description
        }
        
        if self._is_file_path(data):
            # Store file path reference
            table_data["file_path"] = str(data)
            table_data["is_file"] = True
        else:
            # Convert data to JSON-serializable format
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of dicts
                try:
                    table_data["data"] = data.to_dict('records')
                    table_data["metadata"] = {
                        "shape": list(data.shape),
                        "columns": list(data.columns),
                        "dtypes": {str(k): str(v) for k, v in data.dtypes.items()}
                    }
                except Exception:
                    table_data["data"] = data.to_dict('index')
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], dict):
                    # List of dicts
                    table_data["data"] = self._convert_to_json_serializable(data)
                else:
                    # List of lists or other list types
                    table_data["data"] = self._convert_to_json_serializable(data)
            elif isinstance(data, dict):
                # Dictionary - convert to list of key-value pairs or keep as dict
                table_data["data"] = self._convert_to_json_serializable(data)
            else:
                # Fallback: convert to string representation
                table_data["data"] = str(data)
            
            table_data["is_file"] = False
        
        self._current_section["table"] = table_data
        return self
    
    def add_fig(self, source: Union[str, Path, Any], description: Optional[str] = None):
        """
        Add figure to the current section.
        
        Can accept file paths (.png, .jpeg, .jpg) or matplotlib figure objects.
        Each call creates a new section when create_json() is called.
        
        Args:
            source: File path or matplotlib figure object
            description: Optional description text
        """
        if source is None:
            return self
        
        figure_data = {
            "description": description
        }
        
        # Check if it's a matplotlib figure
        try:
            import matplotlib.figure
            if isinstance(source, matplotlib.figure.Figure):
                # Save matplotlib figure to a predictable location
                # Use counter and timestamp for unique but readable names
                import tempfile
                import time
                import uuid
                
                # Create a session-specific temp directory for this report generator instance
                if not hasattr(self, '_temp_fig_dir'):
                    temp_base = tempfile.gettempdir()
                    session_id = str(uuid.uuid4())[:8]
                    self._temp_fig_dir = os.path.join(temp_base, f"reportgen_figs_{session_id}")
                    os.makedirs(self._temp_fig_dir, exist_ok=True)
                
                self._matplotlib_fig_counter += 1
                temp_path = os.path.join(self._temp_fig_dir, f"fig_{self._matplotlib_fig_counter:03d}.png")
                source.savefig(temp_path, dpi=150, bbox_inches='tight')
                figure_data["path"] = temp_path
                figure_data["is_file"] = True
                figure_data["is_matplotlib"] = True
                self._current_section["figure"] = figure_data
                return self
        except (ImportError, AttributeError):
            pass
        
        # Check if it's a file path
        if self._is_file_path(source):
            figure_data["path"] = str(source)
            figure_data["is_file"] = True
            figure_data["is_matplotlib"] = False
        else:
            # Try to treat as file path anyway
            source_str = str(source)
            if os.path.exists(source_str):
                figure_data["path"] = source_str
                figure_data["is_file"] = True
                figure_data["is_matplotlib"] = False
            else:
                # Invalid input
                print(f"Warning: add_fig received invalid input: {type(source)}")
                return self
        
        self._current_section["figure"] = figure_data
        return self
    
    def create_json(self, path: str, file_name: str, write: str = "append"):
        """
        Save current section to JSON file.
        
        Args:
            path: Directory path where JSON file should be saved
            file_name: Name of JSON file
            write: "append" to append to existing JSON, "rewrite" to overwrite
        """
        if write not in ["append", "rewrite"]:
            raise ValueError(f"write must be 'append' or 'rewrite', got '{write}'")
        
        # Ensure path exists
        os.makedirs(path, exist_ok=True)
        
        json_path = os.path.join(path, file_name)
        
        # Load existing data if appending
        sections = []
        if write == "append" and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    sections = json.load(f)
                if not isinstance(sections, list):
                    sections = []
            except Exception as e:
                print(f"Warning: Failed to load existing JSON, starting fresh: {e}")
                sections = []
        
        # Check if current section is a head section
        is_head_section = self._current_section.get("title") == "__HEAD__"
        
        # If it's a head section, check if one already exists
        if is_head_section:
            if self._has_head_section(json_path):
                print("Head section already exists, skipping...")
                self._reset_section()
                return self
        
        # Copy temporary matplotlib figures to persistent location (reports directory)
        figure_data = self._current_section.get("figure")
        if figure_data and figure_data.get("is_matplotlib") and figure_data.get("path"):
            temp_fig_path = figure_data["path"]
            # Check if it's a temporary matplotlib figure path
            if temp_fig_path.startswith('/tmp/reportgen_figs_') and os.path.exists(temp_fig_path):
                # Create a figures subdirectory in reports directory
                figures_dir = os.path.join(path, "figures")
                os.makedirs(figures_dir, exist_ok=True)
                
                # Generate a persistent filename based on counter
                fig_basename = os.path.basename(temp_fig_path)
                persistent_path = os.path.join(figures_dir, fig_basename)
                
                # Copy the temporary figure to persistent location
                try:
                    shutil.copy2(temp_fig_path, persistent_path)
                    # Update the path in current section to use persistent path
                    figure_data["path"] = persistent_path
                    self._current_section["figure"] = figure_data
                except Exception as e:
                    print(f"Warning: Failed to copy matplotlib figure from {temp_fig_path} to {persistent_path}: {e}")
        
        # Add current section if it has at least a title, or if it has content (text, table, figure)
        has_content = (
            self._current_section.get("text") or
            self._current_section.get("table") or
            self._current_section.get("figure") or
            self._current_section.get("samples") or
            self._current_section.get("pipeline_structure") or
            self._current_section.get("best_params") or
            self._current_section.get("param_grid") or
            self._current_section.get("hyperparameter_grid_visualization") or
            self._current_section.get("scores") or
            self._current_section.get("error_matrix")
        )
        if self._current_section.get("title") or has_content:
            sections.append(self._current_section.copy())
        
        # Save to JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sections, f, indent=2, ensure_ascii=False)
            print(f"Saved report section to: {json_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
            raise
        
        # Reset current section
        self._reset_section()
        return self
    
    def compile(self, from_path: str, to_path: str, file_name: str, 
                compile_to_pdf: bool = False, pdf_engine: Optional[str] = None):
        """
        Parse JSON, generate markdown, and copy figures.
        Optionally compile markdown to PDF.
        
        Args:
            from_path: Directory containing JSON file
            to_path: Directory where markdown and figures should be saved
            file_name: Name of JSON file (without extension) - will create .md file
            compile_to_pdf: If True, also compile markdown to PDF (default: False)
            pdf_engine: PDF engine to use ('pandoc', 'xelatex', 'pdflatex', 'lualatex', 'wkhtmltopdf', 'weasyprint').
                       If None, will try to auto-detect available engine (default: None)
        """
        # Load JSON file
        json_path = os.path.join(from_path, file_name)
        if not json_path.endswith('.json'):
            json_path += '.json'
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sections = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")
        
        if not isinstance(sections, list):
            raise ValueError("JSON file must contain a list of sections")
        
        # Create output directory
        os.makedirs(to_path, exist_ok=True)
        
        # Separate head sections from regular sections
        head_sections = [s for s in sections if s.get("title") == "__HEAD__"]
        regular_sections = [s for s in sections if s.get("title") != "__HEAD__"]
        
        # Generate markdown
        md_content = []
        section_slug_counters: Dict[str, int] = {}
        section_fig_counters: Dict[str, int] = {}
        # Track image references: source_path -> list of (section_slug, section_index, description, source_path)
        # We'll use the LAST section's slug for the actual file, but create symlinks for all sections
        image_references: Dict[str, List[tuple]] = {}  # source_path -> [(section_slug, section_index, description, source_path), ...]
        
        # Combine sections for processing (head first, then regular)
        all_sections = head_sections + regular_sections
        
        # First pass: collect all image references and determine section slugs
        for section_index, section in enumerate(all_sections, start=1):
            # Section title
            title = section.get("title", "Untitled Section")
            section_slug_base = self._slugify(title)
            if not section_slug_base:
                section_slug_base = f"section-{section_index}"
            
            slug_count = section_slug_counters.get(section_slug_base, 0)
            section_slug_counters[section_slug_base] = slug_count + 1
            if slug_count > 0:
                section_slug = f"{section_slug_base}-{slug_count + 1}"
            else:
                section_slug = section_slug_base
            
            section_fig_counters.setdefault(section_slug, 0)
            
            # Hyperparameter grid visualization (image) - collect reference
            viz_data = section.get("hyperparameter_grid_visualization")
            if viz_data and viz_data.get("path"):
                viz_path = viz_data["path"]
                description = viz_data.get("description", "Описание изображения")
                
                source_path = None
                if os.path.exists(viz_path):
                    source_path = viz_path
                else:
                    alt_path = os.path.join(from_path, os.path.basename(viz_path))
                    if os.path.exists(alt_path):
                        source_path = alt_path
                    else:
                        # Try relative to current working directory
                        cwd_path = os.path.join(os.getcwd(), os.path.basename(viz_path))
                        if os.path.exists(cwd_path):
                            source_path = cwd_path
                
                if source_path:
                    # Normalize source path to handle relative/absolute paths
                    source_path_normalized = os.path.realpath(os.path.abspath(source_path))
                    # Track this reference - add to list for this source
                    if source_path_normalized not in image_references:
                        image_references[source_path_normalized] = []
                    image_references[source_path_normalized].append((section_slug, section_index, description, source_path))
            
            # Figure content (image) - collect reference
            figure_data = section.get("figure")
            if figure_data and figure_data.get("path"):
                fig_path = figure_data["path"]
                description = figure_data.get("description", "Figure")
                
                source_path = None
                if os.path.exists(fig_path):
                    source_path = fig_path
                else:
                    # Check if it's a matplotlib temp file path
                    if fig_path.startswith('/tmp/reportgen_figs_'):
                        # First, check if current instance has temp dir
                        if hasattr(self, '_temp_fig_dir'):
                            fig_basename = os.path.basename(fig_path)
                            current_temp_path = os.path.join(self._temp_fig_dir, fig_basename)
                            if os.path.exists(current_temp_path):
                                source_path = current_temp_path
                        
                        # Also check in figures subdirectory (where persistent copies are stored)
                        if not source_path:
                            fig_basename = os.path.basename(fig_path)
                            figures_dir = os.path.join(from_path, "figures")
                            persistent_path = os.path.join(figures_dir, fig_basename)
                            if os.path.exists(persistent_path):
                                source_path = persistent_path
                    
                    if not source_path:
                        alt_path = os.path.join(from_path, os.path.basename(fig_path))
                        if os.path.exists(alt_path):
                            source_path = alt_path
                        else:
                            cwd_path = os.path.join(os.getcwd(), os.path.basename(fig_path))
                            if os.path.exists(cwd_path):
                                source_path = cwd_path
                
                if source_path:
                    source_path_normalized = os.path.realpath(os.path.abspath(source_path))
                    if source_path_normalized not in image_references:
                        image_references[source_path_normalized] = []
                    image_references[source_path_normalized].append((section_slug, section_index, description, source_path))
        
        # Now copy images using the LAST section's slug for each unique source file
        # But also create symlinks for each section that references the same source
        source_file_to_dest_name: Dict[str, str] = {}  # source -> actual copied filename (last section's name)
        source_file_to_section_names: Dict[str, Dict[str, str]] = {}  # source -> {section_slug: filename}
        
        for source_path_normalized, refs_list in image_references.items():
            # Get the LAST section that references this source (highest section_index)
            last_ref = max(refs_list, key=lambda x: x[1])  # x[1] is section_index
            last_section_slug, last_section_index, description, source_path = last_ref
            
            # Copy using the LAST section's slug (only once per unique source)
            section_fig_counters[last_section_slug] += 1
            fig_counter = section_fig_counters[last_section_slug]
            _, ext = os.path.splitext(os.path.basename(source_path))
            if not ext:
                ext = ".png"
            unique_fig_name = f"{last_section_slug}__{fig_counter:02d}{ext}"
            dest_path = os.path.join(to_path, unique_fig_name)
            shutil.copy2(source_path, dest_path)
            source_file_to_dest_name[source_path_normalized] = unique_fig_name
            source_file_to_section_names[source_path_normalized] = {}
            
            # Create symlinks/copies for each section that references this source
            actual_path = os.path.join(to_path, unique_fig_name)
            for section_slug, section_index, _, _ in refs_list:
                if section_slug == last_section_slug:
                    # This section's slug matches the actual file name
                    source_file_to_section_names[source_path_normalized][section_slug] = unique_fig_name
                else:
                    # This section needs its own symlink/copy
                    section_fig_name = f"{section_slug}__{fig_counter:02d}{ext}"
                    section_symlink_path = os.path.join(to_path, section_fig_name)
                    if not os.path.exists(section_symlink_path):
                        try:
                            os.symlink(unique_fig_name, section_symlink_path)
                            source_file_to_section_names[source_path_normalized][section_slug] = section_fig_name
                        except OSError:
                            # If symlink fails (e.g., on Windows), just copy the file
                            shutil.copy2(actual_path, section_symlink_path)
                            source_file_to_section_names[source_path_normalized][section_slug] = section_fig_name
                    else:
                        source_file_to_section_names[source_path_normalized][section_slug] = section_fig_name
        
        # Second pass: generate markdown content, referencing the copied images
        for section_index, section in enumerate(all_sections, start=1):
            # Section title
            title = section.get("title", "Untitled Section")
            is_head = title == "__HEAD__"
            
            # For head sections, use head_title_text if available, otherwise "Overview"
            if is_head:
                display_title = section.get("head_title_text", "Overview")
            else:
                display_title = title if title and title != "None" else None
            
            section_slug_base = self._slugify(title) if title and title != "None" else None
            if not section_slug_base:
                section_slug_base = f"section-{section_index}"
            
            slug_count = section_slug_counters.get(section_slug_base, 0)
            if slug_count > 1:
                # Need to recalculate slug to match first pass
                actual_slug_count = sum(1 for i, s in enumerate(all_sections[:section_index], 1) 
                                       if self._slugify(s.get("title", "")) == section_slug_base)
                if actual_slug_count > 1:
                    section_slug = f"{section_slug_base}-{actual_slug_count}"
                else:
                    section_slug = section_slug_base
            else:
                section_slug = section_slug_base
            
            # Only add title if it exists and is not None
            if display_title:
                md_content.append(f"## {display_title}\n\n")
            
            # Hyperparameter grid visualization (image) - reference the copied image
            viz_data = section.get("hyperparameter_grid_visualization")
            if viz_data and viz_data.get("path"):
                viz_path = viz_data["path"]
                description = viz_data.get("description", "Описание изображения")
                
                source_path = None
                if os.path.exists(viz_path):
                    source_path = viz_path
                else:
                    alt_path = os.path.join(from_path, os.path.basename(viz_path))
                    if os.path.exists(alt_path):
                        source_path = alt_path
                    else:
                        cwd_path = os.path.join(os.getcwd(), os.path.basename(viz_path))
                        if os.path.exists(cwd_path):
                            source_path = cwd_path
                
                if source_path:
                    source_path_normalized = os.path.realpath(os.path.abspath(source_path))
                    if source_path_normalized in source_file_to_section_names:
                        # Use this section's specific filename (which may be a symlink)
                        section_names_dict = source_file_to_section_names[source_path_normalized]
                        if section_slug in section_names_dict:
                            unique_fig_name = section_names_dict[section_slug]
                        else:
                            # Fallback to the actual copied filename
                            unique_fig_name = source_file_to_dest_name[source_path_normalized]
                        md_content.append(f"![{description}]({unique_fig_name})\n\n")
                    elif source_path_normalized in source_file_to_dest_name:
                        # Fallback: use the actual copied filename
                        unique_fig_name = source_file_to_dest_name[source_path_normalized]
                        md_content.append(f"![{description}]({unique_fig_name})\n\n")
                    else:
                        fallback_name = os.path.basename(viz_path)
                        md_content.append(f"![{description}]({fallback_name})\n\n")
                        print(f"Warning: Figure not found: {viz_path}")
                else:
                    fallback_name = os.path.basename(viz_path)
                    md_content.append(f"![{description}]({fallback_name})\n\n")
                    print(f"Warning: Figure not found: {viz_path}")
            
            # Samples (events list) - Markdown table
            samples_data = section.get("samples")
            if samples_data and samples_data.get("data"):
                description = samples_data.get("description")
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                # Format as markdown table
                data = samples_data["data"]
                if isinstance(data, list) and len(data) > 0:
                    # Get column names from metadata or infer from first dict
                    if isinstance(data[0], dict):
                        columns = list(data[0].keys())
                        # Use metadata columns if available
                        if samples_data.get("metadata") and samples_data["metadata"].get("columns"):
                            columns = samples_data["metadata"]["columns"]
                        
                        # Create table header
                        md_content.append("| " + " | ".join(columns) + " |\n")
                        md_content.append("|" + "|".join(["----------"] * len(columns)) + "|\n")
                        
                        # Add data rows
                        for item in data:
                            row_values = [str(item.get(col, "")) for col in columns]
                            md_content.append("| " + " | ".join(row_values) + " |\n")
                        md_content.append("\n")
                    else:
                        # Fallback for non-dict data
                        md_content.append(str(data) + "\n\n")
                else:
                    md_content.append(str(data) + "\n\n")
                
                metadata_info = samples_data.get("metadata")
                if metadata_info:
                    md_content.append("**Метаданные выборки:**\n\n")
                    md_content.append("| Свойство | Значение |\n")
                    md_content.append("|----------|----------|\n")
                    shape = metadata_info.get("shape")
                    if shape is not None:
                        md_content.append(f"| Размер | {shape} |\n")
                    columns = metadata_info.get("columns")
                    if columns is not None:
                        md_content.append(f"| Колонки | {', '.join(columns)} |\n")
                    dtypes = metadata_info.get("dtypes")
                    if dtypes:
                        dtype_str = ", ".join([f"{col}: {dtype}" for col, dtype in dtypes.items()])
                        md_content.append(f"| Типы колонок | {dtype_str} |\n")
                    md_content.append("\n")
            
            # Pipeline structure (Python code block) - comes right after samples
            pipeline_data = section.get("pipeline_structure")
            if pipeline_data and pipeline_data.get("steps"):
                steps = pipeline_data["steps"]
                description = pipeline_data.get("description")
                
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                md_content.append("```py\n")
                md_content.append("Pipeline steps:\n")
                for i, step in enumerate(steps):
                    md_content.append(f"  {i+1}. {step['name']}: {step['class_name']}\n")
                md_content.append("```\n")
            
            # Parameter grid (all tested values) - separate section
            param_grid_data = section.get("param_grid")
            if param_grid_data and param_grid_data.get("param_grid"):
                param_grid = param_grid_data.get("param_grid")
                description = param_grid_data.get("description")
                
                if description:
                    md_content.append(f"**{description}:**\n\n")
                else:
                    md_content.append("**Сетка гиперпараметров (все протестированные значения):**\n\n")
                
                merged_param_grid = None
                if isinstance(param_grid, dict):
                    merged_param_grid = param_grid
                elif isinstance(param_grid, list):
                    merged_param_grid = {}
                    for grid in param_grid:
                        if not isinstance(grid, dict):
                            continue
                        for param_name, param_values in grid.items():
                            if isinstance(param_values, (list, tuple, set)):
                                values_iter = list(param_values)
                            else:
                                values_iter = [param_values]
                            if param_name not in merged_param_grid:
                                merged_param_grid[param_name] = []
                            merged_param_grid[param_name].extend(values_iter)
                    if merged_param_grid:
                        for key in list(merged_param_grid.keys()):
                            # Remove duplicates while preserving order
                            seen_values = []
                            for value in merged_param_grid[key]:
                                if value not in seen_values:
                                    seen_values.append(value)
                            merged_param_grid[key] = seen_values
                    else:
                        merged_param_grid = None
                
                if merged_param_grid:
                    # Create two-column table: Parameter | Возможные значения
                    md_content.append("| Параметр | Возможные значения |\n")
                    md_content.append("|----------|-------------------|\n")
                    for param_name, param_values in merged_param_grid.items():
                        # Clean up parameter names
                        display_name = param_name.replace("param_", "").replace("__", " → ")
                        # Format values as a list
                        if isinstance(param_values, list):
                            values_str = ", ".join([str(v) for v in param_values])
                        else:
                            values_str = str(param_values)
                        md_content.append(f"| {display_name} | {values_str} |\n")
                    md_content.append("\n")
                else:
                    md_content.append(str(param_grid) + "\n\n")
            
            # Best parameters - markdown table
            best_params_data = section.get("best_params")
            if best_params_data and best_params_data.get("params"):
                params = best_params_data["params"]
                description = best_params_data.get("description")
                
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                # Format best parameters as markdown table
                if isinstance(params, dict):
                    # Create two-column table: Parameter | Value
                    md_content.append("| Параметр | Значение |\n")
                    md_content.append("|----------|----------|\n")
                    for param_name, param_value in params.items():
                        # Clean up parameter names (remove param_ prefix if present)
                        display_name = param_name.replace("param_", "").replace("__", " → ")
                        md_content.append(f"| {display_name} | {param_value} |\n")
                    md_content.append("\n")
                else:
                    md_content.append(str(params) + "\n\n")
            
            # Scores (markdown table)
            scores_data = section.get("scores")
            if scores_data and scores_data.get("scores"):
                scores = scores_data["scores"]
                description = scores_data.get("description")
                
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                # Format as markdown table
                if isinstance(scores, dict):
                    # Create two-column table: Метрика | Значение
                    md_content.append("| Метрика | Значение |\n")
                    md_content.append("|----------|----------|\n")
                    for key, value in scores.items():
                        # Capitalize first letter for display
                        key_display = key.capitalize() if key else key
                        md_content.append(f"| {key_display} | {value} |\n")
                    md_content.append("\n")
                else:
                    md_content.append(str(scores) + "\n\n")
            
            # Error matrix (table)
            error_matrix_data = section.get("error_matrix")
            if error_matrix_data and error_matrix_data.get("matrix"):
                matrix = error_matrix_data["matrix"]
                description = error_matrix_data.get("description")
                
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                # Format as markdown table
                if isinstance(matrix, list) and len(matrix) > 0:
                    if isinstance(matrix[0], list) and len(matrix[0]) > 0:
                        # 2D matrix - create proper table
                        n_cols = len(matrix[0])
                        # Header row
                        headers = [f"Class {i}" for i in range(n_cols)]
                        md_content.append("| " + " | ".join(["Параметр"] + headers) + " |\n")
                        md_content.append("|" + "|".join(["----------"] + ["----------"] * n_cols) + "|\n")
                        # Data rows
                        row_labels = [f"Class {i}" for i in range(len(matrix))]
                        for i, (row, label) in enumerate(zip(matrix, row_labels)):
                            md_content.append("| " + " | ".join([label] + [str(val) for val in row]) + " |\n")
                    else:
                        # 1D matrix
                        md_content.append("| Параметр | Значение |\n")
                        md_content.append("|----------|----------|\n")
                        for i, val in enumerate(matrix):
                            md_content.append(f"| {i} | {val} |\n")
                    md_content.append("\n")
            
            # Text content
            text_data = section.get("text")
            if text_data:
                description = text_data.get("description")
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                if text_data.get("is_file"):
                    # Read from file
                    file_path = text_data.get("file_path")
                    if file_path and os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                file_content = f.read()
                            md_content.append(f"```\n{file_content}\n```\n\n")
                        except Exception as e:
                            md_content.append(f"*Error reading file {file_path}: {e}*\n\n")
                    else:
                        md_content.append(f"*File not found: {file_path}*\n\n")
                else:
                    # Direct text content
                    content = text_data.get("content", "")
                    md_content.append(f"```\n{content}\n```\n\n")
            
            # Table content
            # Only process table if it's not a wrapper for old methods (scores, best_params, etc.)
            # Old methods store data in both old fields and table, so we skip table if old fields exist
            has_old_method_data = (
                section.get("scores") or 
                section.get("best_params") or 
                section.get("param_grid") or 
                section.get("error_matrix") or
                section.get("samples")
            )
            
            table_data = section.get("table")
            if table_data and not has_old_method_data:
                description = table_data.get("description")
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                if table_data.get("is_file"):
                    # Reference file (user has already saved it in the right structure)
                    file_path = table_data.get("file_path")
                    if file_path:
                        # Just reference the file - assume user has it in the right place
                        file_name = os.path.basename(file_path)
                        md_content.append(f"*Table data from file: {file_name}*\n\n")
                else:
                    # Render table from data
                    data = table_data.get("data")
                    if data:
                        if isinstance(data, list) and len(data) > 0:
                            if isinstance(data[0], dict):
                                # List of dicts - create table
                                columns = list(data[0].keys())
                                md_content.append("| " + " | ".join(columns) + " |\n")
                                md_content.append("|" + "|".join(["----------"] * len(columns)) + "|\n")
                                for item in data:
                                    row_values = [str(item.get(col, "")) for col in columns]
                                    md_content.append("| " + " | ".join(row_values) + " |\n")
                                md_content.append("\n")
                            elif isinstance(data[0], list):
                                # List of lists - create table
                                if len(data) > 0:
                                    n_cols = len(data[0])
                                    md_content.append("| " + " | ".join([f"Col {i+1}" for i in range(n_cols)]) + " |\n")
                                    md_content.append("|" + "|".join(["----------"] * n_cols) + "|\n")
                                    for row in data:
                                        row_values = [str(val) for val in row]
                                        md_content.append("| " + " | ".join(row_values) + " |\n")
                                    md_content.append("\n")
                            else:
                                md_content.append(str(data) + "\n\n")
                        elif isinstance(data, dict):
                            # Dictionary - create key-value table
                            md_content.append("| Key | Value |\n")
                            md_content.append("|----------|----------|\n")
                            for key, value in data.items():
                                md_content.append(f"| {key} | {value} |\n")
                            md_content.append("\n")
                        else:
                            md_content.append(str(data) + "\n\n")
            
            # Figure content
            figure_data = section.get("figure")
            if figure_data:
                description = figure_data.get("description", "Figure")
                fig_path = figure_data.get("path")
                
                # Add description as text before the image (if description exists)
                if description:
                    md_content.append(f"**{description}:**\n\n")
                
                if fig_path:
                    # Find the source path
                    source_path = None
                    if os.path.exists(fig_path):
                        source_path = fig_path
                    else:
                        # Check if it's a matplotlib temp file path
                        if fig_path.startswith('/tmp/reportgen_figs_'):
                            # First, check if current instance has temp dir
                            if hasattr(self, '_temp_fig_dir'):
                                fig_basename = os.path.basename(fig_path)
                                current_temp_path = os.path.join(self._temp_fig_dir, fig_basename)
                                if os.path.exists(current_temp_path):
                                    source_path = current_temp_path
                            
                            # Also check in figures subdirectory (where persistent copies are stored)
                            if not source_path:
                                fig_basename = os.path.basename(fig_path)
                                figures_dir = os.path.join(from_path, "figures")
                                persistent_path = os.path.join(figures_dir, fig_basename)
                                if os.path.exists(persistent_path):
                                    source_path = persistent_path
                        
                        if not source_path:
                            alt_path = os.path.join(from_path, os.path.basename(fig_path))
                            if os.path.exists(alt_path):
                                source_path = alt_path
                            else:
                                cwd_path = os.path.join(os.getcwd(), os.path.basename(fig_path))
                                if os.path.exists(cwd_path):
                                    source_path = cwd_path
                    
                    if source_path:
                        source_path_normalized = os.path.realpath(os.path.abspath(source_path))
                        # Use the already-copied image from first pass if available
                        if source_path_normalized in source_file_to_section_names:
                            section_names_dict = source_file_to_section_names[source_path_normalized]
                            if section_slug in section_names_dict:
                                unique_fig_name = section_names_dict[section_slug]
                            else:
                                unique_fig_name = source_file_to_dest_name[source_path_normalized]
                            md_content.append(f"![{description}]({unique_fig_name})\n\n")
                        elif source_path_normalized in source_file_to_dest_name:
                            unique_fig_name = source_file_to_dest_name[source_path_normalized]
                            md_content.append(f"![{description}]({unique_fig_name})\n\n")
                        else:
                            # Fallback: copy now with section-based naming (shouldn't happen if first pass worked correctly)
                            section_fig_counters[section_slug] += 1
                            fig_counter = section_fig_counters[section_slug]
                            _, ext = os.path.splitext(os.path.basename(source_path))
                            if not ext:
                                ext = ".png"
                            unique_fig_name = f"{section_slug}__{fig_counter:02d}{ext}"
                            dest_path = os.path.join(to_path, unique_fig_name)
                            if os.path.exists(source_path):
                                shutil.copy2(source_path, dest_path)
                                md_content.append(f"![{description}]({unique_fig_name})\n\n")
                            else:
                                print(f"Warning: Figure source not found: {source_path}, skipping figure")
                    else:
                        # Fallback: use section-based naming even if file not found
                        section_fig_counters[section_slug] += 1
                        fig_counter = section_fig_counters[section_slug]
                        _, ext = os.path.splitext(os.path.basename(fig_path)) if fig_path else (None, None)
                        if not ext:
                            ext = ".png"
                        unique_fig_name = f"{section_slug}__{fig_counter:02d}{ext}"
                        print(f"Warning: Figure not found: {fig_path}, using placeholder name: {unique_fig_name}")
                        md_content.append(f"![{description}]({unique_fig_name})\n\n")
            
            # Section separator
            md_content.append("\n---\n\n")
        
        # Write markdown file
        md_file_name = file_name.replace('.json', '.md') if file_name.endswith('.json') else file_name + '.md'
        md_path = os.path.join(to_path, md_file_name)
        
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(''.join(md_content))
            print(f"Compiled markdown report to: {md_path}")
        except Exception as e:
            print(f"Error writing markdown file: {e}")
            raise
        
        # Compile to PDF if requested (before cleanup)
        if compile_to_pdf:
            self._compile_markdown_to_pdf(md_path, to_path, pdf_engine)
        
        # Clean up temporary matplotlib figure directory if it exists (after PDF compilation)
        self._cleanup_temp_figures()
        
        return self
    
    def _cleanup_temp_figures(self):
        """Clean up temporary matplotlib figure directory."""
        if hasattr(self, '_temp_fig_dir') and os.path.exists(self._temp_fig_dir):
            try:
                import shutil
                shutil.rmtree(self._temp_fig_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp figure directory {self._temp_fig_dir}: {e}")
    
    def compile_pdf_from_markdown(self, md_path: str, output_dir: Optional[str] = None, 
                                   pdf_engine: Optional[str] = None):
        """
        Compile an existing markdown file to PDF.
        
        This method allows you to edit the markdown file before compiling to PDF.
        Use this after calling compile() without compile_to_pdf=True.
        
        Args:
            md_path: Path to markdown file (can be relative or absolute)
            output_dir: Directory where PDF should be saved. If None, uses the same directory as markdown file.
            pdf_engine: PDF engine to use ('pandoc', 'xelatex', 'pdflatex', etc.). 
                       If None, will try to auto-detect.
        
        Returns:
            self for method chaining
        
        Example:
            # Generate markdown from JSON
            report.compile("./reports", "./output", "experiment.json")
            
            # Edit markdown file manually...
            
            # Compile PDF from edited markdown
            report.compile_pdf_from_markdown("./output/experiment.md")
        """
        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Markdown file not found: {md_path}")
        
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(md_path))
        
        self._compile_markdown_to_pdf(md_path, output_dir, pdf_engine)
        return self

    def _compile_markdown_to_pdf(self, md_path: str, output_dir: str, pdf_engine: Optional[str] = None):
        """Compile markdown file to PDF (delegates to report_pdf module)."""
        compile_markdown_to_pdf(md_path, output_dir, pdf_engine)

    def _detect_pdf_engine(self) -> Optional[str]:
        """Detect available PDF compilation engine."""
        return detect_pdf_engine()

    def _slugify(self, text: str) -> str:
        """
        Convert text into a filesystem-friendly slug.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '-', text)
        text = text.strip('-')
        return text
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, '__name__'):
            # Handle function/class objects
            return obj.__name__
        elif hasattr(obj, '__class__'):
            # Handle other objects - convert to string representation
            return str(obj)
        else:
            return obj

