# Visualization Package

Visualization and reporting tools for seismic pipeline analysis, hyperparameter tuning, and experiment results.

## Overview

This package provides tools for:

- **Hyperparameter Grid Visualization**: Create comprehensive visualizations of grid search results
- **Report Generation**: Build structured experiment reports with automatic markdown compilation

## Installation

The visualization package is part of the `seismic_pipeline` package. Import it as:

```python
from seismic_pipeline.visualization import ReportGenerator, visualize_hyperparameter_grid_slices
```

## Components

### 1. Hyperparameter Grid Visualizer

Visualize hyperparameter grid search results using multi-dimensional slices.

#### Usage

```python
from seismic_pipeline.visualization import visualize_hyperparameter_grid_slices
from sklearn.model_selection import GridSearchCV

# After running grid search
grid_search = GridSearchCV(estimator, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Generate visualizations
visualize_hyperparameter_grid_slices(grid_search, output_dir='./results')
```

#### Features

- Handles 1D, 2D, and 3D+ parameter spaces
- Automatically creates optimal visualization layout
- Generates high-quality 300 DPI PNG files
- Shows mean cross-validation scores for parameter combinations
- Automatically focuses on preprocessing parameters (label generator, REM calculator,
  feature extractor) unless custom parameter pairs are provided

#### Output

Creates `Hyperparameter_Grid_Search.png` in the specified output directory showing:

- **1D parameter space**: Single bar chart
- **2D parameter space**: Single heatmap
- **3D+ parameter space**: Multiple 2D heatmap slices through parameter space

### 2. Report Generator

Build structured experiment reports with automatic JSON storage and markdown compilation.

#### Basic Usage

```python
from seismic_pipeline.visualization import ReportGeneratorYt
import pandas as pd
import numpy as np

# Create report instance
report = ReportGenerator()

# Add section title
report.title("12 событий")

# Add sample data
df = pd.DataFrame([...])  # Your DataFrame
report.samples(df.head(10), description="First 10 lines of dataset", metadata=True)

# Add pipeline structure
report.pipeline_structure(pipe, description="Pipeline Structure")

# Add parameter grid (all tested values)
param_grid = {'classifier__C': [0.1, 1, 10], 'label_generator__window_days': [1, 2, 3]}
report.param_grid(param_grid, description="Parameter grid")

# Add best hyperparameters
best_params = {'classifier__C': 1, 'label_generator__window_days': 2}
report.best_params(best_params, description="Best parameters")

# Add hyperparameter grid visualization
report.hyperparameter_grid_visualization(
    "./grid.png", 
    description="Hyperparameter grid visualization",
    recreate=True, 
    grid_search=grid_search
)

# Add scores
report.scores({"accuracy": "0.8", "precision": "0.75"}, description="CV scores")

# Add confusion matrix
cm = np.array([[10, 2], [1, 11]])  # Confusion matrix
report.error_matrix(cm, description="Confusion matrix")

# Save to JSON
report.create_json("./reports", "experiment.json", write="append")

# Later, compile to markdown
report.compile("./reports", "./run report", "experiment.json")
```

#### Head Section Methods

Head methods store data only once in a special "head" section that appears at the top of the compiled markdown. If a head section already exists, subsequent calls to head methods are ignored.

```python
# Add head section (only added once, even if called multiple times)
report.head_title("Experiment Overview")
report.head_samples(df.head(5), description="Sample data")
report.head_best_params(best_params, description="Best parameters")
report.head_scores({"accuracy": "0.8"}, description="Overall scores")
report.create_json("./reports", "experiment.json", write="append")

# Later calls to head methods will be ignored if head section exists
report.head_title("Another title")  # This will be ignored
report.create_json("./reports", "experiment.json", write="append")
```

**Available head methods:**
- `head_title(text)` - Set head section title
- `head_samples(df, ...)` - Add DataFrame samples to head
- `head_pipeline_structure(pipe, ...)` - Add pipeline structure to head
- `head_best_params(params, ...)` - Add best parameters to head
- `head_param_grid(grid, ...)` - Add parameter grid to head
- `head_hyperparameter_grid_visualization(path, ...)` - Add visualization to head
- `head_scores(scores, ...)` - Add scores to head
- `head_error_matrix(matrix, ...)` - Add error matrix to head
- `head_add_text(content, ...)` - Add text content to head
- `head_add_table(data, ...)` - Add table data to head
- `head_add_fig(source, ...)` - Add figure to head

#### Additional Content Methods

Add text, tables, or figures to create new sections:

```python
# Add text content (from string or file)
report.title("Text Section")
report.add_text("This is some text content", description="Description")
report.create_json("./reports", "experiment.json", write="append")

# Or from a file
report.title("Log Section")
report.add_text("./logs/experiment.log", description="Experiment log")
report.create_json("./reports", "experiment.json", write="append")

# Add table (from DataFrame, list, dict, or file)
report.title("Table Section")
report.add_table(df, description="Data table")
report.create_json("./reports", "experiment.json", write="append")

# Or from a file
report.title("CSV Table Section")
report.add_table("./data/results.csv", description="Results from file")
report.create_json("./reports", "experiment.json", write="append")

# Add figure (from file path or matplotlib figure)
report.title("Figure Section")
report.add_fig("./plots/result.png", description="Result plot")
report.create_json("./reports", "experiment.json", write="append")

# Or from matplotlib figure object
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
report.title("Matplotlib Figure")
report.add_fig(fig, description="Generated plot")
report.create_json("./reports", "experiment.json", write="append")
```

## API Reference

### ReportGenerator Class

#### Methods

##### `title(text: str) -> ReportGenerator`

Set the section title.

**Parameters:**

- `text`: Title text for the section

**Returns:** `self` for method chaining

---


##### `samples(df: pd.DataFrame, description: Optional[str] = None, n_rows: Optional[int] = None, metadata: bool = False) -> ReportGenerator`
Store DataFrame samples (head or full) as JSON.

**Parameters:**

- `df`: pandas DataFrame to store
- `description`: Optional description text
- `n_rows`: Number of rows to store (default: None, stores full DataFrame)
- `metadata`: If True, include shape, dtypes, column names

**Returns:** `self` for method chaining

**Example:**

```python
report.samples(df.head(10), description="First 10 samples", metadata=True)
```

---


##### `pipeline_structure(pipe: Any, description: Optional[str] = None) -> ReportGenerator`
Extract and store pipeline structure.

**Parameters:**

- `pipe`: PipelineYt or sklearn Pipeline object
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
report.pipeline_structure(pipe, description="Pipeline Structure")
```

---

##### `param_grid(param_grid_dict: Dict[str, Any], description: Optional[str] = None) -> ReportGenerator`

Store parameter grid dictionary (all tested values).

**Parameters:**

- `param_grid_dict`: Dictionary of parameter grid (all tested values)
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
param_grid = {'classifier__C': [0.1, 1, 10], 'label_generator__window_days': [1, 2, 3]}
report.param_grid(param_grid, description="Parameter grid")
```

---

##### `best_params(param_dict: Dict[str, Any], description: Optional[str] = None) -> ReportGenerator`

Store best hyperparameters dictionary.

**Parameters:**

- `param_dict`: Dictionary of best hyperparameters
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
report.best_params(
    {'classifier__C': 1, 'label_generator__window_days': 2},
    description="Best parameters"
)
```

---

##### `hyperparameter_grid_visualization(path: str, description: Optional[str] = None, recreate: bool = False, grid_search: Optional[Any] = None, param_pairs: Optional[List[tuple]] = None, focus_param_prefixes: Optional[List[str]] = None) -> ReportGenerator`

Store hyperparameter grid visualization path, optionally recreate.

**Parameters:**

- `path`: Path to visualization file
- `description`: Optional description text
- `recreate`: If True and grid_search provided, recreate visualization
- `grid_search`: GridSearchCV object for recreating visualization
- `param_pairs`: Optional explicit parameter pairs (with 'param_' prefix) to visualize
- `focus_param_prefixes`: Optional list of parameter prefixes to auto-select when `param_pairs`
  is not provided (defaults to preprocessing components)

**Returns:** `self` for method chaining

**Example:**

```python
report.hyperparameter_grid_visualization(
    "./grid.png",
    description="Hyperparameter grid visualization",
    recreate=True,
    grid_search=grid_search
)
```

---

##### `scores(score_dict: Dict[str, Union[str, float]], description: Optional[str] = None) -> ReportGenerator`

Store score dictionary.

**Parameters:**

- `score_dict`: Dictionary of scores (e.g., `{"accuracy": "0.8", ...}`)
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
report.scores({"accuracy": "0.8", "precision": "0.75"}, description="CV scores")
```

---

##### `error_matrix(matrix: np.ndarray, description: Optional[str] = None) -> ReportGenerator`

Store confusion/error matrix (numpy array) as nested list.

**Parameters:**

- `matrix`: Numpy array representing confusion/error matrix
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
cm = confusion_matrix(y_true, y_pred)
report.error_matrix(cm, description="Confusion matrix")
```

---

##### Head Section Methods

All regular methods have corresponding `head_*` versions that store data in a special head section (displayed as "Overview" at the top of the compiled markdown). Head sections are only added once - if a head section already exists in the JSON file, subsequent calls to head methods are ignored.

**Available head methods:**
- `head_title(text: str) -> ReportGenerator`
- `head_samples(df: pd.DataFrame, ...) -> ReportGenerator`
- `head_pipeline_structure(pipe: Any, ...) -> ReportGenerator`
- `head_best_params(param_dict: Dict[str, Any], ...) -> ReportGenerator`
- `head_param_grid(param_grid_dict: Dict[str, Any], ...) -> ReportGenerator`
- `head_hyperparameter_grid_visualization(path: str, ...) -> ReportGenerator`
- `head_scores(score_dict: Dict[str, Union[str, float]], ...) -> ReportGenerator`
- `head_error_matrix(matrix: np.ndarray, ...) -> ReportGenerator`
- `head_add_text(content: Union[str, Path], ...) -> ReportGenerator`
- `head_add_table(data: Union[pd.DataFrame, List[Dict], Dict, List[List], Path], ...) -> ReportGenerator`
- `head_add_fig(source: Union[str, Path, matplotlib.figure.Figure], ...) -> ReportGenerator`

**Example:**

```python
# Add head section (only added once)
report.head_title("Experiment Overview")
report.head_best_params(best_params)
report.head_scores({"accuracy": "0.8"})
report.create_json("./reports", "experiment.json", write="append")
```

---

##### `add_text(content: Union[str, Path], description: Optional[str] = None) -> ReportGenerator`

Add text content to the current section. Can accept either a string variable or a file path (.txt, .log).

**Parameters:**

- `content`: String text or path to text file (.txt, .log)
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
# From string
report.title("Text Section")
report.add_text("This is some text content", description="Description")
report.create_json("./reports", "experiment.json", write="append")

# From file
report.title("Log Section")
report.add_text("./logs/experiment.log", description="Experiment log")
report.create_json("./reports", "experiment.json", write="append")
```

---

##### `add_table(data: Union[pd.DataFrame, List[Dict], Dict, List[List], Path], description: Optional[str] = None) -> ReportGenerator`

Add table data to the current section. Can accept DataFrame, list of dicts, dict, list of lists, or file path (.csv, .txt).

**Parameters:**

- `data`: Table data (DataFrame, list of dicts, dict, list of lists) or file path
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
# From DataFrame
report.title("Table Section")
report.add_table(df, description="Data table")
report.create_json("./reports", "experiment.json", write="append")

# From list of dicts
report.add_table([{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}])

# From file
report.add_table("./data/results.csv", description="Results from file")
```

---

##### `add_fig(source: Union[str, Path, matplotlib.figure.Figure], description: Optional[str] = None) -> ReportGenerator`

Add figure to the current section. Can accept file paths (.png, .jpeg, .jpg) or matplotlib figure objects.

**Parameters:**

- `source`: File path or matplotlib figure object
- `description`: Optional description text

**Returns:** `self` for method chaining

**Example:**

```python
# From file path
report.title("Figure Section")
report.add_fig("./plots/result.png", description="Result plot")
report.create_json("./reports", "experiment.json", write="append")

# From matplotlib figure
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
report.title("Matplotlib Figure")
report.add_fig(fig, description="Generated plot")
report.create_json("./reports", "experiment.json", write="append")
```

---

##### `create_json(path: str, file_name: str, write: str = "append") -> ReportGenerator`

Save current section to JSON file.

**Parameters:**

- `path`: Directory path where JSON file should be saved
- `file_name`: Name of JSON file
- `write`: `"append"` to append to existing JSON, `"rewrite"` to overwrite

**Returns:** `self` for method chaining

**Example:**

```python
report.create_json("./reports", "experiment.json", write="append")
```

**Note:** After calling `create_json()`, the current section is reset and you can start building a new section.

---

##### `compile(from_path: str, to_path: str, file_name: str, compile_to_pdf: bool = False, pdf_engine: Optional[str] = None) -> ReportGenerator`

Parse JSON, generate markdown, and copy figures. Optionally compile markdown to PDF.

**Parameters:**

- `from_path`: Directory containing JSON file
- `to_path`: Directory where markdown and figures should be saved
- `file_name`: Name of JSON file (with or without .json extension)
- `compile_to_pdf`: If True, also compile markdown to PDF (default: False)
- `pdf_engine`: PDF engine to use ('pandoc', 'xelatex', 'pdflatex', 'lualatex', 'wkhtmltopdf', 'weasyprint').
               If None, will try to auto-detect available engine (default: None)

**Returns:** `self` for method chaining

**Example:**

```python
# Compile to markdown only
report.compile("./reports", "./run report", "experiment.json")

# Compile to markdown and PDF
report.compile("./reports", "./run report", "experiment.json", compile_to_pdf=True)
```

**Note:** This will:

1. Load the JSON file from `from_path`
2. Generate a markdown file in `to_path`
3. Copy all referenced figures to `to_path`
4. Use relative paths for images in the markdown
5. If `compile_to_pdf=True`, compile the markdown to PDF using pandoc (requires pandoc to be installed)

**PDF Compilation Requirements:**

- **pandoc**: Install from https://pandoc.org/installing.html
- **LaTeX engine**: pandoc uses xelatex (or pdflatex as fallback) for PDF generation
  - On Linux: `sudo apt-get install texlive-xetex` or `sudo apt-get install texlive-latex-base`
  - On macOS: `brew install basictex` or install MacTeX
  - On Windows: Install MiKTeX or TeX Live

The PDF will include:
- Table of contents
- All images and figures
- Proper formatting with Unicode support (using xelatex)

---

##### `compile_pdf_from_markdown(md_path: str, output_dir: Optional[str] = None, pdf_engine: Optional[str] = None) -> ReportGenerator`

Compile an existing markdown file to PDF. This allows you to edit the markdown file before compiling to PDF.

**Parameters:**

- `md_path`: Path to markdown file (can be relative or absolute)
- `output_dir`: Directory where PDF should be saved. If None, uses the same directory as markdown file (default: None)
- `pdf_engine`: PDF engine to use ('pandoc', 'xelatex', 'pdflatex', etc.). If None, will try to auto-detect (default: None)

**Returns:** `self` for method chaining

**Example:**

```python
# Step 1: Generate markdown from JSON (without PDF compilation)
report.compile("./reports", "./output", "experiment.json")

# Step 2: Edit the markdown file manually in your editor
# (e.g., edit "./output/experiment.md")

# Step 3: Compile PDF from the edited markdown
report.compile_pdf_from_markdown("./output/experiment.md")

# Or specify a different output directory
report.compile_pdf_from_markdown("./output/experiment.md", output_dir="./pdf_output")
```

**Use Case:**

This method is useful when you want to:
- Edit the generated markdown before creating the final PDF
- Recompile PDF after making manual changes to markdown
- Have more control over the compilation process

**Note:** The markdown file must exist and contain valid markdown. All referenced images must be in the correct relative paths.

---


## JSON Structure

The JSON file stores a list of sections, where each section contains:

```json
{
  "title": "Section title",
  "samples": {
    "data": [...],
    "description": "...",
    "metadata": {
      "shape": [n_rows, n_cols],
      "dtypes": {...},
      "columns": [...]
    }
  },
  "pipeline_structure": {
    "steps": [
      {
        "name": "step_name",
        "class_name": "ClassName",
        "parameters": {...}
      }
    ],
    "description": "..."
  },
  "hyperparameters": {
    "params": {...},
    "description": "..."
  },
  "hyperparameter_grid_visualization": {
    "path": "./grid.png",
    "description": "..."
  },
  "scores": {
    "scores": {"accuracy": "0.8", ...},
    "description": "..."
  },
  "error_matrix": {
    "matrix": [[...]],
    "description": "..."
  },
  "text": {
    "content": "...",
    "file_path": "...",
    "is_file": true/false,
    "description": "..."
  },
  "table": {
    "data": [...],
    "file_path": "...",
    "is_file": true/false,
    "description": "..."
  },
  "figure": {
    "path": "./figure.png",
    "is_file": true,
    "is_matplotlib": false,
    "description": "..."
  }
}
```

## Markdown Output Format

The compiled markdown follows this structure:

- **Head sections** (title == `"__HEAD__"`) are rendered first at the top with "Overview" as the display title
- **Regular sections** follow after head sections
- Each section is separated by `---`

Example structure:

````markdown
## Overview

![Image Description](image.png)

**Описание:**

```py
events = [
    {'rat_id': 'R2', 'date': '2022-11-07'},
    ...
]
```

```shell
Best parameters found:  {...}
Best CV accuracy:  0.8
```

---

## Section Title

![Image Description](image.png)

**Описание:**

```py
events = [
    {'rat_id': 'R2', 'date': '2022-11-07'},
    ...
]
```

```shell
Best parameters found:  {...}
Best CV accuracy:  0.8
```

````

---


## Complete Example

```python
from seismic_pipeline.visualization import ReportGeneratorYt
from seismic_pipeline.mod.grid_searchyt import GridSearchCVYt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# After running your pipeline and grid search
# ... your code ...

# Create report
report = ReportGenerator()

# Section 1: 12 events experiment
report.title("12 событий")
report.samples(events_df.head(12), description="Events list")
report.pipeline_structure(best_pipe, description="Pipeline Structure")

# Extract and add parameter grid
param_grid = {}
if hasattr(grid_search, 'param_grid'):
    if isinstance(grid_search.param_grid, list):
        for pg in grid_search.param_grid:
            param_grid.update(pg)
    elif isinstance(grid_search.param_grid, dict):
        param_grid = grid_search.param_grid

if param_grid:
    report.param_grid(param_grid, description="Parameter grid")

report.best_params(grid_search.best_params_, description="Best parameters")
report.hyperparameter_grid_visualization(
    "./Hyperparameter_Grid_Search.png",
    recreate=True,
    grid_search=grid_search
)
report.scores({
    "Best CV accuracy": str(grid_search.best_score_),
    "Best classifier": type(grid_search.best_estimator_.named_steps['classifier']).__name__
})
cm = confusion_matrix(y_true, y_pred)
report.error_matrix(cm)
report.create_json("./reports", "experiments.json", write="append")

# Section 2: 16 events experiment
report.title("16 событий")
# ... add data for second experiment ...
report.create_json("./reports", "experiments.json", write="append")

# Compile all sections to markdown
report.compile("./reports", "./run report", "experiments.json")

# Or compile to markdown and PDF
report.compile("./reports", "./run report", "experiments.json", compile_to_pdf=True)
```

## File Structure

```text
seismic_pipeline/visualization/
├── __init__.py                        # Package exports
├── hyperparameter_grid_visualizer.py # Grid search visualization
├── report_generator.py                # ReportGenerator class
└── README.md                          # This file
```

## Dependencies

- `numpy` - For array handling
- `pandas` - For DataFrame handling
- `matplotlib` - For visualization
- `scikit-learn` - For GridSearchCV support

## Notes

- All methods return `self` for method chaining
- JSON files use UTF-8 encoding and preserve non-ASCII characters
- Figures are copied to the markdown output directory during compilation
- The report generator automatically handles JSON serialization of numpy types
- Pipeline structure extraction works with both `PipelineYt` and sklearn `Pipeline` objects
- Head sections (created with `head_*` methods) are only added once - if a head section already exists, subsequent calls are ignored
- Head sections are displayed at the top of compiled markdown with "Overview" as the title
- `add_text()`, `add_table()`, and `add_fig()` work like regular methods - they build up `_current_section` and require an explicit `create_json()` call
- Each call to `add_text()`, `add_table()`, or `add_fig()` creates a new section when `create_json()` is called
- PDF compilation requires `pandoc` to be installed. The PDF will include a table of contents and all images

## See Also

- `seismic_pipeline.mod.grid_searchyt.GridSearchCVYt` - Target-aware grid search
- `seismic_pipeline.mod.pipelineyt.PipelineYt` - Target-aware pipeline

