"""
Centralized path management for the project.
Ensures all paths are relative to the project root, making the code portable.
"""
from pathlib import Path

def get_project_root() -> Path:
    """Gets the project root directory."""
    return Path(__file__).resolve().parents[2]

def get_data_dir() -> Path:
    """Gets the data directory."""
    return get_project_root() / "data"

def get_experiments_dir() -> Path:
    """Gets the main experiments directory."""
    return get_project_root() / "experiments"

def get_configs_dir() -> Path:
    """Gets the configs directory."""
    return get_project_root() / "configs"

def get_docs_dir() -> Path:
    """Gets the documentation directory."""
    return get_project_root() / "docs"
