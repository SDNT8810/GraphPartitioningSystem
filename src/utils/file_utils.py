"""
Utilities for cross-platform file system operations.
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Union, Optional


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """
    Ensures that a directory exists, creating it if necessary.
    Handles platform-specific issues and provides better error handling.
    
    Args:
        directory_path: Path to the directory to create
        
    Returns:
        Path object representing the directory
    """
    # Convert to Path object if it's a string
    path_obj = Path(directory_path) if isinstance(directory_path, str) else directory_path
    
    try:
        # Create directory with parents if it doesn't exist
        path_obj.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        logging.error(f"Permission denied when creating directory: {path_obj}")
        logging.info("Check if you have sufficient permissions for this location.")
        raise
    except FileExistsError:
        # This shouldn't happen with exist_ok=True, but just in case
        if path_obj.is_file():
            logging.error(f"Cannot create directory {path_obj}: A file with that name already exists")
            raise
    except OSError as e:
        if sys.platform == 'win32':
            # Windows-specific handling
            if e.winerror == 123:  # Invalid name (e.g., reserved names like COM1, AUX)
                logging.error(f"Cannot create directory with reserved name: {path_obj}")
                raise
            elif e.winerror == 5:  # Access denied
                logging.error(f"Access denied when creating directory: {path_obj}")
                logging.info("Check if you have sufficient permissions or if the path is locked by another process.")
                raise
        else:
            # Generic error handling for other OSes
            logging.error(f"Failed to create directory {path_obj}: {e}")
            raise
            
    return path_obj


def safe_save_path(directory: Union[str, Path], filename: str) -> str:
    """
    Creates a safe, cross-platform path for saving files.
    Ensures the directory exists before returning the path.
    
    Args:
        directory: Directory to save the file in
        filename: Name of the file to save
        
    Returns:
        String representation of the full path
    """
    # Ensure directory exists
    path_obj = ensure_directory_exists(directory)
    
    # Create full path for the file
    full_path = path_obj / filename
    
    # Ensure the parent directory exists
    full_path.parent.mkdir(exist_ok=True, parents=True)
    
    return str(full_path)


def clean_directory(directory_path: Union[str, Path], pattern: str = "*") -> None:
    """
    Safely cleans a directory by removing files matching a pattern.
    
    Args:
        directory_path: Directory to clean
        pattern: Glob pattern for files to remove (default: "*" for all files)
    """
    path_obj = Path(directory_path) if isinstance(directory_path, str) else directory_path
    
    if not path_obj.exists():
        return
        
    if not path_obj.is_dir():
        logging.error(f"Cannot clean {path_obj}: Not a directory")
        return
        
    try:
        for item in path_obj.glob(pattern):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        logging.info(f"Cleaned directory: {path_obj}")
    except Exception as e:
        logging.error(f"Error cleaning directory {path_obj}: {e}")
