"""
I/O utilities for saving and loading data.
"""
import pickle
from pathlib import Path
from typing import Any


def save_pickle(data: Any, filepath: Path, verbose: bool = True) -> None:
    """
    Save data to a pickle file.
    
    Parameters:
    -----------
    data : Any
        Data to save (can be any pickleable object)
    filepath : Path
        Path to save the pickle file
    verbose : bool, optional
        If True, print confirmation message (default: True)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    if verbose:
        print(f"Saved pickle file: {filepath}")


def load_pickle(filepath: Path) -> Any:
    """
    Load data from a pickle file.
    
    Parameters:
    -----------
    filepath : Path
        Path to the pickle file
        
    Returns:
    --------
    Any
        Data loaded from the pickle file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data
