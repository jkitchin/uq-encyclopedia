"""Input/Output utilities for datasets and results."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union
import h5py


def save_dataset(
    filepath: Union[str, Path],
    data_dict: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save dataset to NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Path to save file
    data_dict : dict
        Dictionary of arrays to save
    metadata : dict, optional
        Metadata to save alongside data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Combine data and metadata
    save_dict = data_dict.copy()
    if metadata is not None:
        # Convert metadata to JSON string for storage
        save_dict['_metadata'] = np.array([json.dumps(metadata)])

    np.savez_compressed(filepath, **save_dict)


def load_dataset(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dataset from NPZ file.

    Parameters
    ----------
    filepath : str or Path
        Path to NPZ file

    Returns
    -------
    data : dict
        Dictionary containing arrays and metadata
    """
    filepath = Path(filepath)
    data = {}

    with np.load(filepath, allow_pickle=True) as npz:
        for key in npz.files:
            if key == '_metadata':
                # Parse metadata from JSON
                data['metadata'] = json.loads(str(npz[key][0]))
            else:
                data[key] = npz[key]

    return data


def save_results_csv(
    filepath: Union[str, Path],
    results: pd.DataFrame,
):
    """
    Save results to CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to save file
    results : pd.DataFrame
        Results dataframe
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath, index=False)


def load_results_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load results from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file

    Returns
    -------
    results : pd.DataFrame
        Results dataframe
    """
    return pd.read_csv(filepath)


def save_results_json(
    filepath: Union[str, Path],
    results: Dict[str, Any],
):
    """
    Save results to JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to save file
    results : dict
        Results dictionary
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            return super().default(obj)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


def load_results_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Parameters
    ----------
    filepath : str or Path
        Path to JSON file

    Returns
    -------
    results : dict
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results_hdf5(
    filepath: Union[str, Path],
    arrays: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save results to HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to save file
    arrays : dict
        Dictionary of arrays to save
    metadata : dict, optional
        Metadata to save as attributes
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        # Save arrays
        for key, value in arrays.items():
            f.create_dataset(key, data=value, compression='gzip')

        # Save metadata as attributes
        if metadata is not None:
            for key, value in metadata.items():
                # Convert non-serializable types
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                try:
                    f.attrs[key] = value
                except TypeError:
                    # If still not serializable, convert to string
                    f.attrs[key] = str(value)


def load_results_hdf5(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load results from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file

    Returns
    -------
    data : dict
        Dictionary containing arrays and metadata
    """
    filepath = Path(filepath)
    data = {'arrays': {}, 'metadata': {}}

    with h5py.File(filepath, 'r') as f:
        # Load arrays
        for key in f.keys():
            data['arrays'][key] = f[key][:]

        # Load metadata
        for key, value in f.attrs.items():
            # Try to parse JSON
            if isinstance(value, str):
                try:
                    data['metadata'][key] = json.loads(value)
                except json.JSONDecodeError:
                    data['metadata'][key] = value
            else:
                data['metadata'][key] = value

    return data


def ensure_directory(filepath: Union[str, Path]) -> Path:
    """
    Ensure directory exists for a filepath.

    Parameters
    ----------
    filepath : str or Path
        File path

    Returns
    -------
    filepath : Path
        Path object with directory created
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath
