"""Seed management for reproducibility."""

import hashlib
import numpy as np
from typing import Union, Optional


class SeedManager:
    """
    Centralized seed management for reproducible experiments.

    The SeedManager uses a master seed to deterministically generate
    seeds for different components of the experiment.
    """

    def __init__(self, master_seed: int = 42):
        """
        Initialize seed manager.

        Parameters
        ----------
        master_seed : int
            Master seed for all derived seeds
        """
        self.master_seed = master_seed
        self._seed_log = {}  # Track all generated seeds

    def get_seed(self, experiment_id: str) -> int:
        """
        Get a deterministic seed for a specific experiment.

        Parameters
        ----------
        experiment_id : str
            Unique identifier for the experiment/component

        Returns
        -------
        seed : int
            Deterministic seed derived from master seed and experiment ID
        """
        # Create deterministic seed from master_seed and experiment_id
        combined = f"{self.master_seed}_{experiment_id}"
        hash_obj = hashlib.sha256(combined.encode())
        seed = int(hash_obj.hexdigest(), 16) % (2**32)

        # Log the seed
        self._seed_log[experiment_id] = seed

        return seed

    def set_global_seed(self, seed: Optional[int] = None):
        """
        Set global random seeds for all libraries.

        Parameters
        ----------
        seed : int, optional
            Seed to use. If None, uses master_seed
        """
        if seed is None:
            seed = self.master_seed

        np.random.seed(seed)

        # Try to set seeds for other libraries if available
        try:
            import random
            random.seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

    def get_seed_log(self) -> dict:
        """
        Get log of all generated seeds.

        Returns
        -------
        seed_log : dict
            Dictionary mapping experiment IDs to their seeds
        """
        return self._seed_log.copy()

    def reset_log(self):
        """Clear the seed log."""
        self._seed_log = {}


# Global seed manager instance
_global_seed_manager = None


def get_seed_manager(master_seed: Optional[int] = None) -> SeedManager:
    """
    Get the global seed manager instance.

    Parameters
    ----------
    master_seed : int, optional
        Master seed to use if creating new manager

    Returns
    -------
    seed_manager : SeedManager
        Global seed manager instance
    """
    global _global_seed_manager

    if _global_seed_manager is None:
        if master_seed is None:
            master_seed = 42  # Default master seed
        _global_seed_manager = SeedManager(master_seed)

    return _global_seed_manager


def set_master_seed(master_seed: int):
    """
    Set the master seed for the global seed manager.

    Parameters
    ----------
    master_seed : int
        New master seed
    """
    global _global_seed_manager
    _global_seed_manager = SeedManager(master_seed)


def get_experiment_seed(experiment_id: str) -> int:
    """
    Get a seed for a specific experiment.

    This is a convenience function that uses the global seed manager.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment

    Returns
    -------
    seed : int
        Deterministic seed for the experiment
    """
    manager = get_seed_manager()
    return manager.get_seed(experiment_id)


def set_global_seed(seed: Optional[int] = None):
    """
    Set global random seeds for all libraries.

    This is a convenience function that uses the global seed manager.

    Parameters
    ----------
    seed : int, optional
        Seed to use. If None, uses master seed
    """
    manager = get_seed_manager()
    manager.set_global_seed(seed)
