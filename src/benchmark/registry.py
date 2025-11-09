"""Plugin registration system for datasets, models, and UQ methods."""

from typing import Dict, Type, Callable, Any, Optional
import inspect


class Registry:
    """
    Generic registry for plugins.

    Allows registration and retrieval of classes/functions by name.
    """

    def __init__(self, name: str):
        """
        Initialize registry.

        Parameters
        ----------
        name : str
            Name of the registry
        """
        self.name = name
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: Optional[str] = None,
        **metadata
    ) -> Callable:
        """
        Decorator to register a class or function.

        Parameters
        ----------
        name : str, optional
            Name to register under. If None, uses class/function name
        **metadata
            Additional metadata to store

        Returns
        -------
        decorator : callable
            Decorator function

        Examples
        --------
        >>> dataset_registry = Registry('datasets')
        >>> @dataset_registry.register()
        ... class LineDataset(Dataset):
        ...     pass
        """
        def decorator(obj):
            # Use provided name or class/function name
            obj_name = name if name is not None else obj.__name__

            # Check for duplicates
            if obj_name in self._registry:
                raise ValueError(
                    f"'{obj_name}' already registered in {self.name} registry"
                )

            # Register
            self._registry[obj_name] = obj
            self._metadata[obj_name] = metadata

            return obj

        return decorator

    def get(self, name: str) -> Any:
        """
        Get registered object by name.

        Parameters
        ----------
        name : str
            Name of registered object

        Returns
        -------
        obj : Any
            Registered object

        Raises
        ------
        KeyError
            If name not found in registry
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> list:
        """
        List all registered names.

        Returns
        -------
        names : list
            List of registered names
        """
        return list(self._registry.keys())

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for registered object.

        Parameters
        ----------
        name : str
            Name of registered object

        Returns
        -------
        metadata : dict
            Metadata dictionary
        """
        if name not in self._metadata:
            raise KeyError(f"'{name}' not found in {self.name} registry")
        return self._metadata[name].copy()

    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Get number of registered objects."""
        return len(self._registry)

    def __repr__(self) -> str:
        """String representation."""
        return f"Registry('{self.name}', {len(self)} items)"


# ============================================================================
# Global Registries
# ============================================================================

# Dataset registry
DATASET_REGISTRY = Registry('datasets')

# Model registry
MODEL_REGISTRY = Registry('models')

# UQ method registry
UQ_METHOD_REGISTRY = Registry('uq_methods')


# ============================================================================
# Convenience Functions
# ============================================================================

def register_dataset(name: Optional[str] = None, **metadata):
    """
    Register a dataset class.

    Parameters
    ----------
    name : str, optional
        Name to register under
    **metadata
        Additional metadata

    Returns
    -------
    decorator : callable
        Decorator function
    """
    return DATASET_REGISTRY.register(name, **metadata)


def register_model(name: Optional[str] = None, **metadata):
    """
    Register a model class.

    Parameters
    ----------
    name : str, optional
        Name to register under
    **metadata
        Additional metadata

    Returns
    -------
    decorator : callable
        Decorator function
    """
    return MODEL_REGISTRY.register(name, **metadata)


def register_uq_method(name: Optional[str] = None, **metadata):
    """
    Register a UQ method class.

    Parameters
    ----------
    name : str, optional
        Name to register under
    **metadata
        Additional metadata

    Returns
    -------
    decorator : callable
        Decorator function
    """
    return UQ_METHOD_REGISTRY.register(name, **metadata)


def get_dataset(name: str) -> Type:
    """Get registered dataset class by name."""
    return DATASET_REGISTRY.get(name)


def get_model(name: str) -> Type:
    """Get registered model class by name."""
    return MODEL_REGISTRY.get(name)


def get_uq_method(name: str) -> Type:
    """Get registered UQ method class by name."""
    return UQ_METHOD_REGISTRY.get(name)


def list_datasets() -> list:
    """List all registered dataset names."""
    return DATASET_REGISTRY.list()


def list_models() -> list:
    """List all registered model names."""
    return MODEL_REGISTRY.list()


def list_uq_methods() -> list:
    """List all registered UQ method names."""
    return UQ_METHOD_REGISTRY.list()
