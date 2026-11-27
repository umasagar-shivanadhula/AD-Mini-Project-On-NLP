"""Finds every NLPModule subclass in this folder automatically."""
import importlib
import inspect
import pkgutil

from .base import NLPModule


def load_modules():
    """Import each file here and create one instance of each module class."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, NLPModule) and not inspect.isabstract(cls)
                    and cls.__module__ == module.__name__):
                if cls.name in found:
                    raise ValueError(f"Duplicate module name: {cls.name}")
                found[cls.name] = cls()
    return dict(sorted(found.items(), key=lambda item: (item[1].order, item[0])))
