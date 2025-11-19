"""Digital Nose sample application package."""

from .report import ScentReport

# Import GUI only if tkinter is available
try:
    from .gui import DigitalNoseApp
    __all__ = ["DigitalNoseApp", "ScentReport"]
except ImportError:
    # GUI not available in headless environments
    __all__ = ["ScentReport"]
