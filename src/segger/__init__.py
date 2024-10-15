import importlib.util

cupy_available = importlib.util.find_spec("cupy") is not None

from segger.data import *
from segger.models import *
from segger.training import *
from segger.validation import *

# segger.prediction requires cupy, which is not available in macOS
if cupy_available:
    from segger.prediction import *

__all__ = ["data", "models", "prediction", "training"]
