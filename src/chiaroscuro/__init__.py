from importlib.metadata import PackageNotFoundError, version

from .data_transforms import normalization, standardization
from .predict import predict, predict_chromosome
from .trogdor import TROGDOR

try:
    __version__ = version("trogdor")
except PackageNotFoundError:
    __version__ = "unknown"
