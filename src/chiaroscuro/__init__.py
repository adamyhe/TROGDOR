from importlib.metadata import PackageNotFoundError, version

from .predict import predict, predict_chromosome
from .trogdor import TROGDOR, load_pretrained_model, normalization, standardization

try:
    __version__ = version("trogdor")
except PackageNotFoundError:
    __version__ = "unknown"
