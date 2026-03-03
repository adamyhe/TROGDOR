from .trogdor import TROGDOR, load_pretrained_model, standardization
from .predict import predict, predict_chromosome

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trogdor")
except PackageNotFoundError:
    __version__ = "unknown"
