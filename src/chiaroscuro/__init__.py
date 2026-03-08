from importlib.metadata import PackageNotFoundError, version

from .data_transforms import normalization
from .predict import predict, predict_chromosome, predict_genome
from .trogdor import TROGDOR

try:
    __version__ = version("trogdor")
except PackageNotFoundError:
    __version__ = "unknown"
