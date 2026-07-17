from importlib.metadata import PackageNotFoundError, version

from .data_transforms import normalization  # noqa
from .predict import predict_chromosome, predict_genome  # noqa
from .trogdor import TROGDOR  # noqa

try:
    __version__ = version("trogdor")
except PackageNotFoundError:
    __version__ = "unknown"
