from .asdf.utils import smart_read_yaml, is_mpi_env  # NOQA
from .data.management.skeleton import DataBaseSkeleton  # NOQA
import logging
from .log_util import modify_logger

# Setup the logger
logger = logging.getLogger("gcmt3d")

# Sets up the formatting and everything..
modify_logger(logger)

'''The different levels are

 0: logging.NOTSET
10: logging.DEBUG
15: logging.VERBOSE
20: logging.INFO
30: logging.WARNING
40: logging.ERROR
50: logging.CRITICAL

'''
logger.setLevel(logging.VERBOSE)
