from .schema import Schema

# Setup Null handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
