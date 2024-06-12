from .source import Source
from .kg import KnowledgeGraph

# Setup Null handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
