from .kg import KnowledgeGraph
from .source import Source

# Setup Null handler
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
