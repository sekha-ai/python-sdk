"""
Sekha AI Memory System - Python SDK
"""

from .client import SekhaClient, SyncSekhaClient, ClientConfig, MemoryController, MemoryConfig
from .models import *
from .errors import *

__version__ = "0.5.0"
__author__ = "Sekha AI <team@sekha.ai>"

__all__ = [
    # Client
    "SekhaClient",
    "SyncSekhaClient",
    "ClientConfig",
    "MemoryController",  # Alias for SekhaClient
    "MemoryConfig",      # Alias for ClientConfig
    # Models
    "MessageRole",
    "ConversationStatus",
    "SummaryLevel",
    "MessageDto",
    "NewConversation",
    "ConversationResponse",
    "QueryRequest",
    "QueryResult",
    "QueryResponse",
    "LabelSuggestion",
    "PruningSuggestion",
    "HealthResponse",
    "ImportanceScore",
    "SummaryResponse",
    # Errors
    "SekhaError",
    "SekhaAPIError",
    "SekhaNotFoundError",
    "SekhaAuthError",
    "SekhaConnectionError",
    "SekhaValidationError",
    "SekhaRateLimitError",
]