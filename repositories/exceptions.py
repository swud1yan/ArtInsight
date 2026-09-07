"""Repository-level exceptions for database access."""


class RepositoryError(Exception):
    """Base exception for repository failures."""


class EntityNotFoundError(RepositoryError):
    """Raised when an entity cannot be found by identifier."""


class EntityConflictError(RepositoryError):
    """Raised when a uniqueness or integrity conflict occurs."""