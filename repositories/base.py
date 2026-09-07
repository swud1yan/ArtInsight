"""Generic repository base class for SQLAlchemy ORM models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from repositories.exceptions import EntityConflictError, RepositoryError

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    """Provide common CRUD operations with transaction-safe error handling."""

    def __init__(self, session: Session, model_class: type[ModelType]) -> None:
        """Create a repository bound to an active SQLAlchemy session."""
        self._session = session
        self._model_class = model_class

    def add(self, entity: ModelType) -> ModelType:
        """Add an entity and flush it so generated primary keys are available."""
        try:
            self._session.add(entity)
            self._session.flush()
            return entity
        except IntegrityError as exc:
            self._session.rollback()
            raise EntityConflictError(str(exc)) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise RepositoryError(str(exc)) from exc

    def get_by_id(self, entity_id: int) -> ModelType | None:
        """Return an entity by primary key, or None when it does not exist."""
        try:
            return self._session.get(self._model_class, entity_id)
        except SQLAlchemyError as exc:
            raise RepositoryError(str(exc)) from exc

    def list_all(self) -> list[ModelType]:
        """Return all entities for this repository's model type."""
        try:
            return list(self._session.scalars(select(self._model_class)).all())
        except SQLAlchemyError as exc:
            raise RepositoryError(str(exc)) from exc

    def query_by(self, **filters: object) -> list[ModelType]:
        """Return entities matching equality filters."""
        try:
            statement = select(self._model_class).filter_by(**filters)
            return list(self._session.scalars(statement).all())
        except SQLAlchemyError as exc:
            raise RepositoryError(str(exc)) from exc

    def delete(self, entity: ModelType) -> None:
        """Delete an entity and flush the pending change."""
        try:
            self._session.delete(entity)
            self._session.flush()
        except IntegrityError as exc:
            self._session.rollback()
            raise EntityConflictError(str(exc)) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise RepositoryError(str(exc)) from exc

    def delete_by_id(self, entity_id: int) -> bool:
        """Delete an entity by primary key and report whether it existed."""
        entity = self.get_by_id(entity_id)
        if entity is None:
            return False
        self.delete(entity)
        return True

    def add_many(self, entities: Sequence[ModelType]) -> list[ModelType]:
        """Add multiple entities and flush them in one unit of work."""
        try:
            self._session.add_all(entities)
            self._session.flush()
            return list(entities)
        except IntegrityError as exc:
            self._session.rollback()
            raise EntityConflictError(str(exc)) from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise RepositoryError(str(exc)) from exc