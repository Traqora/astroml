from typing import Any


class AstroMLError(Exception):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.context = context


class IngestionError(AstroMLError):
    pass


class FeatureError(AstroMLError):
    pass


class ModelError(AstroMLError):
    pass


class DatabaseError(AstroMLError):
    pass
