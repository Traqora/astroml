from typing import Any, Dict, List, Optional, Union, Callable
class AstroMLError(Exception):
    def __init__(self, message -> Any: str, **context):
        super().__init__(message)
        self.context = context

class IngestionError(AstroMLError): pass
class FeatureError(AstroMLError): pass
class ModelError(AstroMLError): pass
class DatabaseError(AstroMLError): pass
