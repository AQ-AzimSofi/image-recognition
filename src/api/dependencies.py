from functools import lru_cache

from ..db import Database
from ..rekognition import RekognitionClient


@lru_cache()
def get_db() -> Database:
    return Database()


@lru_cache()
def get_rekognition_client() -> RekognitionClient:
    return RekognitionClient()
