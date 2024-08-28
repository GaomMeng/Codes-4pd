# models/requests.py
from pydantic import BaseModel

class TtsRequest(BaseModel):
    transcription: str
    language: str
