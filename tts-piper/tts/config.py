# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class LogSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_prefix="LOG_")
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

class TtsSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_prefix="TTS_")
    model_path: str = "./mnt/models/piper_tts_model.onnx"
    punctuation_to_ignore: str = ",.;!?"

log_settings = LogSettings()
tts_settings = TtsSettings()