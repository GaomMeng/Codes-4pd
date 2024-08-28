import logging
import os

from fastapi import FastAPI, Request
from typing import Dict, List
from pydantic import BaseModel

import argostranslate.package
import argostranslate.translate

def install_package(from_code: str, to_code: str):
    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())


# Logging config
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=LOG_LEVEL, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class TranslateRequest(BaseModel):
    parameter: Dict

class TranslateResponse(BaseModel):
    trans_results: List[str]

app = FastAPI()

@app.get("/ready")
def ready():    
    return {"status": "OK"}

@app.post("/v1/translate")
def translate(request: TranslateRequest) -> TranslateResponse:
    text = request.parameter["text"][0]    
    from_lang = request.parameter["from"]
    to_lang = request.parameter["to"]

    # Install package if not installed
    # if not argostranslate.package.is_package_installed(from_lang, to_lang):
    #     install_package(from_lang, to_lang)

    logger.info(f"Received request with text of length {len(text)}...")

    try:
        translated_text = argostranslate.translate.translate(
            text,
            from_lang,
            to_lang,
        )
        return TranslateResponse(
            trans_results = [translated_text]
        )
    except Exception as e:
        logger.error(f"Translation failed due to {e}.")
        logger.error(f"Empty return to avoid submit failure.")
        return TranslateResponse(
            trans_results = [""]
        )


