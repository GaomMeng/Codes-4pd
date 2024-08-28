# server.py
import logging
import os
import uuid

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse

from tts.config import log_settings, tts_settings
from tts.models import TtsRequest
from tts.synthesizer import generate_speech_with_piper
from tts.utils import replace_punctuation_with_space

# Logger configuration
logger = logging.getLogger(__name__)
log_level = getattr(logging, log_settings.level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format=log_settings.format,
    datefmt=log_settings.datefmt
)

app = FastAPI()

service_running = False
tts_model_path = tts_settings.model_path
punctuation_to_ignore = tts_settings.punctuation_to_ignore

@app.get("/ready")
def ready():
    global service_running
    if service_running:
        return {"status": "OK"}
    else:
        try:
            # Test service availability be generating once.
            # This could also warm up everything.
            text_to_generate = "Banishment, This, World!"
            generate_speech_with_piper(
                text_to_generate, 
                tts_model_path, 
                "ready_output.wav"
            )

            if os.path.exists("ready_output.wav"):
                service_running = True
                return {"status": "OK"}
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Service not ready"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/")
def tts(request: TtsRequest):
    try:
        # Assign a uuid to the job.
        job_id = str(uuid.uuid4())
        logger.info(
            f"Received TTS request for language code {request.language}."
        )

        logger.info(
            f"The original length of text was {len(request.transcription)}."
        )

        text_to_generate = replace_punctuation_with_space(
            request.transcription,
            punctuation_to_ignore
        )

        generate_speech_with_piper(
            text_to_generate,
            tts_model_path,
            f"/tmp/{job_id}.wav"
        )
        
        logger.info(
            f"File generated at /tmp/{job_id}.wav. Sending to client..."
        )

        return FileResponse(
            path=f"/tmp/{job_id}.wav",
            filename="output.wav",
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
