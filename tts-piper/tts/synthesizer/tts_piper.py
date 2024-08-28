# synthesizer/tts_piper.py
import logging
import os
import subprocess

from tts.config import log_settings

# Logger configuration
logger = logging.getLogger(__name__)
log_level = getattr(logging, log_settings.level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format=log_settings.format,
    datefmt=log_settings.datefmt
)

def generate_speech_with_piper(text, model_path, output_file_path):
    try:
        # Construct the command
        command = (
            f'echo "{text}" | ' +
            f"piper" +
            f" --model {model_path}" +
            f" --output_file {output_file_path}"
        )
        
        # Execute the command
        results = subprocess.run(
            command, 
            shell=True, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if os.path.exists(output_file_path):
            logger.info(f"Successfully generated speech file at {output_file_path}.")
        else:
            logger.error(f"Failed to find the voice file generated.")
        
        return output_file_path
    except subprocess.CalledProcessError as e:
        logger.error(f"An error occurred while generating the speech file: {e}.")
