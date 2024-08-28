
# TTS Piper Project

This project is a Text-to-Speech (TTS) service. It provides a FastAPI server that generates speech audio files from text input using TTS engines (currently only `piper`).

## Features

- Converts text input to speech audio files.
- Provides a REST API for text-to-speech conversion.
- Dockerized for easy deployment.
- Currently only support one language type per instance. Will support multilingual in the future.

## Prerequisites

- Python 3.10 (Since `piper-tts` only works on *some* version of Python, the latest being 3.10. If you try to install on higher version of Python, `pip install piper-tts` is likely to fail.)
- Docker (for containerized deployment)
- A piper TTS model (downloadable from [piper-voices](https://huggingface.co/rhasspy/piper-voices)) and its json config

## Installation

### Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://gitlab.4pd.io/liuxinyang/tts-piper.git
   cd tts-piper
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install piper-tts
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prepare the TTS model**:
   1. Download a model (together with the json file of *SAME* filename) from [piper-voices](https://huggingface.co/rhasspy/piper-voices), and put it in an accessible folder. 
   2. Set the environmental variable TTS_MODEL_PATH to the .onnx file (e.g., `/tmp/models/my_piper_tts_model.onnx`, and the file `my_piper_tts_model.onnx.json` must be in the same folder).

5. **Run the FastAPI server**:
   ```bash
   uvicorn tts.server:app --host 0.0.0.0 --port 8000
   ```

### Running with Docker

1. **Build the Docker image**:
   ```bash
   docker build -t tts-piper:latest .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -d -p 80:80 \
   -v your/model/folder/path:container/model/folder/path \
   -e TTS_MODEL_PATH=path/to/onnx/model/file \
   tts-piper:latest
   ```

## API Endpoints

### Check Service Readiness

- **Endpoint**: `/ready`
- **Method**: `GET`
- **Description**: Checks if the TTS service is ready.
- **Response**: `{"status": "OK"}` if ready, otherwise `500 Internal Server Error`.

### Generate Speech

- **Endpoint**: `/`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "language": "en",
    "transcription": "Hello, world!"
  }
  ```
- **Response**: A `.wav` file containing the generated speech.

## Project Structure

```plaintext
.
├── Dockerfile
├── README.md
├── requirements.txt
├── tts
│   ├── models
│   │   └── __init__.py
│   │   └── requests.py
│   ├── synthesizer
│   │   ├── __init__.py
│   │   └── tts_piper.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── str_utils.py
│   ├── config.py
│   └── server.py
```

- **`Dockerfile`**: Docker configuration for the project.
- **`requirements.txt`**: List of Python dependencies.
- **`tts/`**: Application source code.
  - **`models/`**: Contains `BaseModel` for API request validation.
  - **`synthesizer/`**: Contains the main TTS function in `tts_piper.py`.
  - **`utils/`**: Utility functions, such as string manipulation in `str_utils.py`.
  - **`config.py`**: Configuration settings for logging and TTS.
  - **`server.py`**: Main FastAPI server implementation.

## Example Usage

### Checking Service Readiness

```bash
curl -X GET http://localhost:8000/ready
```

### Generating Speech

```bash
curl -X POST http://localhost:8000/ -H "Content-Type: application/json" -d '{"language": "en", "transcription": "Hello, world!"}' --output output.wav
```

This command will save the generated speech file as `output.wav`.

## Configuration

The project uses `pydantic` for configuration management. Configuration settings are defined in `config.py`.

### LogSettings

- **`level`**: Logging level (default: `"INFO"`)
- **`format`**: Logging format (default: `"%(asctime)s - %(levelname)s - %(message)s"`)
- **`datefmt`**: Date format for logs (default: `"%Y-%m-%d %H:%M:%S"`)

### TtsSettings

- **`model_path`**: Path to the TTS model (default: `"./mnt/models/piper_tts_model.onnx"`, there ***MUST*** be a json of the same filename in the folder, that is something like `./mnt/models/piper_tts_model.onnx.json`)
- **`punctuation_to_ignore`**: Punctuation to ignore (default: `",.;!?"`)
