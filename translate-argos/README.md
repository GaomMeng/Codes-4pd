
# Translation Service with Argos Translate

This project provides a FastAPI-based translation service using Argos Translate. The service allows you to translate text between specified languages. The Dockerfile is designed to create a flexible container that can handle different language pairs based on the build or runtime configuration.

## Prerequisites

- Docker
- Docker Compose (optional)
- Python 3.8 or higher

## Project Structure

- `Dockerfile`: Defines the Docker image.
- `app.py`: The FastAPI application that handles translation requests.
- `requirements.txt`: Python dependencies for the project.
- `README.md`: This file.

## Running the Project 

### 1. Install the required packages

```sh
pip install -r requirements.txt
```

### 2. Install the desired language pairs using argospm

View [argospm index](https://www.argosopentech.com/argospm/index/) for the available language pairs.

```sh
argospm update
argospm install translate-ja_en translate-en_zh
```

### 3. Run the FastAPI application using Uvicorn

```sh
uvicorn app:app --host 0.0.0.0 --port 80
```

## Building the Docker Image

To build the Docker image with specific language pairs, use the following commands. You can specify the language pairs during the build process using build arguments.

### Example: Japanese to English and English to Chinese

View [argospm index](https://www.argosopentech.com/argospm/index/) for the language pairs that's available. If you are trying to build the service to translate between languages without an existing option here, try install language A to English and English to language B, and argostranslate will automatically translate using English as the medium.

```sh
docker build --build-arg LANG_PAIR1=translate-ja_en --build-arg LANG_PAIR2=translate-en_zh -t my_translator .
```

### General Build Command

```sh
docker build --build-arg LANG_PAIR1=<language_pair_1> --build-arg LANG_PAIR2=<language_pair_2> -t my_translator .
```

Replace `<language_pair_1>` and `<language_pair_2>` with the desired language pairs. Language pairs are specified in the format `translate-<from_lang>_<to_lang>`.

## Running the Docker Container

You can run the Docker container with the specified language pairs as environment variables:

```sh
docker run -e LANG_PAIRS="translate-ja_en translate-en_zh" -p 80:80 my_translator
```

## Using the Translation Service

Once the container is running, you can use the following endpoints:

### Check if the service is ready

```sh
GET /ready
```

Response:

```json
{
    "status": "OK"
}
```

### Translate Text

```sh
POST /v1/translate
```

Request Body:

```json
{
    "parameter": {
        "text": ["駆逐してやる!! この世から･･･一匹残らず!!"],
        "from": "ja",
        "to": "zh"
    }
}
```

Response:

```json
{
    "trans_results": ["Translated text"]
}
```

## Logging

The logging level can be configured using the `LOG_LEVEL` environment variable. By default, it is set to `WARNING`.

Example:

```sh
docker run -e LOG_LEVEL=INFO -e LANG_PAIRS="translate-en_fa translate-en_es" -p 80:80 my_translator
```

## Conclusion

This project provides a flexible and customizable translation service using Argos Translate and FastAPI. By leveraging Docker, you can easily build and deploy the service with different language pairs as needed.
