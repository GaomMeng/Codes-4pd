# Use the official Python 3.10 image from the Docker Hub
# piper-tts does NOT support higher version of Python!
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt (if you have other dependencies)
COPY requirements.txt ./

RUN pip install piper-tts

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt 

COPY tts tts

# Command to run your application
CMD ["uvicorn", "tts.server:app", "--host", "0.0.0.0", "--port", "80"]
