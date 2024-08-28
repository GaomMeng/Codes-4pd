FROM ubuntu:24.04

WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment in the container
RUN python3 -m venv /opt/venv

# Activate the virtual environment
# Every command that uses Python or pip should now be prefixed with this activation
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install \
    --no-cache-dir \
    -r requirements.txt 

# Install argospm
RUN pip install argostranslate

# Specify language to install
ARG LANG_PAIR1=translate-ja_en
ARG LANG_PAIR2=translate-en_zh

# Download and install the translation packages manually
RUN wget https://www.argosopentech.com/datasets/${LANG_PAIR1}.argosmodel -P /app/models/
RUN wget https://www.argosopentech.com/datasets/${LANG_PAIR2}.argosmodel -P /app/models/
RUN argospm install /app/models/${LANG_PAIR1}.argosmodel
RUN argospm install /app/models/${LANG_PAIR2}.argosmodel

# Expose port 80
EXPOSE 80

# Copy the application code
COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
