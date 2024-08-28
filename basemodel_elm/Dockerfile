FROM docker.4pd.io/zhaoxuezhi/vllm-openai:v0.4.2

ENV BUILD_CUDA_EXT=0
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./requirements.txt /workspace/requirements.txt
COPY ./logging.conf     /workspace/logging.conf
COPY ./app              /workspace/app
COPY ./run.sh           /workspace/run.sh

RUN pip install \
    --no-cache-dir \
    -U \
    -i https://nexus.4pd.io/repository/pypi-all/simple \
    -r /workspace/requirements.txt

# vllm config
ENV NUM_GPU=1
ENV MAX_MODEL_LEN=10000
ENV GPU_MEMORY_UTILIZATION=0.9
ENV GEN_MAX_TOKENS=1024
ENV GEN_TEMPERATURE=0.001
ENV GEN_TOP_P=0.999

# model config
ENV MODEL_PATH=""

# prompt config
ENV SYSTEM_PROMPT=""
ENV PREFIX_PROMPT=""
ENV SUFFIX_PROMPT=""
ENV FEW_SHOT_PATH=""
ENV REFLECTION_PROMPT=""
ENV CHOICE_TYPE_ENABLE="False"

# other config
ENV CONTEST_TYPE="literature"
ENV MODE="model_hub"
ENV MODE_NAME="llm"

EXPOSE 80

ENTRYPOINT [ "/bin/bash", "/workspace/run.sh" ]