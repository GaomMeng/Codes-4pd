"""A strategy server based on local vLLM server."""

import json
import logging
import os
import urllib.error
import urllib.request

from fastapi import FastAPI, HTTPException, Request
from openai import OpenAI

logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

MODE = os.environ["MODE"]

# Model config
if MODE == "model_hub":
    logger.info(os.environ["LEADERBOARD_MODELHUB_KEY2INFO"])
    model_info = json.loads(os.environ["LEADERBOARD_MODELHUB_KEY2INFO"])
    model_name = os.environ["MODEL_NAME"]
    TOKEN = model_info["token"]
    BASE_URL = model_info["entrypoint"]
    MODEL_PATH = model_info["model_key2info"][model_name]["modelId"]
else:
    MODEL_PATH = os.environ["MODEL_PATH"]
    BASE_URL = "http://0.0.0.0:8000/v1"
    TOKEN = "deadbeef"

# Generation config
GEN_MAX_TOKENS = int(os.environ["GEN_MAX_TOKENS"])
GEN_TEMPERATURE = float(os.environ["GEN_TEMPERATURE"])
GEN_TOP_P = float(os.environ["GEN_TOP_P"])

# Prompt config
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
PREFIX_PROMPT = os.getenv("PREFIX_PROMPT", "")
SUFFIX_PROMPT = os.getenv("SUFFIX_PROMPT", "")
FEW_SHOT_PATH = os.getenv("FEW_SHOT_PATH", "")
REFLECTION_PROMPT = os.getenv("REFLECTION_PROMPT", "")
CHOICE_TYPE_ENABLE = os.getenv("CHOICE_TYPE_ENABLE", "False").lower() in (
    "true",
    "1",
    "yes",
)

# Other config
CONTEST_TYPE = os.getenv("CONTEST_TYPE", "literature")
FEW_SHOT_ENABLE = False

# Check path
if MODE != "model_hub":
    logger.info("MODEL_PATH is: %s", MODEL_PATH)
    assert os.path.isdir(MODEL_PATH)
    logger.info("found the model at: %s", MODEL_PATH)

if len(FEW_SHOT_PATH) != 0:
    logger.info("FEW_SHOT_PATH is: %s", FEW_SHOT_PATH)
    assert os.path.isfile(FEW_SHOT_PATH)
    FEW_SHOT_ENABLE = True
    logger.info("found the few shot file at: %s", FEW_SHOT_PATH)

# Print MODE
print()
logger.info("MODE: %s", MODE)
logger.info("MODE_PATH: %s", MODEL_PATH)
logger.info("BASE_URL: %s", BASE_URL)
logger.info("TOKEN: %s", TOKEN)
print()

# Print config
print()
logger.info("%s prompt config: ", CONTEST_TYPE)
logger.info("SYSTEM_PROMPT: %s", SYSTEM_PROMPT)
logger.info("PREFIX_PROMPT: %s", PREFIX_PROMPT)
logger.info("SUFFIX_PROMPT: %s", SUFFIX_PROMPT)
logger.info("REFLECTION_PROMPT: %s", REFLECTION_PROMPT)
logger.info("CHOICE_TYPE_ENABLE: %s", CHOICE_TYPE_ENABLE)
logger.info("FEW_SHOT_ENABLE: %s", FEW_SHOT_ENABLE)
print()

# Create instances
client = OpenAI(api_key=TOKEN, base_url=BASE_URL)
app = FastAPI()


def generate(messages):
    completions = client.chat.completions.create(
        model=MODEL_PATH,
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        messages=messages,
        stop=None,
        stream=False,
    )
    return completions.choices[0].message.content


def parse_content(content):
    separator = "@@@"
    if separator in content:
        type_, question = content.split(separator, 1)
        return type_, question
    else:
        return "", content


def make_single_content(type_, question):
    if CHOICE_TYPE_ENABLE:
        return f"{PREFIX_PROMPT}{type_}{question}{SUFFIX_PROMPT}"
    else:
        return f"{PREFIX_PROMPT}{question}{SUFFIX_PROMPT}"


def make_messages(type_, question):
    messages = []

    if len(SYSTEM_PROMPT) != 0:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if FEW_SHOT_ENABLE:
        with open(FEW_SHOT_PATH, "r", encoding="utf-8") as fs:
            for shot in fs.read().split("---"):
                shot_info = [x for x in shot.split("\n") if x]
                assert len(shot_info) == 3
                messages.append(
                    {
                        "role": "user",
                        "content": make_single_content(shot_info[0], shot_info[1]),
                    }
                )
                messages.append({"role": "assistant", "content": shot_info[2]})

    messages.append({"role": "user", "content": make_single_content(type_, question)})

    if len(REFLECTION_PROMPT) != 0:
        first_answer = generate(messages)
        messages.append({"role": "assistant", "content": first_answer})
        messages.append({"role": "user", "content": REFLECTION_PROMPT})

    return messages


# API
@app.get("/ready")
async def ready():
    """Check if server is ready."""
    if MODE == "model_hub":
        return "Ready"

    try:
        with urllib.request.urlopen(f"{BASE_URL}/models"):
            return "Ready"
    except urllib.error.URLError as err:
        raise HTTPException(status_code=503, detail="Not ready") from err


@app.post("/v1/chat/completions")
async def post_chat_completions(request: Request):
    body = await request.json()
    content = body["messages"][0]["content"]

    # By default: type is empty, content is the question.
    type_ = ""
    question = content

    if CHOICE_TYPE_ENABLE:
        type_, question = parse_content(content)

    # If necessary, perform preprocessing here
    if CONTEST_TYPE == "english":
        pass
    elif CONTEST_TYPE == "literature":
        pass
    else:
        pass

    messages = make_messages(type_, question)
    answer = generate(messages)

    return {"choices": [{"message": {"content": answer}}]}
