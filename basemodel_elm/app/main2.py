import json
import logging
import os
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
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", 128))
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", 1.0))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", 1.0))

# Create instances
client = OpenAI(api_key=TOKEN, base_url=BASE_URL)
app = FastAPI()

def generate(messages):
    try:
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
    except Exception as e:
        logger.error(f"Error in generate function: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generate function: {str(e)}")

def make_translation_prompt(from_lang, to_lang, text):
    return f"将这句话从{from_lang}语言翻译为{to_lang}语言，只显示翻译内容：{text}"

@app.post("/v1/chat/completions")
async def post_chat_completions(request: Request):
    try:
        body = await request.json()
        data = body.get("data", [])
        
        results = []
        
        for item in data:
            from_lang = item["from"]
            to_lang = item["to"]
            texts = item["texts"]
            
            translated_texts = []
            for text in texts:
                prompt = make_translation_prompt(from_lang, to_lang, text)
                messages = [{"role": "user", "content": prompt}]
                translation = generate(messages)
                translated_texts.append([translation])
            
            results.append({
                "from": from_lang,
                "to": to_lang,
                "texts": texts,
                "translated": translated_texts
            })
        
        return {"choices": [{"message": {"content": json.dumps({"data": results})}}]}
    except Exception as e:
        logger.error(f"Error in post_chat_completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in post_chat_completions: {str(e)}")

    
@app.get("/ready")
async def ready():
    """Check if server is ready."""
    if MODE == "model_hub":
        return "Ready"

    try:
        # Simple test to check if the model is accessible
        test_message = [{"role": "user", "content": "Hello"}]
        generate(test_message)
        return "Ready"
    except Exception as e:
        logger.error(f"Error in ready check: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

