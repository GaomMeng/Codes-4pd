import shutil
import argparse
import os
import torch
from torch import distributed, nn
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Form
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union
import time 
import numpy as np 
import logging
# from openai import OpenAI, AsyncOpenAI, AsyncAzureOpenAI
import uuid
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)
import re

def sanitize_filename(input_str):
    # Replace spaces with underscores
    sanitized_str = input_str.replace(' ', '_')
    # Remove any character that is not alphanumeric, underscore, or hyphen
    sanitized_str = re.sub(r'[^\w\-]', '', sanitized_str)
    return sanitized_str

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "/root/test/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "/root/test/mobius", 
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')


MODEL_PATH = "/root/test/model_zoo/Qwen1.5-7B-Chat"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained(MODEL_PATH)


async def translate_prompt(prompt):
    prefix = '请把下面一段话翻译成英文；如果原本就是英文那么请直接输出原文：\n'
    full_prompt = prefix + prompt + "\n翻译成英文结果："
    print(full_prompt)
    inputs = tokenizer(full_prompt,return_tensors='pt').to('cuda')
    generate_ids = model.generate(inputs.input_ids, generation_config=generation_config)
    pred = tokenizer.batch_decode(generate_ids,skip_special_tokens=True)
    print("\npred:\n")
    print(pred)
    return pred[0].split("翻译成英文结果：")[-1]

async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _predict(prompt,name, prompt_2='', seed_0=0, randomize_seed_0=True, step=40, base_refiner_ratio=0.8, seed=0, randomize_seed=True, motion_bucket_id=50, noise_aug_strength=0.1, frames_per_second=5):
    # en_prompt = Translator(from_lang="ZH",to_lang="EN-US").translate(prompt)
    en_prompt = await translate_prompt(prompt=prompt)
    print(en_prompt)
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        width=1024,
        height=1024,
        guidance_scale=7,
        num_inference_steps=50,
        clip_skip=3
    ).images[0]
    
    image_file = f"/root/test/{sanitize_filename(en_prompt[:10])}.png"
    image.save(image_file)
    return image_file

@app.post("/predict")
async def predict(prompt: str=Form(...)):
    name = uuid.uuid4().hex
    # prompt = request.prompt
    print(prompt)
    video_file = await _predict(prompt,name)
    print(video_file)
    final_video_file = os.path.basename(video_file)
    
    #final_video_file = f"{name}.mp4"
    print(final_video_file)
    # shutil.copy(video_file, os.path.join('/video-workspace/videos/', final_video_file))
    return {
        "success": True,
        "filename": final_video_file, 
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
