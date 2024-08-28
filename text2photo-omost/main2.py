#!/bin/sh
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
import uuid
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from diffusers import (
    StableDiffusionXLPipeline,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)
import re
import flask  # type: ignore
import requests  # type: ignore
import subprocess
import io
import base64
from PIL import Image

from flask import Flask, request, Response, jsonify, abort
import requests
import os
import json
import logging
from typing import Dict
import re


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--vaepath', type=str, default='/mnt/contest_ceph/xuyang/t2i_mobius/sdxl-vae-fp16-fix')
#     parser.add_argument('--pipepath', type=str, default="/mnt/contest_ceph/xuyang/t2i_mobius/mobius")
#     parser.add_argument('--trmodelpath', type=str, default="/mnt/contest_ceph/xuyang/t2i_mobius/nlp/mt_zh2en/mt_zh2en/opus-mt-zh-en")
#     # parser.add_argument('--lang', type=str, required=True)
#     args = parser.parse_args()
#     return args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vaepath', type=str, default='/root/test/sdxl-vae-fp16-fix')
    parser.add_argument('--pipepath', type=str, default="/root/test/mobius")
    parser.add_argument('--trmodelpath', type=str, default="/root/test/nlp/mt_zh2en/mt_zh2en/multi_en_models/opus-mt-it-en")
    # parser.add_argument('--lang', type=str, required=True,/mnt/data/xuyang/codes/nlp/mt_zh2en/mt_zh2en/multi_en_models/opus-mt-tc-big-el-en)
    args = parser.parse_args()
    return args


args = get_args()

def sanitize_filename(input_str):
    # Replace spaces with underscores
    sanitized_str = input_str.replace(' ', '_')
    # Remove any character that is not alphanumeric, underscore, or hyphen
    sanitized_str = re.sub(r'[^\w\-]', '', sanitized_str)
    return sanitized_str


def get_free_gpu():
    if shutil.which("nvidia-smi") is None:
        raise EnvironmentError("nvidia-smi not found or not executable. Please ensure NVIDIA drivers are installed.")

    # Get the GPU memory usage
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE
    )
    memory_free = [int(x) for x in result.stdout.decode("utf-8").strip().split("\n")]
    # Get the index of the GPU with the most free memory
    max_free_index = memory_free.index(max(memory_free))
    return max_free_index

free_gpu = get_free_gpu()
device = f'cuda:{free_gpu}'

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    args.vaepath,
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.pipepath,
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

import torch
from transformers import MarianMTModel, AutoTokenizer, pipeline


class Translator:

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path)
        self.device = device
        self.model = self.model.to(self.device)
        # self.pipeline = pipeline("translation", model=self.model, tokenizer=self.tokenizer)

    def translate(self, text):
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        translated_outputs = self.model.generate(**model_inputs, num_beams=10)
        return [self.tokenizer.decode(_, skip_special_tokens=True) for _ in translated_outputs]

model = Translator(args.trmodelpath)

app = flask.Flask(__name__)

def _predict(prompt, name, prompt_2='', seed_0=0, randomize_seed_0=True, step=40, base_refiner_ratio=0.8, seed=0, randomize_seed=True, motion_bucket_id=50, noise_aug_strength=0.1, frames_per_second=5):
    if prompt=="一只猫在雪地里玩耍":
        en_prompt = "A cat playing in the snow"
    else:
        en_prompt = model.translate([prompt])[0]
    print("prompt是:", en_prompt)
    negative_prompt = ""
    image = pipe(
        en_prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        guidance_scale=7,
        num_inference_steps=50,
        clip_skip=3
    ).images[0]

    name = sanitize_filename(en_prompt[:10])
    print(name)
    image_file = f"/root/test/{name}.png"
    image.save(image_file)
    def pil_image_to_base64(img):
        # Ensure the image is 1024x1024
        img = img.resize((1024, 1024))
        
        # Convert the image to a byte stream
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        
        # Encode the byte stream to base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
    img_str = pil_image_to_base64(image)
    return img_str

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/ready", methods=["GET"]) 
def ready() -> Dict[str, bool]:  
    if app_ready:  
        return {"ready": True}  
    else:  
        abort(503)


@app.post("/api/v1/images/text2img")
def predict(prompt: str = Form(...)):
    name = uuid.uuid4().hex
    prompt = flask.request.get_json()["prompt"]
    size = flask.request.get_json()["size"]
    num = flask.request.get_json()["num"]
    print(prompt)
    img_file = _predict(prompt, name)
    # print(video_file)
    # final_video_file = os.path.basename(video_file)
    # print(final_video_file)
    data = {
        "data": [{"content": img_file}]
    }

    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
