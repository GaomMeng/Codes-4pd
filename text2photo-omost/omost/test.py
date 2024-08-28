import lib_omost.canvas as omost_canvas
import os
import base64
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

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
HF_TOKEN = None

import lib_omost.memory_management as memory_management
import uuid

import torch
import numpy as np
import gradio as gr
import tempfile
from gradio_client import Client

gradio_temp_dir = './gradio'
os.makedirs(gradio_temp_dir, exist_ok=True)

from threading import Thread

# Phi3 Hijack
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel

Phi3PreTrainedModel._supports_sdpa = True

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from lib_omost.pipeline import StableDiffusionXLOmostPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList

# SDXL

# sdxl_name = 'D:\\LiblibAI\\Omost\\mobius'
sdxl_name = '/root/test/mobius'

tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
# text_encoder = CLIPTextModel.from_pretrained(
    # sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder")
# text_encoder_2 = CLIPTextModel.from_pretrained(
    # sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(sdxl_name, subfolder="text_encoder_2")
# vae = AutoencoderKL.from_pretrained(
#     sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
vae = AutoencoderKL.from_pretrained(sdxl_name, subfolder="vae")
# unet = UNet2DConditionModel.from_pretrained(
#     sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")
unet = UNet2DConditionModel.from_pretrained(sdxl_name, subfolder="unet")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

# LLM

# llm_name = 'D:\\LiblibAI\\Omost\\omost-dolphin-2.9-llama3-8b-4bits'
llm_name = '/root/test/omost/omost-dolphin-2.9-llama3-8b-4bits'

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.bfloat16,  # This is computation type, not load/memory type. The loading quant type is baked in config.
    token=HF_TOKEN,
    device_map="auto"  # This will load model to gpu with an offload system
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
    token=HF_TOKEN
)

memory_management.unload_all_models(llm_model)


@torch.inference_mode()
def chat_fn(message: str, history: list, seed:int, temperature: float, top_p: float, max_new_tokens: int) -> str:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]

    for user, assistant in history:
        if isinstance(user, str) and isinstance(assistant, str):
            if len(user) > 0 and len(assistant) > 0:
                conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    
    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)

    input_ids = llm_tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    streamer = TextIteratorStreamer(llm_tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    def interactive_stopping_criteria(*args, **kwargs) -> bool:
        if getattr(streamer, 'user_interrupted', False):
            print('User stopped generation')
            return True
        else:
            return False

    stopping_criteria = StoppingCriteriaList([interactive_stopping_criteria])

    def interrupter():
        streamer.user_interrupted = True
        return

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    if temperature == 0:
        generate_kwargs['do_sample'] = False

    llm_model.generate(**generate_kwargs)

    outputs = []
    for text in streamer:
        outputs.append(text)

    return "".join(outputs)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def diffusion_fn(canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt):

    use_initial_latent = False
    eps = 0.05

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])
    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    # print(canvas_outputs)
    # print(canvas_outputs['bag_of_conditions'])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])
    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
            initial_latent=latents,
            strength=highres_denoise,
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        unique_hex = uuid.uuid4().hex
        image_path = os.path.join(gradio_temp_dir, f"{unique_hex}_{i}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            # 把base64字符串保存到文件
            with open(os.path.join(gradio_temp_dir, f"{unique_hex}_{i}") + '.txt', 'w') as f:
                f.write(image_data)
                
            return image_data
    return

# prompt = chat_fn("rabbit", [], 12345, 0.6, 0.9, 4096)

# canvas = omost_canvas.Canvas.from_bot_response(prompt)

# canvas_state = canvas.process()

# # print(canvas_state)

# diffusion_fn(canvas_state, 1, 12345, 1024, 1024, 1.0, 25, 5, 20, 0.9, "lowres, bad anatomy, bad hands, cropped, worst quality")

def _predict(prompt, name, prompt_2='', seed_0=0, randomize_seed_0=True, step=40, base_refiner_ratio=0.8, seed=0, randomize_seed=True, motion_bucket_id=50, noise_aug_strength=0.1, frames_per_second=5):
    negative_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    
    canvas = omost_canvas.Canvas.from_bot_response(prompt)
    canvas_state = canvas.process()
    
    # image = diffusion_fn(
    #     prompt,
    #     negative_prompt=negative_prompt,
    #     width=1024,
    #     height=1024,
    #     guidance_scale=7,
    #     num_inference_steps=50,
    #     clip_skip=3
    # ).images[0]
    
    return diffusion_fn(canvas_state, 1, 12345, 1024, 1024, 1.0, 25, 5, 20, 0.9, negative_prompt)

    # name = sanitize_filename(en_prompt[:10])
    # print(name)
    # image_file = f"/root/test/{name}.png"
    # image.save(image_file)
    # def pil_image_to_base64(img):
    #     # Ensure the image is 1024x1024
    #     img = img.resize((1024, 1024))
        
    #     # Convert the image to a byte stream
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="PNG")
        
    #     # Encode the byte stream to base64
    #     img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    #     return img_str
    # img_str = pil_image_to_base64(image)
    # return img_str

app = flask.Flask(__name__)

app.route("/health")
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
    # print(prompt)
    prompt = chat_fn("rabbit", [], 12345, 0.6, 0.9, 4096)
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