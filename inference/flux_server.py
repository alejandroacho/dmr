"""
FLUX.1-dev inference server for the Blackwell Gateway.
Loads the model from /models (mounted read-only) and serves POST /generate.
"""

import base64
import io
import logging
import os
import random
import time

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/models")
PORT = int(os.getenv("PORT", "8004"))

app = FastAPI(title="FLUX.1-dev inference server")
pipe: FluxPipeline | None = None


@app.on_event("startup")
async def load_model() -> None:
    global pipe
    logger.info("Loading FLUX.1-dev from %s ...", MODEL_PATH)
    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    logger.info("Model loaded in %.1fs", time.time() - t0)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    cfg_scale: float = 7.5
    seed: int | None = None
    agent_id: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "flux1-dev", "ready": pipe is not None}


@app.post("/generate")
async def generate(req: GenerateRequest) -> JSONResponse:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    logger.info(
        "Generating %dx%d image | steps=%d | seed=%d | agent=%s",
        req.width, req.height, req.steps, seed, req.agent_id,
    )
    t0 = time.time()

    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.cfg_scale,
        generator=generator,
    )

    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    elapsed = (time.time() - t0) * 1000
    logger.info("Image generated in %.1fms", elapsed)

    return JSONResponse({
        "images": [b64],
        "seed": seed,
        "width": req.width,
        "height": req.height,
        "processing_time_ms": elapsed,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
