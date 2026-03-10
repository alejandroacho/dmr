"""
LTX-Video inference server for the Blackwell Gateway.
Loads the model from /models (mounted read-only) and serves POST /generate.
"""

import base64
import io
import logging
import os
import random
import time
import tempfile

import numpy as np
import torch
from diffusers import LTXVideoPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "/models")
PORT = int(os.getenv("PORT", "8005"))

app = FastAPI(title="LTX-Video inference server")
pipe: LTXVideoPipeline | None = None


@app.on_event("startup")
async def load_model() -> None:
    global pipe
    logger.info("Loading LTX-Video from %s ...", MODEL_PATH)
    t0 = time.time()
    pipe = LTXVideoPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    logger.info("Model loaded in %.1fs", time.time() - t0)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 768
    height: int = 512
    num_frames: int = 81
    fps: int = 24
    steps: int = 50
    cfg_scale: float = 7.0
    seed: int | None = None
    agent_id: str | None = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "ltx-video", "ready": pipe is not None}


@app.post("/generate")
async def generate(req: GenerateRequest) -> JSONResponse:
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator("cuda").manual_seed(seed)

    logger.info(
        "Generating %dx%d video | %d frames @ %dfps | steps=%d | seed=%d | agent=%s",
        req.width, req.height, req.num_frames, req.fps, req.steps, seed, req.agent_id,
    )
    t0 = time.time()

    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=req.width,
        height=req.height,
        num_frames=req.num_frames,
        num_inference_steps=req.steps,
        guidance_scale=req.cfg_scale,
        generator=generator,
    )

    # result.frames is a list of PIL images (one per frame)
    frames: list = result.frames[0] if isinstance(result.frames[0], list) else result.frames

    # Export to MP4 via imageio
    import imageio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    writer = imageio.get_writer(tmp_path, fps=req.fps, format="FFMPEG", codec="libx264")
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.unlink(tmp_path)

    elapsed = (time.time() - t0) * 1000
    logger.info("Video generated in %.1fms", elapsed)

    return JSONResponse({
        "video_base64": b64,
        "fps": req.fps,
        "num_frames": len(frames),
        "width": req.width,
        "height": req.height,
        "seed": seed,
        "processing_time_ms": elapsed,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
