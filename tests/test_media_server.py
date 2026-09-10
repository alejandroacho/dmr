"""
Tests for inference/media_server.py — the ComfyUI workflow builders.

The graphs are the part most likely to break silently: a wrong socket index, a
misnamed autogrow key or a dict where a bare combo key belongs produces a
ComfyUI error at request time, not at import. Everything asserted here was taken
from an authoritative source, and the comments say which — several encodings are
NOT discoverable from /object_info, so a name-only check passes while the request
still fails.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "media_server", Path(__file__).parent.parent / "inference" / "media_server.py"
)
media_server = importlib.util.module_from_spec(_SPEC)
sys.modules["media_server"] = media_server
_SPEC.loader.exec_module(media_server)

AVRequest = media_server.AVRequest
MusicRequest = media_server.MusicRequest
ImageRequest = media_server.ImageRequest


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def _uploads(**overrides) -> dict[str, list[str]]:
    base = {
        "first_frame": [], "last_frame": [],
        "ref_images": [], "ref_videos": [], "ref_video_audios": [], "ref_audios": [],
    }
    base.update(overrides)
    return base


def _av(req_kwargs=None, **upload_overrides):
    req = AVRequest(prompt="a lighthouse in a storm", **(req_kwargs or {}))
    graph, save_id = media_server._build_av_workflow(req, 1234, _uploads(**upload_overrides))
    return graph, save_id, req


def _music(**kwargs):
    req = MusicRequest(prompt="lofi hip hop, mellow piano", **kwargs)
    graph, save_id = media_server._build_music_workflow(req, 1234)
    return graph, save_id, req


def _image(refs=None, **kwargs):
    req = ImageRequest(prompt="a noir portrait", **kwargs)
    graph, save_id = media_server._build_image_workflow(req, 1234, refs or [])
    return graph, save_id, req


def _links(graph):
    """Every [node_id, socket] reference in the graph."""
    out = []
    for nid, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                out.append((nid, key, value[0], value[1]))
    return out


# ──────────────────────────────────────────────────────
#  Graph integrity, all modalities
# ──────────────────────────────────────────────────────

@pytest.mark.parametrize("builder", [
    lambda: _av(),
    lambda: _av(first_frame=["a.png"], last_frame=["b.png"]),
    lambda: _av(ref_images=["r.png"], ref_videos=["v.mp4"], ref_audios=["a.wav"]),
    lambda: _music(),
    lambda: _music(lyrics="la la la", duration=30.0),
    lambda: _image(),
    lambda: _image(variant="base"),
    lambda: _image(refs=["r1.png", "r2.png"]),
])
def test_every_link_points_at_a_real_node(builder):
    """A dangling reference is a ComfyUI 400 at request time — catch it here."""
    graph, save_id, _ = builder()

    for nid, key, target, socket in _links(graph):
        assert target in graph, f"{nid}.{key} -> unknown node '{target}'"
        assert isinstance(socket, int), f"{nid}.{key} socket must be an int"
    assert save_id in graph


@pytest.mark.parametrize("builder,save_type", [
    (lambda: _av(), "SaveVideo"),
    (lambda: _music(), "SaveAudioMP3"),
    (lambda: _image(), "SaveImage"),
])
def test_each_modality_ends_in_its_save_node(builder, save_type):
    graph, save_id, _ = builder()
    assert graph[save_id]["class_type"] == save_type


# ──────────────────────────────────────────────────────
#  av — MiniMax-H3
# ──────────────────────────────────────────────────────

def test_av_loaders_and_output_chain():
    graph, save_id, req = _av()

    assert graph["clip"]["inputs"]["type"] == "minimax"
    assert graph["clip"]["inputs"]["clip_name"] == media_server.H3_TEXT_ENCODER
    # int8-convrot is detected from the state dict; forcing fp8 would break it
    assert graph["unet"]["inputs"]["weight_dtype"] == "default"
    assert graph["sampler"]["inputs"]["model"] == ["shift", 0]

    # One nested AV latent, two decoders: video half then audio half
    assert graph["decode_video"]["inputs"] == {"samples": ["sampler", 0], "vae": ["vae_video", 0]}
    assert graph["decode_audio"]["inputs"] == {"samples": ["sampler", 0], "vae": ["vae_audio", 0]}
    assert graph["mux"]["inputs"]["fps"] == float(req.fps)
    # DYNAMICCOMBO takes the bare option key; a dict is silently dropped
    assert graph[save_id]["inputs"]["codec"] == "auto"


def test_av_conditioning_sockets():
    """MiniMaxH3ImageToVideo returns (positive, latent) — indexes 0 and 1."""
    graph, _, _ = _av()
    assert graph["sampler"]["inputs"]["positive"] == ["cond", 0]
    assert graph["sampler"]["inputs"]["latent_image"] == ["cond", 1]
    # H3 emits no negative; CFG needs one, zeroed from the positive
    assert graph["negative"]["class_type"] == "ConditioningZeroOut"


def test_av_defaults():
    graph, _, req = _av()
    s = graph["sampler"]["inputs"]
    assert (s["sampler_name"], s["scheduler"]) == ("res_multistep", "simple")
    assert (s["steps"], s["cfg"]) == (20, 1.0)
    assert graph["shift"]["inputs"]["shift_video"] == 12.0
    assert graph["shift"]["inputs"]["shift_audio"] == 3.0
    assert (req.width, req.height, req.num_frames, req.fps) == (1344, 768, 124, 24)


def test_av_mode_selects_checkpoint():
    assert _av()[0]["unet"]["inputs"]["unet_name"] == media_server.CKPT_FL2VA
    assert _av(first_frame=["a.png"])[0]["unet"]["inputs"]["unet_name"] == media_server.CKPT_FL2VA
    assert _av(ref_images=["r.png"])[0]["unet"]["inputs"]["unet_name"] == media_server.CKPT_REF2VA


def test_av_keyframes_wired():
    graph, _, _ = _av(first_frame=["a.png"], last_frame=["b.png"])
    assert graph["cond"]["inputs"]["first_frame"] == ["load_first_frame", 0]
    assert graph["cond"]["inputs"]["last_frame"] == ["load_last_frame", 0]

    graph, _, _ = _av(first_frame=["only.png"])
    assert "last_frame" not in graph["cond"]["inputs"]


def test_av_ref_autogrow_keys_are_dotted_and_zero_indexed():
    """H3 uses TemplatePrefix, so keys are <container>.<prefix><i> from 0.

    The bare "ref_image_0" form reaches execute() as an unexpected kwarg.
    """
    graph, _, _ = _av(ref_images=["r0.png", "r1.png"], ref_videos=["v.mp4"],
                      ref_video_audios=["s.wav"], ref_audios=["a.wav"])
    cond = graph["cond"]["inputs"]

    assert cond["ref_images.ref_image_0"] == ["load_ref_img_0", 0]
    assert cond["ref_images.ref_image_1"] == ["load_ref_img_1", 0]
    # GetVideoComponents → (images, audio, fps, bit_depth)
    assert cond["ref_videos.ref_video_0"] == ["split_ref_vid_0", 0]
    assert cond["ref_video_audios.ref_video_audio_0"] == ["load_ref_vid_audio_0", 0]
    assert cond["ref_audios.ref_audio_0"] == ["load_ref_audio_0", 0]
    assert cond["audio_vae"] == ["vae_audio", 0]


def test_av_ref_video_falls_back_to_its_own_soundtrack():
    graph, _, _ = _av(ref_videos=["clip.mp4"])
    assert graph["cond"]["inputs"]["ref_video_audios.ref_video_audio_0"] == ["split_ref_vid_0", 1]


# ──────────────────────────────────────────────────────
#  music — ACE-Step 1.5 XL Turbo
# ──────────────────────────────────────────────────────

def test_music_matches_the_official_blueprint():
    """Values from ComfyUI's "Text to Audio (ACE-Step 1.5)" blueprint."""
    graph, save_id, req = _music()

    # Two encoders, in this order, type "ace"
    clip = graph["clip"]["inputs"]
    assert graph["clip"]["class_type"] == "DualCLIPLoader"
    assert clip["clip_name1"] == media_server.ACE_CLIP_1
    assert clip["clip_name2"] == media_server.ACE_CLIP_2
    assert clip["type"] == "ace"

    assert graph["shift"]["class_type"] == "ModelSamplingAuraFlow"
    assert graph["shift"]["inputs"]["shift"] == 3.0

    s = graph["sampler"]["inputs"]
    assert (s["steps"], s["cfg"]) == (8, 1.0)          # XL Turbo: 8 steps, no CFG
    assert (s["sampler_name"], s["scheduler"]) == ("euler", "simple")
    assert s["model"] == ["shift", 0]
    assert s["latent_image"] == ["latent", 0]

    assert graph["decode"]["class_type"] == "VAEDecodeAudio"
    assert graph[save_id]["inputs"]["quality"] == req.mp3_quality


def test_music_conditioning_fields():
    graph, _, _ = _music(lyrics="hello", bpm=140, key_scale="E minor", time_signature="3")
    c = graph["cond"]["inputs"]

    assert graph["cond"]["class_type"] == "TextEncodeAceStepAudio1.5"
    assert c["tags"] == "lofi hip hop, mellow piano"   # prompt → tags
    assert c["lyrics"] == "hello"
    assert (c["bpm"], c["keyscale"], c["timesignature"]) == (140, "E minor", "3")
    assert c["clip"] == ["clip", 0]


def test_music_has_two_distinct_cfg_values():
    """The conditioning node's cfg_scale drives the audio-code LM; the sampler's
    cfg drives diffusion. The blueprint uses 2.0 and 1.0 respectively."""
    graph, _, _ = _music()
    assert graph["cond"]["inputs"]["cfg_scale"] == 2.0
    assert graph["sampler"]["inputs"]["cfg"] == 1.0


def test_music_duration_reaches_latent_and_conditioning():
    graph, _, _ = _music(duration=45.0)
    assert graph["latent"]["inputs"]["seconds"] == 45.0
    assert graph["cond"]["inputs"]["duration"] == 45.0


# ──────────────────────────────────────────────────────
#  image — HiDream-O1
# ──────────────────────────────────────────────────────

def test_image_uses_custom_sampling_not_ksampler():
    """HiDream-O1 needs ModelNoiseScale + SamplerCustom + BasicScheduler.

    A plain KSampler graph does not reproduce the official templates.
    """
    graph, _, _ = _image()

    assert "KSampler" not in {n["class_type"] for n in graph.values()}
    assert graph["sample"]["class_type"] == "SamplerCustom"
    assert graph["noise"]["class_type"] == "ModelNoiseScale"
    assert graph["sigmas"]["class_type"] == "BasicScheduler"
    # All-in-one checkpoint: MODEL 0, CLIP 1, VAE 2
    assert graph["ckpt"]["class_type"] == "CheckpointLoaderSimple"
    assert graph["positive"]["inputs"]["clip"] == ["ckpt", 1]
    assert graph["decode"]["inputs"]["vae"] == ["ckpt", 2]
    # SamplerCustom returns (output, denoised_output); templates take output
    assert graph["decode"]["inputs"]["samples"] == ["sample", 0]


def test_image_dev_variant_defaults():
    """From image_hidream_o1_dev.json: LCM, 28 steps, cfg 1, noise scale 7.6."""
    graph, _, _ = _image(variant="dev")

    assert graph["ckpt"]["inputs"]["ckpt_name"] == media_server.IMAGE_CKPT_DEV
    assert graph["sampler_sel"]["class_type"] == "SamplerLCM"
    assert graph["sampler_sel"]["inputs"]["noise_clip_std"] == 2.5
    assert graph["sigmas"]["inputs"]["steps"] == 28
    assert graph["sample"]["inputs"]["cfg"] == 1.0
    assert graph["noise"]["inputs"]["noise_scale"] == 7.6
    # dev has no seam smoothing
    assert "seam" not in graph
    assert graph["sample"]["inputs"]["model"] == ["noise", 0]


def test_image_base_variant_defaults():
    """From image_hidream_o1.json: dpmpp_2m_sde_gpu, 40 steps, cfg 5, scale 8."""
    graph, _, _ = _image(variant="base")

    assert graph["ckpt"]["inputs"]["ckpt_name"] == media_server.IMAGE_CKPT_BASE
    assert graph["sampler_sel"]["class_type"] == "KSamplerSelect"
    assert graph["sampler_sel"]["inputs"]["sampler_name"] == "dpmpp_2m_sde_gpu"
    assert graph["sigmas"]["inputs"]["steps"] == 40
    assert graph["sample"]["inputs"]["cfg"] == 5.0
    assert graph["noise"]["inputs"]["noise_scale"] == 8.0
    # base adds the seam-smoothing patch, and sampling runs off it
    assert graph["seam"]["class_type"] == "HiDreamO1PatchSeamSmoothing"
    assert graph["sample"]["inputs"]["model"] == ["seam", 0]


def test_image_native_canvas_and_overrides():
    _, _, req = _image()
    assert (req.width, req.height) == (2048, 2048)

    graph, _, _ = _image(steps=12, cfg_scale=3.0, noise_scale=5.0, width=1024, height=1536)
    assert graph["sigmas"]["inputs"]["steps"] == 12
    assert graph["sample"]["inputs"]["cfg"] == 3.0
    assert graph["noise"]["inputs"]["noise_scale"] == 5.0
    assert graph["latent"]["inputs"]["width"] == 1024
    assert graph["latent"]["inputs"]["height"] == 1536


def test_image_reference_autogrow_keys_are_one_indexed():
    """HiDreamO1ReferenceImages uses TemplateNames image_1..image_10 — base 1,
    unlike H3's zero-based TemplatePrefix."""
    graph, _, _ = _image(refs=["a.png", "b.png"])
    refs = graph["refs"]["inputs"]

    assert graph["refs"]["class_type"] == "HiDreamO1ReferenceImages"
    assert refs["images.image_1"] == ["load_ref_1", 0]
    assert refs["images.image_2"] == ["load_ref_2", 0]
    assert "images.image_0" not in refs
    # Both conditionings are rewritten, so sampling must read from the ref node
    assert graph["sample"]["inputs"]["positive"] == ["refs", 0]
    assert graph["sample"]["inputs"]["negative"] == ["refs", 1]


def test_image_without_references_skips_the_node():
    graph, _, _ = _image()
    assert "refs" not in graph
    assert graph["sample"]["inputs"]["positive"] == ["positive", 0]
    assert graph["sample"]["inputs"]["negative"] == ["negative", 0]


# ──────────────────────────────────────────────────────
#  Retention
# ──────────────────────────────────────────────────────

def _age(path: Path, hours: float) -> None:
    """Backdates a file's mtime by *hours*."""
    import os
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_sweep_removes_only_old_files(tmp_path):
    fresh = tmp_path / "fresh.mp4"
    stale = tmp_path / "stale.mp4"
    fresh.write_bytes(b"x" * 100)
    stale.write_bytes(b"y" * 500)
    _age(stale, 48)

    files, freed = media_server.sweep_dir(tmp_path, max_age_s=24 * 3600)

    assert (files, freed) == (1, 500)
    assert fresh.exists()
    assert not stale.exists()


def test_sweep_is_recursive_but_keeps_directories(tmp_path):
    sub = tmp_path / "media"
    sub.mkdir()
    old = sub / "old.png"
    old.write_bytes(b"z" * 10)
    _age(old, 30)

    files, _ = media_server.sweep_dir(tmp_path, max_age_s=3600)

    assert files == 1
    assert not old.exists()
    # ComfyUI creates and expects some of these to exist
    assert sub.is_dir()


def test_sweep_respects_a_pattern(tmp_path):
    ours = tmp_path / "in_image_abc.png"
    theirs = tmp_path / "example.png"
    for f in (ours, theirs):
        f.write_bytes(b"q")
        _age(f, 99)

    files, _ = media_server.sweep_dir(tmp_path, max_age_s=3600, pattern="in_*")

    assert files == 1
    assert not ours.exists()
    # ComfyUI ships bundled inputs its own templates reference
    assert theirs.exists()


def test_sweep_disabled_when_age_is_zero(tmp_path):
    f = tmp_path / "ancient.mp4"
    f.write_bytes(b"k")
    _age(f, 10_000)

    assert media_server.sweep_dir(tmp_path, max_age_s=0) == (0, 0)
    assert f.exists()


def test_sweep_tolerates_a_missing_directory(tmp_path):
    assert media_server.sweep_dir(tmp_path / "nope", max_age_s=3600) == (0, 0)


def test_cleanup_once_targets_outputs_and_own_uploads(tmp_path, monkeypatch):
    """The sweep must not reach outside output/media and input/in_*."""
    (tmp_path / "output" / "media").mkdir(parents=True)
    (tmp_path / "input").mkdir()

    generated = tmp_path / "output" / "media" / "av_00001_.mp4"
    marker = tmp_path / "output" / "_output_images_will_be_put_here"
    uploaded = tmp_path / "input" / "in_image_deadbeef.png"
    bundled = tmp_path / "input" / "example.png"
    for f in (generated, marker, uploaded, bundled):
        f.write_bytes(b"d" * 4)
        _age(f, 72)

    monkeypatch.setattr(media_server, "COMFY_ROOT", str(tmp_path))
    monkeypatch.setattr(media_server, "RETENTION_HOURS", 24.0)

    files, _ = media_server.cleanup_once()

    assert files == 2
    assert not generated.exists() and not uploaded.exists()
    # Neither ComfyUI's output marker nor its bundled inputs
    assert marker.exists() and bundled.exists()


# ──────────────────────────────────────────────────────
#  Request limits
# ──────────────────────────────────────────────────────

def test_token_estimate_matches_comfyuis_grid():
    """T = ((frames-5)/17)*5+2 after snapping frames up to 17k+5."""
    # 124 frames -> T=37, 1344/16=84, 768/16=48
    assert media_server.av_latent_tokens(1344, 768, 124) == 37 * 48 * 84
    assert media_server.AV_DEFAULT_TOKENS == 149_184
    # The request that wedged the node for 9 hours
    assert media_server.av_latent_tokens(1344, 768, 999) > 1_000_000


def test_frame_count_capped_to_the_trained_range():
    """999 frames is ~8x the tokens of the default and outside what H3 was
    trained on, so it burns hours and returns junk. Reject it up front."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AVRequest(prompt="x", num_frames=999)
    with pytest.raises(ValidationError):
        AVRequest(prompt="x", num_frames=4)
    # The documented ceiling is still allowed
    assert AVRequest(prompt="x", num_frames=media_server.AV_MAX_FRAMES).num_frames == 362


def test_canvas_capped_to_h3_pixels():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="pixels"):
        AVRequest(prompt="x", width=1920, height=1080)
    # The native canvas passes, in either orientation
    assert AVRequest(prompt="x", width=1344, height=768)
    assert AVRequest(prompt="x", width=768, height=1344)


def test_other_av_bounds():
    from pydantic import ValidationError

    for kwargs in ({"steps": 0}, {"steps": 500}, {"fps": 0}, {"width": 16}):
        with pytest.raises(ValidationError):
            AVRequest(prompt="x", **kwargs)


def test_music_and_image_bounds():
    from pydantic import ValidationError

    for kwargs in ({"duration": 0}, {"duration": 5000}, {"steps": 0}, {"bpm": 5}):
        with pytest.raises(ValidationError):
            MusicRequest(prompt="x", **kwargs)

    for kwargs in ({"width": 8192}, {"batch_size": 0}, {"batch_size": 99}, {"steps": 0}):
        with pytest.raises(ValidationError):
            ImageRequest(prompt="x", **kwargs)


# ──────────────────────────────────────────────────────
#  Generation deadline
#
#  A 15s clip at full canvas runs ~55 min in one pass, so the old shared 1800s
#  value cut off exactly the jobs it was meant to protect — after the GPU had
#  already done the work. These pin the two halves of the fix: no deadline by
#  default, and a per-call cap on ComfyUI that is a separate knob.
# ──────────────────────────────────────────────────────

async def test_no_generation_deadline_by_default(monkeypatch):
    """With GENERATE_TIMEOUT_S=0 the poll loop must run, not fall straight through.

    Regression: `deadline = time.time() + 0` made `while time.time() < deadline`
    false on the first evaluation, so a 0 meant "give up at once" — a 504 before
    a single poll — rather than "wait as long as it takes".
    """
    monkeypatch.setattr(media_server, "GENERATE_TIMEOUT_S", 0)
    monkeypatch.setattr(media_server, "POLL_INTERVAL_S", 0)

    calls = {"n": 0}

    async def fake_get(path, **kwargs):
        calls["n"] += 1
        if calls["n"] < 4:
            return {}                       # still running
        return {"p1": {"outputs": {"save": {"images": [{"filename": "a.png"}]}}}}

    monkeypatch.setattr(media_server, "_comfy_get", fake_get)

    outputs = await media_server._await_result("p1")
    assert outputs == {"save": {"images": [{"filename": "a.png"}]}}
    assert calls["n"] == 4, "should have kept polling instead of timing out"


async def test_deadline_still_enforced_when_configured(monkeypatch):
    """Removing the default must not remove the ability to set one."""
    from fastapi import HTTPException

    monkeypatch.setattr(media_server, "GENERATE_TIMEOUT_S", 1)
    monkeypatch.setattr(media_server, "POLL_INTERVAL_S", 0)

    async def never_finishes(path, **kwargs):
        return {}

    monkeypatch.setattr(media_server, "_comfy_get", never_finishes)

    with pytest.raises(HTTPException) as exc:
        await media_server._await_result("p1")
    assert exc.value.status_code == 504
    assert "GENERATE_TIMEOUT_S" in str(exc.value.detail)


def test_comfy_call_cap_is_a_separate_knob():
    """The per-call cap must stay finite even with no generation deadline.

    Conflating the two is what made "no generation timeout" inexpressible: the
    same value bounded a 55-minute job and a 20ms /history poll.
    """
    assert media_server.GENERATE_TIMEOUT_S == 0
    assert media_server.COMFY_HTTP_TIMEOUT_S > 0


# ──────────────────────────────────────────────────────
#  HiDream-O1 variant availability
# ──────────────────────────────────────────────────────

def _object_info(checkpoints: list[str]) -> dict:
    """A minimal /object_info: every required node, and the loaders' combo lists."""
    info: dict = {n: {} for nodes in media_server.MODALITY_NODES.values() for n in nodes}
    loaders = {
        "UNETLoader": ("unet_name", [media_server.CKPT_FL2VA, media_server.ACE_DIT]),
        "CLIPLoader": ("clip_name", [media_server.H3_TEXT_ENCODER,
                                     media_server.ACE_CLIP_1, media_server.ACE_CLIP_2]),
        "VAELoader": ("vae_name", [media_server.H3_VIDEO_VAE, media_server.H3_AUDIO_VAE,
                                   media_server.ACE_VAE]),
        "CheckpointLoaderSimple": ("ckpt_name", checkpoints),
    }
    for node, (field, files) in loaders.items():
        info[node] = {"input": {"required": {field: [files]}}}
    return info


async def _probe_with(monkeypatch, checkpoints: list[str]):
    async def fake_get(path, **kwargs):
        return _object_info(checkpoints)

    monkeypatch.setattr(media_server, "_comfy_get", fake_get)
    await media_server._probe_comfy()


async def test_one_image_variant_is_enough_for_the_modality(monkeypatch):
    """The download script invites fetching only one checkpoint; that must work."""
    await _probe_with(monkeypatch, [media_server.IMAGE_CKPT_DEV])

    assert media_server._capabilities["image"] is True
    assert media_server._image_variants == {"dev": True, "base": False}


async def test_missing_variant_is_rejected_before_reaching_comfyui(monkeypatch):
    """Regression: only the dev checkpoint was checked, so /health advertised
    image as available and a variant="base" request died inside ComfyUI as an
    opaque 502 — the exact class of late failure the health probe exists to
    prevent."""
    from fastapi import HTTPException

    await _probe_with(monkeypatch, [media_server.IMAGE_CKPT_DEV])

    media_server._require_image_variant("dev")          # present — no raise

    with pytest.raises(HTTPException) as exc:
        media_server._require_image_variant("base")
    assert exc.value.status_code == 503
    assert media_server.IMAGE_CKPT_BASE in str(exc.value.detail)
    assert "dev" in str(exc.value.detail), "should name what IS available"


async def test_image_unavailable_when_no_checkpoint_is_present(monkeypatch):
    await _probe_with(monkeypatch, [])

    assert media_server._capabilities["image"] is False
    assert media_server._image_variants == {"dev": False, "base": False}
    # av and music are unaffected — their weights are still there.
    assert media_server._capabilities["av"] is True
    assert media_server._capabilities["music"] is True


# ──────────────────────────────────────────────────────
#  ref2va reference limits
#
#  The per-container maxima come from the Autogrow templates in
#  comfy_extras/nodes_minimax_h3.py: ref_images max 9, and 3 each for
#  ref_videos, ref_video_audios and ref_audios. Validating here only makes
#  ComfyUI's rejection legible — except for the pairing rule, which ComfyUI
#  does not reject at all, it just ignores the extras.
# ──────────────────────────────────────────────────────

@pytest.fixture
def av_client(monkeypatch):
    """A client with the av modality available and the GPU path stubbed out.

    All the limit checks run before the first upload, so nothing below reaches
    the stubs — they exist so a request that *passes* validation doesn't try to
    talk to ComfyUI.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setitem(media_server._capabilities, "av", True)

    async def fake_upload(payload_b64, kind):
        return f"in_{kind}_stub.bin"

    async def fake_run(graph, extensions, request=None):
        return [{"filename": "h3_00001_.mp4"}], [b"fake-mp4"], 1.0

    monkeypatch.setattr(media_server, "_upload_asset", fake_upload)
    monkeypatch.setattr(media_server, "_run", fake_run)
    return TestClient(media_server.app)


def _av_body(**kwargs):
    return {"prompt": "a lighthouse", "num_frames": 124, **kwargs}


def test_ref_video_audios_capped_at_three(av_client):
    """Regression: this container was the only one left unbounded, so a caller
    could send any number. The node's Autogrow template maxes it at 3, so
    ComfyUI would have rejected the graph after the uploads were already done."""
    resp = av_client.post("/generate", json=_av_body(
        ref_videos=["v1", "v2", "v3"],
        ref_video_audios=["a1", "a2", "a3", "a4"],
    ))
    assert resp.status_code == 400
    assert "3 reference video soundtracks" in resp.json()["detail"]


def test_ref_video_audios_must_pair_with_a_video(av_client):
    """ref_video_audio_N is the soundtrack *of* ref_video_N — the node pairs them
    by index, so an extra with no matching video is silently dropped. That is the
    one limit ComfyUI will not complain about, which makes it the one worth
    catching: the caller pays the upload and never learns it was ignored."""
    resp = av_client.post("/generate", json=_av_body(
        ref_videos=["v1"],
        ref_video_audios=["a1", "a2"],
    ))
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "2 ref_video_audios for 1 ref_videos" in detail
    assert "ref_audios" in detail, "should point at the right field for standalone audio"


def test_soundtracks_count_toward_the_total(av_client):
    """A supplied soundtrack is a file the caller sent, so it counts. 9 images +
    3 videos + 1 soundtrack = 13, which the previous sum scored as 12."""
    resp = av_client.post("/generate", json=_av_body(
        ref_images=[f"i{n}" for n in range(9)],
        ref_videos=["v1", "v2", "v3"],
        ref_video_audios=["a1"],
    ))
    assert resp.status_code == 400
    assert "at most 12 reference files, got 13" in resp.json()["detail"]


def test_paired_soundtracks_within_the_total_are_accepted(av_client):
    """The bounds must not reject a legal request: 6 images + 3 videos + 3 of
    their soundtracks = 12 files, all four containers within their maxima."""
    resp = av_client.post("/generate", json=_av_body(
        ref_images=[f"i{n}" for n in range(6)],
        ref_videos=["v1", "v2", "v3"],
        ref_video_audios=["a1", "a2", "a3"],
    ))
    assert resp.status_code == 200, resp.text
    assert resp.json()["mode"] == "ref2va"


def test_per_container_maxima_match_the_node_schema(av_client):
    """Pinned against the Autogrow templates: 9 / 3 / 3 / 3."""
    for field, limit in (("ref_images", 9), ("ref_videos", 3), ("ref_audios", 3)):
        over = av_client.post("/generate", json=_av_body(
            **{field: [f"x{n}" for n in range(limit + 1)]}))
        assert over.status_code == 400, f"{field} past {limit} should be rejected"
