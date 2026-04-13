"""
Patches vllm 0.17.1 (NVIDIA 26.03 image) to add Gemma 4 support
by registering the model files copied from vllm 0.19.0.
"""
import os
import re

VLLM = os.environ["VLLM"]


def patch_after(path, marker, insertion, unique_check, label):
    """Insert `insertion` immediately after `marker`."""
    with open(path) as f:
        content = f.read()
    if unique_check in content:
        print(f"{label}: already patched")
        return
    if marker not in content:
        raise RuntimeError(f"{label}: marker not found in {path!r}:\n  {marker!r}")
    content = content.replace(marker, marker + insertion, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"{label}: patched OK")


def patch_before(path, marker, insertion, unique_check, label):
    """Insert `insertion` immediately before `marker`."""
    with open(path) as f:
        content = f.read()
    if unique_check in content:
        print(f"{label}: already patched")
        return
    if marker not in content:
        raise RuntimeError(f"{label}: marker not found in {path!r}:\n  {marker!r}")
    content = content.replace(marker, insertion + marker, 1)
    with open(path, "w") as f:
        f.write(content)
    print(f"{label}: patched OK")


# ── 4a. Model registry — text models dict (_MODELS) ─────────────────────
patch_after(
    path=f"{VLLM}/model_executor/models/registry.py",
    marker='"Gemma3ForCausalLM": ("gemma3", "Gemma3ForCausalLM"),',
    insertion='\n    "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),',
    unique_check='"Gemma4ForCausalLM"',
    label="registry.py (_MODELS)",
)

# ── 4b. Model registry — multimodal dict (_MULTIMODAL_MODELS) ────────────
patch_after(
    path=f"{VLLM}/model_executor/models/registry.py",
    marker='"Gemma3ForConditionalGeneration": ("gemma3_mm", "Gemma3ForConditionalGeneration"),  # noqa: E501',
    insertion='\n    "Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),  # noqa: E501',
    unique_check='"Gemma4ForConditionalGeneration"',
    label="registry.py (_MULTIMODAL_MODELS)",
)

# ── 5. Rotary embedding __init__.py ───────────────────────────────────────
patch_after(
    path=f"{VLLM}/model_executor/layers/rotary_embedding/__init__.py",
    marker="from .base import RotaryEmbedding",
    insertion="\nfrom .gemma4_rope import Gemma4RotaryEmbedding",
    unique_check="gemma4_rope",
    label="rotary_embedding/__init__.py",
)

# ── 5b. Rotary embedding get_rope() — add "proportional" scaling type ─────
patch_before(
    path=f"{VLLM}/model_executor/layers/rotary_embedding/__init__.py",
    marker='    else:\n        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")',
    insertion=(
        '    elif scaling_type == "proportional":\n'
        '        rotary_emb = Gemma4RotaryEmbedding(\n'
        '            head_size,\n'
        '            rotary_dim,\n'
        '            max_position,\n'
        '            base,\n'
        '            is_neox_style,\n'
        '            dtype,\n'
        '        )\n'
    ),
    unique_check='"proportional"',
    label="rotary_embedding/__init__.py (get_rope proportional)",
)

# ── 6. tool_parsers/__init__.py ───────────────────────────────────────────
patch_before(
    path=f"{VLLM}/tool_parsers/__init__.py",
    marker='"hermes": (',
    insertion=(
        '"gemma4": (\n'
        '        "gemma4_tool_parser",\n'
        '        "Gemma4ToolParser",\n'
        '    ),\n'
        '    '
    ),
    unique_check='"gemma4"',
    label="tool_parsers/__init__.py",
)

# ── 6b. Patch gemma4_tool_parser.py for vLLM 0.17.1 compatibility ─────────
# The parser was copied from 0.19.0 which imports Tool from abstract_tool_parser
# and expects `tools` in __init__.  In 0.17.1 neither exists.
tool_parser_path = f"{VLLM}/tool_parsers/gemma4_tool_parser.py"
with open(tool_parser_path) as f:
    tp_content = f.read()

if "# patched-for-017" not in tp_content:
    # Fix import: remove Tool from the import
    tp_content = tp_content.replace(
        "from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser",
        "from vllm.tool_parsers.abstract_tool_parser import ToolParser  # patched-for-017",
    )
    # Fix __init__ signature: remove tools param and super() call
    tp_content = tp_content.replace(
        "def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):\n"
        "        super().__init__(tokenizer, tools)",
        "def __init__(self, tokenizer: TokenizerLike, **kwargs):\n"
        "        super().__init__(tokenizer)",
    )
    with open(tool_parser_path, "w") as f:
        f.write(tp_content)
    print("gemma4_tool_parser.py: patched for 0.17.1 compatibility")
else:
    print("gemma4_tool_parser.py: already patched for 0.17.1")

# ── 7. reasoning/__init__.py ──────────────────────────────────────────────
patch_before(
    path=f"{VLLM}/reasoning/__init__.py",
    marker='"deepseek_r1": (',
    insertion=(
        '"gemma4": (\n'
        '        "gemma4_reasoning_parser",\n'
        '        "Gemma4ReasoningParser",\n'
        '    ),\n'
        '    '
    ),
    unique_check='"gemma4"',
    label="reasoning/__init__.py",
)

# ── 8. model_arch_config_convertor.py ─────────────────────────────────────
src_path  = "/tmp/vllm_src/vllm/transformers_utils/model_arch_config_convertor.py"
dest_path = f"{VLLM}/transformers_utils/model_arch_config_convertor.py"

with open(src_path) as f:
    src_lines = f.read().split("\n")
with open(dest_path) as f:
    dest_content = f.read()

if "Gemma4ModelArchConfigConvertor" in dest_content:
    print("model_arch_config_convertor.py: already patched")
else:
    # Extract Gemma4ModelArchConfigConvertor class from 0.19.0 source.
    # Stop at the next top-level non-blank, non-comment, non-class line
    # (the MODEL_ARCH_CONFIG_CONVERTORS dict definition).
    start = next(i for i, l in enumerate(src_lines) if "class Gemma4ModelArchConfigConvertor" in l)
    end = start + 1
    while end < len(src_lines):
        l = src_lines[end]
        stripped = l.strip()
        if stripped and not l[0].isspace() and not stripped.startswith("#") and not stripped.startswith("class "):
            break
        end += 1
    gemma4_class = "\n".join(src_lines[start:end]).rstrip()

    # Append class before the convertor dict
    dest_content = dest_content.rstrip("\n") + f"\n\n{gemma4_class}\n"

    # Register in the dict after the last Gemma3 entry
    dest_content = dest_content.replace(
        '"gemma3n": Gemma3nModelArchConfigConvertor,',
        (
            '"gemma3n": Gemma3nModelArchConfigConvertor,\n'
            '    "gemma4": Gemma4ModelArchConfigConvertor,\n'
            '    "gemma4_text": Gemma4ModelArchConfigConvertor,'
        ),
    )
    with open(dest_path, "w") as f:
        f.write(dest_content)
    print("model_arch_config_convertor.py: patched OK")
