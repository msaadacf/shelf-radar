from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from json_repair import repair_json
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


@dataclass(frozen=True)
class BookCandidate:
    title: str
    author: Optional[str]
    confidence: float

# Here I am specifying exact instructions to VLM regarding output so as to decrease later cleaning work. I found this more useful to do.
INSTR_TEMPLATE = """You are extracting book spine text from a bookshelf photo.
Return ONLY valid JSON with this schema:
{{
  "books": [
    {{"title": "...", "author": null or "...", "confidence": 0.0-1.0}}
  ]
}}
Rules:
- Prefer full titles; omit words you are unsure about.
- If author is not visible, set author to null.
- Output max {max_items} items.
"""


def _parse_model_json(raw: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        fixed = repair_json(blob)
        return json.loads(fixed)


def load_qwen2_vl(model_id: str):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    return model, processor


@torch.inference_mode()
def extract_candidates(
    image_path: str,
    model,
    processor,
    max_items: int = 30,
    max_new_tokens: int = 600,
) -> List[BookCandidate]:
    img = Image.open(image_path).convert("RGB")
    instr = INSTR_TEMPLATE.format(max_items=max_items)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": instr},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    data = _parse_model_json(raw)
    books = data.get("books", []) or []

    out: List[BookCandidate] = []
    for b in books:
        title = (b.get("title") or "").strip()
        if len(title) < 3:
            continue
        out.append(
            BookCandidate(
                title=title,
                author=(b.get("author") or None),
                confidence=float(b.get("confidence") or 0.0),
            )
        )

    out.sort(key=lambda x: x.confidence, reverse=True)
    return out[:max_items]
