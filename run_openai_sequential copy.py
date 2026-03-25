#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequential LFW experiment runner (non-Batch, image data URLs) — argparse version.

Requirements:
  pip install openai pandas pillow tqdm

ENV:
  export OPENAI_API_KEY=...

CSV format:
  columns: image1, image2[, pair_id]
  image1/image2 may be relative to DATASET_DIR or absolute paths.

Example:
  python run_seq.py \
    --csv .data/LFW/lfw_short_genuine.csv \
    --dataset-dir /mnt/scratch/.../LFW/deepfunneled \
    --system text_generators/prompts/.../prompt_system.txt \
    --user   text_generators/prompts/.../prompt_user_match.txt \
    --out .data/results/lfw_sequential_results_genuine.jsonl \
    --model gpt-4o-mini \
    --temp 0.7 --max-tokens 2000 --retries 4 \
    --img-max-side 160 --img-quality 80 --img-format JPEG
"""

from __future__ import annotations
import os
import io
import json
import time
import base64
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from openai import OpenAI


# --------------------------
# Helpers
# --------------------------

def sha256_text(s: str) -> str:
    """_summary_

    Args:
        s (str): _description_

    Returns:
        str: _description_
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_text(p: str) -> str:
    """_summary_

    Args:
        p (str): _description_

    Returns:
        str: _description_
    """
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def compress_to_data_url(path: str, *, fmt: str, max_side: int, quality: int) -> str:
    """
    Load image, downscale so longest side = max_side, strip metadata,
    encode as compact base64 data URL.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side) if max_side > 0 else 1.0
    if scale > 1.0:
        new_w, new_h = int(round(w / scale)), int(round(h / scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    bio = io.BytesIO()
    if fmt.upper() == "WEBP":
        img.save(bio, format="WEBP", quality=quality, method=6)
        mime = "image/webp"
    else:
        img.save(bio, format="JPEG", quality=quality, optimize=True, progressive=True)
        mime = "image/jpeg"
    b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"

from typing import List, Dict, Any


def build_messages_gpt52(
    system_prompt: str,
    user_prompt: str,
    data_url1: str,
    data_url2: str,
) -> List[Dict[str, Any]]:
    """
    Build input messages for GPT-5.2 Responses API with system prompt,
    user prompt, and two images.
    """

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt,
                },
                {
                    "type": "input_image",
                    "image_url": data_url1,
                },
                {
                    "type": "input_image",
                    "image_url": data_url2,
                },
            ],
        },
    ]

    
def build_messages(system_prompt: str, user_prompt: str, data_url1: str, data_url2: str) -> List[Dict[str, Any]]:
    """
    Build a sequence of chat messages containing a system prompt and a user message with text and two image URLs.
    Parameters
    ----------
    system_prompt : str
        Text that sets the assistant's behavior or context (system message).
    user_prompt : str
        The textual content of the user's message.
    data_url1 : str
        URL of the first image to include in the user's message.
    data_url2 : str
        URL of the second image to include in the user's message.
    Returns
    -------
    List[Dict[str, Any]]

    """

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url1}},
                {"type": "image_url", "image_url": {"url": data_url2}},
            ],
        },
    ]


def extract_text_finish_usage(resp) -> tuple[str, str, Optional[dict]]:
    """
    Robust extraction across SDK versions.
    """
    choice = resp.choices[0]
    if hasattr(choice, "message") and hasattr(choice.message, "content"):
        text = choice.message.content
    else:
        text = choice["message"]["content"]
    finish = getattr(choice, "finish_reason", None) or choice.get("finish_reason")
    usage = getattr(resp, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    return text, (str(finish) if finish else "stop"), usage


def to_responses_input(messages):
    """
    Convert Chat Completions-style messages -> Responses API input.
    - "text"      -> {"type":"input_text","text":...}
    - "image_url" -> {"type":"input_image","image_url":"<URL or data:...>"}
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", [])
        parts = []

        if isinstance(content, str):
            parts.append({"type": "input_text", "text": content})
        else:
            for part in content:
                t = part.get("type")
                if t == "text":
                    parts.append({"type": "input_text", "text": part.get("text", "")})
                elif t == "image_url":
                    # Accept either {"image_url":{"url":"..."}} or {"image_url":"..."}
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url")  # <- Chat format
                    else:
                        url = iu            # <- already a string
                    if not isinstance(url, str) or not url:
                        # skip malformed image parts instead of sending an object
                        continue
                    parts.append({"type": "input_image", "image_url": url})
                # silently ignore any other part types

        out.append({"role": role, "content": parts})
    return out

# --------------------------
# Core runner
# --------------------------


def run_sequential(
    csv_path: str,
    dataset_dir: str,
    system_path: str,
    user_path: str,
    out_jsonl: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_segments: int,
    retries: int,
    img_max_side: int,
    img_quality: int,
    img_format: str,
    resume: bool,
    start_index: int,
    limit: Optional[int],
) -> None:
    """Runs the sequential explanation generation from OpenAI experiment."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = load_text(system_path)
    user_prompt = load_text(user_path)
    sys_hash = sha256_text(system_prompt)
    usr_hash = sha256_text(user_prompt)

    df = pd.read_csv(csv_path)
    # if "pair_id" not in df.columns:
    #     df["pair_id"] = [f"pair_{i:06d}" for i in range(len(df))]

    # normalize absolute paths
    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(dataset_dir, p)

    df["image1"] = df["image1"].apply(_abs)
    df["image2"] = df["image2"].apply(_abs)

    # slicing controls
    if start_index > 0:
        df = df.iloc[start_index:]
    if limit is not None:
        df = df.iloc[:limit]

    # resume support
    done = set()
    out_dir = os.path.dirname(out_jsonl) or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if resume and os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    pid = obj.get("pair_id")
                    if pid:
                        done.add(pid)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse line in existing output: {ln[:100]}...")

    with open(out_jsonl, "a", encoding="utf-8") as out_fp:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing pairs sequentially"):
            pair_id = row["pair_id"]
            if resume and pair_id in done:
                continue

            img1_path, img2_path = row["image1"], row["image2"]

            # prepare data URLs (compressed)
            try:
                url1 = compress_to_data_url(img1_path, fmt=img_format, max_side=img_max_side, quality=img_quality)
                url2 = compress_to_data_url(img2_path, fmt=img_format, max_side=img_max_side, quality=img_quality)
            except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as e:
                rec = {
                    "pair_id": pair_id,
                    "image1": img1_path, "image2": img2_path,
                    "model": model,
                    "request_ts": datetime.utcnow().isoformat() + "Z",
                    "system_sha256": sys_hash, "user_sha256": usr_hash,
                    "response": None,
                    "error": f"image_encode_error: {type(e).__name__}: {e}",
                }
                out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fp.flush()
                continue

            # core request with retries + auto-continue on truncation
            for attempt in range(1, retries + 1):
                try:
                    messages = build_messages(system_prompt, user_prompt, url1, url2)

                    full_text = ""
                    segments = 0
                    finish = None
                    usages = []

                    while segments < max_segments:
                        if str(model).lower().startswith("gpt-5"):
                            # ✅ GPT-5 uses the Responses API and Responses content types
                            r = client.responses.create(
                                model=model,
                                input=to_responses_input(messages),
                                temperature=1,
                                max_output_tokens=max_tokens,   # note: Responses uses max_output_tokens
                            )
                            print(r)
                            # print(r.output_text[:200])  # optional peek
                            # Extract
                            text = getattr(r, "output_text", "") or ""
                            finish = getattr(r, "finish_reason", None) or "stop"
                            usage = getattr(r, "usage", None)
                            if usage is not None and hasattr(usage, "model_dump"):
                                usage = usage.model_dump()
                        else:
                            # Other models via Chat Completions
                            resp = client.chat.completions.create(
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                messages=messages,
                            )
                            text, finish, usage = extract_text_finish_usage(resp)

                        # print(text[:200])  # optional peek
                        full_text += (text or "")
                        if usage:
                            usages.append(usage)

                        if str(finish).lower() != "length":
                            break

                        # else continue with a short rolling window to avoid prompt bloat
                        messages = messages + [
                            {"role": "assistant", "content": full_text[-2000:]},
                            {"role": "user", "content": [{"type": "text", "text": "Continue."}]},
                        ]
                        segments += 1

                    rec = {
                        "pair_id": pair_id,
                        "image1": img1_path, "image2": img2_path,
                        "model": model,
                        "request_ts": datetime.utcnow().isoformat() + "Z",
                        "system_sha256": sys_hash, "user_sha256": usr_hash,
                        "response": {
                            "text": full_text,
                            "finish_reason": finish,
                            "segments_used": segments + 1,
                            "usage": (usages[-1] if usages else None),
                        },
                        "error": None,
                    }
                    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fp.flush()
                    break  # success → stop retry loop

                except Exception as e:
                    if attempt == retries:
                        rec = {
                            "pair_id": pair_id,
                            "image1": img1_path, "image2": img2_path,
                            "model": model,
                            "request_ts": datetime.utcnow().isoformat() + "Z",
                            "system_sha256": sys_hash, "user_sha256": usr_hash,
                            "response": None,
                            "error": str(e),
                        }
                        out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_fp.flush()
                        break
                    time.sleep(1.5 * attempt)  # backoff


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential LFW experiment runner (non-Batch, image data URLs).")

    # Paths
    p.add_argument("--csv", dest="csv_path", required=False, default=None, help="Path to CSV (image1,image2[,pair_id]).")
    p.add_argument("--dataset-name", type=str, help="Dataset name", default="lfw")
    p.add_argument("--dataset-dir", dest="dataset_dir", required=False, default=None, help="Root directory of images.")
    p.add_argument("--system_path", dest="system_path", required=False, help="Path to system prompt file.")
    p.add_argument("--user_path", dest="user_path", required=False, help="Path to user prompt file.")
    p.add_argument("--out", dest="out_jsonl", required=False, help="Output JSONL to append results.")

    # Model / generation
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Vision model (default: env OPENAI_MODEL or gpt-4o-mini).")
    p.add_argument("--temp", dest="temperature", type=float, default=0.7, help="Sampling temperature.")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=2000, help="Max completion tokens per segment.")
    p.add_argument("--max-segments", dest="max_segments", type=int, default=3, help="Auto-continue segments on truncation.")
    p.add_argument("--subset", type=str, help="Subset name for output file naming", choices=['genuine', 'impostor', 'direct'])
    p.add_argument("--retries", type=int, default=4, help="Retry attempts per pair on error.")

    # Image compression
    p.add_argument("--img-max-side", dest="img_max_side", type=int, default=160, help="Resize longest side to this many px (0 = no resize).")
    p.add_argument("--img-quality", dest="img_quality", type=int, default=80, help="JPEG/WEBP quality (higher = larger).")
    p.add_argument("--img-format", dest="img_format", choices=["JPEG", "WEBP"], default="JPEG", help="Output format for inline images.")

    # Flow control
    p.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip pair_ids already in output (default True).")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume; process all lines.")
    p.add_argument("--start-index", dest="start_index", type=int, default=0, help="Start from this row index (0-based).")
    p.add_argument("--limit", dest="limit", type=int, default=None, help="Limit number of rows to process from start-index.")

    return p.parse_args()


def main():
    args = parse_args()

    args.out_jsonl = f"GeneratedOutputs/{args.model}/{args.dataset_name}_{args.subset}_explanations.jsonl"

    run_sequential(
        csv_path=args.csv_path,
        dataset_dir=args.dataset_dir,
        system_path=args.system_path,
        user_path=args.user_path,
        out_jsonl=args.out_jsonl,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_segments=args.max_segments,
        retries=args.retries,
        img_max_side=args.img_max_side,
        img_quality=args.img_quality,
        img_format=args.img_format,
        resume=args.resume,
        start_index=args.start_index,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
