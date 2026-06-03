from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "numind/NuExtract3"


def clean_csv_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:csv)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def write_csv_text(path: Path, text: str) -> None:
    rows = []

    for row in csv.reader(clean_csv_text(text).splitlines()):
        if any(cell.strip() for cell in row):
            rows.append([cell.strip() for cell in row])

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def run(png_path: Path, out_csv: Path) -> None:
    image = Image.open(png_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    prompt = """
Extract the table from this image.

Return only valid CSV.
Do not add explanations.
Do not use markdown fences.
Preserve the table structure as well as possible.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

    output_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    output_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    write_csv_text(out_csv, output_text)

    md_path = out_csv.with_suffix(".md")
    md_path.write_text(output_text, encoding="utf-8")

    print(f"CSV written to: {out_csv}")
    print(f"Raw output written to: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    run(Path(args.png), Path(args.out))