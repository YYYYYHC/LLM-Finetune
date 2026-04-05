#!/usr/bin/env python3
"""Simple inference script for contour LoRA model using transformers + peft."""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Inference with trained contour LoRA model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B", help="Base model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--output_file", type=str, default=None, help="Save output JSON to file")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    prompt = "Generate B-spline contour slices for a 3D shape in UDF contour token format:"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Disable thinking mode if Qwen3 adds <think> tokens
    if "/no_think" not in text and "think" in text.lower():
        # Try adding enable_thinking=False
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            pass

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"Generating with temperature={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens}...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\nGenerated {len(generated_ids)} tokens")
    print("=" * 60)

    # Try to parse as JSON
    # Strip any <think>...</think> block if present
    clean = response
    if "<think>" in clean:
        think_end = clean.find("</think>")
        if think_end != -1:
            clean = clean[think_end + len("</think>"):].strip()

    try:
        parsed = json.loads(clean)
        print("Valid JSON output!")
        print(f"  format: {parsed.get('format', 'N/A')}")
        print(f"  bounding_box_xz: {parsed.get('bounding_box_xz', 'N/A')}")
        slices = parsed.get("slices", [])
        print(f"  num_slices: {len(slices)}")
        if slices:
            print(f"  first slice y: {slices[0].get('y', 'N/A')}")
            contours = slices[0].get("contours", [])
            print(f"  first slice contours: {len(contours)}")

        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(parsed, f, indent=2)
            print(f"\nSaved to {out_path}")
    except json.JSONDecodeError as e:
        print(f"Not valid JSON: {e}")
        print(f"First 500 chars of response:\n{clean[:500]}")
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                f.write(clean)
            print(f"\nRaw output saved to {out_path}")


if __name__ == "__main__":
    main()
