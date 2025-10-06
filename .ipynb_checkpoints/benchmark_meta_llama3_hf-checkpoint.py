"""Benchmark Meta-Llama-3-8B-Instruct via Hugging Face transformers."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULT_PATTERN = re.compile(r"\((\d+)\)")


@dataclass
class Example:
    prompt: str
    answer: str

    @property
    def gold_count(self) -> int:
        match = RESULT_PATTERN.search(self.answer)
        if not match:
            raise ValueError(f"Invalid gold answer format: {self.answer}")
        return int(match.group(1))


def load_dataset(path: Path, limit: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                break
            record = json.loads(line)
            examples.append(Example(prompt=record["prompt"], answer=record["answer"]))
    if not examples:
        raise ValueError(f"Dataset at {path} is empty or limit=0")
    return examples


def extract_count(text: str) -> int | None:
    match = RESULT_PATTERN.search(text)
    if match:
        return int(match.group(1))
    # fallback: bare integer
    digits = re.findall(r"\d+", text)
    if len(digits) == 1:
        return int(digits[0])
    return None


def device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a precise assistant. Answer with a single number in parentheses, e.g. `(3)`.",
        },
        {"role": "user", "content": prompt},
    ]


def run_benchmark(
    model_name: str,
    dataset: Iterable[Example],
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    output_path: Path,
    batch_size: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    model.to(device)

    total = 0
    correct = 0
    no_parse = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fh:
        for example in dataset:
            messages = build_messages(example.prompt)
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_tokens = outputs[0, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

            predicted = extract_count(decoded)
            correct_prediction = predicted == example.gold_count if predicted is not None else False

            if predicted is None:
                no_parse += 1
            elif correct_prediction:
                correct += 1

            record = {
                "prompt": example.prompt,
                "gold_answer": example.answer,
                "model_response": decoded,
                "parsed_count": predicted,
                "correct": correct_prediction,
            }
            out_fh.write(json.dumps(record) + "\n")

            total += 1
            if total % 50 == 0:
                accuracy = correct / total
                print(
                    f"Processed {total} examples | Accuracy: {accuracy:.3f} | Unparsed: {no_parse}"
                )

    final_accuracy = correct / total
    print("\n=== Benchmark Summary ===")
    print(f"Examples evaluated: {total}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Unparsed responses: {no_parse}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Meta-Llama-3-8B-Instruct via transformers")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/counting_dataset.jsonl"),
        help="JSONL dataset file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of examples to evaluate (use -1 for full dataset)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1,
        help="Top-p sampling cutoff",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device identifier (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/meta_llama3_hf_predictions.jsonl"),
        help="Where to save per-example model responses",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (currently only 1 supported; placeholder for future)",
    )
    args = parser.parse_args()
    if args.limit == -1:
        args.limit = None
    return args


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.limit)
    dtype = getattr(torch, args.dtype)
    device = device_from_arg(args.device)
    run_benchmark(
        model_name=args.model,
        dataset=dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        dtype=dtype,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
