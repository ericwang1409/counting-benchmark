"""Benchmark Meta-Llama-3-8B-Instruct on the counting dataset using llama.cpp bindings."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from llama_cpp import Llama

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
    return None


def run_benchmark(
    model_path: Path,
    dataset: Iterable[Example],
    max_tokens: int,
    temperature: float,
    repeat_penalty: float,
    n_gpu_layers: int,
) -> None:
    llama = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        verbose=False,
    )

    total = 0
    correct = 0
    no_parse = 0

    for example in dataset:
        response = llama.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. "
                        "Answer with a single count inside parentheses and nothing else."
                    ),
                },
                {"role": "user", "content": example.prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            repeat_penalty=repeat_penalty,
        )
        choice = response["choices"][0]["message"]["content"].strip()
        predicted = extract_count(choice)
        if predicted is None:
            no_parse += 1
        else:
            if predicted == example.gold_count:
                correct += 1
        total += 1

        # Simple running report every 50 examples.
        if total % 50 == 0:
            accuracy = correct / total
            print(f"Processed {total} examples | Accuracy: {accuracy:.3f} | Unparsed: {no_parse}")

    final_accuracy = correct / total
    print("\n=== Benchmark Summary ===")
    print(f"Examples evaluated: {total}")
    print(f"Accuracy: {final_accuracy:.4f}")
    print(f"Unparsed responses: {no_parse}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Meta-Llama-3-8B on counting dataset")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to Meta-Llama-3-8B-Instruct GGUF weights (e.g., .gguf)",
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
        default=8,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.0,
        help="Repeat penalty parameter for llama.cpp",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to keep on GPU/Metal (-1 lets llama.cpp decide)",
    )
    args = parser.parse_args()

    if args.limit == -1:
        args.limit = None
    return args


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.limit)
    run_benchmark(
        model_path=args.model,
        dataset=dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repeat_penalty=args.repeat_penalty,
        n_gpu_layers=args.n_gpu_layers,
    )


if __name__ == "__main__":
    main()
