"""Causal mediation analysis using TransformerLens.

The script compares a "clean" prompt (with several on-type words) against a
"corrupt" prompt (where those words are replaced with distractors). It locates
layers whose hidden states mediate the correct count by patching the clean
activations into the corrupt run and measuring how the probability of the clean
answer changes.

Example (requires ``transformer-lens`` and ``torch``):

    python analysis/causal_mediation.py \
        --model meta-llama/Llama-2-7b-hf \
        --category fruit \
        --clean_words apple banana pillow lamp \
        --corrupt_words chair banana pillow lamp
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from transformer_lens import HookedTransformer

from generate_dataset import CATEGORIES


@dataclass
class PromptConfig:
    category: str
    clean_words: Sequence[str]
    corrupt_words: Sequence[str]

    def build_prompt(self, words: Sequence[str]) -> str:
        return (
            "Count the number of words in the following list that match the given type, "
            "and put the numerical answer in parentheses.\n"
            f"Type: {self.category}\n"
            f"List: [{' '.join(words)}]\n"
            "Answer: "
        )

    @property
    def clean_prompt(self) -> str:
        return self.build_prompt(self.clean_words)

    @property
    def corrupt_prompt(self) -> str:
        return self.build_prompt(self.corrupt_words)

    @property
    def clean_count(self) -> int:
        vocab = set(CATEGORIES[self.category])
        return sum(word in vocab for word in self.clean_words)

    @property
    def corrupt_count(self) -> int:
        vocab = set(CATEGORIES[self.category])
        return sum(word in vocab for word in self.corrupt_words)


def device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_model(model_name: str, device: torch.device, dtype: torch.dtype) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)
    model.eval()
    return model


def tokenize_with_offsets(tokenizer, prompt: str, device: torch.device):
    encoding = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoding.pop("offset_mapping")[0].tolist()
    tokens = encoding["input_ids"].to(device)
    return tokens, offsets


def find_word_token_spans(
    tokenizer,
    prompt: str,
    words: Sequence[str],
    offsets: List[Tuple[int, int]],
) -> List[List[int]]:
    spans: List[List[int]] = []
    cursor = 0
    for word in words:
        start = prompt.index(word, cursor)
        end = start + len(word)
        token_indices: List[int] = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            if tok_start >= end:
                break
            if tok_end <= start:
                continue
            token_indices.append(idx)
        if not token_indices:
            raise ValueError(f"Could not align word '{word}' to token indices")
        spans.append(token_indices)
        cursor = end
    return spans


def extract_digit_token_id(tokenizer, digit: int) -> int:
    tokenized = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
    if len(tokenized) != 1:
        raise ValueError(
            f"Tokenizer splits digit {digit}; consider adjusting evaluation logic."
        )
    return tokenized[0]


def next_token_prob(logits: torch.Tensor, token_id: int) -> float:
    probs = torch.softmax(logits[0, -1], dim=-1)
    return float(probs[token_id].item())


def cumulative_pairs(
    word_spans_clean: Sequence[Sequence[int]],
    word_spans_corrupt: Sequence[Sequence[int]],
) -> List[List[Tuple[int, int]]]:
    pairs: List[List[Tuple[int, int]]] = []
    cumulative: List[Tuple[int, int]] = []
    for clean_span, corrupt_span in zip(word_spans_clean, word_spans_corrupt):
        if len(clean_span) != len(corrupt_span):
            raise ValueError(
                "Tokenization mismatch between clean and corrupt prompts; "
                "choose word lists whose tokenization aligns."
            )
        cumulative.extend(zip(clean_span, corrupt_span))
        pairs.append(list(cumulative))
    return pairs


def activation_patch(
    model: HookedTransformer,
    clean_cache,
    corrupt_tokens: torch.Tensor,
    mapping: Sequence[Tuple[int, int]],
) -> List[torch.Tensor]:
    patched_logits: List[torch.Tensor] = []
    for layer_idx in range(model.cfg.n_layers):
        clean_resid = clean_cache["resid_post", layer_idx][0]

        def patch_fn(resid, hook):
            resid = resid.clone()
            for clean_idx, corrupt_idx in mapping:
                resid[0, corrupt_idx, :] = clean_resid[clean_idx]
            return resid

        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(hook_name, patch_fn)],
            return_type="logits",
            remove_batch_dim=False,
        )
        patched_logits.append(logits.detach())
    return patched_logits


def run_experiment(args):
    device = device_from_arg(args.device)
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    config = PromptConfig(
        category=args.category,
        clean_words=args.clean_words,
        corrupt_words=args.corrupt_words,
    )

    if len(config.clean_words) != len(config.corrupt_words):
        raise ValueError("clean_words and corrupt_words must have the same length")

    if config.clean_count == config.corrupt_count:
        print(
            "[warn] Clean and corrupt prompts have the same count; "
            "activation patching may show weak effects."
        )

    model = load_model(args.model, device, dtype)
    tokenizer = model.tokenizer

    clean_tokens, clean_offsets = tokenize_with_offsets(tokenizer, config.clean_prompt, device)
    corrupt_tokens, corrupt_offsets = tokenize_with_offsets(
        tokenizer, config.corrupt_prompt, device
    )

    clean_spans = find_word_token_spans(
        tokenizer, config.clean_prompt, config.clean_words, clean_offsets
    )
    corrupt_spans = find_word_token_spans(
        tokenizer, config.corrupt_prompt, config.corrupt_words, corrupt_offsets
    )

    clean_logits, clean_cache = model.run_with_cache(
        clean_tokens,
        return_type="logits",
        remove_batch_dim=False,
    )
    corrupt_logits, _ = model.run_with_cache(
        corrupt_tokens,
        return_type="logits",
        remove_batch_dim=False,
    )

    clean_digit_id = extract_digit_token_id(tokenizer, config.clean_count)
    clean_prob = next_token_prob(clean_logits, clean_digit_id)
    corrupt_prob = next_token_prob(corrupt_logits, clean_digit_id)

    print("=== Prompt Summary ===")
    print(f"Clean prompt: {config.clean_prompt}")
    print(f"Corrupt prompt: {config.corrupt_prompt}")
    print(f"Clean count: {config.clean_count} | Corrupt count: {config.corrupt_count}")
    print(f"Model clean prob for {config.clean_count}: {clean_prob:.4f}")
    print(f"Model corrupt prob for {config.clean_count}: {corrupt_prob:.4f}\n")

    cumulative_mappings = cumulative_pairs(clean_spans, corrupt_spans)

    for idx, mapping in enumerate(cumulative_mappings, start=1):
        word_label = config.clean_words[idx - 1]
        patched_logits_per_layer = activation_patch(
            model=model,
            clean_cache=clean_cache,
            corrupt_tokens=corrupt_tokens,
            mapping=mapping,
        )
        deltas = []
        for layer_idx, logits in enumerate(patched_logits_per_layer):
            prob = next_token_prob(logits, clean_digit_id)
            deltas.append((layer_idx, prob, prob - corrupt_prob))
        top = sorted(deltas, key=lambda item: item[2], reverse=True)[: args.topk]
        print(f"=== Cumulative patch through word #{idx} ({word_label}) ===")
        for layer_idx, prob, delta in top:
            print(
                f"Layer {layer_idx:02d}: prob={prob:.4f} | delta={delta:+.4f} "
                f"(corrupt baseline {corrupt_prob:.4f})"
            )
        print()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="TransformerLens model name")
    parser.add_argument(
        "--category",
        type=str,
        default="fruit",
        choices=sorted(CATEGORIES.keys()),
        help="Dataset category for prompts",
    )
    parser.add_argument(
        "--clean_words",
        nargs="+",
        default=["apple", "banana", "pillow", "lamp"],
        help="Words used in the clean prompt list",
    )
    parser.add_argument(
        "--corrupt_words",
        nargs="+",
        default=["chair", "banana", "pillow", "lamp"],
        help="Words used in the corrupt prompt list; same length as clean_words",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device identifier (auto/cpu/cuda/mps)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top layers to display per cumulative patch",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_experiment(parse_args())
