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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
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


def get_model_prediction(logits: torch.Tensor, tokenizer) -> Tuple[int, float]:
    """Get the model's actual prediction and its probability."""
    probs = torch.softmax(logits[0, -1], dim=-1)
    predicted_token = torch.argmax(probs).item()
    predicted_digit = int(tokenizer.decode([predicted_token]))
    probability = float(probs[predicted_token].item())
    return predicted_digit, probability


def get_word_pairs(
    word_spans_clean: Sequence[Sequence[int]],
    word_spans_corrupt: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """Get token pairs for each word position (not cumulative)."""
    pairs: List[Tuple[int, int]] = []
    for clean_span, corrupt_span in zip(word_spans_clean, word_spans_corrupt):
        if len(clean_span) != len(corrupt_span):
            raise ValueError(
                "Tokenization mismatch between clean and corrupt prompts; "
                "choose word lists whose tokenization aligns."
            )
        # Add pairs for this word only (not cumulative)
        pairs.extend(zip(clean_span, corrupt_span))
    return pairs


def activation_patch_layer(
    model: HookedTransformer,
    clean_cache,
    corrupt_tokens: torch.Tensor,
    mapping: Sequence[Tuple[int, int]],
    layer_idx: int,
) -> torch.Tensor:
    """Patch activations from clean run into corrupt run at a specific layer."""
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
    )
    return logits.detach()


def activation_patch_position(
    model: HookedTransformer,
    clean_cache,
    corrupt_tokens: torch.Tensor,
    clean_position: int,
    corrupt_position: int,
    layer_idx: int,
) -> torch.Tensor:
    """Patch activation at corrupt_position from clean_position."""
    clean_resid = clean_cache["resid_post", layer_idx][0]

    def patch_fn(resid, hook):
        resid = resid.clone()
        resid[0, corrupt_position, :] = clean_resid[clean_position]
        return resid

    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(hook_name, patch_fn)],
        return_type="logits",
    )
    return logits.detach()


def get_position_after_word(word_spans: List[List[int]], word_idx: int) -> int:
    """Get the last token position of the word_idx-th word (1-indexed)."""
    if word_idx == 0:
        raise ValueError("word_idx should be 1-indexed (1 to N)")
    return word_spans[word_idx - 1][-1]  # Last token of the word


def test_running_count_representation(
    model: HookedTransformer,
    clean_cache,
    corrupt_tokens: torch.Tensor,
    tokenizer,
    clean_spans: List[List[int]],
    corrupt_spans: List[List[int]],
    config: PromptConfig,
) -> None:
    """Test for running count representation by patching at positions after each word."""
    print("=== Running Count Representation Analysis ===")
    print("Testing if there's a layer that maintains a running count...")
    
    # Track which layers work at each position
    layer_performance = {}  # layer_idx -> {word_count: (pred, prob, is_correct)}
    
    # Test patching at positions after each word
    for num_words in range(1, len(config.clean_words) + 1):
        print(f"\n--- Testing after processing {num_words} word(s) ---")
        
        # Get position after processing first num_words
        clean_pos = get_position_after_word(clean_spans, num_words)
        corrupt_pos = get_position_after_word(corrupt_spans, num_words)
        
        # Calculate expected count for first num_words
        expected_count = sum(
            word in CATEGORIES[config.category] 
            for word in config.clean_words[:num_words]
        )
        
        print(f"Position after word {num_words}: clean_pos={clean_pos}, corrupt_pos={corrupt_pos}")
        print(f"Expected count for first {num_words} words: {expected_count}")
        
        # Test each layer
        layer_results = []
        for layer_idx in range(model.cfg.n_layers):
            # Patch at the position after the word
            patched_logits = activation_patch_position(
                model=model,
                clean_cache=clean_cache,
                corrupt_tokens=corrupt_tokens,
                clean_position=clean_pos,
                corrupt_position=corrupt_pos,
                layer_idx=layer_idx,
            )
            
            patched_pred, patched_prob = get_model_prediction(patched_logits, tokenizer)
            is_correct = (patched_pred == expected_count)
            
            layer_results.append((layer_idx, patched_pred, patched_prob, is_correct))
            
            # Track performance for summary
            if layer_idx not in layer_performance:
                layer_performance[layer_idx] = {}
            layer_performance[layer_idx][num_words] = (patched_pred, patched_prob, is_correct)
        
        # Sort by correctness, then by probability
        layer_results.sort(key=lambda x: (x[3], x[2]), reverse=True)
        
        print(f"Results (looking for prediction = {expected_count}):")
        for layer_idx, pred, prob, is_correct in layer_results[:10]:  # Top 10
            status = "✓" if is_correct else "✗"
            print(f"  Layer {layer_idx:02d}: pred={pred} prob={prob:.4f} {status}")
        
        # Check if any layer consistently gives the right count
        correct_layers = [lr for lr in layer_results if lr[3]]
        if correct_layers:
            print(f"  → {len(correct_layers)} layers correctly predict count={expected_count}")
        else:
            print(f"  → No layers correctly predict count={expected_count}")
    
    # Summary analysis
    print(f"\n=== Summary: Which layers consistently maintain running count? ===")
    print("Looking for layers that work across multiple word positions...")
    
    for layer_idx in range(model.cfg.n_layers):
        if layer_idx in layer_performance:
            results = layer_performance[layer_idx]
            correct_count = sum(1 for (pred, prob, is_correct) in results.values() if is_correct)
            total_tests = len(results)
            
            if correct_count > 0:
                print(f"Layer {layer_idx:02d}: {correct_count}/{total_tests} correct")
                for word_count, (pred, prob, is_correct) in results.items():
                    status = "✓" if is_correct else "✗"
                    print(f"  After {word_count} word(s): pred={pred} prob={prob:.4f} {status}")
    
    # Find layers that work consistently
    consistent_layers = []
    for layer_idx in range(model.cfg.n_layers):
        if layer_idx in layer_performance:
            results = layer_performance[layer_idx]
            correct_count = sum(1 for (pred, prob, is_correct) in results.values() if is_correct)
            if correct_count >= 2:  # Works for at least 2 positions
                consistent_layers.append((layer_idx, correct_count))
    
    if consistent_layers:
        print(f"\nLayers that work consistently (≥2 positions):")
        for layer_idx, count in sorted(consistent_layers, key=lambda x: x[1], reverse=True):
            print(f"  Layer {layer_idx:02d}: {count} positions correct")
    else:
        print(f"\nNo layers work consistently across multiple positions.")
        print("This suggests the model does NOT maintain a running count representation.")


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

    # Run both prompts and get their actual predictions
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

    # Get actual model predictions
    clean_pred, clean_prob = get_model_prediction(clean_logits, tokenizer)
    corrupt_pred, corrupt_prob = get_model_prediction(corrupt_logits, tokenizer)

    print("=== Prompt Summary ===")
    print(f"Clean prompt: {config.clean_prompt}")
    print(f"Corrupt prompt: {config.corrupt_prompt}")
    print(f"Clean count: {config.clean_count} | Corrupt count: {config.corrupt_count}")
    print(f"Model clean prediction: {clean_pred} (prob: {clean_prob:.4f})")
    print(f"Model corrupt prediction: {corrupt_pred} (prob: {corrupt_prob:.4f})")
    print()

    # Test: Does patching individual layers recover the correct count?
    print("=== Layer-wise Activation Patching ===")
    print("Testing which layers contain counting representations...")
    
    # Get all word pairs for patching
    all_word_pairs = get_word_pairs(clean_spans, corrupt_spans)
    
    # Test each layer individually
    layer_results = []
    for layer_idx in range(model.cfg.n_layers):
        patched_logits = activation_patch_layer(
            model=model,
            clean_cache=clean_cache,
            corrupt_tokens=corrupt_tokens,
            mapping=all_word_pairs,
            layer_idx=layer_idx,
        )
        
        patched_pred, patched_prob = get_model_prediction(patched_logits, tokenizer)
        
        # Calculate improvement: does patching this layer make the model predict correctly?
        is_correct = (patched_pred == config.clean_count)
        prob_improvement = patched_prob - corrupt_prob
        
        layer_results.append((layer_idx, patched_pred, patched_prob, is_correct, prob_improvement))
    
    # Sort by whether the layer recovers the correct prediction
    layer_results.sort(key=lambda x: (x[3], x[4]), reverse=True)
    
    print(f"Results (sorted by correctness, then by probability improvement):")
    for layer_idx, pred, prob, is_correct, improvement in layer_results:
        status = "✓" if is_correct else "✗"
        print(f"Layer {layer_idx:02d}: pred={pred} prob={prob:.4f} {status} (Δ={improvement:+.4f})")
    
    # Test: Which individual words matter for counting?
    print(f"\n=== Individual Word Analysis ===")
    print("Testing which words are crucial for counting...")
    
    for word_idx, (clean_word, corrupt_word) in enumerate(zip(config.clean_words, config.corrupt_words)):
        # Patch just this word
        word_pairs = list(zip(clean_spans[word_idx], corrupt_spans[word_idx]))
        
        word_results = []
        for layer_idx in range(model.cfg.n_layers):
            patched_logits = activation_patch_layer(
                model=model,
                clean_cache=clean_cache,
                corrupt_tokens=corrupt_tokens,
                mapping=word_pairs,
                layer_idx=layer_idx,
            )
            
            patched_pred, patched_prob = get_model_prediction(patched_logits, tokenizer)
            is_correct = (patched_pred == config.clean_count)
            word_results.append((layer_idx, patched_pred, patched_prob, is_correct))
        
        # Find best layer for this word
        best_layer = max(word_results, key=lambda x: (x[3], x[2]))
        print(f"Word '{clean_word}' → '{corrupt_word}': Best layer {best_layer[0]} → pred={best_layer[1]} prob={best_layer[2]:.4f} {'✓' if best_layer[3] else '✗'}")
    
    print()
    
    # NEW: Test for running count representation
    test_running_count_representation(
        model=model,
        clean_cache=clean_cache,
        corrupt_tokens=corrupt_tokens,
        tokenizer=tokenizer,
        clean_spans=clean_spans,
        corrupt_spans=corrupt_spans,
        config=config,
    )


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
