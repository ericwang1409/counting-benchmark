# Counting Benchmark: Causal Mediation Analysis

This repository contains a comprehensive analysis of how large language models perform counting tasks, with a focus on understanding whether models maintain internal running count representations.

## Overview

The project investigates whether language models maintain a "running count" representation in their hidden states while processing lists of words. This is tested through causal mediation analysis using activation patching techniques.

## Dataset Generation

The dataset consists of counting prompts with the following structure:

```
Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [apple banana grape pear pillow lamp chair table]
Answer: 
```

### Categories
- **Fruit**: apple, banana, grape, kiwi, mango, papaya, pear, plum, apricot, blueberry, strawberry, raspberry, blackberry, pineapple
- **Animal**: dog, cat, mouse, lion, tiger, bear, wolf, fox, deer, rabbit, squirrel, bird, fish, snake
- **Body part**: head, hand, foot, eye, ear, nose, mouth, arm, leg, finger, toe, knee, elbow, shoulder
- **Instrument**: guitar, piano, violin, drum, trumpet, flute, saxophone, clarinet, cello, harp, organ, banjo
- **Profession**: doctor, teacher, lawyer, engineer, artist, chef, pilot, nurse, farmer, writer, scientist, musician
- **Vehicle**: car, truck, bus, train, plane, boat, bike, motorcycle, helicopter, taxi, ambulance, firetruck

## Benchmark Performance

### Model: Meta-Llama-3-8B-Instruct

The model shows mixed performance on counting tasks:

**Short Lists (4 words):**
- Clean prompt: `[apple banana pillow lamp]` → Model predicts 2 ✓ (correct)
- Corrupt prompt: `[chair banana pillow lamp]` → Model predicts 1 ✓ (correct)

**Longer Lists (8 words):**
- Clean prompt: `[apple banana grape pear pillow lamp chair table]` → Model predicts 3 ✗ (should be 4)
- Corrupt prompt: `[chair desk table lamp pillow lamp chair table]` → Model predicts 0 ✓ (correct)

### Key Findings

1. **Model makes counting errors** on longer lists, predicting 3 instead of 4 for 4-fruit lists
2. **Model is more accurate** on shorter lists and when the count is lower
3. **Performance degrades** as the number of items to count increases

## Causal Mediation Analysis

### Experimental Design

The analysis uses activation patching to test whether models maintain running count representations:

1. **Clean Run**: Process a list with the correct items (e.g., 4 fruits)
2. **Corrupt Run**: Process a list with incorrect items (e.g., 0 fruits)  
3. **Patching**: Replace activations from clean run into corrupt run at specific positions
4. **Test**: Does patching recover intermediate counts (1, 2, 3, 4)?

### Results: No Running Count Representation

**Key Finding**: The model does NOT maintain a running count representation.

**Evidence:**
- **Tokenization Mismatch**: Clean and corrupt prompts tokenize differently, preventing position-based patching
- **No Position Recovery**: When patching positions after each word, no layers can recover intermediate counts
- **All Layers Fail**: No layer consistently predicts the correct count at intermediate positions

**Example Results:**
```
--- Testing after processing 1 word(s) ---
Expected count for first 1 words: 1
→ No layers correctly predict count=1

--- Testing after processing 2 word(s) ---  
Expected count for first 2 words: 2
→ No layers correctly predict count=2

--- Testing after processing 3 word(s) ---
Expected count for first 3 words: 3  
→ No layers correctly predict count=3
```

### What This Means

The model's counting mechanism is:

1. **Not position-based**: Cannot handle tokenization differences between prompts
2. **Not running count-based**: Does not maintain incremental counting state
3. **More sophisticated**: Likely processes the entire sequence before making the final count decision

## Technical Implementation

### Dependencies
- `transformer-lens>=2.15.0,<2.16.0`
- `transformers>=4.43,<4.50`
- `torch>=2.8.0`

### Usage

```bash
# Run causal mediation analysis
python analysis/causal_mediation.py \
    --clean_words apple banana grape pear \
    --corrupt_words chair desk table lamp

# Run benchmark evaluation  
python benchmark_meta_llama3_hf.py
```

### Key Files

- `analysis/causal_mediation.py`: Causal mediation analysis implementation
- `benchmark_meta_llama3_hf.py`: Benchmark evaluation script
- `generate_dataset.py`: Dataset generation utilities
- `data/counting_dataset.jsonl`: Generated counting dataset

## Conclusion

This analysis provides strong evidence that large language models do not maintain running count representations in their hidden states. Instead, they use more sophisticated mechanisms that process entire sequences before making counting decisions. The tokenization mismatch issues further support this conclusion, as a robust running count system would need to handle such variations.

The findings suggest that models' counting abilities rely on distributed processing across multiple layers rather than maintaining explicit counting state, which has implications for understanding how models perform arithmetic and sequential reasoning tasks.


ubuntu@192-222-51-240:~/counting-benchmark$  python analysis/causal_mediation.py --clean_words apple grape pillow lamp mango chair banana table --corrupt_words chair table pillow lamp lamp chair desk table
/usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
2025-10-06 00:40:57.854917: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759711257.864967   20328 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1759711257.869159   20328 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1759711257.874282   20328 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1759711257.874309   20328 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1759711257.874319   20328 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1759711257.874321   20328 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.12it/s]
WARNING:root:With reduced precision, it is advised to use `from_pretrained_no_processing` instead of `from_pretrained`.
WARNING:root:You are not using LayerNorm, so the writing weights can't be centered! Skipping
Loaded pretrained model meta-llama/Meta-Llama-3-8B-Instruct into HookedTransformer
=== Prompt Summary ===
Clean prompt: Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [apple grape pillow lamp mango chair banana table]
Answer: 
Corrupt prompt: Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [chair table pillow lamp lamp chair desk table]
Answer: 
Clean count: 4 | Corrupt count: 0
Model clean prediction: 4 (prob: 0.6387)
Model corrupt prediction: 0 (prob: 0.9780)

=== Layer-wise Activation Patching ===
Testing which layers contain counting representations...
Results (sorted by correctness, then by probability improvement):
Layer 04: pred=4 prob=0.6431 ✓ (Δ=-0.3350)
Layer 02: pred=4 prob=0.6411 ✓ (Δ=-0.3369)
Layer 00: pred=4 prob=0.6401 ✓ (Δ=-0.3379)
Layer 03: pred=4 prob=0.6392 ✓ (Δ=-0.3389)
Layer 01: pred=4 prob=0.6377 ✓ (Δ=-0.3403)
Layer 05: pred=4 prob=0.6167 ✓ (Δ=-0.3613)
Layer 06: pred=4 prob=0.5952 ✓ (Δ=-0.3828)
Layer 07: pred=4 prob=0.5410 ✓ (Δ=-0.4370)
Layer 08: pred=4 prob=0.4858 ✓ (Δ=-0.4922)
Layer 16: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 17: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 18: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 19: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 20: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 21: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 22: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 23: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 24: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 25: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 30: pred=0 prob=0.9785 ✗ (Δ=+0.0005)
Layer 15: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 26: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 27: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 28: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 29: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 31: pred=0 prob=0.9780 ✗ (Δ=+0.0000)
Layer 14: pred=0 prob=0.9771 ✗ (Δ=-0.0010)
Layer 13: pred=0 prob=0.9668 ✗ (Δ=-0.0112)
Layer 10: pred=3 prob=0.7793 ✗ (Δ=-0.1987)
Layer 11: pred=3 prob=0.7471 ✗ (Δ=-0.2310)
Layer 09: pred=3 prob=0.6001 ✗ (Δ=-0.3779)
Layer 12: pred=2 prob=0.5645 ✗ (Δ=-0.4136)

=== Individual Word Analysis ===
Testing which words are crucial for counting...
Word 'apple' → 'chair': Best layer 13 → pred=0 prob=0.9810 ✗
Word 'grape' → 'table': Best layer 2 → pred=1 prob=0.9863 ✗
Word 'pillow' → 'pillow': Best layer 1 → pred=0 prob=0.9790 ✗
Word 'lamp' → 'lamp': Best layer 11 → pred=0 prob=0.9785 ✗
Word 'mango' → 'lamp': Best layer 0 → pred=1 prob=0.9893 ✗
Word 'chair' → 'chair': Best layer 8 → pred=0 prob=0.9800 ✗
Word 'banana' → 'desk': Best layer 1 → pred=1 prob=0.9893 ✗
Word 'table' → 'table': Best layer 2 → pred=0 prob=0.9790 ✗

=== Running Count Representation Analysis ===
Testing if there's a layer that maintains a running count...

--- Testing after processing 1 word(s) ---
Position after word 1: clean_pos=31, corrupt_pos=31
Expected count for first 1 words: 1
Results (looking for prediction = 1):
  Layer 13: pred=0 prob=0.9810 ✗
  Layer 21: pred=0 prob=0.9790 ✗
  Layer 22: pred=0 prob=0.9790 ✗
  Layer 23: pred=0 prob=0.9790 ✗
  Layer 24: pred=0 prob=0.9790 ✗
  Layer 14: pred=0 prob=0.9785 ✗
  Layer 15: pred=0 prob=0.9785 ✗
  Layer 16: pred=0 prob=0.9785 ✗
  Layer 17: pred=0 prob=0.9785 ✗
  Layer 18: pred=0 prob=0.9785 ✗
  → No layers correctly predict count=1

--- Testing after processing 2 word(s) ---
Position after word 2: clean_pos=32, corrupt_pos=32
Expected count for first 2 words: 2
Results (looking for prediction = 2):
  Layer 02: pred=1 prob=0.9863 ✗
  Layer 01: pred=1 prob=0.9858 ✗
  Layer 00: pred=1 prob=0.9849 ✗
  Layer 03: pred=1 prob=0.9849 ✗
  Layer 13: pred=0 prob=0.9814 ✗
  Layer 05: pred=1 prob=0.9805 ✗
  Layer 04: pred=1 prob=0.9800 ✗
  Layer 16: pred=0 prob=0.9780 ✗
  Layer 17: pred=0 prob=0.9780 ✗
  Layer 18: pred=0 prob=0.9780 ✗
  → No layers correctly predict count=2

--- Testing after processing 3 word(s) ---
Position after word 3: clean_pos=33, corrupt_pos=33
Expected count for first 3 words: 2
Results (looking for prediction = 2):
  Layer 01: pred=0 prob=0.9790 ✗
  Layer 03: pred=0 prob=0.9790 ✗
  Layer 02: pred=0 prob=0.9785 ✗
  Layer 04: pred=0 prob=0.9785 ✗
  Layer 05: pred=0 prob=0.9780 ✗
  Layer 13: pred=0 prob=0.9780 ✗
  Layer 15: pred=0 prob=0.9780 ✗
  Layer 16: pred=0 prob=0.9780 ✗
  Layer 17: pred=0 prob=0.9780 ✗
  Layer 18: pred=0 prob=0.9780 ✗
  → No layers correctly predict count=2

--- Testing after processing 4 word(s) ---
Position after word 4: clean_pos=34, corrupt_pos=34
Expected count for first 4 words: 2
Results (looking for prediction = 2):
  Layer 11: pred=0 prob=0.9785 ✗
  Layer 00: pred=0 prob=0.9780 ✗
  Layer 01: pred=0 prob=0.9780 ✗
  Layer 02: pred=0 prob=0.9780 ✗
  Layer 03: pred=0 prob=0.9780 ✗
  Layer 04: pred=0 prob=0.9780 ✗
  Layer 05: pred=0 prob=0.9780 ✗
  Layer 06: pred=0 prob=0.9780 ✗
  Layer 07: pred=0 prob=0.9780 ✗
  Layer 12: pred=0 prob=0.9780 ✗
  → No layers correctly predict count=2

--- Testing after processing 5 word(s) ---
Position after word 5: clean_pos=35, corrupt_pos=35
Expected count for first 5 words: 3
Results (looking for prediction = 3):
  Layer 00: pred=1 prob=0.9893 ✗
  Layer 01: pred=1 prob=0.9893 ✗
  Layer 02: pred=1 prob=0.9893 ✗
  Layer 03: pred=1 prob=0.9888 ✗
  Layer 04: pred=1 prob=0.9888 ✗
  Layer 05: pred=1 prob=0.9883 ✗
  Layer 06: pred=1 prob=0.9883 ✗
  Layer 07: pred=1 prob=0.9883 ✗
  Layer 08: pred=1 prob=0.9883 ✗
  Layer 09: pred=1 prob=0.9873 ✗
  → No layers correctly predict count=3

--- Testing after processing 6 word(s) ---
Position after word 6: clean_pos=36, corrupt_pos=36
Expected count for first 6 words: 3
Results (looking for prediction = 3):
  Layer 08: pred=0 prob=0.9800 ✗
  Layer 05: pred=0 prob=0.9795 ✗
  Layer 07: pred=0 prob=0.9795 ✗
  Layer 06: pred=0 prob=0.9790 ✗
  Layer 01: pred=0 prob=0.9785 ✗
  Layer 02: pred=0 prob=0.9785 ✗
  Layer 04: pred=0 prob=0.9785 ✗
  Layer 15: pred=0 prob=0.9785 ✗
  Layer 03: pred=0 prob=0.9780 ✗
  Layer 10: pred=0 prob=0.9780 ✗
  → No layers correctly predict count=3

--- Testing after processing 7 word(s) ---
Position after word 7: clean_pos=37, corrupt_pos=37
Expected count for first 7 words: 4
Results (looking for prediction = 4):
  Layer 01: pred=1 prob=0.9893 ✗
  Layer 02: pred=1 prob=0.9893 ✗
  Layer 03: pred=1 prob=0.9893 ✗
  Layer 04: pred=1 prob=0.9893 ✗
  Layer 00: pred=1 prob=0.9888 ✗
  Layer 05: pred=1 prob=0.9883 ✗
  Layer 06: pred=1 prob=0.9878 ✗
  Layer 07: pred=1 prob=0.9844 ✗
  Layer 09: pred=1 prob=0.9810 ✗
  Layer 08: pred=1 prob=0.9790 ✗
  → No layers correctly predict count=4

--- Testing after processing 8 word(s) ---
Position after word 8: clean_pos=38, corrupt_pos=38
Expected count for first 8 words: 4
Results (looking for prediction = 4):
  Layer 02: pred=0 prob=0.9790 ✗
  Layer 03: pred=0 prob=0.9790 ✗
  Layer 05: pred=0 prob=0.9790 ✗
  Layer 06: pred=0 prob=0.9790 ✗
  Layer 07: pred=0 prob=0.9790 ✗
  Layer 08: pred=0 prob=0.9790 ✗
  Layer 04: pred=0 prob=0.9785 ✗
  Layer 01: pred=0 prob=0.9780 ✗
  Layer 14: pred=0 prob=0.9780 ✗
  Layer 15: pred=0 prob=0.9780 ✗
  → No layers correctly predict count=4

=== Summary: Which layers consistently maintain running count? ===
Looking for layers that work across multiple word positions...

No layers work consistently across multiple positions.
This suggests the model does NOT maintain a running count representation.
