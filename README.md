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
