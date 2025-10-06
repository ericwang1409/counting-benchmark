# Counting Experiments

This repository aims to tackle this task:

Given the following type of prompt, a sufficiently large language model will be able to answer with the correct number.

```
Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.
Type: fruit
List: [dog apple cherry bus cat grape bowl]
Answer: (
```

Your task:

1. create a dataset of several thousand examples like this.
2. benchmark some open-weight LMs on solving this task zero-shot (without reasoning tokens)
3. for a single model, create a causal mediation analysis experiment (patching from one run to another) to answer: "is there a hidden state layer that contains a representation of the running count of matching words, while processing the list of words?"

# Question 1

I was able to create a dataset of several thousand examples of this programmatically. By creating a few distinct categories filled with unique words, the program is able to generate many different permutations of these in order to create a wide range of questions of the above type.

# Question 2

I benchmarked meta-llama/Meta-Llama-3-8B-Instruct on this task. I am working on a Macbook Pro M1 machine, so I first started with using the Q4_K_M quantized version. I found that the results were very low, around 10% accurate. I found that the given prompt was a bit confusing because of the last ")" in the prompt, leading to around 60% of responses being in the incorrect format (without the parentheses). By removing that, the accuracy shot up to 77%. Here is the terminal output for a benchmark run of the Q4_K_M run.

```
Processed 50 examples | Accuracy: 0.720 | Unparsed: 0
Processed 100 examples | Accuracy: 0.770 | Unparsed: 0
Processed 150 examples | Accuracy: 0.800 | Unparsed: 0
Processed 200 examples | Accuracy: 0.805 | Unparsed: 0
Processed 250 examples | Accuracy: 0.776 | Unparsed: 0
Processed 300 examples | Accuracy: 0.777 | Unparsed: 0
Processed 350 examples | Accuracy: 0.783 | Unparsed: 0
Processed 400 examples | Accuracy: 0.782 | Unparsed: 0
Processed 450 examples | Accuracy: 0.767 | Unparsed: 0
Processed 500 examples | Accuracy: 0.770 | Unparsed: 0

=== Benchmark Summary ===
Examples evaluated: 500
Accuracy: 0.7700
Unparsed responses: 0
```

Later on, I ended up spinning up a GH200 on Lambda Cloud for start 3. I redid the benchmarking with the full model from hugging face, and the accuracy dropped a bit. This might be due to chat template formatting or random differences.

```
Processed 50 examples | Accuracy: 0.560 | Unparsed: 0
Processed 100 examples | Accuracy: 0.610 | Unparsed: 0
Processed 150 examples | Accuracy: 0.640 | Unparsed: 0
Processed 200 examples | Accuracy: 0.645 | Unparsed: 0
Processed 250 examples | Accuracy: 0.616 | Unparsed: 0
Processed 300 examples | Accuracy: 0.617 | Unparsed: 0
Processed 350 examples | Accuracy: 0.623 | Unparsed: 0
Processed 400 examples | Accuracy: 0.622 | Unparsed: 0
Processed 450 examples | Accuracy: 0.607 | Unparsed: 0
Processed 500 examples | Accuracy: 0.610 | Unparsed: 0

=== Benchmark Summary ===
Examples evaluated: 500
Accuracy: 0.6150
Unparsed responses: 0
```

# Question 3

This was my methodology for the causal mediation analysis experiment:

1. **Clean Run**: Process a list with the correct items (e.g., 4 fruits)
2. **Corrupt Run**: Process a list with incorrect items (e.g., 0 fruits)
3. **Patching**: Replace activations from clean run into corrupt run at specific positions
4. **Test**: Does patching recover intermediate counts (1, 2, 3, 4)?

I found that there is no running count representation. When patching positions after each word, no layers can recover intermediate counts. Furthermore, no layer consistently predicts the correct count at intermediate positions. The model does not maintain a running count representation. Across multiple experiments with different word combinations, no layer at any position successfully recovered intermediate counts. This suggests the model uses distributed processing rather than explicit state tracking. The counting information appears spread across positions and early layers (0-8) rather than accumulated incrementally at specific positions. The counting emerges holistically rather than through a sequential accumulation mechanism."

# Example output:

```
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
```
