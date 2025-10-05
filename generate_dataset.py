from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = {
    "fruit": [
        "apple",
        "banana",
        "grape",
        "kiwi",
        "mango",
        "papaya",
        "pear",
        "plum",
        "apricot",
        "blueberry",
        "strawberry",
        "raspberry",
        "blackberry",
        "pineapple",
    ],
    "animal": [
        "dog",
        "cat",
        "mouse",
        "lion",
        "tiger",
        "bear",
        "zebra",
        "horse",
        "shark",
        "dolphin",
        "eagle",
        "falcon",
        "otter",
        "giraffe",
    ],
    "vehicle": [
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
        "boat",
        "scooter",
        "train",
        "subway",
        "airplane",
        "tractor",
        "skateboard",
        "helicopter",
        "sled",
    ],
    "profession": [
        "doctor",
        "teacher",
        "lawyer",
        "engineer",
        "nurse",
        "chef",
        "pilot",
        "artist",
        "firefighter",
        "scientist",
        "architect",
        "plumber",
        "mechanic",
        "dentist",
    ],
    "instrument": [
        "guitar",
        "piano",
        "violin",
        "drum",
        "flute",
        "trumpet",
        "trombone",
        "cello",
        "clarinet",
        "saxophone",
        "harp",
        "tuba",
        "banjo",
        "xylophone",
    ],
    "body part": [
        "hand",
        "arm",
        "leg",
        "foot",
        "head",
        "ear",
        "nose",
        "mouth",
        "finger",
        "toe",
        "knee",
        "elbow",
        "ankle",
    ],
}

DISTRACTOR_BANK = [
    "table",
    "chair",
    "spoon",
    "fork",
    "bowl",
    "plate",
    "lamp",
    "book",
    "computer",
    "pillow",
    "door",
    "window",
    "couch",
    "blanket",
    "clock",
    "pen",
    "pencil",
    "paper",
    "magnet",
    "battery",
    "bottle",
    "camera",
    "mirror",
    "wallet",
    "phone",
]

CATEGORY_KEYS = list(CATEGORIES.keys())
VOCABULARY = list({word for words in CATEGORIES.values() for word in words} | set(DISTRACTOR_BANK))


def build_word_list(category: str, target_count: int, length: int) -> list[str]:
    target_words = CATEGORIES[category]
    target_count = min(target_count, len(target_words))
    targets = random.sample(target_words, k=target_count)

    distractor_pool = [w for w in VOCABULARY if w not in target_words]
    needed = length - target_count
    if needed > len(distractor_pool):
        raise ValueError("Not enough distinct distractors to fill the list")
    distractors = random.sample(distractor_pool, k=needed)

    words = targets + distractors
    random.shuffle(words)
    return words


def generate_example() -> dict:
    category = random.choice(CATEGORY_KEYS)
    length = random.randint(3, 5)
    target_count = random.randint(0, min(length, len(CATEGORIES[category])))
    words = build_word_list(category, target_count, length)
    # Re-compute from the final list to guard against bookkeeping mistakes.
    actual_count = sum(word in CATEGORIES[category] for word in words)
    prompt = (
        "Count the number of words in the following list that match the given type, "
        "and put the numerical answer in parentheses.\n"
        f"Type: {category}\n"
        f"List: [{' '.join(words)}]\n"
        "Answer: ("
    )
    answer = f"({actual_count})"
    return {
        "prompt": prompt,
        "answer": answer,
        "metadata": {
            "type": category,
            "list": words,
            "count": actual_count,
        },
    }


def generate_dataset(num_examples: int, seed: int | None = None) -> list[dict]:
    if seed is not None:
        random.seed(seed)
    return [generate_example() for _ in range(num_examples)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate counting dataset")
    parser.add_argument("--n", type=int, default=5000, help="Number of examples to generate")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/counting_dataset.jsonl"),
        help="Where to write the resulting JSONL dataset",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    dataset = generate_dataset(args.n, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for example in dataset:
            fh.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
