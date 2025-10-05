from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CATEGORIES = {
    "fruit": [
        "apple",
        "orange",
        "banana",
        "grape",
        "pear",
        "kiwi",
        "mango",
        "cherry",
        "peach",
        "plum",
        "apricot",
        "lime",
        "melon",
        "papaya",
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
    "color": [
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "brown",
        "white",
        "black",
        "teal",
        "violet",
        "indigo",
        "silver",
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
        "eye",
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
    words = []
    target_words = CATEGORIES[category]
    # place the matching words
    for _ in range(target_count):
        words.append(random.choice(target_words))
    # fill the rest with distractors that are not target words
    distractor_pool = [w for w in VOCABULARY if w not in target_words]
    for _ in range(length - target_count):
        words.append(random.choice(distractor_pool))
    random.shuffle(words)
    return words


def generate_example() -> dict:
    category = random.choice(CATEGORY_KEYS)
    length = random.randint(5, 14)
    target_count = random.randint(0, length)
    if target_count > length:
        target_count = length
    words = build_word_list(category, target_count, length)
    # re-count in case duplicates lead to unexpected matches
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
