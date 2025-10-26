# This utility mirrors the original convert2hf_dataset.py from ACE‑Step.
# It converts a folder of MP3 audio files and their associated
# "_prompt.txt" and "_lyrics.txt" files into a HuggingFace dataset
# saved to disk.  It is provided here so that users can run the
# conversion independently of ComfyUI if desired.

import argparse
from pathlib import Path

from datasets import Dataset

try:  # pragma: no cover - defensive import for standalone execution
    from .dataset_utils import collect_examples, repeat_examples
except ImportError:  # pragma: no cover
    from dataset_utils import collect_examples, repeat_examples


def create_dataset(
    data_dir: str = "./data",
    repeat_count: int = 2000,
    output_name: str = "zh_lora_dataset",
):
    """Create a HuggingFace dataset that mirrors the official tooling."""

    examples = collect_examples(Path(data_dir))
    payload = repeat_examples(examples, repeat_count)
    ds = Dataset.from_list(payload)
    ds.save_to_disk(output_name)

def main():
    parser = argparse.ArgumentParser(description="Create a dataset from audio files.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing the audio files.")
    parser.add_argument("--repeat_count", type=int, default=1, help="Number of times to repeat the dataset.")
    parser.add_argument("--output_name", type=str, default="zh_lora_dataset", help="Name of the output dataset.")
    args = parser.parse_args()
    create_dataset(data_dir=args.data_dir, repeat_count=args.repeat_count, output_name=args.output_name)

if __name__ == "__main__":
    main()
