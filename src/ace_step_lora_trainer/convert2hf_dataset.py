# This utility mirrors the original convert2hf_dataset.py from ACE‑Step.
# It converts a folder of MP3 audio files and their associated
# "_prompt.txt" and "_lyrics.txt" files into a HuggingFace dataset
# saved to disk.  It is provided here so that users can run the
# conversion independently of ComfyUI if desired.

from datasets import Dataset
from pathlib import Path
import os
import argparse

def create_dataset(data_dir: str = "./data", repeat_count: int = 2000, output_name: str = "zh_lora_dataset"):
    data_path = Path(data_dir)
    all_examples = []
    for song_path in data_path.glob("*.mp3"):
        prompt_path = str(song_path).replace(".mp3", "_prompt.txt")
        lyric_path = str(song_path).replace(".mp3", "_lyrics.txt")
        try:
            assert os.path.exists(prompt_path), f"Prompt file {prompt_path} does not exist."
            assert os.path.exists(lyric_path), f"Lyrics file {lyric_path} does not exist."
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
            keys = song_path.stem
            example = {
                "keys": keys,
                "filename": str(song_path),
                "tags": prompt.split(", "),
                "speaker_emb_path": "",
                "norm_lyrics": lyrics,
                "recaption": {},
            }
            all_examples.append(example)
        except AssertionError:
            continue
    ds = Dataset.from_list(all_examples * repeat_count)
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