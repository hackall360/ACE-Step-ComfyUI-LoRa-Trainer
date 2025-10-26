"""Utility helpers for working with ACE-Step training datasets.

The ACE-Step LoRA training instructions describe a lightweight on-disk
dataset format where each audio sample is stored as three files that
share a common stem:

``sample.mp3``
    The raw audio used for training.

``sample_prompt.txt``
    A comma separated list of tags describing the track.  These tags are
    fed into the text encoder during training.

``sample_lyrics.txt``
    The normalised lyrics for the track.  These are used by the lyric
    encoder.

This module centralises the logic for scanning directories that follow
this convention so that both the standalone CLI utility and the ComfyUI
nodes behave identically.  Keeping the parsing logic in one place makes
it easier to stay aligned with updates to the official training
documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class DatasetExample:
    """Represents a single ACE-Step training example.

    The fields mirror what the official ``convert2hf_dataset.py`` script
    writes into the HuggingFace dataset.  Using a dataclass simplifies
    testing and keeps the code readable.
    """

    keys: str
    filename: str
    tags: List[str]
    speaker_emb_path: str
    norm_lyrics: str
    recaption: Dict[str, str]


def _normalise_tags(raw: str) -> List[str]:
    """Normalise a comma separated tag string into a list.

    The ACE-Step guide recommends comma separated descriptors.  The
    formatting in the wild is not always consistent so we strip whitespace
    and ignore empty tags to make the behaviour forgiving.
    """

    tags = [tag.strip() for tag in raw.split(",")]
    return [tag for tag in tags if tag]


def collect_examples(data_dir: Path) -> List[DatasetExample]:
    """Scan ``data_dir`` for ACE-Step training triples.

    Parameters
    ----------
    data_dir:
        Directory that should contain ``*.mp3`` files and the associated
        ``*_prompt.txt`` and ``*_lyrics.txt`` files.

    Returns
    -------
    list[DatasetExample]
        Ordered list of discovered samples.  The order is deterministic so
        that repeated conversions yield the same dataset.
    """

    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(
            f"Data directory '{data_dir}' does not exist or is not a directory."
        )

    examples: List[DatasetExample] = []
    for song_path in sorted(data_dir.glob("*.mp3")):
        base = song_path.with_suffix("")
        prompt_path = base.with_name(base.name + "_prompt.txt")
        lyric_path = base.with_name(base.name + "_lyrics.txt")

        if not prompt_path.exists() or not lyric_path.exists():
            # Skip incomplete triples – the official tooling does the same.
            continue

        prompt = prompt_path.read_text(encoding="utf-8").strip()
        lyrics = lyric_path.read_text(encoding="utf-8").strip()

        example = DatasetExample(
            keys=song_path.stem,
            filename=str(song_path.resolve()),
            tags=_normalise_tags(prompt),
            speaker_emb_path="",
            norm_lyrics=lyrics,
            recaption={},
        )
        examples.append(example)

    return examples


def repeat_examples(examples: Iterable[DatasetExample], repeat_count: int) -> List[Dict[str, object]]:
    """Repeat the example list ``repeat_count`` times and convert to dicts.

    HuggingFace's :func:`datasets.Dataset.from_list` expects a list of
    dictionaries.  Returning dictionaries here avoids duplicating the
    conversion logic across callers.
    """

    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")

    materialised = list(examples)
    if not materialised:
        raise ValueError(
            "No valid audio samples were found. Ensure your files follow the "
            "'<name>.mp3', '<name>_prompt.txt', '<name>_lyrics.txt' convention."
        )

    payload: List[Dict[str, object]] = []
    for _ in range(int(repeat_count)):
        for example in materialised:
            payload.append(example.__dict__)
    return payload


__all__ = ["DatasetExample", "collect_examples", "repeat_examples"]

