# Copyright 2025 The ACE‑Step Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ComfyUI custom nodes for training Low‑Rank Adaptation (LoRA) adapters with
ACE‑Step models.

This module exposes two nodes:

* ``AceStepDatasetConverter`` converts a directory of MP3 files along
  with ``*_prompt.txt`` and ``*_lyrics.txt`` metadata into a
  HuggingFace dataset on disk【647056671281071†L2-L31】.  The logic mirrors
  the `convert2hf_dataset.py` script from the ACE‑Step repository【864267642857010†L4-L38】.

* ``AceStepLoRaTrainer`` launches the bundled `trainer.py` script to
  fine‑tune LoRA adapters for ACE‑Step.  All of the hyper‑parameters
  documented in ``TRAIN_INSTRUCTION.md`` are exposed as node inputs
  so that you can configure training from within ComfyUI【647056671281071†L139-L193】.

Both nodes follow the conventions required by ComfyUI: a class
variable ``CATEGORY`` specifying where the node appears in the UI, an
``INPUT_TYPES`` class method describing input widgets, a
``RETURN_TYPES`` tuple naming the return value types, and a
``FUNCTION`` string indicating which method implements the node's
behaviour【529967675490930†L165-L171】.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


class AceStepDatasetConverter:
    """
    Convert a directory of MP3 files and corresponding prompt/lyrics files
    into a HuggingFace dataset on disk.

    The converter expects a directory where each audio sample contains three
    files: ``filename.mp3``, ``filename_prompt.txt`` and
    ``filename_lyrics.txt``【647056671281071†L2-L31】.  The ``_prompt.txt``
    files should contain comma‑separated audio tags describing the track,
    and the ``_lyrics.txt`` files should contain the song lyrics.  The
    resulting dataset will be repeated ``repeat_count`` times and saved to
    ``output_name`` as a HuggingFace dataset【864267642857010†L4-L38】.
    """

    CATEGORY = "ACE‑Step/Training"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        """
        Define the inputs for the dataset conversion node.

        * ``data_dir`` – a string path to the directory containing audio data.
        * ``repeat_count`` – an integer specifying how many times to repeat the
          dataset.  Repeating small datasets helps balance the number of
          training steps.
        * ``output_name`` – the directory name where the dataset will be
          written.

        All inputs are required.
        """
        return {
            "required": {
                "data_dir": (
                    "STRING",
                    {
                        "default": "./data",
                        "placeholder": "Path to directory of MP3, prompt and lyrics files",
                    },
                ),
                "repeat_count": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 1,
                        "max": 1_000_000,
                        "step": 1,
                    },
                ),
                "output_name": (
                    "STRING",
                    {
                        "default": "zh_lora_dataset",
                        "placeholder": "Name of the output HuggingFace dataset directory",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"

    def convert(self, data_dir: str, repeat_count: int, output_name: str) -> Tuple[str]:
        """
        Perform dataset conversion.  This method walks the ``data_dir``
        directory looking for MP3 files and their associated prompt/lyrics
        files, builds a list of examples, repeats the list ``repeat_count``
        times, and writes a HuggingFace dataset to ``output_name``【864267642857010†L4-L38】.

        Parameters
        ----------
        data_dir:
            Directory containing ``*.mp3``, ``*_prompt.txt`` and
            ``*_lyrics.txt`` files.
        repeat_count:
            Number of times to repeat the dataset.
        output_name:
            Name of the output directory for the HuggingFace dataset.

        Returns
        -------
        tuple[str]:
            A tuple containing the path to the created dataset.
        """
        # Import datasets lazily to avoid pulling in the dependency if the
        # converter node is never executed.
        from datasets import Dataset  # type: ignore

        data_path = Path(data_dir)
        if not data_path.exists() or not data_path.is_dir():
            raise ValueError(
                f"Data directory '{data_dir}' does not exist or is not a directory."
            )

        all_examples = []
        for song_path in data_path.glob("*.mp3"):
            prompt_path = song_path.with_suffix("").as_posix() + "_prompt.txt"
            lyric_path = song_path.with_suffix("").as_posix() + "_lyrics.txt"
            if not os.path.exists(prompt_path) or not os.path.exists(lyric_path):
                continue
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            with open(lyric_path, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()

            example = {
                "keys": song_path.stem,
                "filename": str(song_path),
                "tags": [tag.strip() for tag in prompt.split(",")],
                "speaker_emb_path": "",
                "norm_lyrics": lyrics,
                "recaption": {},
            }
            all_examples.append(example)

        if not all_examples:
            raise ValueError(
                f"No valid samples found in '{data_dir}'. "
                "Ensure files follow the naming convention described in the training documentation."
            )

        dataset = Dataset.from_list(all_examples * int(repeat_count))
        dataset.save_to_disk(output_name)
        return (output_name,)


class AceStepLoRaTrainer:
    """
    Train a Low‑Rank Adaptation (LoRA) adapter for ACE‑Step.

    This node wraps the ``trainer.py`` script included in this package
    and exposes its many command‑line arguments as inputs so that
    training can be configured from within a ComfyUI workflow【647056671281071†L139-L193】.
    The training parameters mirror those described in the ACE‑Step
    training instructions.

    When executed, this node launches the trainer as a subprocess.
    Training progress and checkpoints will be written to the directories
    specified by ``logger_dir`` and ``checkpoint_dir``.  The return
    value is a simple string indicating completion; if an error occurs
    during training it will be raised so that ComfyUI can report it to
    the user.
    """

    CATEGORY = "ACE‑Step/Training"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        """
        Define the required and optional inputs for the LoRA trainer node.
        Each entry corresponds to a command‑line argument of ``trainer.py``
        described in the ACE‑Step documentation【647056671281071†L139-L193】.
        """
        return {
            "required": {
                "dataset_path": (
                    "STRING",
                    {
                        "default": "./zh_lora_dataset",
                        "placeholder": "HuggingFace dataset directory",
                    },
                ),
                "lora_config_path": (
                    "STRING",
                    {
                        "default": "config/zh_rap_lora_config.json",
                        "placeholder": "Path to LoRA config JSON",
                    },
                ),
                "exp_name": (
                    "STRING",
                    {
                        "default": "ace_step_lora",
                        "placeholder": "Experiment name",
                    },
                ),
                "trainer_script": (
                    "STRING",
                    {
                        "default": "trainer.py",
                        "placeholder": "Path to ACE‑Step trainer.py",
                    },
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 1e-4,
                        "min": 1e-6,
                        "max": 1e-1,
                        "step": 1e-6,
                    },
                ),
                "num_workers": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 256,
                    },
                ),
                "epochs": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 10_000,
                    },
                ),
                "max_steps": (
                    "INT",
                    {
                        "default": 2_000_000,
                        "min": 1,
                        "max": 10_000_000,
                    },
                ),
                "every_n_train_steps": (
                    "INT",
                    {
                        "default": 2_000,
                        "min": 1,
                        "max": 1_000_000,
                    },
                ),
            },
            "optional": {
                "num_nodes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                    },
                ),
                "shift": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                    },
                ),
                "precision": (
                    (["32", "16", "bf16"],),
                    {
                        "default": "32",
                        "tooltip": "Floating‑point precision. 32 for FP32, 16 for FP16 or bf16",
                    },
                ),
                "accumulate_grad_batches": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                    },
                ),
                "gradient_clip_val": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                    },
                ),
                "gradient_clip_algorithm": (
                    (["norm", "value"],),
                    {
                        "default": "norm",
                    },
                ),
                "devices": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                    },
                ),
                "logger_dir": (
                    "STRING",
                    {
                        "default": "./exps/logs/",
                        "placeholder": "Directory to write training logs",
                    },
                ),
                "ckpt_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Resume checkpoint path (optional)",
                    },
                ),
                "checkpoint_dir": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Directory to save new checkpoints",
                    },
                ),
                "reload_dataloaders_every_n_epochs": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "every_plot_step": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 1,
                        "max": 10_000_000,
                    },
                ),
                "val_check_interval": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1_000_000,
                        "tooltip": "Number of steps between validations (0 disables periodic validation)",
                    },
                ),
                "adapter_name": (
                    "STRING",
                    {
                        "default": "lora_adapter",
                        "placeholder": "Name of the new LoRA adapter",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "train_lora"

    def train_lora(
        self,
        dataset_path: str,
        lora_config_path: str,
        exp_name: str,
        trainer_script: str,
        learning_rate: float,
        num_workers: int,
        epochs: int,
        max_steps: int,
        every_n_train_steps: int,
        num_nodes: int = 1,
        shift: float = 3.0,
        precision: str = "32",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float = 0.5,
        gradient_clip_algorithm: str = "norm",
        devices: int = 1,
        logger_dir: str = "./exps/logs/",
        ckpt_path: str = "",
        checkpoint_dir: str = "",
        reload_dataloaders_every_n_epochs: int = 1,
        every_plot_step: int = 2000,
        val_check_interval: int = 0,
        adapter_name: str = "lora_adapter",
    ) -> Tuple[str]:
        """
        Launch the ACE‑Step LoRA trainer as a subprocess.  All parameters map
        directly onto the command‑line flags of ``trainer.py``【647056671281071†L139-L193】.

        Returns
        -------
        tuple[str]:
            A tuple containing a status message indicating where logs or
            checkpoints have been written.
        """
        cmd = ["python", trainer_script]
        cmd += ["--num_nodes", str(num_nodes)]
        cmd += ["--shift", str(shift)]
        cmd += ["--learning_rate", str(learning_rate)]
        cmd += ["--num_workers", str(num_workers)]
        cmd += ["--epochs", str(epochs)]
        cmd += ["--max_steps", str(max_steps)]
        cmd += ["--every_n_train_steps", str(every_n_train_steps)]
        cmd += ["--dataset_path", dataset_path]
        cmd += ["--exp_name", exp_name]
        cmd += ["--precision", precision]
        cmd += ["--accumulate_grad_batches", str(accumulate_grad_batches)]
        cmd += ["--gradient_clip_val", str(gradient_clip_val)]
        cmd += ["--gradient_clip_algorithm", gradient_clip_algorithm]
        cmd += ["--devices", str(devices)]
        cmd += ["--logger_dir", logger_dir]
        if ckpt_path:
            cmd += ["--ckpt_path", ckpt_path]
        if checkpoint_dir:
            cmd += ["--checkpoint_dir", checkpoint_dir]
        cmd += ["--reload_dataloaders_every_n_epochs", str(reload_dataloaders_every_n_epochs)]
        cmd += ["--every_plot_step", str(every_plot_step)]
        if val_check_interval and int(val_check_interval) > 0:
            cmd += ["--val_check_interval", str(val_check_interval)]
        cmd += ["--lora_config_path", lora_config_path]
        cmd += ["--adapter_name", adapter_name]

        # Resolve the trainer script path.  If the provided path is
        # relative and does not exist, attempt to find it alongside this
        # nodes file.  This allows packaging the script within this
        # module.
        script_path = Path(trainer_script)
        if not script_path.exists():
            candidate = Path(__file__).parent / trainer_script
            if candidate.exists():
                script_path = candidate
            else:
                raise FileNotFoundError(
                    f"Could not find trainer script at '{trainer_script}'. "
                    "Provide an absolute path or place the script alongside this node file."
                )
        cmd[1] = str(script_path)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        for stdout_line in process.stdout:
            stdout_lines.append(stdout_line.rstrip())
        _, remaining_err = process.communicate()
        if remaining_err:
            stderr_lines.append(remaining_err)
        return_code = process.returncode
        if return_code != 0:
            error_message = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
            raise RuntimeError(
                f"ACE‑Step training failed with exit code {return_code}.\n{error_message}"
            )

        status = (
            f"ACE‑Step LoRA training complete.  Logs written to '{logger_dir}'. "
            f"Checkpoints saved to '{checkpoint_dir or 'default checkpoint directory'}'."
        )
        return (status,)


# Register nodes with ComfyUI
NODE_CLASS_MAPPINGS = {
    "ACE‑Step Dataset Converter": AceStepDatasetConverter,
    "ACE‑Step LoRA Trainer": AceStepLoRaTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE‑Step Dataset Converter": "ACE‑Step: Create Dataset",
    "ACE‑Step LoRA Trainer": "ACE‑Step: Train LoRA",
}

__all__ = [
    "AceStepDatasetConverter",
    "AceStepLoRaTrainer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]