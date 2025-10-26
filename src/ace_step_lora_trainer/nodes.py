# Copyright 2025 The ACE-Step Authors.
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

"""ComfyUI custom nodes that mirror the ACE-Step LoRA training workflow.

The official ACE-Step LoRA training guide explains how to prepare data,
configure LoRA adapters and run the training loop【647056671281071†L2-L193】.
This module provides dedicated nodes for each step so that the entire
process can be orchestrated inside a ComfyUI workflow.  Every node follows
ComfyUI's conventions – ``CATEGORY`` determines where the node appears in
the UI, ``INPUT_TYPES`` describes the widgets, ``RETURN_TYPES`` names the
outputs and ``FUNCTION`` points at the implementation method【529967675490930†L165-L171】.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from safetensors.torch import save_file as save_safetensors

from .dataset_utils import collect_examples, repeat_examples


class AceStepDatasetConverter:
    """Convert raw ACE-Step training triples into a HuggingFace dataset."""

    CATEGORY = "ACE-Step/Data"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "data_dir": (
                    "STRING",
                    {
                        "default": "./data",
                        "placeholder": "Directory of *.mp3 + _prompt/_lyrics files",
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
                        "default": "./zh_lora_dataset",
                        "placeholder": "Destination directory for the dataset",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"

    def convert(self, data_dir: str, repeat_count: int, output_name: str) -> Tuple[str]:
        data_path = Path(data_dir)
        examples = collect_examples(data_path)
        payload = repeat_examples(examples, repeat_count)
        dataset = Dataset.from_list(payload)
        dataset.save_to_disk(output_name)
        return (str(Path(output_name).resolve()),)


class AceStepDatasetUpdater:
    """Append new samples to an existing ACE-Step dataset on disk."""

    CATEGORY = "ACE-Step/Data"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "dataset_path": (
                    "STRING",
                    {
                        "default": "./zh_lora_dataset",
                        "placeholder": "Existing HuggingFace dataset path",
                    },
                ),
                "new_data_dir": (
                    "STRING",
                    {
                        "default": "./new_data",
                        "placeholder": "Directory containing additional MP3/prompt/lyric triples",
                    },
                ),
                "repeat_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1_000_000,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "create_backup": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Copy the dataset before applying updates",
                    },
                ),
                "backup_dir": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional directory for the backup copy",
                    },
                ),
                "overwrite_backup": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Replace an existing backup directory if it exists",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    FUNCTION = "update_dataset"

    def update_dataset(
        self,
        dataset_path: str,
        new_data_dir: str,
        repeat_count: int,
        create_backup: bool = True,
        backup_dir: str = "",
        overwrite_backup: bool = False,
    ) -> Tuple[str, int]:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset path '{dataset_path}' does not exist. Run the converter first."
            )

        if create_backup:
            backup_path = Path(backup_dir) if backup_dir else dataset_dir.with_name(dataset_dir.name + ".bak")
            if backup_path.exists():
                if not overwrite_backup:
                    raise FileExistsError(
                        f"Backup directory '{backup_path}' already exists. Enable overwrite to replace it."
                    )
                shutil.rmtree(backup_path)
            shutil.copytree(dataset_dir, backup_path)

        existing_dataset = load_from_disk(str(dataset_dir))
        new_examples = collect_examples(Path(new_data_dir))
        payload = repeat_examples(new_examples, repeat_count)
        additional = Dataset.from_list(payload)
        merged = concatenate_datasets([existing_dataset, additional])
        merged.save_to_disk(str(dataset_dir))
        return (str(dataset_dir.resolve()), len(payload))


class AceStepDatasetArchiver:
    """Create a tar archive of a HuggingFace dataset directory."""

    CATEGORY = "ACE-Step/Data"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "dataset_path": (
                    "STRING",
                    {
                        "default": "./zh_lora_dataset",
                        "placeholder": "Dataset directory to archive",
                    },
                ),
                "archive_path": (
                    "STRING",
                    {
                        "default": "./archives/zh_lora_dataset.tar.gz",
                        "placeholder": "Destination archive path",
                    },
                ),
            },
            "optional": {
                "compression": (
                    (["gz", "bz2", "xz", "none"],),
                    {
                        "default": "gz",
                        "tooltip": "Compression algorithm for the tar archive",
                    },
                ),
                "create_parents": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Create parent directories if they do not exist",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Replace the archive if it already exists",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "archive_dataset"

    def archive_dataset(
        self,
        dataset_path: str,
        archive_path: str,
        compression: str = "gz",
        create_parents: bool = True,
        overwrite: bool = False,
    ) -> Tuple[str]:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist")

        archive_file = Path(archive_path)
        if archive_file.exists():
            if overwrite:
                archive_file.unlink()
            else:
                raise FileExistsError(
                    f"Archive '{archive_file}' already exists. Enable overwrite to replace it."
                )

        if create_parents:
            archive_file.parent.mkdir(parents=True, exist_ok=True)

        mode = {
            "gz": "w:gz",
            "bz2": "w:bz2",
            "xz": "w:xz",
            "none": "w",
        }[compression]

        with tarfile.open(archive_file, mode) as tar:
            tar.add(dataset_dir, arcname=dataset_dir.name)

        return (str(archive_file.resolve()),)


class AceStepLoRaConfigBuilder:
    """Create a PEFT-compatible LoRA configuration JSON file."""

    CATEGORY = "ACE-Step/Config"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "output_path": (
                    "STRING",
                    {
                        "default": "./config/ace_step_lora_config.json",
                        "placeholder": "Location to save the config JSON",
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 2048,
                        "step": 1,
                    },
                ),
                "alpha": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "dropout": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
            "optional": {
                "bias": (
                    (["none", "all", "lora_only"],),
                    {
                        "default": "none",
                        "tooltip": "Bias handling strategy used by PEFT",
                    },
                ),
                "task_type": (
                    ([
                        "AUDIO_GENERATION",
                        "AUDIO_CLASSIFICATION",
                        "SEQ_2_SEQ_LM",
                        "CAUSAL_LM",
                    ],),
                    {
                        "default": "AUDIO_GENERATION",
                        "tooltip": "Task type embedded in the config",
                    },
                ),
                "target_modules": (
                    "STRING",
                    {
                        "default": "to_q,to_k,to_v,to_out.0",
                        "placeholder": "Comma separated module names",
                    },
                ),
                "modules_to_save": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional comma separated modules to keep in full precision",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_config"

    def build_config(
        self,
        output_path: str,
        rank: int,
        alpha: int,
        dropout: float,
        bias: str = "none",
        task_type: str = "AUDIO_GENERATION",
        target_modules: str = "to_q,to_k,to_v,to_out.0",
        modules_to_save: str = "",
    ) -> Tuple[str]:
        config_path = Path(output_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config: Dict[str, Any] = {
            "r": int(rank),
            "lora_alpha": int(alpha),
            "lora_dropout": float(dropout),
            "bias": bias,
            "task_type": task_type,
            "target_modules": [module.strip() for module in target_modules.split(",") if module.strip()],
        }
        modules = [module.strip() for module in modules_to_save.split(",") if module.strip()]
        if modules:
            config["modules_to_save"] = modules

        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return (str(config_path.resolve()),)


class AceStepLoRaTrainer:
    """Launch the bundled ACE-Step ``trainer.py`` script."""

    CATEGORY = "ACE-Step/Training"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
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
                        "placeholder": "Experiment name (used for logging)",
                    },
                ),
                "trainer_script": (
                    "STRING",
                    {
                        "default": "trainer.py",
                        "placeholder": "Path to ACE-Step trainer.py",
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
                        "step": 1,
                    },
                ),
                "epochs": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 10_000,
                        "step": 1,
                    },
                ),
                "max_steps": (
                    "INT",
                    {
                        "default": 2_000_000,
                        "min": 1,
                        "max": 10_000_000,
                        "step": 1,
                    },
                ),
                "every_n_train_steps": (
                    "INT",
                    {
                        "default": 2_000,
                        "min": 1,
                        "max": 1_000_000,
                        "step": 1,
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
                        "step": 1,
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
                        "tooltip": "Floating point precision for PyTorch Lightning",
                    },
                ),
                "accumulate_grad_batches": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 256,
                        "step": 1,
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
                        "step": 1,
                    },
                ),
                "logger_dir": (
                    "STRING",
                    {
                        "default": "./exps/logs/",
                        "placeholder": "Directory to write TensorBoard logs and checkpoints",
                    },
                ),
                "ckpt_path": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Optional checkpoint to resume from",
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
                        "step": 1,
                    },
                ),
                "every_plot_step": (
                    "INT",
                    {
                        "default": 2000,
                        "min": 1,
                        "max": 10_000_000,
                        "step": 1,
                    },
                ),
                "val_check_interval": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1,
                        "tooltip": "Number of steps between validation runs (0 disables periodic validation)",
                    },
                ),
                "adapter_name": (
                    "STRING",
                    {
                        "default": "lora_adapter",
                        "placeholder": "Name used when saving the adapter",
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
        cmd: List[str] = ["python", trainer_script]
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
        if val_check_interval:
            cmd += ["--val_check_interval", str(val_check_interval)]
        cmd += ["--lora_config_path", lora_config_path]
        cmd += ["--adapter_name", adapter_name]

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

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        for line in process.stdout:
            stdout_lines.append(line.rstrip())
        _, remaining_err = process.communicate()
        if remaining_err:
            stderr_lines.append(remaining_err)

        if process.returncode != 0:
            error_message = "\n".join(stderr_lines) if stderr_lines else "Unknown error"
            raise RuntimeError(
                f"ACE-Step training failed with exit code {process.returncode}.\n{error_message}"
            )

        status = (
            f"ACE-Step LoRA training complete. Logs written to '{logger_dir}'. "
            f"Checkpoints saved to '{checkpoint_dir or 'default checkpoint directory'}'."
        )
        return (status,)


class AceStepLoRaCheckpointExporter:
    """Export the LoRA adapter saved during training for deployment."""

    CATEGORY = "ACE-Step/Training"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        return {
            "required": {
                "checkpoint_dir": (
                    "STRING",
                    {
                        "default": "./exps/logs/latest/checkpoints/epoch=0-step=0_lora",
                        "placeholder": "Directory containing adapter_config.json and adapter_model.bin",
                    },
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": "./exports/ace_step_lora",
                        "placeholder": "Where to copy or convert the adapter",
                    },
                ),
            },
            "optional": {
                "format": (
                    (["peft", "safetensors"],),
                    {
                        "default": "safetensors",
                        "tooltip": "Keep the PEFT .bin weights or convert to safetensors",
                    },
                ),
                "overwrite": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Replace files in the output directory if they already exist",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    FUNCTION = "export_adapter"

    def export_adapter(
        self,
        checkpoint_dir: str,
        output_dir: str,
        format: str = "safetensors",
        overwrite: bool = False,
    ) -> Tuple[str, str]:
        source = Path(checkpoint_dir)
        if not source.exists() or not source.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' does not exist or is not a directory."
            )

        adapter_config = source / "adapter_config.json"
        adapter_weights = source / "adapter_model.bin"
        if not adapter_config.exists() or not adapter_weights.exists():
            raise FileNotFoundError(
                "The checkpoint directory does not contain 'adapter_config.json' and 'adapter_model.bin'. "
                "Ensure you point to the LoRA adapter folder produced during training."
            )

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        config_target = destination / "adapter_config.json"
        if config_target.exists() and not overwrite:
            raise FileExistsError(f"'{config_target}' already exists. Enable overwrite to replace it.")
        shutil.copy2(adapter_config, config_target)

        if format == "peft":
            weights_target = destination / "adapter_model.bin"
            if weights_target.exists() and not overwrite:
                raise FileExistsError(f"'{weights_target}' already exists. Enable overwrite to replace it.")
            shutil.copy2(adapter_weights, weights_target)
        else:
            weights_target = destination / "adapter_model.safetensors"
            if weights_target.exists() and not overwrite:
                raise FileExistsError(
                    f"'{weights_target}' already exists. Enable overwrite to replace it."
                )
            state_dict = torch.load(adapter_weights, map_location="cpu")
            save_safetensors(state_dict, str(weights_target))

        return (str(destination.resolve()), str(weights_target.resolve()))


NODE_CLASS_MAPPINGS = {
    "ACE-Step Dataset Converter": AceStepDatasetConverter,
    "ACE-Step Dataset Updater": AceStepDatasetUpdater,
    "ACE-Step Dataset Archiver": AceStepDatasetArchiver,
    "ACE-Step LoRA Config Builder": AceStepLoRaConfigBuilder,
    "ACE-Step LoRA Trainer": AceStepLoRaTrainer,
    "ACE-Step LoRA Exporter": AceStepLoRaCheckpointExporter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACE-Step Dataset Converter": "ACE-Step: Create Dataset",
    "ACE-Step Dataset Updater": "ACE-Step: Update Dataset",
    "ACE-Step Dataset Archiver": "ACE-Step: Archive Dataset",
    "ACE-Step LoRA Config Builder": "ACE-Step: Build LoRA Config",
    "ACE-Step LoRA Trainer": "ACE-Step: Train LoRA",
    "ACE-Step LoRA Exporter": "ACE-Step: Export LoRA",
}

__all__ = [
    "AceStepDatasetArchiver",
    "AceStepDatasetConverter",
    "AceStepDatasetUpdater",
    "AceStepLoRaCheckpointExporter",
    "AceStepLoRaConfigBuilder",
    "AceStepLoRaTrainer",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]

