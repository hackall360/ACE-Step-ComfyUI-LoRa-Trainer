# ACE‑Step LoRA Trainer for ComfyUI

This repository packages a set of **ComfyUI custom nodes** that make it easy
to train [LoRA](https://arxiv.org/abs/2106.09685) adapters for the
[ACE‑Step](https://ace-step.github.io/) music generation model directly
inside your ComfyUI workflows.  The nodes wrap the official data
preparation and training scripts from the ACE‑Step project and expose
their parameters through the ComfyUI interface.  By bundling the
trainer and dataset converter into a single package, you can
conveniently install the nodes via their Git URL and avoid manual
configuration.

## Features

* **Dataset Converter** – Point the node at a directory of
  `.mp3` files with accompanying `_prompt.txt` and `_lyrics.txt` files
  and it will create a HuggingFace dataset on disk.  The behaviour
  matches the official `convert2hf_dataset.py` utility.

* **LoRA Trainer** – Launches the ACE‑Step `trainer.py` script with
  configurable hyper‑parameters, precision settings and logging
  options.  All of the command‑line arguments documented in the
  official training instructions are exposed as node inputs.

* **Bundled Scripts** – The actual training script (`trainer.py`) and
  dataset conversion utility (`convert2hf_dataset.py`) are included in
  this package so you don't need to clone the ACE‑Step repository
  separately.

## Installation

You can install these nodes into your ComfyUI environment by cloning
the repository into your `custom_nodes` directory or by using the
ComfyUI manager to install via Git URL.  For example:

```bash
cd <path‑to‑ComfyUI>/custom_nodes
git clone https://github.com/your‑org/ace-step-lora-trainer.git
```

Restart ComfyUI and open the node search.  You should see two new
nodes under **ACE‑Step/Training**: *ACE‑Step: Create Dataset* and
*ACE‑Step: Train LoRA*.

## Usage

1. **Prepare your data**: Collect a set of MP3 files and create
   corresponding `*_prompt.txt` and `*_lyrics.txt` files for each
   audio file.  Place all three files in the same directory.  See the
   official training guide for details on the file format and
   naming conventions【647056671281071†L2-L31】.

2. **Create a dataset**: Add the *ACE‑Step: Create Dataset* node to
   your graph.  Specify the directory containing your audio files,
   choose how many times to repeat the dataset (this is helpful for
   small datasets) and the name of the output directory where the
   dataset will be saved.  Run the node to produce a HuggingFace
   dataset on disk【864267642857010†L4-L38】.

3. **Train a LoRA adapter**: Add the *ACE‑Step: Train LoRA* node.
   Provide the path to the dataset created in the previous step, a
   LoRA configuration JSON, an experiment name and any other
   hyper‑parameters you wish to adjust.  When executed, the node will
   run the training loop and write logs and checkpoints to the
   specified directories【647056671281071†L139-L193】.

For more information about the available hyper‑parameters, see the
`TRAIN_INSTRUCTION.md` file in the ACE‑Step repository.

## License

This project is licensed under the Apache License 2.0.  See the
`LICENSE` file for the full license text.