# ACE‑Step LoRA Trainer for ComfyUI

This repository packages a suite of **ComfyUI custom nodes** that mirror
the official ACE‑Step documentation for training
[LoRA](https://arxiv.org/abs/2106.09685) adapters.  The nodes wrap the
data preparation utilities, training entry points and export steps so
that you can orchestrate the entire workflow inside ComfyUI.

## Features

* **Dataset Converter** – Convert directories of `.mp3` files plus
  `_prompt.txt` and `_lyrics.txt` metadata into on-disk HuggingFace
  datasets, matching the official `convert2hf_dataset.py` utility.
* **Dataset Updater & Archiver** – Append new material to an existing
  dataset while keeping automatic backups, then package the dataset into
  a compressed tarball for safekeeping or sharing.
* **LoRA Config Builder** – Create PEFT-compatible LoRA configuration
  JSON files without leaving ComfyUI.  Configure rank, alpha, dropout
  and target modules interactively.
* **LoRA Trainer** – Launch the bundled ACE‑Step `trainer.py` script
  with every hyper-parameter exposed as node inputs so you can fine-tune
  from scratch or resume from a checkpoint.
* **LoRA Exporter** – Copy the resulting adapter to a deployment folder
  as either raw PEFT weights or a `safetensors` file ready for use in
  inference pipelines.
* **Bundled Scripts** – The training (`trainer.py`) and dataset
  conversion (`convert2hf_dataset.py`) entry points are included so you
  do not need a separate ACE‑Step checkout.

## Installation

You can install these nodes into your ComfyUI environment by cloning
the repository into your `custom_nodes` directory or by using the
ComfyUI manager to install via Git URL.  For example:

```bash
cd <path‑to‑ComfyUI>/custom_nodes
git clone https://github.com/your‑org/ace-step-lora-trainer.git
```

Restart ComfyUI and open the node search.  You should see new entries
under **ACE‑Step/Data**, **ACE‑Step/Config** and **ACE‑Step/Training**
covering the full workflow.

## Usage

1. **Prepare your data**: Collect a set of MP3 files and create
   corresponding `*_prompt.txt` and `*_lyrics.txt` files for each
   audio file.  Place all three files in the same directory.  See the
   official training guide for details on the file format and
   naming conventions【647056671281071†L2-L31】.

2. **Create a dataset**: Add the *ACE‑Step: Create Dataset* node.
   Specify the directory containing your audio files, choose how many
   times to repeat the dataset (useful for small corpora) and the output
   directory.  Execute the node to produce a HuggingFace dataset on
   disk【864267642857010†L4-L38】.

3. **Iterate on the dataset** *(optional)*: Use the *ACE‑Step: Update
   Dataset* node to append newly curated material.  The node can back up
   the original dataset before applying updates.  The *ACE‑Step: Archive
   Dataset* node packages the dataset into a `.tar` file for storage or
   distribution.

4. **Build a LoRA config**: Run the *ACE‑Step: Build LoRA Config* node
   to generate a PEFT configuration JSON with your preferred rank,
   alpha, dropout and module selections.  This mirrors the options from
   the ACE‑Step `TRAIN_INSTRUCTION.md` guide.

5. **Train or resume**: Add the *ACE‑Step: Train LoRA* node.  Point it
   at your dataset and config file, set logging/checkpoint directories
   and tweak the exposed hyper-parameters to match the ACE‑Step
   instructions【647056671281071†L139-L193】.  Provide `ckpt_path` to
   continue training from an existing checkpoint.

6. **Export the adapter**: Once training finishes, run the *ACE‑Step:
   Export LoRA* node on the desired checkpoint directory.  It copies the
   `adapter_config.json` file and either keeps the PEFT `.bin` weights or
   converts them to `adapter_model.safetensors` for ComfyUI inference.

For more information about the available hyper‑parameters, see the
`TRAIN_INSTRUCTION.md` file in the ACE‑Step repository.

## License

This project is licensed under the Apache License 2.0.  See the
`LICENSE` file for the full license text.
