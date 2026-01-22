# How to use the `src` code locally

This project uses **Hydra** for configuration management.  
All preprocessing and training steps are driven by configuration files that are composed at runtime.

The training entrypoint combines **paths**, **data**, **preprocessing**, and **training** configs and runs a fully modular pipeline.

---

## Data Preprocessing (`data.py`)

`data.py` runs the preprocessing pipeline for the *Voice of Birds* dataset  
(source: https://www.kaggle.com/code/dima806/bird-species-by-sound-detection/notebook).

The purpose of the preprocessing step is to convert raw audio files into **model-ready tensors** in a reproducible and configurable way.

---

### `data.py`

The preprocessing pipeline consists of the following steps:

1. **Load raw audio**
   - Reads audio files (`.wav`, `.mp3`) from `paths.raw_dir`

2. **Chunking**
   - Long recordings are split into fixed-length clips
   - With optional overlap availeble

3. **Feature extraction**
   - Computes **log-mel spectrograms**

4. **Global normalization**
   - Computes dataset-wide mean and standard deviation
   - Applies normalization to all spectrograms
   - Improves training stability and convergence

5. **Group-aware dataset splitting**
   - Data is split **by recording ID**, not by individual chunks
   - Prevents leakage between train / validation / test sets

6. **Save processed tensors**
   - Outputs are saved to `paths.processed_dir`:
     - `train_x.pt`, `train_y.pt`
     - `val_x.pt`, `val_y.pt`
     - `test_x.pt`, `test_y.pt`
   - Class labels and metadata are stored alongside the tensors

---

### Running preprocessing

Run for repo root
To run the default data preprocessing pipeline use
```bash
uvr data
```
The default configs can be inspected with this command
```bash
uvr data --cfg job
```
### Change parameters
Run experiments with different configs do:
```bash
uvr data preprocessing.sr=15999
uvr data data.train_split=0.7
```
### Default configs 
```yaml
preprocessing:
  sr: 16000
  clip_sec: 5.0
  n_fft: 1024
  hop_length: 512
  n_mels: 64
  fq_min: 20
  fq_max: 8000
  min_rms: 0.005
  min_mel_std: 0.10
  min_samples: 50

data:
  train_split: 0.8
  test_split: 0.1
  seed: 4
  clip_sec: 5.0
  stride_sec: 2.5
  pad_last: true
```

## Training (`train.py`)
`train.py` runs the model in `src/call_of_birds_autobird/model.py` with its training loop defined in `src/call_of_func/train/train_engine.py` and other functions. 

### Commands

#### Run training

Run for repo root
To run the default model use
```bash
uvr train
```
The default configs can be inspected with this command 
```bash
uvr train --cfg job
```

To run experiments with different hyperparameter, optimizers and scheduler modify parameter under train:
Training config lives under the `train` namespace in the composed Hydra config.

#### Change hyperparameters
Examples:
```bash
uvr train train.hp.epochs=100
```
#### Default hyperparameter settings
```yaml
epochs: 3
lr: 0.0003
batch_size: 32
d_model: 64
n_heads: 2
n_layers: 1
pin_memory: false
shuffle_train: true
shuffle_val: false
num_workers: 2
amp: true
grad_clip: 1.0
use_wandb: true
```