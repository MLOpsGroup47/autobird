# How use src code local

This project uses **Hydra** for training configuration.  
The training entrypoint composes a full config (paths, data, preprocessing, and train settings) and runs the modular training engine.

## Data Preprocessing ('data.py')


## Training (`train.py`)


### Commands

#### Run training

To run the default model use
```bash
uvr train
```
The default setting can be inspected with this command 
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
```bash
.epochs: 3
.lr: 0.0003
.batch_size: 32
.sample_min: 50
.d_model: 64
.n_heads: 2
.n_layers: 1
.pin_memory: false
.shuffle_train: true
.shuffle_val: false
.num_workers: 2
.amp: true
.grad_clip: 1.0
.use_wandb: true
```

