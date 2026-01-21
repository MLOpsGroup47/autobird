# Training (`train.py`)

This project uses **Hydra** for training configuration.  
The training entrypoint composes a full config (paths, data, preprocessing, and train settings) and runs the modular training engine.

## Command

### Run training

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

### Change hyperparameters
Examples:
```bash
uvr train hyperparams.hyperparameters.epochs=10
uvr train hyperparams.hyperparameters.lr=0.001
uvr train train.hyperparams.hyperparameters.batch_size=64
uvr train train.hyperparams.hyperparameters.sample_min=100
uvr train train.hyperparams.hyperparameters.amp=false
uvr train train.hyperparams.hyperparameters.grad_clip=0.5
```
