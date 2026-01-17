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


