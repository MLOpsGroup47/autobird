from pathlib import Path

import torch
import typer
from call_of_func.dataclasses.pathing import PathConfig
from call_of_func.dataclasses.training import hyperparameter
from call_of_func.utils.get_configs import _load_cfg
from torch.utils.data import DataLoader, TensorDataset

#cfg = PathConfig()


def build_dataloader(cfg: str = typer.Option("config"), test: bool = False):


    cfg = _load_cfg(config_name=cfg)

    paths = PathConfig(
        root=Path(cfg.paths.root),
        raw_dir=Path(cfg.paths.raw_dir),
        processed_dir=Path(cfg.paths.processed_dir),
        reports_dir=Path(cfg.paths.reports_dir),
        ckpt_dir=Path(cfg.paths.ckpt_dir),
        x_train=Path(cfg.paths.x_train),
        y_train=Path(cfg.paths.y_train),
        x_val= Path(cfg.paths.x_val),
        y_val=Path(cfg.paths.y_val),
    )
    hp = hyperparameter()

    x_train = torch.load(paths.x_train)
    x_val   = torch.load(paths.x_val)
    y_train = torch.load(paths.y_train)
    y_val   = torch.load(paths.y_val)


    # add test
    #if test:
    #    x_test  = torch.load(paths.x_test)
    #    y_test  = torch.load(paths.y_test)
    #    test_ds = torch.TensorDataset(x_test, y_test)
    #val_loader = DataLoader(
    #    test_df, 
    #    batch_size=hp.batch_size, 
    #    shuffle=hp.shuffle_val, 
    #    num_workers=hp.num_workers, 
    #    pin_memory=hp.pin_memory,
    #)


    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val,y_val)

    train_loader = DataLoader(
        train_ds, 
        batch_size=hp.batch_size, 
        shuffle=hp.shuffle_train, 
        num_workers=hp.num_workers, 
        pin_memory=hp.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=hp.batch_size, 
        shuffle=hp.shuffle_val, 
        num_workers=hp.num_workers, 
        pin_memory=hp.pin_memory,
    )

    return train_loader, val_loader