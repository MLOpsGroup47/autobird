from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch


def rm_rare_classes(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    classes: list[str],
    min_samples: int = 20,
):
    # y_* must be torch.long
    y_train = y_train.long()
    y_val = y_val.long()

    # counts per class in TRAIN
    train_counts = torch.bincount(y_train, minlength=len(classes))  # [C]

    keep = (train_counts >= min_samples).nonzero(as_tuple=True)[0]  # tensor of kept class ids
    keep_set = set(keep.tolist())

    print(f"Keeping {len(keep_set)}/{len(classes)} classes with >= {min_samples} train samples")

    # filter train/val to kept classes
    train_mask = torch.tensor([int(int(y) in keep_set) for y in y_train], dtype=torch.bool)
    val_mask = torch.tensor([int(int(y) in keep_set) for y in y_val], dtype=torch.bool)

    x_train2 = x_train[train_mask]
    y_train2 = y_train[train_mask]
    x_val2 = x_val[val_mask]
    y_val2 = y_val[val_mask]

    # remap old labels -> new contiguous labels
    keep_sorted = sorted(keep_set)
    old_to_new = {old: new for new, old in enumerate(keep_sorted)}

    y_train2 = torch.tensor([old_to_new[int(y)] for y in y_train2], dtype=torch.long)
    y_val2 = torch.tensor([old_to_new[int(y)] for y in y_val2], dtype=torch.long)

    new_classes = [classes[old] for old in keep_sorted]

    # stats
    new_counts = torch.bincount(y_train2)
    print("New train min/max:", int(new_counts.min()), int(new_counts.max()))
    print(f"After pruning: train={len(y_train2)}, val={len(y_val2)}, classes={len(new_classes)}")

    return x_train2, y_train2, x_val2, y_val2, new_classes
