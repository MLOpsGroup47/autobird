from typing import Optional

import torch


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def rm_rare_classes(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    min_samples: int = 20,
    class_names: Optional[list[str]] = None,  # optional
):
    y_train = y_train.long()
    y_val = y_val.long()

    num_classes = int(y_train.max().item()) + 1
    train_counts = torch.bincount(y_train, minlength=num_classes)

    keep = (train_counts >= min_samples).nonzero(as_tuple=True)[0]   # tensor of kept class ids
    keep_sorted = keep.tolist()
    keep_set = set(keep_sorted)

    print(f"Keeping {len(keep_sorted)}/{num_classes} classes with >= {min_samples} train samples")

    # Fast masks
    train_mask = torch.isin(y_train, keep)
    val_mask = torch.isin(y_val, keep)

    x_train2 = x_train[train_mask]
    y_train2 = y_train[train_mask]
    x_val2 = x_val[val_mask]
    y_val2 = y_val[val_mask]

    # Remap old -> new contiguous
    old_to_new = torch.full((num_classes,), -1, dtype=torch.long)
    old_to_new[keep] = torch.arange(len(keep), dtype=torch.long)

    y_train2 = old_to_new[y_train2]
    y_val2 = old_to_new[y_val2]

    # Optional: map names too 
    new_class_names = None
    if class_names is not None:
        new_class_names = [class_names[i] for i in keep_sorted]

    new_counts = torch.bincount(y_train2, minlength=len(keep_sorted))
    print(new_counts.shape)
    #print("New train min/max:", int(new_counts.min().item()), int(new_counts.max().item()))
    print(f"After pruning: train={len(y_train2)}, val={len(y_val2)}, classes={len(keep_sorted)}")

    return x_train2, y_train2, x_val2, y_val2, new_class_names
