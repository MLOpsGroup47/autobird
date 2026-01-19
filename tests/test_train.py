import pytest
import torch
from call_of_birds_autobird.model import Model
from call_of_func.train.train_engine import accuracy
from call_of_func.train.train_helper import rm_rare_classes


def test_accuracy():
    """Test the accuracy calculation."""
    # Batch size 4, 3 classes
    logits = torch.tensor([
        [10.0, 1.0, 0.0],  # Pred: 0
        [1.0, 10.0, 0.0],  # Pred: 1
        [0.0, 1.0, 10.0],  # Pred: 2
        [10.0, 0.0, 0.0],  # Pred: 0
    ])
    
    # All correct
    y_true = torch.tensor([0, 1, 2, 0])
    acc = accuracy(logits, y_true)
    assert acc == 1.0, "Accuracy should be 1.0 for all correct predictions"
    
    # None correct
    y_wrong = torch.tensor([1, 2, 0, 1])
    acc_wrong = accuracy(logits, y_wrong)
    assert acc_wrong == 0.0, "Accuracy should be 0.0 for all incorrect predictions"
    
    # Half correct
    y_half = torch.tensor([0, 0, 2, 1]) # Correct, Wrong, Correct, Wrong
    acc_half = accuracy(logits, y_half)
    assert acc_half == 0.5, "Accuracy should be 0.5 for half correct predictions"


def test_rm_rare_classes():
    """Test removing rare classes and remapping labels."""
    # 3 classes: 0, 1, 2
    classes = ["class0", "class1", "class2"]
    
    # Class 0: 5 samples
    # Class 1: 2 samples (Rare, threshold=3)
    # Class 2: 5 samples
    
    # Create dummy data (12 mels, 10 time steps)
    feature_shape = (1, 12, 10)
    
    # Train set
    y_train = torch.tensor([0]*5 + [1]*2 + [2]*5, dtype=torch.long)
    N_train = len(y_train)
    x_train = torch.randn(N_train, *feature_shape)
    
    # Val set (just some samples)
    y_val = torch.tensor([0, 1, 2], dtype=torch.long)
    N_val = len(y_val)
    x_val = torch.randn(N_val, *feature_shape)
    
    min_samples = 20

    x_train_new, y_train_new, x_val_new, y_val_new, classes_new = rm_rare_classes(
        x_train = x_train, 
        y_train = y_train,
        x_val = x_val,
        y_val = y_val,
        class_names = classes,
        min_samples=min_samples
    )
    
    # Checks:
    # 1. Class 1 should be gone. Only 0 and 2 remain.
    # 2. Labels should be remapped: 0->0, 2->1 (since 1 is removed)
    # 3. Size of train should be 5+5=10
    # 4. Size of val should be 1 (for 0) + 1 (for 2) = 2. (sample for 1 removed)
    
    assert len(y_train_new) == 10, "Train set size incorrect after removing rare classes"
    assert len(y_val_new) == 2, "Val set size incorrect after removing rare classes"
    assert len(classes_new) == 2, "Number of classes incorrect after removing rare classes"
    assert classes_new == ["class0", "class2"], "Class names incorrect after removing rare classes"
    
    # Verify labels in new set are 0 or 1
    assert set(y_train_new.tolist()) == {0, 1}, "Train labels not remapped correctly"
    assert set(y_val_new.tolist()) == {0, 1}, "Val labels not remapped correctly"


def test_single_train_step():
    """Smoke test for a single training step."""
    device = torch.device("cpu") # Test on CPU for CI compatibility
    n_classes = 5
    batch_size = 4
    
    # Initialize components
    model = Model(n_classes=n_classes, d_model=32, n_heads=2, n_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy batch [B, 1, Mels, Time]
    # Model expects [B, 1, Mels, Frames]
    # Mels=64 (default in some configs, but flexible), Time=128
    input_shape = (batch_size, 1, 64, 128)
    x = torch.randn(*input_shape).to(device)
    y = torch.randint(0, n_classes, (batch_size,)).to(device)
    
    # Initial mode
    model.train()
    
    # Forward
    logits = model(x)
    assert logits.shape == (batch_size, n_classes), "Logits shape is incorrect, expected (batch_size, n_classes)"
    
    # Loss
    loss = criterion(logits, y)
    assert not torch.isnan(loss), "Loss is NaN"
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Just checking it runs without error and gradients are populated
    # Not all parameters may receive gradients if they are unused, 
    # but at least some should.
    grads_found = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
             grads_found = True
             break
    assert grads_found, "No gradients were computed!"
