import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from typing import Literal

def setup_trainer(
    lr: float = 1e-4, 
    weight_decay: float = 1e-4, 
    device: torch.device = torch.device('cpu'), 
    scheduler_mode: Literal['min', 'max']='min', 
    class_weights: torch.Tensor | None = None
)-> tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LRScheduler, nn.Module]:
    '''Config a resnet18 model, and return it. Can be replace by more complex function later.'''
    resnet_model = models.resnet18(weights='DEFAULT')
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2)) # type: ignore
    resnet_model = resnet_model.to(device)
    optimizer = optim.AdamW(resnet_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.5, patience=4)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    return (resnet_model, optimizer, scheduler, criterion)