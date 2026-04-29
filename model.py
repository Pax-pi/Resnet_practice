import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.models as models

def setup_trainer(lr: float = 1e-4, weight_decay: float = 1e-4, device: torch.device = torch.device('cpu'), class_weights: torch.Tensor | None = None):
    '''Config a resnet18 model, and return it. Can be replace by more complex function later.'''
    resnet_model = models.resnet18(pretrained = True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_ftrs, 2)) # type: ignore
    resnet_model = resnet_model.to(device)
    optimizer = torch.optim.AdamW(resnet_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    return (resnet_model, optimizer, scheduler, criterion)