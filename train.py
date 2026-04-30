import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device:torch.device) -> float:
    '''Train one epoch for given model.'''
    model.train()
    train_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device:torch.device)-> dict:
    model.eval()
    val_loss = 0.0
    metrics = MetricCollection({
        'accuracy': Accuracy(task='binary'),
        'precision': Precision(task='binary'),
        'recall': Recall(task='binary'),
        'f1': F1Score(task='binary')
    }).to(device)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.long().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            metrics.update(predicted, labels)
    final_metrics = metrics.compute()
    return {
        "val_loss": val_loss / len(dataloader),
        "val_accuracy": final_metrics['accuracy'].item(),
        "val_precision": final_metrics['precision'].item(),
        "val_recall": final_metrics['recall'].item(),
        "val_f1": final_metrics['f1'].item()
    }
    