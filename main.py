#from loader import data_setup
from dataset import ResNetDataset
from model import setup_trainer
from train import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
from utils import seed_everything, seed_worker
import albumentations as A
import torch
import time
import pandas as pd

#Hyperparameters
num_epochs = 12
batch_size = 128
num_workers = 4
global_seed = 42
early_stop_patience = 4
lr = 1e-4
weight_decay = 1e-4

#fixed parameters
featurename = 'patientId'
labelname = 'Target'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#legacy code, do not use
#label_sheet, img_dir = data_setup(env='colab', data_type='competition', kaggle_dir='rsna-pneumonia-detection-challenge', df_dir='stage_2_train_labels.csv', img_dir='stage_2_train_images')

seed_everything(global_seed)
g = torch.Generator()
g.manual_seed(global_seed)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.5),
], seed=global_seed)
val_transform = None

label_sheet = pd.read_csv('../data/rsna/stage_2_train_labels.csv')
img_dir = Path('../data/rsna/stage_2_train_images')
df_unique = label_sheet.groupby(featurename, as_index=False)[labelname].max()

train, validation = train_test_split(df_unique, random_state=42, stratify=df_unique[labelname])

train_dataset = ResNetDataset(df=train, image_dir=img_dir, featurename=featurename, labelname=labelname, dcm=True, grey=True, transform=train_transform)
val_dataset = ResNetDataset(df=validation, image_dir=img_dir, featurename=featurename, labelname=labelname, dcm=True, grey=True, transform=val_transform)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True,  worker_init_fn=seed_worker)

N_total = len(train)
N_0 = len(train[train['Target']==0])
N_1 = len(train[train['Target']==1])

C = 2
W_0 = N_total / (C * N_0)
W_1 = N_total / (C * N_1)

print(f"Computed class weights -> Class 0: {W_0:.4f}, Class 1: {W_1:.4f}")

class_weights = torch.tensor([W_0, W_1], dtype = torch.float32).to(device)

model, optimizer, scheduler, criterion = setup_trainer(device=device, class_weights=class_weights, lr=lr, weight_decay=weight_decay)

print('Start training……')

best_val_acc = 0.0
best_val_recall = 0.0
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
    
    val_dict = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device)
    val_loss = val_dict['val_loss']
    val_accuracy = val_dict['val_accuracy'] * 100
    val_recall = val_dict['val_recall'] * 100
    
    epoch_time = time.time() - start_time

    print(f'Epoch [{epoch+1}/{num_epochs}] | ',
          f'Time taken: {epoch_time:.1f}s |',
          f'Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ',
          f'Validation Accuracy: {val_accuracy:.2f}% | Validation Recall: {val_recall:.2f}%'
         )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New record! Saving weights. Val Loss: {val_loss:.4f} | Validation Recall: {val_recall:.2f}%')
    else:
        patience_counter += 1
        print(f'No Val Loss improvement, early stop counter : {patience_counter}/{early_stop_patience}')
        if patience_counter >= early_stop_patience:
            print("Early Stopping.")
            break
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning rate: {current_lr}")
    
model.load_state_dict(torch.load('best_model.pth'))
print(f'Traning complete. Loaded best record. Accuracy: {best_val_acc:.2f}%.')