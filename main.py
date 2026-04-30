from loader import data_setup
from dataset import ResNetDataset
from model import setup_trainer
from train import train_one_epoch, evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import time

num_epochs = 12
featurename = 'patientId'
labelname = 'Target'
batch_size = 128
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_sheet, img_dir = data_setup(env='colab', data_type='competition', kaggle_dir='rsna-pneumonia-detection-challenge', df_dir='stage_2_train_labels.csv', img_dir='stage_2_train_images')

df_unique = label_sheet.groupby(featurename, as_index=False)[labelname].max()

train, validation = train_test_split(df_unique, random_state=42, stratify=df_unique[labelname])

train_dataset = ResNetDataset(df=train, image_dir=img_dir, featurename=featurename, labelname=labelname, dcm=True)
val_dataset = ResNetDataset(df=validation, image_dir=img_dir, featurename=featurename, labelname=labelname, dcm=True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers, pin_memory=True)

model, optimizer, scheduler, criterion = setup_trainer(device=device)

best_val_acc = 0.0
best_val_recall = 0.0

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss = train_one_epoch(model=model, dataloader=train_loader, optimizer=optimizer, criterion=criterion, device=device)
    
    val_dict = evaluate(model=model, dataloader=val_loader, criterion=criterion, device=device)
    val_loss = val_dict['val_loss']
    val_accuracy = val_dict['val_accuracy']
    val_recall = val_dict['val_recall']
    
    epoch_time = time.time() - start_time

    print(f'Epoch [{epoch+1}/{num_epochs}] | ',
          f'Time taken: {epoch_time:.1f}s |',
          f'Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | ',
          f'Validation Accuracy: {val_accuracy:.2f}% | Validation Recall: {val_recall:.2f}%'
         )
    if val_accuracy > 75.0:
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New Record! Saving weights | accuracy: {val_accuracy:.2f}% | recall: {val_recall:.2f}%')
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning rate: {current_lr}")
    
model.load_state_dict(torch.load('best_model.pth'))
print(f'Traning complete. Loaded best record. Accuracy: {best_val_acc:.2f}%.')