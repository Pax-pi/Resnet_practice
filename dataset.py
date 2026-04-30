import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Any
import pydicom
from pathlib import Path
from matplotlib.image import imread

class ResNetDataset(Dataset):
    '''Dataset class for ResNet model, a simple one'''
    
    def __init__(self, df: pd.DataFrame, image_dir: Path, featurename: str, labelname: str, transform: Callable[[Any], Any] | None = None, dcm: bool = False, grey: bool = False) -> None:
        '''Initialize an instance'''
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.featurename = featurename
        self.labelname = labelname
        self.dcm = dcm
        self.grey = grey

    def __len__(self) -> int:
        '''Return the length'''
        return len(self.df)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get item'''
        file_id = self.df.loc[idx, self.featurename]
        target = self.df.loc[idx, self.labelname]
        
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        if self.dcm:
            dcm_path = self.image_dir / f'{file_id}.dcm'
            dicom = pydicom.dcmread(str(dcm_path))
            img = dicom.pixel_array.astype(np.float32)
            img = cv2.resize(img, (224, 224))
            if self.grey:
                img = (img - np.min(img))/(np.max(img) - np.min(img) + 1e-6)
                img = np.stack([img, img, img], axis = -1)
            else:
                img_min = np.min(img, axis=(0, 1), keepdims=True)
                img_max = np.max(img, axis=(0, 1), keepdims=True)
                img = (img - img_min) / (img_max - img_min + 1e-6)
        else:
            file_path = self.image_dir / f'{file_id}.jpg' #assuming non medical image are stored in JPG
            img = imread(str(file_path))
            img = cv2.resize(img, (224, 224))
            if self.grey:
                img = (img - np.min(img))/(np.max(img) - np.min(img) + 1e-6)
                img = np.stack([img, img, img], axis = -1)
            else:
                img_min = np.min(img, axis=(0, 1), keepdims=True)
                img_max = np.max(img, axis=(0, 1), keepdims=True)
                img = (img - img_min) / (img_max - img_min + 1e-6)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if self.transform:
            augmented = self.transform(image = img) # type: ignore
            img = augmented['image']
            
        img_tensor = torch.tensor(img).permute(2, 0, 1)

        return img_tensor, target_tensor