import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BaseDatasets(Dataset):
    def __init__(self, file_list, img_folder, msk_folder=None, transform=None):
        self.file_list = file_list
        self.img_folder = img_folder
        self.msk_folder = msk_folder
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load image and convert to RGB
        img_path = os.path.join(self.img_folder, self.file_list[idx][0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        if self.msk_folder:
            # Load mask and convert to grayscale
            msk_path = os.path.join(self.msk_folder, self.file_list[idx][1])
            msk = Image.open(msk_path).convert("L")
            msk = np.array(msk, dtype=np.float32) / 255.0  # Normalize to [0, 1]

            # Apply the same transformation to the mask if any
            if self.transform:
                msk = self.transform(Image.fromarray(msk * 255).convert("L"))
            
            msk = np.array(msk, dtype=np.float32) / 255.0
            msk = (msk > 0).astype(np.float32)
#             msk = np.expand_dims(msk, axis=0)  # Shape: [1, H, W]
            msk = torch.from_numpy(msk).float()
            
            # Calculate the background mask
            background_msk = 1.0 - msk.numpy()[0]  # Access the first channel to invert it

            # Stack the foreground and background masks
            msk = np.stack([msk.numpy()[0], background_msk], axis=0)  # Shape: [2, H, W]
            msk = torch.from_numpy(msk).float()

            return img, msk
        else:
            return img