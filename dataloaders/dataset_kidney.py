import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt

class BaseDatasets(Dataset):
    def __init__(self, file_list, img_folder, msk_folder=None, size=256):
        self.file_list = file_list
        self.img_folder = img_folder
        self.msk_folder = msk_folder
        self.size = size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 讀取影像，若是有 mask，則使用第一個連結;若沒有，則直接使用 path
        img_path = os.path.join(self.img_folder, self.file_list[idx][0] if self.msk_folder else self.file_list[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img,(self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        # 若是有 mask，則進入
        if self.msk_folder:
            msk_path = os.path.join(self.msk_folder, self.file_list[idx][1])
            msk = cv2.imread(msk_path)
            msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            msk = msk[:, :, 0]            
            
            return img, msk
        else:
            return img