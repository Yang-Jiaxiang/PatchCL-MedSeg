img_size = 400
supervised_epochs = 100
patch_size = 32
contrastive_batch_size = 256
batch_size = 2
classes = 2
base_path = '/home/louis/Documents/project/PatchCL-MedSeg'
data_dir = base_path + '/0_data_dataset_voc_950_kidney/'
output_dir = base_path + '/dataset/splits/kidney/1-3/'

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import math
import time
import sys
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(base_path)
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
from dataloaders.dataset_kidney import BaseDatasets  
from utils.queues import Embedding_Queues
from utils.CELOSS import CE_loss
from utils.patch_utils import _get_patches
from utils.aug_utils import batch_augment
from utils.get_embds import get_embeddings
from utils.const_reg import consistency_cost
from utils.plg_loss import PCGJCL
from utils.torch_poly_lr_decay import PolynomialLRDecay
from utils.loss_file import save_loss, check_loss_file

dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)

IMG_folder_path = data_dir 
msk_folder_path = data_dir
# Load file lists
with open(os.path.join(output_dir, "labeled.txt"), 'r') as file:
    labeled_files = [line.strip().split(' ') for line in file.readlines()]

with open(os.path.join(output_dir, "unlabeled.txt"), 'r') as file:
    unlabeled_files = [line.strip() for line in file.readlines()]

# Define transformations if needed
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# # Create datasets and dataloaders
labeled_dataset = BaseDatasets(labeled_files, IMG_folder_path, msk_folder_path, transform)
unlabeled_dataset = BaseDatasets(unlabeled_files, IMG_folder_path, transform=transform)

labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
print('===========================================================')
print('number of labeled_dataset: ', len(labeled_dataset))
print('number of unlabeled_dataset: ', len(unlabeled_dataset))
print('===========================================================')

# for imgs, masks in labeled_dataloader:
#     break
# print('imgs.shape: ', imgs.shape)
# print('masks.shape: ', masks.shape)
# print('===========================================================')



# check loss file before training, if not exist, create one
check_loss_file(base_path + '/supervised pre training_loss.csv')
check_loss_file(base_path + '/SSL_loss.csv')


stochastic_approx = StochasticApprox(classes,0.5,0.8)

model = Network()
teacher_model = Network()

#Turning off gradients for teacher model
for param in teacher_model.parameters():
    param.requires_grad=False
    #Esuring mothe the models have same weight
teacher_model.load_state_dict(model.state_dict())
model.contrast=False
teacher_model.contrast = False

model = nn.DataParallel(model)
model = model.to(dev)
teacher_model = nn.DataParallel(teacher_model)
teacher_model=teacher_model.to(dev)

embd_queues = Embedding_Queues(classes)

cross_entropy_loss=CE_loss()
metrics=[smp.utils.metrics.IoU(threshold=0.5)]

optimizer_pretrain=torch.optim.Adam(model.parameters(),lr=0.001)
optimizer_ssl=torch.optim.SGD(model.parameters(),lr=0.007)
optimizer_contrast=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = PolynomialLRDecay(optimizer=optimizer_ssl, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)


for c_epochs in range(200):  # 200 個 epoch 的半監督學習
    step = 0
    min_loss = math.inf
    epoch_loss = 0
    # print('Epoch ',c_epochs)

    labeled_iterator = iter(labeled_dataloader)
    for imgs in unlabeled_dataloader:
        t1 = time.time()
        with torch.no_grad():
            # 將 imgs 發送到設備
            imgs = imgs.to(dev)
            # 將模型設置為評估模式
            model.eval()
            # 獲取偽標籤
            model.module.contrast = False
            p_masks = model(imgs)
            p_masks = p_masks.detach()  # 確保 p_masks 在 GPU 上，因為 model 的輸出在 GPU 上

            # 由於我們同樣使用有標籤數據進行 PCGJCL
            imgs2, masks2 = next(labeled_iterator)
            imgs2, masks2 = imgs2.to(dev), masks2.to(dev)

            # 連接無標籤和有標籤集
            combined_imgs = torch.cat([imgs, imgs2], dim=0).cpu()  # 移动到 CPU
            combined_masks = torch.cat([p_masks, masks2], dim=0).cpu()  # 移动到 CPU

            # 獲取類別相關的 patch 列表
            patch_list = _get_patches(combined_imgs, combined_masks, classes, True, img_size, patch_size)

            # 隨機近似過濾和閾值更新
            qualified_patch_list = stochastic_approx.update(patch_list)

            # 為老師模型製作增強
            augmented_patch_list = batch_augment(qualified_patch_list, contrastive_batch_size)

            # 轉換為張量
            aug_tensor_patch_list = []
            qualified_tensor_patch_list = []
            for i in range(len(augmented_patch_list)):
                if augmented_patch_list[i] is not None:
                    aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
                    qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
                else:
                    aug_tensor_patch_list.append(None)
                    qualified_tensor_patch_list.append(None)

        # 將模型設回訓練模式
        model.train()
        model.module.contrast = True
        student_emb_list = get_embeddings(model, qualified_tensor_patch_list, True, batch_size)

        # 通過老師模型獲取增強 patch 的嵌入
        teacher_model.train()
        teacher_model.module.contrast = True
        teacher_embedding_list = get_embeddings(teacher_model, aug_tensor_patch_list, False, batch_size)

        # 入隊這些
        embd_queues.enqueue(teacher_embedding_list)

        # 計算 PCGJCL 損失
        PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 1, 10, psi=4096)
        PCGJCL_loss = PCGJCL_loss.to(dev)
        print('PCGJCL_loss: ', PCGJCL_loss)

        model.module.contrast = False
        # 計算監督損失，只使用有標籤數據
        imgs2, masks2 = imgs2.to(dev), masks2.to(dev)

        # 前向傳播，只針對有標籤數據
        out = model(imgs2)
        print('out.shape:', out.shape)
        # 调整 masks2 为类别索引
        if masks2.dim() == 4 and masks2.shape[1] == 2:  # 假设 masks2 是 one-hot 编码
            masks2 = masks2.argmax(1)
        masks2 = masks2.long()  # 确保 masks2 是 LongTensor
        

        supervised_loss = cross_entropy_loss(out, masks2)
        print('supervised_loss: ', supervised_loss)
        
        teacher_model.module.contrast = False
        consistency_loss = consistency_cost(model, teacher_model, imgs, p_masks)
        print('consistency_loss:', consistency_loss)

        # 總損失
        loss = supervised_loss + 0.5 * PCGJCL_loss + 4 * consistency_loss
        epoch_loss+=loss.item()
        # 反向傳播
        loss.backward()
        optimizer_ssl.step()
        scheduler.step()

        # 更新老師模型參數
        for param_stud, param_teach in zip(model.parameters(), teacher_model.parameters()):
            param_teach.data.copy_(0.001 * param_stud + 0.999 * param_teach)

        # 附加信息
        t2 = time.time()
        print('step ', step, 'loss: ', loss.item(), ' & time: ', t2 - t1)
        step += 1
    
    save_loss(epoch_loss/len(unlabeled_dataloader), base_path + '/SSL_loss.csv')
    if epoch_loss < min_loss:
        torch.save(model, './best_contrast_SSL.pth')
