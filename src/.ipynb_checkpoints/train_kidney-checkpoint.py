base_path = '/home/u5169119/PatchCL-MedSeg'

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
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt

sys.path.append(base_path)
from utils.stochastic_approx import StochasticApprox
from utils.model import Network
# from dataloaders.dataset_kidney_binary_mask import BaseDatasets  
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

# +
img_size = 400
supervised_epochs = 100
patch_size = 64
contrastive_batch_size = 256
batch_size = 16
classes = 2
data_dir = base_path + '/0_data_dataset_voc_950_kidney/'
output_dir = base_path + '/dataset/splits/kidney/1-3/'

# Loss 紀錄
supervised_loss_path = base_path + '/supervised pre training_loss.csv'
SSL_loss_path = base_path + '/SSL_loss.csv'

best_contrast_supervised_model = base_path + '/best_contrast_supervised.pth'

# check_loss_file(supervised_loss_path)
# check_loss_file(SSL_loss_path)

# +
dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)

stochastic_approx = StochasticApprox(classes,0.5,0.8)
embd_queues = Embedding_Queues(classes)
cross_entropy_loss=CE_loss()
metrics=[smp.utils.metrics.IoU(threshold=0.5)]

# +
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

# +
# 初始化模型
model = Network()
teacher_model = Network()

# # 加载预训练的模型
# model = torch.load(best_contrast_supervised_model, map_location='cuda:0')
# teacher_model = torch.load(best_contrast_supervised_model, map_location='cuda:0')

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
# -

optimizer_pretrain=torch.optim.Adam(model.parameters(),lr=0.001)
optimizer_ssl=torch.optim.SGD(model.parameters(),lr=0.007)
optimizer_contrast=torch.optim.Adam(model.parameters(),lr=0.001)
scheduler = PolynomialLRDecay(optimizer=optimizer_ssl, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)

# # Supervised Learning

for c_epochs in range(supervised_epochs): #100 epochs supervised pre training
    step=0
    min_loss = math.inf
    epoch_loss=0
    
    print('Epoch: ', c_epochs)

    total_supervised_loss = 0
    total_contrastive_loss = 0

    for imgs, masks in labeled_dataloader:
        print('imgs shape: ', imgs.shape)
        print('masks shape: ', masks.shape)
        t1=time.time()
        p_masks = masks
        imgs = imgs

        # 一個 classes 一個 index
        patch_list = _get_patches(imgs, p_masks, classes, True, img_size, patch_size)

        # 隨機近似過濾和閾值更新
        qualified_patch_list = stochastic_approx.update(patch_list)
        qualified_patch_list = patch_list

        # augmentation 給 teacher model
        augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)


        # 轉換為 tensor
        aug_tensor_patch_list=[]
        qualified_tensor_patch_list=[]
        for i in range(len(augmented_patch_list)):
            if augmented_patch_list[i] is not None:
                aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
                qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
            else:
                aug_tensor_patch_list.append(None)
                qualified_tensor_patch_list.append(None)
        
        #get embeddings of qualified patches through student model
        model = model.train()
        model.module.contrast=True

        print(qualified_tensor_patch_list[0].shape)
        student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True,batch_size)
        print('student_emb_list: ', len(student_emb_list))

        #get embeddings of augmented patches through teacher model
        teacher_model.train()
        teacher_model.module.contrast = True
        teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False,batch_size)
        print('teacher_embedding_list: ', len(teacher_embedding_list))

        #enqueue these
        embd_queues.enqueue(teacher_embedding_list)
        
        #calculate PCGJCL loss
        PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)
        total_contrastive_loss += PCGJCL_loss.item()
        print('PCGJCL_loss: ', PCGJCL_loss)

        PCGJCL_loss = PCGJCL_loss.to(dev)  # 確保這個損失也在正確的設備上

        imgs, masks =imgs.to(dev), masks.to(dev)

        model.module.contrast=False
        out = model(imgs)
        
        masks_3 = masks
        if masks_3.dim() == 4:
            masks_3 = masks_3.argmax(dim=1)
        masks_3 = masks_3.long()  # Ensuring the correct type for cross_entropy_loss
        supervised_loss = cross_entropy_loss(out,masks_3)
        total_supervised_loss += supervised_loss.item()
        print('supervised_loss: ', supervised_loss)

        Alpha_consistency=0.5
        #total loss
        loss = supervised_loss + 0.5*PCGJCL_loss
        epoch_loss+=loss.item()

        #backpropagate
        loss.backward()
        optimizer_contrast.step()
                
        
        t2=time.time()
        print('step ', step, 'loss: ',loss.item(), ' & time: ',t2-t1)
        step+=1

    
    avg_epoch_loss = epoch_loss / len(labeled_dataloader)
    avg_supervised_loss = total_supervised_loss / len(labeled_dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(labeled_dataloader)

    save_loss(total_loss = f"{avg_epoch_loss:.4f}", 
              supervised_loss=f"{avg_supervised_loss:.4f}", 
              contrastive_loss=f"{avg_contrastive_loss:.4f}", 
              consistency_loss = 0 ,
              filename=supervised_loss_path)
    
    if epoch_loss < min_loss:
        torch.save(model,best_contrast_supervised_model)

# # SSL

for c_epochs in range(200):  # 200 個 epoch 的半監督學習
    step = 0
    min_loss = math.inf
    epoch_loss = 0
    print('Epoch ',c_epochs)

    labeled_iterator = iter(labeled_dataloader)

    total_supervised_loss = 0
    total_contrastive_loss = 0
    total_consistency_loss = 0
    
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
        model.to(dev)
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
        total_contrastive_loss += PCGJCL_loss.item()
        print('PCGJCL_loss: ', PCGJCL_loss.item())

        model.module.contrast = False
        # 計算監督損失，只使用有標籤數據
        imgs2, masks2 = imgs2.to(dev), masks2.to(dev)

        # 前向傳播，只針對有標籤數據
        out = model(imgs2)
        # 调整 masks2 为类别索引
        if masks2.dim() == 4 and masks2.shape[1] == 2:  # 假设 masks2 是 one-hot 编码
            masks2 = masks2.argmax(1)
        masks2 = masks2.long()  # 确保 masks2 是 LongTensor
        

        supervised_loss = cross_entropy_loss(out, masks2)
        total_supervised_loss+= supervised_loss.item()
        print('supervised_loss: ', supervised_loss.item())
        
        teacher_model.module.contrast = False
        consistency_loss = consistency_cost(model, teacher_model, imgs, p_masks)
        total_consistency_loss+=consistency_loss.item()
        print('consistency_loss:', consistency_loss.item())

        # 總損失
        loss = supervised_loss + 0.5 * PCGJCL_loss + 4 * consistency_loss
        epoch_loss+=loss.item()
        
        
        #backpropagate
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
    
    
    avg_epoch_loss = epoch_loss / len(unlabeled_dataloader)
    avg_supervised_loss = total_supervised_loss / len(unlabeled_dataloader)
    avg_contrastive_loss = total_contrastive_loss / len(unlabeled_dataloader)
    avg_consistency_loss = total_consistency_loss / len(unlabeled_dataloader)

    save_loss(total_loss= f"{avg_epoch_loss:.4f}", 
              supervised_loss=f"{avg_supervised_loss:.4f}", 
              contrastive_loss=f"{avg_contrastive_loss:.4f}", 
              consistency_loss=f"{avg_consistency_loss:.4f}",
              filename=SSL_loss_path)
    
    if epoch_loss < min_loss:
        torch.save(model, './best_contrast_SSL.pth')


