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

for imgs, masks in labeled_dataloader:
    break
print('imgs.shape: ', imgs.shape)
print('masks.shape: ', masks.shape)
print('===========================================================')



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


for c_epochs in range(supervised_epochs): #100 epochs supervised pre training
    step=0
    min_loss = math.inf
    epoch_loss=0
    
    print('Epoch: ', c_epochs)
    for imgs, masks in labeled_dataloader:
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
        print('supervised_loss: ', supervised_loss)

        Alpha_consistency=0.5
        #total loss
        loss = (supervised_loss * (1-Alpha_consistency)) + (Alpha_consistency * PCGJCL_loss)
        
        epoch_loss+=loss.item()
        
        t2=time.time()
        print('step ', step, 'loss: ',loss.item(), ' & time: ',t2-t1)
        step+=1
        
    save_loss(epoch_loss/len(labeled_dataloader), base_path + '/supervised pre training_loss.csv')
    if epoch_loss < min_loss:
        torch.save(model,'./best_contrast.pth')