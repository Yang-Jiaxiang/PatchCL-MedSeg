base_path = '/tf/PatchCL-MedSeg'

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

img_size = 400
patch_size = 32
contrastive_batch_size = 256
batch_size = 8
classes = 2
data_dir = base_path + '/0_data_dataset_voc_950_kidney/'
output_dir = base_path + '/dataset/splits/kidney/1-3/'
    
if __name__=="__main__":
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(dev)
    
    stochastic_approx = StochasticApprox(21,0.5,0.8)
    
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

    embd_queues = Embedding_Queues(21)

    cross_entropy_loss=CE_loss()
    metrics=[smp.utils.metrics.IoU(threshold=0.5)]

    optimizer_pretrain=torch.optim.Adam(model.parameters(),lr=0.001)
    optimizer_ssl=torch.optim.SGD(model.parameters(),lr=0.007)
    scheduler = PolynomialLRDecay(optimizer_ssl, max_decay_steps=200, end_learning_rate=0.0001, power=2.0)

    contrastive_batch_size = 128
    
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

    # # 輸入皆為 RGB
    # # Create datasets and dataloaders
    labeled_dataset = BaseDatasets(labeled_files, IMG_folder_path, msk_folder_path, transform)
    unlabeled_dataset = BaseDatasets(unlabeled_files, IMG_folder_path, transform=transform)

    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

    
    print('number of labeled_dataset: ', len(labeled_dataset))
    print('number of unlabeled_dataset: ', len(unlabeled_dataset))
    
    check_loss_file(base_path + '/supervised pre training_loss.csv')
    check_loss_file(base_path + '/SSL_loss.csv')
    
    #CONTRASTIVE PRETRAINING (warm up)
    #torch.autograd.set_detect_anomaly(True)
    for c_epochs in range(100): #100 epochs supervised pre training
        step=0
        min_loss = math.inf
        epoch_loss=0
        #print('Epoch ',c_epochs)

        for imgs, masks in labeled_dataloader:

            t1=time.time()
            with torch.no_grad():

                #Send psudo masks & imgs to cpu
                p_masks=masks
                imgs = imgs
                print('p_masks shape: ', p_masks.shape)

                #get classwise patch list
                patch_list = _get_patches(imgs, p_masks, classes, True, img_size, patch_size)
                print('patch_list len: ', len(patch_list[0])) # 一類別，output 1

                #stochastic approximation filtering and threshold update
                qualified_patch_list = stochastic_approx.update(patch_list)
                print('qualified_patch_list len: ', len(qualified_patch_list[0]))
                
                qualified_patch_list = patch_list
                print('patch_list len: ', len(patch_list[0])) # 一類別，output 1


                #make augmentations for teacher model
                augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)


                #convert to tensor
                aug_tensor_patch_list=[]
                qualified_tensor_patch_list=[]
                for i in range(len(augmented_patch_list)):
                    if augmented_patch_list[i] is not None:
                        aug_tensor_patch_list.append(torch.tensor(augmented_patch_list[i]))
                        qualified_tensor_patch_list.append(torch.tensor(qualified_patch_list[i]))
                    else:
                        aug_tensor_patch_list.append(None)
                        qualified_tensor_patch_list.append(None)
                        
                print('aug_tensor_patch_list len: ', len(aug_tensor_patch_list[0]))
                print('qualified_tensor_patch_list len: ', len(qualified_tensor_patch_list[0]))


            #get embeddings of qualified patches through student model
            model=model.train()
            model.module.contrast=True
            student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

            #get embeddings of augmented patches through teacher model
            teacher_model.train()
            teacher_model.contrast = True
            teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

            #enqueue these
            embd_queues.enqueue(teacher_embedding_list)
            print('student_emb_list: ', len(student_emb_list))

            #calculate PCGJCL loss
            PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 0.2 , 4, psi=4096)

            #calculate supervied loss
            imgs, masks =imgs.to(dev), masks.to(dev)
            out = model(imgs)
            supervised_loss = cross_entropy_loss(out,masks)

            #total loss
            loss = supervised_loss + 0.5*PCGJCL_loss

            epoch_loss+=loss

            #backpropagate
            loss.backward()
            optimizer_contrast.step()


            for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
                param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

            #Extras
            t2=time.time()
            print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
            step+=1
            
        save_loss(loss, base_path + '/supervised pre training.csv')
        if epoch_loss < min_loss:
            torch.save(model,'./best_contrast.pth')


    for c_epochs in range(200): #200 epochs supervised SSL
        step=0
        min_loss = math.inf
        epoch_loss=0
        #print('Epoch ',c_epochs)

        labeled_iterator = iter(labelled_dataloader)
        for imgs in unlabeled_dataloader:

            t1=time.time()
            with torch.no_grad():

                #send imgs to dev
                imgs = imgs.to(dev)

                #set model in Eval mode
                model = model.eval()

                #Get pseudo masks
                model.module.contrast=False
                p_masks = model(imgs)
                
                print('p_masks shape: ', p_masks.shape)

                #Send psudo masks & imgs to cpu
                p_masks=masks
                p_masks = p_masks.to('cpu').detach()
                imgs = imgs.to('cpu').detach()

                #Since we use labeled data for PCGJCL as well
                imgs2, masks2 = labeled_iterator.next()

                #concatenating unlabeled and labeled sets
                p_masks = torch.cat([p_masks,masks2],dim=0)
                imgs = torch.cat([imgs,imgs2],dim=0)

                #get classwise patch list
                patch_list = _get_patches(imgs,p_masks)

                #stochastic approximation filtering and threshold update
                qualified_patch_list = stochastic_approx.update(patch_list)



                #make augmentations for teacher model
                augmented_patch_list = batch_augment(qualified_patch_list,contrastive_batch_size)


                #convert to tensor
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
            model=model.train()
            model.module.contrast=True
            student_emb_list = get_embeddings(model,qualified_tensor_patch_list,True)

            #get embeddings of augmented patches through teacher model
            teacher_model.train()
            teacher_model.contrast = True
            teacher_embedding_list = get_embeddings(teacher_model,aug_tensor_patch_list,False)

            #enqueue these
            embd_queues.enqueue(teacher_embedding_list)

            #calculate PCGJCL loss
            PCGJCL_loss = PCGJCL(student_emb_list, embd_queues, 128, 1 , 10, alpha=1)


            #calculate supervied loss
            imgs2, masks2 =imgs2.to(dev), masks2.to(dev)
            out = model(imgs)
            supervised_loss = cross_entropy_loss(out,masks2)


            #Consistency Loss
            consistency_loss=consistency_cost(model,teacher_model,imgs,p_masks)


            #total loss
            loss = supervised_loss + 0.5*PCGJCL_loss + 4*consistency_loss

            #backpropagate
            loss.backward()
            optimizer_ssl.step()
            scheduler.step()


            for param_stud, param_teach in zip(model.parameters(),teacher_model.parameters()):
                param_teach.data.copy_(0.001*param_stud + 0.999*param_teach)

            #Extras
            t2=time.time()
            print('step ', step, 'loss: ',loss, ' & time: ',t2-t1)
            step+=1
            
        save_loss(loss, base_path + '/supervised SSL.csv')
        if epoch_loss < min_loss:
            torch.save(model,'./best_contrast.pth')
