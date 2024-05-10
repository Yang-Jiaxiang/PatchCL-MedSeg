# PatchCL-MedSeg

Pretrained weights are provided as best_model.pth.
You can use this to fine-tune on a dataset specific to your task

If you have any questions, just drop an email to hbasak@cs.stonybrook.edu

# Training equipment

I use the DOCKER Image provided by TWCC: pytorch-23.05-py3:latest  
Run using Tesla V100 GPU  
Python 3.10.6  
Pytorch 2.0.0

# Need to modify

I use Pascal VOC data format for training data, please modify the 'dataloaders/dataset_kidney.py' file according to your data structure.  
And modify the 'base_path' folder path in the 'src/train_kidney.py' file.  
And modify the img_size, patch_size, contrastive_batch_size, batch_size, classes, data_dir, output_dir parameters in the 'src/train_kidney.py' file.
And modify the 'classes'parameters in the 'utils/model.py' file.

# Waiting for resolution

Dual GPU training architecture may cause "ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 2048])". This problem is usually caused by using batch normalization (Batch Normalization) This error occurs when the batch size is 1. Batch Normalization is a technique used to train deep neural networks that speeds up training by normalizing the mean and variance of input features. However, when the batch size is 1, since there is only one data point, the variance cannot be calculated, so the batch normalization layer does not work properly.

# Suggestion

Data sets can be converted to PASCAL VOC format,reference https://docs.ultralytics.com/datasets/detect/voc/ .  
Please note that the 'classes' parameter in 'src/train_kidney.py' contains background.  
If the markup contains only observation targets, you must also mark the background as a 'class'. Please refer to the 'background_msk' section of 'dataloaders/dataset_kidney.py'
