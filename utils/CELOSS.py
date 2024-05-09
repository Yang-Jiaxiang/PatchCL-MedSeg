import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import utils

class CE_loss(utils.losses.CrossEntropyLoss):
    def __init__(self):
        super(CE_loss,self).__init__()
        #self.weights=torch.ones(21)
        #self.weights[0]=0.05
        self.ce_loss=utils.losses.CrossEntropyLoss()

    def forward(self,prediction,target):
        target=target.long()
        return self.ce_loss(prediction,target)