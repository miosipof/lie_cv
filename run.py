import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import gc
import sys


from src.inference import CUBInference, VOCSegInference, ViTInference


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ### ------------------------------------------------------------------------------
    # ### Inference with CUB_200_2011 dataset for classification task
    # ### ------------------------------------------------------------------------------
    # infer = CUBInference(device, 
    #                      coupling_constant=1e6, 
    #                      alpha=1.0,
    #                      beta=1.0,
    #                      v=0.01,
    #                      coarse_factor=1)
    
    # infer.run(train_epochs=None, eps=500, detection_steps=10, max_samples=None)



    # ### ------------------------------------------------------------------------------
    # ### Inference with VOCSeg dataset for segmentation task
    # ### ------------------------------------------------------------------------------
    # infer = VOCSegInference(device, 
    #                      coupling_constant=1e8, 
    #                      alpha=1.0,
    #                      beta=1.0,
    #                      v=0.01,
    #                      coarse_factor=1)
    
    # infer.run(train_epochs=None, eps=500, detection_steps=10, max_samples=None, thresh=1e-2)



    ### ------------------------------------------------------------------------------
    ### Inference with
    ### ------------------------------------------------------------------------------


    infer = ViTInference(device, 
                         coupling_constant=1e8, 
                         alpha=1.0,
                         beta=1.0,
                         v=0.01,
                         coarse_factor=1, 
                         num_steps=50)
    
    infer.run(eps=0.005, detection_steps=5, max_samples=5, thresh=1e-3)