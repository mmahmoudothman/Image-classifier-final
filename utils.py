
# Imports here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import time
from PIL import Image
import seaborn as sb



def show_images(loader):
    images, labels = next(iter(loader))
    print(images.shape)
    print(labels)
    fig, axes = plt.subplots(figsize=(10,4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        helper.imshow(images[ii], ax=ax)

def load_cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
  
    factor = 256.00 / min(image.width, image.height)
    size = (int(image.width * factor), int(image.height * factor))
    image = image.resize(size)
    padding_width = (image.width - 224) / 2
    padding_height = (image.height - 224) / 2
    box = (padding_width, padding_height, padding_width + 224, padding_height + 224)
    image = image.crop(box)   
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std      
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax