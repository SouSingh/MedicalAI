import sys
from PIL import Image
from cog import BasePredictor, Input, Path
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism


val_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class MedNISTDataset(Dataset):
    def __init__(self, image_files, transforms):
        self.image_files = image_files
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index])

def image2we(image):
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=6).to(device)
    model.load_state_dict(torch.load('best_metric_model.pth'))
    model.eval()

    # Wrap the image in a list to make it a batch of size 1
    image_batch = [image]

    # Create a DataLoader with the single image batch
    val_loader = DataLoader(MedNISTDataset(image_batch, val_transforms), batch_size=1, num_workers=0)
    lis = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
    with torch.no_grad():
        for test_data in val_loader:
            test_images = test_data.to(device)
            pred = model(test_images).argmax(dim=1)
            return lis[pred.item()]




class Predictor(BasePredictor):
    def predict(
        self, 
        image: Path = Input(description="Input image"),        
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(str(image))
        image.save('out.png')
        pathed = 'out.png'
        out = image2we(pathed)
        os.remove(pathed)
        return out

