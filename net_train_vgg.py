import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from net import ContrastiveLoss


vgg_data_path = os.path.join('D:\\', 'bigdata', 'vggface2_test', 'test')
folder_dataset = dset.ImageFolder(root=vgg_data_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SiameseNetworkDatasetVGG(Dataset):
    def __init__(self, folder, transform=None, invert=False):
        self.folder = folder
        self.transform = transform
        self.invert = invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.folder.imgs)
        should_get_same_class = random.randint(0, 1)

        while True:
            img1_tuple = random.choice(self.folder.imgs)
            if (img0_tuple[1] == img1_tuple[1]) == should_get_same_class\
                    and img0_tuple[0] != img1_tuple[1]:
                break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        if self.invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.Tensor([not should_get_same_class])

    def __len__(self):
        return len(self.folder.imgs)


vgg_siamese_dataset = SiameseNetworkDatasetVGG(
    folder=folder_dataset,
    transform=transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    invert=False
)

train_dataloader = DataLoader(
    dataset=vgg_siamese_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


inp1, inp2, classes = next(iter(train_dataloader))
out = torchvision.utils.make_grid(torch.cat((inp1, inp2), dim=2))
imshow(out, title=[int(x.item()) for x in classes])










