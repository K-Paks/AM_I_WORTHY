import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision import models

from net import ContrastiveLoss


vgg_data_path = os.path.join('D:\\', 'bigdata', 'vggface2_test', 'test_data')
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
        transforms.Resize((224, 224)),
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


# inp1, inp2, classes = next(iter(train_dataloader))
# out = torchvision.utils.make_grid(torch.cat((inp1, inp2), dim=2))
# imshow(out, title=[int(x.item()) for x in classes])

model_ft = models.resnet18(pretrained=True).to(device)
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, 5)
model_ft = model_ft.to(device)

criterion = ContrastiveLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.001)

def to_device(device, *args):
    return [x.to(device) for x in args]

# train


num_epochs = 5
loss_hist = []
epoch_loss_hist = []
data_len = len(train_dataloader)

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    model_ft.train()

    running_loss = 0.0
    # running_corrects = 0

    for i, (inp1, inp2, labels) in enumerate(train_dataloader):
        if i%500 == 0:
            print(f'{i} out of {data_len}')

        inp1, inp2, labels = to_device(device, inp1, inp2, labels)

        optimizer.zero_grad()

        out1 = model_ft(inp1)
        out2 = model_ft(inp2)
        loss = criterion(out1, out2, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inp1.size(0)
    epoch_loss = running_loss / len(train_dataloader)
    epoch_loss_hist.append(epoch_loss)
    print('{} Loss: {:.4f}'.format(
        "train", epoch_loss))

















