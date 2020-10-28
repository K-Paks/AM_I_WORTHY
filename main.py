import os
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from PIL import Image
import PIL.ImageOps
import torch
import numpy as np
import matplotlib.pyplot as plt
from net import SiameseNetwork

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()



class Datapath():
    data_path = os.path.join('C:\\', 'Users', 'User', 'OneDrive', 'Programowanie', 'data', 'Faces_Oneshot')
    train_dir = os.path.join(data_path, 'faces', 'training')
    test_dir = os.path.join(data_path, 'faces', 'testing')


class SiameseNetworkDataset(Dataset):

    def __init__(self, imgFolderDS, transform=None, invert=True):
        self.imageFolderDS = imgFolderDS
        self.transform = transform
        self.invert = invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDS.imgs)
        # make sure approx. 50% of the images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looking for image from the same class
                img1_tuple = random.choice(self.imageFolderDS.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looking for different image

                img1_tuple = random.choice(self.imageFolderDS.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert('L')
        img1 = img1.convert('L') # TODO check what it does

        if self.invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDS.imgs)


folder_dataset = dset.ImageFolder(root=Datapath.train_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


siamese_dataset = SiameseNetworkDataset(imgFolderDS=folder_dataset,
                                        transform=transforms.Compose([
                                            transforms.Resize((100, 100)),
                                            transforms.ToTensor(),
                                        ]),
                                        invert=True)

vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=0,
                            batch_size=8)

dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
imshow(torchvision.utils.make_grid(concatenated))




class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive



torch.manual_seed(42)
np.random.seed(42)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=64)

counter = []
loss_history = []
iteration_number = 0

def cuda_maybe(device, *args):
    return [arg.to(device) for arg in args]

for epoch in range(100):
    for i, data in enumerate(train_dataloader):
        img0, img1, label = data
        img0, img1, label = cuda_maybe(device, img0, img1, label)

        optimizer.zero_grad()
        out1, out2 = model(img0, img1)
        loss_contrastive = criterion(out1, out2, label)
        loss_contrastive.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch {epoch}, loss {loss_contrastive.item()}')
            iteration_number+=10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

show_plot(counter, loss_history)

#################
import pickle
with open('FaceModel.pickle', 'wb') as f:
    pickle.dump(model, f)




###

from torch.autograd import Variable
folder_dataset_test = dset.ImageFolder(root=Datapath.test_dir)
siamese_dataset = SiameseNetworkDataset(imgFolderDS=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                        ,invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)

for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = model(Variable(x0).cuda(), Variable(x1).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

#######

import cv2
me1 = cv2.imread('1.jpg')
me1 = cv2.resize(me1, (100, 100))
me1 = cv2.cvtColor(me1, cv2.COLOR_BGR2GRAY)

me2 = cv2.imread('2.jpg')
me2 = cv2.resize(me2, (100, 100))
me2 = cv2.cvtColor(me2, cv2.COLOR_BGR2GRAY)
plt.imshow(me1)
plt.show()

me1 = me1.astype(np.float32)
me2 = me2.astype(np.float32)

me1 = torch.from_numpy(me1)
me2 = torch.from_numpy(me2)
me1, me2 = cuda_maybe(device, me1, me2)
me1, me2 = me1.reshape(1, 1, 100, 100), me2.reshape(1, 1, 100, 100)

x1, x2 = model(me1, me2)
euc_dist = F.pairwise_distance(x1, x2)
euc_dist

cat = torch.cat((me1/255, me2/255), 0).cpu()
imshow(torchvision.utils.make_grid(cat), f'Dissim: {euc_dist.item()}')

####

ja = cv2.imread('1.jpg')
ja_gray = cv2.cvtColor(ja, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
    ja_gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)


for (x, y, w, h) in faces:
    cv2.rectangle(ja_gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

status = cv2.imwrite('faces_detected.jpg', ja_gray)
