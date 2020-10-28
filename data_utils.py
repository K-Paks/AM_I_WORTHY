import os
import random
import PIL.ImageOps
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

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
