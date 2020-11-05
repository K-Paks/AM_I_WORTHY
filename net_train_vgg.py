import torchvision.datasets as dset
from torch.utils.data import Dataset
import random
import os

from PIL import Image



vgg_data_path = os.path.join('D:\\', 'bigdata', 'vggface2_test', 'test')
folder_dataset = dset.ImageFolder(root=vgg_data_path)

class SiameseNetworkDatasetVGG(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __getitem__(self, index):
        img0_tuple = random.choice(self.folder.imgs)
        should_get_same_class = random.randint(0, 1)

        while True:
            img1_tuple = random.choice(self.folder.imgs)
            if (img0_tuple[1] == img1_tuple[1]) == should_get_same_class:
                break

        img0 = Image.open(img0_tuple[0]).convert('L')
        img1 = Image.open(img1_tuple[0]).convert('L')
        return img0, img1

    def __len__(self):
        return len(self.folder.imgs)


vgg_siamese_dataset = SiameseNetworkDatasetVGG(folder=folder_dataset)


