import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
import os

from PIL import Image, ImageOps


vgg_data_path = os.path.join('D:\\', 'bigdata', 'vggface2_test', 'test')
folder_dataset = dset.ImageFolder(root=vgg_data_path)

class SiameseNetworkDatasetVGG(Dataset):
    def __init__(self, folder, transform=None, invert=True):
        self.folder = folder
        self.transform = transform
        self.invert = invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.folder.imgs)
        should_get_same_class = random.randint(0, 1)

        while True:
            img1_tuple = random.choice(self.folder.imgs)
            if (img0_tuple[1] == img1_tuple[1]) == should_get_same_class:
                break

        img0 = Image.open(img0_tuple[0]).convert('L')
        img1 = Image.open(img1_tuple[0]).convert('L')

        if self.invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1

    def __len__(self):
        return len(self.folder.imgs)


vgg_siamese_dataset = SiameseNetworkDatasetVGG(
    folder=folder_dataset,
    transform=transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ]),
    invert=True
)


