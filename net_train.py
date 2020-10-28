from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import torch
import numpy as np

from net import SiameseNetwork, ContrastiveLoss
from utils import imshow, show_plot, cuda_maybe
from data_utils import Datapath, SiameseNetworkDataset



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




torch.manual_seed(42)
np.random.seed(42)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=64)



# training
counter = []
loss_history = []
iteration_number = 0

for epoch in range(100):
    for i, data in enumerate(train_dataloader):
        img0, img1, label = cuda_maybe(device, *data)

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

