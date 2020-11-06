from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

from net import SiameseNetwork, ContrastiveLoss
from utils import show_plot, cuda_maybe
from data_utils import Datapath, SiameseNetworkDataset

folder_dataset = dset.ImageFolder(root=Datapath.train_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

siamese_dataset = SiameseNetworkDataset(imgFolderDS=folder_dataset,
                                        transform=transforms.Compose([
                                            transforms.Resize((100, 100)),
                                            transforms.ToTensor(),
                                        ]),
                                        invert=False)
model = SiameseNetwork().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=0,
                              batch_size=4)

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
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# show_plot(counter, loss_history)
torch.save(model.state_dict(), 'model.pth')

#################
# Validation

# from torch.autograd import Variable
#
# how_many = 60
# max_dist = 2
# score = 0
#
# folder_dataset_test = dset.ImageFolder(root=Datapath.test_dir)
# siamese_dataset = SiameseNetworkDataset(imgFolderDS=folder_dataset_test,
#                                         transform=transforms.Compose([transforms.Resize((100, 100)),
#                                                                       transforms.ToTensor()
#                                                                       ])
#                                         , invert=False)
#
# test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
# dataiter = iter(test_dataloader)
#
# for i in range(how_many):
#     x0, x1, label2 = next(dataiter)
#     concatenated = torch.cat((x0, x1), 0)
#
#     output1, output2 = model(Variable(x0).cuda(), Variable(x1).cuda())
#     euc_d = F.pairwise_distance(output1, output2)
#
#
#     if not label2 and euc_d.item() < max_dist:
#         score+= 1
#     elif label2 and euc_d.item() > max_dist:
#         score+=1


#######
