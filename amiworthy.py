import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from net import SiameseNetwork
from utils import prep_face, FaceCropper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# load model
# model = SiameseNetwork().cuda()
# model.load_state_dict(torch.load('model.pth'))
model = models.resnet18().to(device)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
model.load_state_dict(torch.load('model_vgg.pth'))

faceCropper = FaceCropper(
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(100, 100),
    face_size=(224, 224),
    device=device
)

frameW = 1280
frameH = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameW)
cap.set(4, frameH)

# mePrep
me_day = prep_face('3day.jpg')
# me_night = prep_face('me_night.jpg')


while True:
    success, img = cap.read()
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('3day.jpg', img_face)

    face_data = faceCropper.get_face(img)
    if face_data is not None:
        timg, x, y, w, h = face_data
        x1 = model(timg)
        x2 = model(me_day)
        # x1, x2 = model(timg, me_day)
        euc_dist = F.pairwise_distance(x1, x2)
        print(euc_dist.item())

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        if euc_dist.item() < 0.5:
            cv2.putText(img, "Karol", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
        else:
            print(euc_dist.item())

    cv2.imshow('Cam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
