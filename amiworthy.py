import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from net import SiameseNetwork
from utils import prep_face, FaceCropper

# load model
model = SiameseNetwork().cuda()
model.load_state_dict(torch.load('model.pth'))
faceCropper = FaceCropper(
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(100, 100)
)

frameW = 640
frameH = 480
cap = cv2.VideoCapture(1)
cap.set(3, frameW)
cap.set(4, frameH)

# mePrep
me_day = prep_face('me_day.jpg')
# me_night = prep_face('me_night.jpg')


while True:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imwrite('3day.jpg', img_face)

    face_data = faceCropper.get_face(img_gray)
    if face_data is not None:
        timg, x, y, w, h = face_data
        x1, x2 = model(timg, me_day)
        euc_dist = F.pairwise_distance(x1, x2)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 150, 0), 3)
        if euc_dist.item() < 2.7:
            cv2.putText(img, "Karol", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 2)
        else:
            print(euc_dist.item())

    cv2.imshow('Cam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
