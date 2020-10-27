import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

from net import SiameseNetwork

# load model
model = SiameseNetwork().cuda()
model.load_state_dict(torch.load('model.pth'))

print(model)

frameW = 640
frameH = 480
cap = cv2.VideoCapture(1)
cap.set(3, frameW)
cap.set(4, frameH)

# mePrep
me1 = cv2.imread('3.jpg')
me1 = cv2.resize(me1, (100, 100))
me1 = cv2.cvtColor(me1, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('1_2.jpg', me1)
me1 = me1.astype(np.float32)
me1 = torch.from_numpy(me1).reshape(1, 1, 100, 100)
me1 = me1.cuda()


while True:
    success, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(100, 100)
    )



    for (x, y, w, h) in faces:
        img_face = img[y:y + h, x:x + w]
        img_face = cv2.resize(img_face, (100, 100))
        npimg = img_face.astype(np.float32)/255
        timg = torch.from_numpy(npimg).reshape(1, 1, 100, 100)
        timg = timg.cuda()

        # cv2.imwrite('3.jpg', img_face)


        x1, x2 = model(timg, me1)
        euc_dist = F.pairwise_distance(x1, x2)

        if euc_dist.item() < 2:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Karol', (70, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 150, 0), 2)
        # else:
        #     print(euc_dist.item())





    # cv2.imshow("cropped", crop_img)
    cv2.imshow('cam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# x1, x2 = model(me1, me2)
# euc_dist = F.pairwise_distance(x1, x2)
# euc_dist

