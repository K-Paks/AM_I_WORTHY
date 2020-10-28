import cv2
import torch
import numpy as np

def prep_face(file):
    face = cv2.imread(file)
    face = cv2.resize(face, (100, 100))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('1_2.jpg', face)
    face = face.astype(np.float32)
    face = torch.from_numpy(face).reshape(1, 1, 100, 100)
    return face.cuda()


class FaceCropper():
    def __init__(self, scaleFactor, minNeighbors, minSize):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def get_face(self, img):
        faces = self.faceCascade.detectMultiScale(
            img,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )

        for (x, y, w, h) in faces:
            img_face = img[y:y + h, x:x + w]
            img_face = cv2.resize(img_face, (100, 100))
            npimg = img_face.astype(np.float32) / 255
            timg = torch.from_numpy(npimg).reshape(1, 1, 100, 100)
            timg = timg.cuda()
            return timg, x, y, w, h