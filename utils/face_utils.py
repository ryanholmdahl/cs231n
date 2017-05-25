import cv2
import numpy as np


class Cropper:
    def __init__(self, cascade_path, flip_image=False):
        self.classifier = cv2.CascadeClassifier(cascade_path)
        self.flip_image = flip_image

    def get_crop(self, image):
        gray = image
        if self.flip_image:
            gray = np.fliplr(gray)
        faces = self.classifier.detectMultiScale(gray, 1.1)
        retval = None
        max_w = 0
        for (x,y,w,h) in faces:
            w = max(w,h)
            h = max(w,h)
            if w > max_w:
                retval = gray[y:y+h, x:x+w]
                max_w = w
        return retval

