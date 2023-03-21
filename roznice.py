from PyQt5 import uic
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QLabel, QApplication, QFileDialog, QMessageBox, QSlider, \
    QTableWidgetItem  # pylint: disable=no-name-in-module, import error, bad-option-value
from PyQt5.QtCore import pyqtSlot  # pylint: disable=no-name-in-module, import error, bad-option-value
from PyQt5.QtGui import QIcon, QPixmap
import math
from pathlib import Path

from skimage.color import rgb2gray
from skimage import data

import numpy as np
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import sys
import sys
matplotlib.use('QT5Agg')
cls, wnd = uic.loadUiType('roznice.ui')


class roznice(wnd, cls):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imageA = 0
        self.imageB = None



    def on_pushButton_pressed(self):
        global name_1
        name_1 = self.open_dialog_box()
        self.imageA = plt.imread(name_1)
        label = QLabel(self)
        pixmap = QPixmap(name_1.name)
        self.label.setPixmap(pixmap)






    def on_pushButton_6_pressed(self):
        global name_2
        name_2 = self.open_dialog_box()
        self.imageB = plt.imread(name_2)
        label_2 = QLabel(self)
        pixmap_2 = QPixmap(name_2.name)
        self.label_2.setPixmap(pixmap_2)

    def wykrywanie_krawedzi(self, ImageC):
        imageC = plt.imread(ImageC)
        edges = cv2.Canny(imageC, 100, 2500, apertureSize=5)

        invert = cv2.bitwise_not(edges)
        cv2.imwrite('Krawedzie.jpg', invert)
        #cv2.imwrite('Krawedzie.jpg', edges1)



    def on_pushButton_8_pressed(self):
        self.wykrywanie_krawedzi(name_1)
        label = QLabel(self)
        pixmap_3 = QPixmap('Krawedzie.jpg')
        self.label.setPixmap(pixmap_3)

    def on_pushButton_9_pressed(self):
        self.wykrywanie_krawedzi(name_2)
        label = QLabel(self)
        pixmap_4 = QPixmap('Krawedzie.jpg')
        self.label_2.setPixmap(pixmap_4)



    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'Obrazy (*.jpg; *.jpeg; *.png)')
        filename = Path(filename[0])
        print(filename)
        return filename



    def on_pushButton_3_pressed(self):

        # convert the images to grayscale
        grayA = cv2.cvtColor(self.imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(self.imageB, cv2.COLOR_BGR2GRAY)

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        global score
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))

        if score==1:
            print('Te obrazki są takie same')


        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(self.imageA, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(self.imageB, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # show the output images

        imageA = cv2.cvtColor(self.imageA, cv2.COLOR_RGB2BGR)

        cv2.imwrite('Oryginal.jpg', imageA)

        label_1 = QLabel(self)
        pixmap_3 = QPixmap('Oryginal.jpg')
        self.label.setPixmap(pixmap_3)
        self.label_3.setText(str(round(score, 2)))

        imageB = cv2.cvtColor(self.imageB, cv2.COLOR_RGB2BGR)
        cv2.imwrite('Modified.jpg', imageB)

        label_2 = QLabel(self)
        pixmap_3 = QPixmap('Modified.jpg')
        self.label_2.setPixmap(pixmap_3)
        self.label_3.setText(str(round(score, 2)))

    def on_pushButton_2_pressed(self):
        self.label.clear()
        self.label_2.clear()
        self.label_3.clear()




    def on_pushButton_4_pressed(self):
        self.imageA = plt.imread(name_1)
        label = QLabel(self)
        pixmap = QPixmap(name_1.name)
        self.label.setPixmap(pixmap)


    def on_pushButton_7_pressed(self):
        self.imageB = plt.imread(name_2)
        #label_2 = QLabel(self)
        pixmap_2 = QPixmap(name_2.name)
        self.label_2.setPixmap(pixmap_2)



app = QApplication([])
okno = roznice()
okno.show()
app.exec_()
