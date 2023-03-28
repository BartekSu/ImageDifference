from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import tkinter as tk
from tkinter import messagebox

matplotlib.use('QT5Agg')
cls, wnd = uic.loadUiType('roznice.ui')


class Differences(wnd, cls):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.imageA = 0
        self.imageB = None

    def on_pushButton_pressed(self):
        global name_1
        name_1 = self.open_dialog_box()
        self.imageA = plt.imread(name_1)
        pixmap = QPixmap(name_1.name)
        self.label.setPixmap(pixmap)

    def on_pushButton_6_pressed(self):
        global name_2
        name_2 = self.open_dialog_box()
        self.imageB = plt.imread(name_2)
        pixmap_2 = QPixmap(name_2.name)
        self.label_2.setPixmap(pixmap_2)

    def edge_detection(self, ImageC):
        imageC = plt.imread(ImageC)
        edges = cv2.Canny(imageC, 100, 2500, apertureSize=5)

        invert = cv2.bitwise_not(edges)
        cv2.imwrite('Krawedzie.jpg', invert)

    def on_pushButton_8_pressed(self):
        self.edge_detection(name_1)
        pixmap_3 = QPixmap('Krawedzie.jpg')
        self.label.setPixmap(pixmap_3)

    def on_pushButton_9_pressed(self):
        self.edge_detection(name_2)
        pixmap_4 = QPixmap('Krawedzie.jpg')
        self.label_2.setPixmap(pixmap_4)

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName(self, 'Open File', '', 'Obrazy (*.jpg; *.jpeg; *.png)')
        filename = Path(filename[0])
        print(filename)
        return filename

    def popup(self, title="", sentence=""):
        tk.Tk().withdraw()
        messagebox.showinfo(title=title, message=sentence)

    # noinspection PyGlobalUndefined
    def on_pushButton_3_pressed(self):
        # convert the images to grayscale
        grayA = cv2.cvtColor(self.imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(self.imageB, cv2.COLOR_BGR2GRAY)

        if grayA.shape != grayB.shape:
            self.popup('Error', 'Images shape is not the same')

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        global score
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        if 1 == score:
            self.popup("Error", "Images are the same")
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

        # convert colours to RGB
        imageA = cv2.cvtColor(self.imageA, cv2.COLOR_RGB2BGR)

        cv2.imwrite('Oryginal.jpg', imageA)

        pixmap_3 = QPixmap('Oryginal.jpg')
        self.label.setPixmap(pixmap_3)
        self.label_3.setText(str(round(score, 2)))

        imageB = cv2.cvtColor(self.imageB, cv2.COLOR_RGB2BGR)
        cv2.imwrite('Modified.jpg', imageB)

        pixmap_3 = QPixmap('Modified.jpg')
        self.label_2.setPixmap(pixmap_3)
        self.label_3.setText(str(round(score, 2)))

    def on_pushButton_2_pressed(self):
        self.label.clear()
        self.label_2.clear()
        self.label_3.clear()

    def on_pushButton_4_pressed(self):
        self.imageA = plt.imread(name_1)
        pixmap = QPixmap(name_1.name)
        self.label.setPixmap(pixmap)

    def on_pushButton_7_pressed(self):
        self.imageB = plt.imread(name_2)
        pixmap_2 = QPixmap(name_2.name)
        self.label_2.setPixmap(pixmap_2)


app = QApplication([])
okno = Differences()
okno.show()
app.exec_()
