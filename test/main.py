#coding=utf-8
#import libs
import sys
import os
from   tkinter import *
import tkinter.ttk
import tkinter.font
import core as video
import cv2
from PIL import Image, ImageTk

#Add your Varial Here: (Keep This Line of comments)
#Define UI Class
class  Project1:
    def StatusLabel(self, text):
        self.statusText.set("상태 : " + text)

    def UpdateImage(self, origin, plate, crop):
        # print(origin)
        oImage = ImageTk.PhotoImage(image=Image.fromarray(origin).resize((542, 252)))
        self.Canvas_11.config(image=oImage)
        self.Canvas_11.image = oImage

        if not (plate is None):
            oImage = ImageTk.PhotoImage(image=Image.fromarray(plate).resize((230, 82)))
            self.Canvas_9.config(image=oImage)
            self.Canvas_9.image = oImage

        if not (crop is None):
            oImage = ImageTk.PhotoImage(image=Image.fromarray(crop).resize((230, 82)))
            self.Canvas_10.config(image=oImage)
            self.Canvas_10.image = oImage

        print("update!")
        pass

    def __init__(self,root,isTKroot = True):
        uiName = self.__class__.__name__
        self.root = root
        if isTKroot == True:
            root.title("자동차 번호판 인식 영역 추출")
            root['background'] = '#efefef'

        self.statusText = StringVar()
        self.statusText.set("상태 : ")
        Form_1 = tkinter.Canvas(root, width=10, height=4)
        Form_1.place(x=0, y=0, width=640, height=480)
        Form_1.configure(bg="#efefef")
        Form_1.configure(highlightthickness=0)
        # Create the elements of root
        LabelFrame_5 = tkinter.LabelFrame(Form_1, text="관리", takefocus=True, width=10, height=4)
        LabelFrame_5.place(x=24, y=16, width=596, height=58)
        LabelFrame_5.configure(relief="groove")
        Button_6 = tkinter.Button(LabelFrame_5, text="Webcam", width=10, height=4)
        Button_6.place(x=39, y=5, width=100, height=28)
        Button_6.configure(command=lambda: video.WebCamOn(self))

        Button_8 = tkinter.Button(LabelFrame_5, text="Webcam Close", width=10, height=4)
        Button_8.place(x=150, y=5, width=100, height=28)
        Button_8.configure(command=lambda: video.WebCamOff())

        self.Label_12 = tkinter.Label(LabelFrame_5, textvariable=self.statusText, width=10, height=4)
        self.Label_12.place(x=319, y=8, width=200, height=20)
        self.Label_12.configure(relief="flat")

        self.Canvas_9 = tkinter.Label(Form_1)
        self.Canvas_9.place(x=38, y=106, width=230, height=82)
        # self.Canvas_9.configure(bg="#eadf57")
        # self.Canvas_9_Container = self.Canvas_9.create_image(0,0, anchor="nw")
        self.Canvas_10 = tkinter.Label(Form_1)
        self.Canvas_10.place(x=350, y=109, width=230, height=82)
        # self.Canvas_10.configure(bg="#61bae0")
        # self.Canvas_10_Container = self.Canvas_10.create_image(0,0, anchor="nw")

        self.Canvas_11 = tkinter.Label(Form_1)
        self.Canvas_11.place(x=38, y=215, width=542, height=252)
        # self.Canvas_11.configure(bg="#f05155")
        # self.Canvas_11_Container = self.Canvas_11.create_image(0,0, anchor="nw")

        # Inital all element's Data
        # Add Some Logic Code Here: (Keep This Line of comments)

#Create the root of Kinter
if  __name__ == '__main__':
    root = tkinter.Tk()
    root.geometry('640x480')
    MyDlg = Project1(root)
    root.mainloop()
