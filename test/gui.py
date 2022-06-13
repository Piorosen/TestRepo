#coding=utf-8
#import libs
import sys
import os
import tkinter
from   tkinter import *
import tkinter.ttk
import tkinter.font
import main as video

#Add your Varial Here: (Keep This Line of comments)
#Define UI Class
class  Project1:
    def __init__(self,root,isTKroot = True):
        uiName = self.__class__.__name__
        self.root = root

        if isTKroot == True:
            root.title("Form1")
            root['background'] = '#efefef'
        Form_1= tkinter.Canvas(root,width = 10,height = 4)
        Form_1.place(x = 0,y = 0,width = 640,height = 480)
        Form_1.configure(bg = "#efefef")
        Form_1.configure(highlightthickness = 0)
        #Create the elements of root
        LabelFrame_5 = tkinter.LabelFrame(Form_1,text="Text",takefocus = True,width = 10,height = 4)
        LabelFrame_5.place(x = 24,y = 16,width = 596,height = 58)
        LabelFrame_5.configure(relief = "groove")
        Button_6 = tkinter.Button(LabelFrame_5,text="Webcam",width = 20,height = 4)
        Button_6.place(x = 39,y = 5,width = 100,height = 28)
        Button_6.configure(command=lambda:video.WebCamOn())

        Button_8 = tkinter.Button(LabelFrame_5,text="Webcam Close",width = 20,height = 4)
        Button_8.place(x = 150,y = 5,width = 100,height = 28)
        Button_8.configure(command=lambda:video.WebCamOff())
        Frame_9 = tkinter.Frame(Form_1)
        Frame_9.place(x = 29,y = 100,width = 277,height = 355)
        Frame_9.configure(bg = "#eadf57")
        Frame_9.configure(relief = "flat")
        Frame_10 = tkinter.Frame(Form_1)
        Frame_10.place(x = 329,y = 106,width = 280,height = 355)
        Frame_10.configure(bg = "#61bae0")
        Frame_10.configure(relief = "flat")
        #Inital all element's Data
        #Add Some Logic Code Here: (Keep This Line of comments)


#Create the root of Kinter
if  __name__ == '__main__':
    root = tkinter.Tk()
    root.geometry('640x480')
    MyDlg = Project1(root)
    root.mainloop()
