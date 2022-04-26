import tkinter as tk
from tkinter.font import BOLD
from urllib import response
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk
import cv2
from numpy import choose

def uploadImageFile():
    global img
    global scale
    imPath = filedialog.askopenfilename(
        title = "Select image",
        initialdir = "images", 
        filetypes = (("JPG File","*.jpg"), ("All File Types","*.*")) 
    )

    if imPath:
        img = Image.open(imPath)
        scale = img.size[0] / 320
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
        img.save("interface/input.jpg")
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.pack()


def calculateResult():
    global result

    # Apply Viola Jones
    faceDetection()
    eyeDetection()

    result = ImageTk.PhotoImage(Image.open("interface/result.jpg"))
    res_label.config(image=result)
    res_label.pack()


def faceDetection ():
    input = cv2.imread("interface/input.jpg", 0)
    haar_cascade_face = cv2.CascadeClassifier(
        'env/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml'
    )
    faces_rects = haar_cascade_face.detectMultiScale(
        input, 
        scaleFactor=1.2, 
        minNeighbors=5
    )

    res = cv2.imread("interface/input.jpg", 1)
    for (x, y, w, h) in faces_rects:
         cv2.rectangle(res, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite("interface/result.jpg", res)



def eyeDetection ():
    input = cv2.imread("interface/input.jpg", 0)
    haar_cascade_face = cv2.CascadeClassifier(
        'env/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml'
    )
    faces_rects = haar_cascade_face.detectMultiScale(
        input, 
        scaleFactor=1.2, 
        minNeighbors=5
    )

    res = cv2.imread("interface/result.jpg", 1)
    for (x, y, w, h) in faces_rects:
        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("interface/result.jpg", res)


def App(appName:str, WIDTH:int, HEIGHT:int):
    global img_label
    global res_label
    
    window = tk.Tk()
    window.config(background="#f8e1fc")
    window.columnconfigure([0, 1], minsize=320, weight=1)
    window.rowconfigure(0, minsize=20, weight=1)
    window.rowconfigure(1, minsize=426, weight=1)
    window.rowconfigure(2, minsize=5, weight=1)

    window.title(appName)
    
    window.geometry(str(WIDTH)+'x'+str(HEIGHT))
    
    # input frame/label/btn
    img_frame = tk.Frame(width=320, height=426, relief=tk.RAISED, borderwidth=5)
    img_frame.grid(row=1, column=0, padx=5)
    img_frame.config(background="#deecff")
    img_label = tk.Label(img_frame)
    btn = tk.Button(window, text=f"Upload image", command=uploadImageFile, background="#deecff")
    btn.grid(row=2, column=0, pady=5, sticky="n")


    # result frame/label/btn
    res_frame = tk.Frame(width=320, height=426, relief=tk.RAISED, borderwidth=5)
    res_frame.grid(row=1, column=1, padx=5)
    res_frame.config(background="#deecff")
    res_label = tk.Label(res_frame)    
    btn = tk.Button(window, text=f"Calculate result", command=calculateResult, background="#deecff")
    btn.grid(row=2, column=1, pady=5, sticky="n")
    

    window.mainloop()

App('Viola Jones', 800, 540)
