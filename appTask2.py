# interfase must looks like
# image scaleimg histimg gradimg dftimg dctimg parrallel
# img    img      img      img    img     img    img  
# matching result
#        img      img      img    img     img    img
# for 3 the same
# cross valid change, must be test photo (from 1 to 40) and result of accuracy
from glob import glob
import tkinter as tk
from tkinter.font import BOLD
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk
import cv2
from cv2 import split
import numpy as np
from scipy import fftpack
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def printAllData():
    global data
    global optParam
    global optParamPar
    global weights
    global features
    global parFeatures
    optParam = {
        'scale': [280, 4.200000000000001], 
        'hist': [360, 11], 
        'grad': [360, 17], 
        'dft': [320, 11], 
        'dct': [360, 5]
    }
    optParamPar = {
        "scale": 3.4000000000000017,
        "hist": 13,
        "grad": 17,
        "dft": 11,
        "dct": 5
    }
    weights = {
        "scale": 0.72,
        "hist": 0.4425,
        "grad": 0.94,
        "dft": 0.135,
        "dct": 0.17
    }
    data = loadData("orl_faces")
    mask = loadDataMask("orl_faces_with_mask")
    fawkes = loadDataFawkes("orl_faces_high_cloaked")
    features = {}
    parFeatures = {}
    for mtd in optParamPar.keys():
        print(f"train {mtd}")
        train, test = splitData(data, optParam[mtd][0] // 40)
        features[mtd] = fit(train, mtd, optParam[mtd][1])
        train, test = splitData(data, 8)
        print(f"train {mtd} for parrallel")
        parFeatures[mtd] = fit(train, mtd, optParamPar[mtd])
    
    data = {
        "test": data,
        "masks": mask,
        "fawkes": fawkes
    }
    



def hist(image, columns):
    return cv2.calcHist([image], [0], None, [columns], [0, 256])


def fit(train, method, param):
    methods = {
        "hist": hist,
        "dft": dft,
        "dct": dct,
        "scale": scale,
        "grad": grad
    }
    features = []
    # count = 0
    for person in train:
        pers = []
        for img in person:
            pers.append(methods[method](img, param))
        features.append(pers)
        # count += 1
        #print(f"train {count}/{40}")
    return features


def predict(features, test, method, param):
    methods = {
        "hist": hist,
        "dft": dft,
        "dct": dct,
        "scale": scale,
        "grad": grad
    }
    yes = 0.0
    no = 0.0
    plt.ion()
    for i in range(len(test)):
        for img in test[i]:
            plt.subplot(221)
            plt.imshow(img)
            plt.title("image")
            plt.axis("off")
            plt.pause(0.01)
            feature = methods[method](img, param)
            plt.subplot(212)
            if method == "dft":
                plt.imshow(feature[:, :optParam["dft"][1]])
            else:
                plt.imshow(feature)
            plt.title(method)
            plt.axis("off")
            plt.pause(0.01)
            no += 1
            minLoss = np.inf
            res = 0
            for j in range(len(features)):
                for index, train in enumerate(features[j]):
                   loss = calcLoss(feature, train) 
                   if loss < minLoss:
                       minLoss = loss
                       res = j
                       plt.subplot(222)
                       plt.imshow(data["test"][j][index])
                       plt.title("result")
                       plt.axis("off")
                       plt.pause(0.01)
            if res == i:
                yes += 1
        # print(f"predict {i}/40, {yes}, {no}, {res}, {i}")
    plt.ioff()
    plt.show()
    return yes / no


def testScale():
    if choise == "test":
        train, test = splitData(data[choise], optParam["scale"][0] // 40)
        predict(features["scale"], test, "scale", optParam["scale"][1])
    else:
        predict(features["scale"], data[choise], "scale", optParam["scale"][1]) 


def testHist():
    if choise == "test":
        train, test = splitData(data[choise], optParam["hist"][0] // 40)
        predict(features["hist"], test, "hist", optParam["hist"][1])
    else:
        predict(features["hist"], data[choise], "hist", optParam["hist"][1]) 


def testDct():
    if choise == "test":
        train, test = splitData(data[choise], optParam["dct"][0] // 40)
        predict(features["dct"], test, "dct", optParam["dct"][1])
    else:
        predict(features["dct"], data[choise], "dct", optParam["dct"][1]) 


def testGrad():
    if choise == "test":
        train, test = splitData(data[choise], optParam["grad"][0] // 40)
        predict(features["grad"], test, "grad", optParam["grad"][1])
    else:
        predict(features["grad"], data[choise], "grad", optParam["grad"][1]) 


def testDft():
    if choise == "test":
        train, test = splitData(data[choise], optParam["dft"][0] // 40)
        predict(features["dft"], test, "dft", optParam["dft"][1])
    else:
        predict(features["dft"], data[choise], "dft", optParam["dft"][1]) 


def predictParallel(features, test, params, weights):
    methods = {
        "hist": hist,
        "dft": dft,
        "dct": dct,
        "scale": scale,
        "grad": grad
    }
    yes = 0.0
    no = 0.0
    plt.ion()
    for i in range(len(test)):
        for img in test[i]:
            plt.clf()
            plt.subplot(321)
            plt.imshow(img)
            plt.title("image")
            plt.axis("off")
            plt.pause(0.01)
            no += 1
            feature = {}
            result = {}
            for index, mtd in enumerate(methods.keys()):
                feature = methods[mtd](img, params[mtd])
                plt.subplot(3,5, index+6)
                if mtd == "dft":
                    plt.imshow(feature[:, :optParam["dft"][1]])
                else:
                    plt.imshow(feature)
                plt.title(mtd)
                plt.axis("off")
                plt.pause(0.001)
                minLoss = np.inf
                res = 0
                for j in range(len(features[mtd])):
                    for ind, train in enumerate(features[mtd][j]):
                        loss = calcLoss(feature, train) 
                        if loss < minLoss:
                            minLoss = loss
                            res = j
                            plt.subplot(3, 5, index+11)
                            plt.imshow(data["test"][j][ind])
                            plt.title("result")
                            plt.axis("off")
                            plt.pause(0.001)
                if res in result.keys():
                    result[res] += weights[mtd]
                else:
                    result[res] = weights[mtd]
            res = 0
            ans = 0
            for key, value in result.items():
                if value > res:
                    res = value
                    ans = key
            if ans == i:
                yes += 1
            plt.subplot(322)
            plt.imshow(data["test"][ans][0])
            plt.title("result")
            plt.axis("off")
            plt.pause(1)
    plt.ioff()
    plt.show()
    return yes / no

def testParallel():
    if choise == "test":
        train, test = splitData(data[choise], 8)
        predictParallel(parFeatures, test, optParamPar, weights)
    else:
        predictParallel(parFeatures, data[choise], optParamPar, weights)


def dft(image, shape):
    return np.abs(
        np.fft.fft2(image)[0:shape][0:shape]
    )


def dct(image, shape):
    return fftpack.dct(fftpack.dct(image.T).T)[0:shape, 0:shape]


def scale(image, scaling = 10):
    return cv2.resize(image, 
        (
            int(image.shape[1] / scaling), 
            int(image.shape[0] / scaling)
        )
    )


def grad(image, kernel):
    if kernel % 2 == 0:
        kernel += 1
    return np.uint8(
        np.abs(
            cv2.Laplacian(image, cv2.CV_64F, ksize=kernel)
        )
    )


def calcLoss(image, test):
    return np.sqrt(np.sum((test.astype(np.int32) - image) ** 2))


def loadData(orlPath):
    data = []
    for i in range(1, 41):
        person = []
        for j in range(1, 11):
            person.append(cv2.imread(f"{orlPath}/s{i}/{j}.jpg", 0))
        # random.shuffle(person)
        data.append(person)
        print(f"loaded {i}/40")
    return data


def loadDataMask(orlPath):
    data = []
    for i in range(1, 41):
        person = []
        for j in range(1, 11):
            person.append(cv2.imread(f"{orlPath}/s{i}/{j}-with-mask.jpg", 0))
        # random.shuffle(person)
        data.append(person)
        print(f"loaded {40+i}/80")
    return data


def loadDataFawkes(orlPath):
    data = []
    for i in range(1, 41):
        person = []
        for j in range(1, 11):
            person.append(cv2.imread(f"{orlPath}/s{i}/{j}_cloaked.jpg", 0))
        # random.shuffle(person)
        data.append(person)
        print(f"loaded {80+i}/120")
    return data


def splitData(data, trainNum):
    train = []
    test = []
    for person in data:
        train.append(person[:trainNum])
        test.append(person[trainNum:])
    return train, test


def choiseChange(event):
    global choise
    choise = chooseChoise.get()
    print(choise)


def App(appName:str, WIDTH:int, HEIGHT:int):
    global choise
    global chooseChoise
    
    window = tk.Tk()
    window.config(background="#f8e1fc")
    window.columnconfigure([0, 1], minsize=20, weight=1)
    window.rowconfigure([0], minsize=50, weight=1)

    window.title(appName)
    
    window.geometry(str(WIDTH)+'x'+str(HEIGHT))
    
    printAllData()
    btn_frame = tk.Frame(window, background="#f8e1fc")
    btn_frame.grid(row=0, column=1)
    btn = tk.Button(btn_frame, command=testScale, text=f"Test Scale", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testHist, text=f"Test Hist", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testGrad, text=f"Test Grad", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testDft, text=f"Test Dft", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testDct, text=f"Test Dct", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testParallel, text=f"Test Parallel", background="#deecff", width=10)
    btn.pack(side="top", pady=1)

    choise = "test"
    choise_frame = tk.Frame(window)
    choise_frame.config(background="#f8e1fc")
    choise_frame.grid(row=0, column=0)
    chooseChoise = ttk.Combobox(choise_frame,   
                            values=[
                                "test",
                                "masks",
                                "fawkes"
                            ],
                            state="readonly",
                            font=BOLD,
                            background="#deecff")
    text = tk.Label(choise_frame, text="Choose dataset:", background="#f8e1fc", font=BOLD)
    text.pack()
    chooseChoise.pack()
    chooseChoise.current(0)
    chooseChoise.bind("<<ComboboxSelected>>", choiseChange)

    window.mainloop()

App('Face recognition', 400, 240)
