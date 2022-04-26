# interfase must looks like
# image scaleimg histimg gradimg dftimg dctimg parrallel
# img    img      img      img    img     img    img  
# matching result
#        img      img      img    img     img    img
# for 3 the same
# cross valid change, must be test photo (from 1 to 40) and result of accuracy
import tkinter as tk
from tkinter.font import BOLD
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
matplotlib.use("TkAgg")

def printAllData():
    global data
    global param
    global weights
    global features
    param = {
        "random": 1000,
        "orb": 600,
        "rgbHist": 2
    }
    weights = {
        "random": 1.0,
        "orb": 0.8,
        "rgbHist": 0.6
    }
    dataGray = loadData("painters")
    dataRgb = loadData("painters", -1)
    features = {}
    trainG, testG = splitData(dataGray, 9)
    trainR, testR = splitData(dataRgb, 9)
    for mtd in param.keys():
        print(f"train {mtd}")
        if mtd == "orb":
            features[mtd] = fit(trainG, mtd, param[mtd])
        else:
            features[mtd] = fit(trainR, mtd, param[mtd])
    data = {
        "gray": dataGray,
        "rgb": dataRgb
    }
    


def random(image, amount):
    global choice
    np.random.seed(4)
    if len(choice) != amount:
        choice = np.random.choice(len(image[0].reshape(-1)), amount, replace=False)
    res = []
    for i in range(3):
        color = []
        for elem in choice:
            color.append(image[i].reshape(-1)[elem])
        res.append(color)
    return np.array(res)


def orb(image, numKp):
    ORB = cv2.ORB_create(nfeatures=numKp)
    _, des = ORB.detectAndCompute(image, None)
    des = cv2.resize(des, (200, 200))
    # print(len(des), des[0])
    # des = des.reshape(250, 250)
    return np.array(des).reshape((200, 200))


def rgbHist(image, columns):
    hist = []
    for i in range(3):
        hist.append(cv2.calcHist([image[i]], [0], None, [columns], [0, 256]))
    return np.array(hist)


def loadData(dataPath, rgb = 0):
    data = []
    if rgb == 0:
        for i in range(1, 6):
            author = []
            for j in range(1, 11):
                author.append(cv2.imread(f"{dataPath}/{i}/{j}.jpg", 0))
            data.append(author)
            print(f"loaded {i}/5")
    else:
        for i in range(1, 6):
            author = []
            for j in range(1, 11):
                paint = cv2.imread(f"{dataPath}/{i}/{j}.jpg")
                author.append(cv2.split(paint))
            data.append(author)
            print(f"loaded {i+5}/10")
    return data


def splitData(data, trainNum):
    train = []
    test = []
    for author in data:
        train.append(author[:trainNum])
        test.append(author[trainNum:])
    return train, test


def calcLoss(image, test):
    return np.sqrt(np.sum((image.astype(np.int32) - test) ** 2))


def fit(train, method, param):
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    features = []
    for author in train:
        #print(author)
        athr = []
        for paint in author:
            #print(paint)
            athr.append(methods[method](paint, param))
        features.append(athr)
    return features


def predict(features, test, method, param):
    painters = ["Aйвазовский", "Моне", "ван Гог", "Магритт", "Шишкин"]
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    plt.ion()
    for i in range(len(test)):
        for img in test[i]:
            plt.subplot(221)
            plt.imshow(cv2.merge(data["rgb"][i][9]))
            plt.title(f"image {painters[i]}")
            plt.axis("off")
            plt.pause(0.05)   
            feature = methods[method](img, param)   
            plt.subplot(212)      
            if method == "random":
                plt.imshow(feature.reshape(30, 100))
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
                        plt.imshow(cv2.merge(data["rgb"][j][index]))
                        plt.title(f"result {painters[j]}")
                        plt.axis("off")
                        plt.pause(0.5)
            if res == i:
                yes += 1
            plt.pause(2)
    return yes / no


def predictParallel(features, testG, testR, params, weights):
    painters = ["Aйвазовский", "Моне", "ван Гог", "Магритт", "Шишкин"]
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    plt.ion()
    for i in range(len(testG)):
        for indx, img in enumerate(testG[i]):
            plt.clf()
            plt.subplot(321)
            plt.imshow(cv2.merge(data["rgb"][i][9]))
            plt.title(f"image {painters[i]}")
            plt.axis("off")
            plt.pause(0.01)
            no += 1
            feature = {}
            result = {}
            for index, mtd in enumerate(methods.keys()):
                if mtd == "orb":
                    feature = methods[mtd](img, params[mtd])
                else:
                    feature = methods[mtd](testR[i][indx], params[mtd])
                plt.subplot(3,3, index+4)
                if mtd == "random":
                    plt.imshow(feature.reshape(30, 100))
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
                            plt.subplot(3, 3, index+7)
                            plt.imshow(cv2.merge(data["rgb"][j][ind]))
                            plt.title(f"result {painters[j]}")
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
            plt.imshow(cv2.merge(data["rgb"][ans][0]))
            plt.title(f"result {painters[ans]}")
            plt.axis("off")
            plt.pause(5)
    return yes / no


def testOrb():
    trainG, testG = splitData(data["gray"], 9)
    predict(features["orb"], testG, "orb", param["orb"])

def testRgbHist():
    trainR, testR = splitData(data["rgb"], 9)
    predict(features["rgbHist"], testR, "rgbHist", param["rgbHist"])

def testRandom():
    trainR, testR = splitData(data["rgb"], 9)
    predict(features["random"], testR, "random", param["random"])

def testParallel():
    trainG, testG = splitData(data["gray"], 9)
    trainR, testR = splitData(data["rgb"], 9)
    predictParallel(features, testG, testR, param, weights)


def App(appName:str, WIDTH:int, HEIGHT:int):
    
    window = tk.Tk()
    window.config(background="#f8e1fc")
    window.columnconfigure([0, 1], minsize=20, weight=1)
    window.rowconfigure([0], minsize=50, weight=1)

    window.title(appName)
    img_frame = tk.Frame(window, background="#f8e1fc", relief=tk.RAISED, borderwidth=5)
    img_frame.grid(row=0, column=0)
    img_label = tk.Label(img_frame)
    img = ImageTk.PhotoImage(Image.open("painters/4/2.jpg"))
    img_label.config(image=img)
    img_label.pack()
    
    window.geometry(str(WIDTH)+'x'+str(HEIGHT))
    global choice  
    choice = np.empty(0)
    printAllData()
    btn_frame = tk.Frame(window, background="#f8e1fc")
    btn_frame.grid(row=0, column=1)
    btn = tk.Button(btn_frame, command=testOrb, text=f"Test Orb", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testRgbHist, text=f"Test RgbHist", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testRandom, text=f"Test Random", background="#deecff", width=10)
    btn.pack(side="top", pady=1)
    btn = tk.Button(btn_frame, command=testParallel, text=f"Test Parallel", background="#deecff", width=10)
    btn.pack(side="top", pady=1)

    window.mainloop()

App('Author recognition', 500, 340)
