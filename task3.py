from cProfile import label
from cgi import test
from re import M
from tkinter import W
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


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
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    for i in range(len(test)):
        for img in test[i]:
            feature = methods[method](img, param)
            no += 1
            minLoss = np.inf
            res = 0
            for j in range(len(features)):
                for train in features[j]:
                    loss = calcLoss(feature, train) 
                    if loss < minLoss:
                        minLoss = loss
                        res = j
            if res == i:
                yes += 1
    return yes / no


def predictStack(features, test, method, param):
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    result = []
    match = []
    for i in range(len(test)):
        for img in test[i]:
            feature = methods[method](img, param)
            no += 1
            minLoss = np.inf
            res = 0
            for j in range(len(features)):
                for train in features[j]:
                    loss = calcLoss(feature, train) 
                    if loss < minLoss:
                        minLoss = loss
                        res = j
            if res == i:
                yes += 1
            result.append(yes/no)
            match.append((i, res))
    return (result, match) 


def predictParallel(features, testG, testR, params, weights):
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    for i in range(len(testG)):
        for index, img in enumerate(testG[i]):
            no += 1
            feature = {}
            result = {}
            for mtd in methods.keys():
                if mtd == "orb":
                    feature = methods[mtd](img, params[mtd])
                else:
                    feature = methods[mtd](testR[i][index], params[mtd])
                minLoss = np.inf
                res = 0
                for j in range(len(features[mtd])):
                    for train in features[mtd][j]:
                        loss = calcLoss(feature, train) 
                        if loss < minLoss:
                            minLoss = loss
                            res = j
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
    return yes / no


def predictParallelStack(features, testG, testR, params, weights):
    methods = {
        "rgbHist": rgbHist,
        "orb": orb,
        "random": random
    }
    yes = 0.0
    no = 0.0
    rslt = []
    match = []
    for i in range(len(testG)):
        for index, img in enumerate(testG[i]):
            no += 1
            feature = {}
            result = {}
            for mtd in methods.keys():
                if mtd == "orb":
                    feature = methods[mtd](img, params[mtd])
                else:
                    feature = methods[mtd](testR[i][index], params[mtd])
                minLoss = np.inf
                res = 0
                for j in range(len(features[mtd])):
                    for train in features[mtd][j]:
                        loss = calcLoss(feature, train) 
                        if loss < minLoss:
                            minLoss = loss
                            res = j
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
            match.append((i, ans))
            rslt.append(yes/no)
    return (rslt, match)


# preparation
global choice
choice = np.empty(0)
# program starts here
dataGray = loadData("painters")
dataRgb = loadData("painters", -1)
# img = cv2.imread("painters/1/1.jpg")
# img = cv2.split(img)
# for i in range(3):
#     print(np.sum(img[i].astype(np.int32) - dataRgb[0][0][i]))
# hist1 = rgbHist(img, 50)
# print(len(hist1))
# print()
# hist2 = fit([[img]], "rgbHist", 50)
# print(len(hist2[0][0]))
# print(predict(hist2, [[img]], "rgbHist", 50, True))
# for i in range(3):
#     print(np.sqrt(np.sum((hist1[i].astype(np.int32) - hist2[0][0][i]))**2))
#     print(calcLoss(hist1[i], hist2[0][0][i]))
# print(dataGray)
# print(dataRgb)
# print(type(dataGray[0][0]), len(dataRgb[0][0]))
print("----------------------------------------------------------------------")
trainG, testG = splitData(dataGray, 9)
print(len(trainG[0][0][0]), len(testG[0][0][0]))
trainR, testR = splitData(dataRgb, 9)
print(len(trainR[0]), len(testR[0]))
mtds = [
    ("rgbHist", 1, 1, 50), 
    ("random", 1000, 100, 4000),
    ("orb", 100, 100, 1000)
]
resTrain = []
resTest = []
for (mtd, _, _, end) in mtds:
    print(f"train {mtd}")
    if mtd == "orb":
        features = fit(trainG, mtd, end)
    else:    
        features = fit(trainR, mtd, end)
    #print(features)
    # print(len(features[4][0]), len(features[0]))
    print(f"test {mtd} on train")
    if mtd == "orb":
        resTrain.append(predict(features, trainG, mtd, end))
    else:
        resTrain.append(predict(features, trainR, mtd, end))
    print(f"result = {resTrain[-1]}")
    print(f"test {mtd} on test")
    print("----------------------------------------------------------------------")
    if mtd == "orb":
        resTest.append(predict(features, testG, mtd, end))
    else:
        resTest.append(predict(features, testR, mtd, end))

plt.ion()
plt.scatter(["rgbHist", "random", "orb"], resTrain, label="Train")
plt.scatter(["rgbHist", "random", "orb"], resTest, label="Test")
plt.xlabel("method")
plt.ylabel("accuracy")
plt.legend()    
plt.title("Test on train and test")
plt.grid(True)
plt.savefig(f"figures2/firstCheck.png")
plt.pause(1)


trainSize = [5 * i for i in range(1, 10)]
# print("cross validation")

# optParam = {}
# accuracy = {
#     "rgbHist": [],
#     "random": [],
#     "orb": []
# }
# params = {
#     "rgbHist": [],
#     "random": [],
#     "orb": [] 
# }
# for i in range(1, 10):
#     trainG, testG = splitData(dataGray, i)
#     trainR, testR = splitData(dataRgb, i)
#     print(f"train with trainSize = {trainSize[i - 1]}")
#     print("----------------------------------------------------------------------")
#     for mtd, start, step, end in mtds:
#         print(f"train {mtd}")
#         s = start
#         accuracyTemp = []
#         param = []
#         while s <= end:
#             print(f"train with param = {s}")
#             if mtd == "orb":
#                 features = fit(trainG, mtd, s)
#             else:    
#                 features = fit(trainR, mtd, s)
#             if mtd == "orb":
#                 res = predict(features, testG, mtd, s)
#             else:
#                 res = predict(features, testR, mtd, s)
#             param.append(s)
#             accuracyTemp.append(res)
#             s += step

#         accuracy[mtd].append(max(accuracyTemp))
#         params[mtd].append(param[accuracyTemp.index(max(accuracyTemp))])
# for key in accuracy:
#     for elem in accuracy[key]:
#         print(elem, end=" ")
#     print()
# for key in params:
#     for elem in params[key]:
#         print(elem, end=" ")
#     print()
# plt.clf()
# for mtd in accuracy.keys():
#     plt.plot(trainSize, accuracy[mtd], label=mtd)
#     index = accuracy[mtd].index(max(accuracy[mtd]))
#     optParam[mtd] = [trainSize[index], params[mtd][index]]

# print(optParam)
# plt.legend()
# plt.xlabel("train size")
# plt.ylabel("accuracy")
# plt.grid(True)
# plt.title("Cross validation")
# plt.savefig("figures2/crossValidate.png")
# plt.pause(0.01)

# 0.37777777777777777 0.45 0.45714285714285713 0.43333333333333335 0.44 0.6 0.6666666666666666 0.8 0.6 
# 0.24444444444444444 0.35 0.3142857142857143 0.43333333333333335 0.44 0.45 0.4666666666666667 0.5 0.6 
# 0.4 0.4 0.42857142857142855 0.5 0.52 0.45 0.4666666666666667 0.6 0.8 
# 3 19 17 4 2 5 5 16 2 
# 1100 1700 1000 1700 1300 1000 1000 1000 1000  
# 500 600 200 200 200 200 200 200 600
params = {
    "random": [
        1100, 1700, 1000, 1700, 1300, 1000, 1000, 1000, 1000 
    ],
    "orb": [500, 600, 200, 200, 200, 200, 200, 200, 600],
    "rgbHist": [3, 19, 17, 4, 2, 5, 5, 16, 2]
}
accuracy = {
    "random": [
        0.24444444444444444, 
        0.35, 
        0.3142857142857143, 
        0.43333333333333335, 
        0.44, 
        0.45, 
        0.4666666666666667, 
        0.5, 
        0.6 
    ],
    "orb": [
        0.4, 
        0.4, 
        0.42857142857142855, 
        0.5, 
        0.52, 
        0.45, 
        0.4666666666666667, 
        0.6, 
        0.8 
    ],
    "rgbHist": [
        0.37777777777777777, 
        0.45,
        0.45714285714285713, 
        0.43333333333333335, 
        0.44,
        0.6, 
        0.6666666666666666, 
        0.8, 
        0.6 
    ]
}

optParam = {'rgbHist': [40, 16], 'random': [45, 1000], 'orb': [45, 600]}

print("----------------------------------------------------------------------")
# train all methods with optimal params and max train size
features = {}
mtds = ["random", "orb", "rgbHist"]
testRes = []
trainG, testG = splitData(dataGray, 9)
trainR, testR = splitData(dataRgb, 9)
print("train all methods with optimal params on max train size")
for mtd in mtds:
    print(f"train {mtd}")
    if mtd == "orb":
        features[mtd] = fit(trainG, mtd, params[mtd][8])
    else:
        features[mtd] = fit(trainR, mtd, params[mtd][8])
    print(f"check {mtd} on test")
    if mtd == "orb":
        testRes.append(predict(features[mtd], testG, mtd, params[mtd][8]))
    else:
        testRes.append(predict(features[mtd], testR, mtd, params[mtd][8]))

print(testRes)
plt.clf()
plt.plot(mtds, testRes, label="max trainSize", marker="*")
# train all methods with optimal params and 40 train size
features = {}
testRes = []
trainG, testG = splitData(dataGray, 8)
trainR, testR = splitData(dataRgb, 8)
print("train all methods with optimal params on 40 train size")
for mtd in mtds:
    print(f"train {mtd}")
    if mtd == "orb":
        features[mtd] = fit(trainG, mtd, params[mtd][7])
    else:
        features[mtd] = fit(trainR, mtd, params[mtd][7])
    print(f"check {mtd} on test")
    if mtd == "orb":
        testRes.append(predict(features[mtd], testG, mtd, params[mtd][7]))
    else:
        testRes.append(predict(features[mtd], testR, mtd, params[mtd][7]))
print(testRes)
plt.plot(mtds, testRes, label="40 trainSize", marker="+")

# train all methods with optimal params and optimal train size
features = {}
testRes = []
print("train all methods with optimal params on max train size")
for mtd in mtds:
    print(f"train {mtd}")
    if mtd == "orb":
        trainG, testG = splitData(dataGray, optParam[mtd][0] // 5)
        features[mtd] = fit(trainG, mtd, optParam[mtd][1])
    else:
        trainR, testR = splitData(dataRgb, optParam[mtd][0] // 5)
        features[mtd] = fit(trainR, mtd, optParam[mtd][1])
    print(f"check {mtd} on test")
    if mtd == "orb":
        testRes.append(predict(features[mtd], testG, mtd, optParam[mtd][1]))
    else:
        testRes.append(predict(features[mtd], testR, mtd, optParam[mtd][1]))
print(testRes)
plt.plot(mtds, testRes, label="optimal trainSize", marker="x")

plt.xlabel("method")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on max train size")
plt.savefig("figures2/bestAccuracy.png")
plt.pause(3)

# [0.6, 0.8, 0.6]
# [0.5, 0.6, 0.8]
# [0.6, 0.8, 0.8]

print("----------------------------------------------------------------------")
print("test methods on stack of people")
plotTest = {}
trainG, testG = splitData(dataGray, 9)
trainR, testR = splitData(dataRgb, 9)
for mtd in mtds:
    print(f"check {mtd} on test")
    if mtd == "orb":
        plotTest[mtd], match = predictStack(features[mtd], testG, mtd, optParam[mtd][1])
    else:
        plotTest[mtd], match = predictStack(features[mtd], testR, mtd, optParam[mtd][1])
    print(f"{mtd} results")
    print(match)

plt.clf()
for mtd in mtds:
    if mtd == "random":
        plt.plot(plotTest[mtd], label=mtd, marker="*")
    else:
        plt.plot(plotTest[mtd], label=mtd)
plt.xlabel("pictures amount")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on stack of pictures")
plt.savefig("figures2/testStackMtds.png")
plt.pause(3)
# random results
# [(0, 0), (1, 2), (2, 2), (3, 0), (4, 4)]
# check orb on test
# orb results
# [(0, 2), (1, 1), (2, 2), (3, 3), (4, 4)]
# check rgbHist on test
# rgbHist results
# [(0, 0), (1, 2), (2, 2), (3, 3), (4, 4)]
print("----------------------------------------------------------------------")
print("parallel system")
print("----------------------------------------------------------------------")
print("""test parallel system with differet train size 
and optimal params from cross validation""")
results = []
parallelFeatures = {}
weights = {}
param = {}
for i in range(1, 10):
    print(f"train with train size = {trainSize[i-1]}")
    trainG, testG = splitData(dataGray, i)
    trainR, testR = splitData(dataRgb, i)
    for mtd in mtds:
        print(f"train {mtd}")
        if mtd == "orb":
            parallelFeatures[mtd] = fit(trainG, mtd, params[mtd][i-1])
            weights[mtd] = accuracy[mtd][i-1]
            param[mtd] = params[mtd][i-1]
        else:
            parallelFeatures[mtd] = fit(trainR, mtd, params[mtd][i-1])
            weights[mtd] = accuracy[mtd][i-1]
            param[mtd] = params[mtd][i-1]
    weights = {
        "random": 1, 
        "orb": 1, 
        "rgbHist": 1
    }
    trainG, testG = splitData(dataGray, 9)
    trainR, testR = splitData(dataRgb, 9)
    results.append(predictParallel(parallelFeatures, testG, testR, param, weights))
    print(f"accuracy {results[-1]}")
    print("----------------------------------------------------------------------")

plt.clf()
plt.plot(trainSize, results)
plt.xlabel("train size")
plt.ylabel("accuracy")
plt.grid(True)
plt.title("Parallel system accuracy")
plt.savefig("figures2/parallelSystemSize.png")
plt.pause(3)

# print("----------------------------------------------------------------------")
print("test optimal train size vs 40 train size vs max train size")
# max train size
print("test on trained max train size")
results = []
weights = {
    "random": 0.6, 
    "orb": 0.8, 
    "rgbHist": 0.6
}
results.append(predictParallel(parallelFeatures, testG, testR, param, weights))

parallelFeatures40 = {}
weights40 = {}
param40 = {}
print("train on 40 train size")
trainG, testG = splitData(dataGray, 8)
trainR, testR = splitData(dataRgb, 8)
for mtd in mtds:
    if mtd == "orb":
        parallelFeatures40[mtd] = fit(trainG, mtd, params[mtd][7])
        weights40[mtd] = accuracy[mtd][7]
        param40[mtd] = params[mtd][7]
    else:
        parallelFeatures40[mtd] = fit(trainR, mtd, params[mtd][7])
        weights40[mtd] = accuracy[mtd][7]
        param40[mtd] = params[mtd][7]
weights40 = {
    "random": 0.5, 
    "orb": 0.6, 
    "rgbHist": 0.8
}
trainG, testG = splitData(dataGray, 9)
trainR, testR = splitData(dataRgb, 9)
results.append(predictParallel(parallelFeatures40, testG, testR, param40, weights40))

print("train on optimal train size")
weightsOpt = {
    "random": 0.6, 
    "orb": 0.8, 
    "rgbHist": 0.8
}
parallelFeaturesOpt = {}
paramOpt = {}
for mtd in mtds:
    if mtd == "orb":
        trainG, testG = splitData(dataGray, optParam[mtd][0] // 5)
        parallelFeaturesOpt[mtd] = fit(trainG, mtd, optParam[mtd][1])
        paramOpt[mtd] = optParam[mtd][1]
    else:
        trainR, testR = splitData(dataRgb, optParam[mtd][0] // 5)
        parallelFeaturesOpt[mtd] = fit(trainR, mtd, optParam[mtd][1])
        paramOpt[mtd] = optParam[mtd][1]
        
trainG, testG = splitData(dataGray, 9)
trainR, testR = splitData(dataRgb, 9)
results.append(predictParallel(parallelFeaturesOpt, testG, testR, paramOpt, weightsOpt))

plt.clf()
plt.plot(["max trainSize", "40 trainSize", "optimal trainSize"], results)
plt.xlabel("test group")
plt.ylabel("accuracy")
plt.grid(True)
plt.title("Accuracy on optimal params and different train size")
plt.savefig("figures2/parralelMaxAccuracy.png")
plt.pause(3)

print("----------------------------------------------------------------------")
print("test parallel system on stack of paints")
results = {}
results["max"], match = predictParallelStack(parallelFeatures, testG, testR, param, weights)
print("max results")
print(match)
results["40"], match = predictParallelStack(parallelFeatures40, testG, testR, param40, weights40)
print("40 results")
print(match)
results["opt"], match = predictParallelStack(parallelFeaturesOpt, testG, testR, paramOpt, weightsOpt)
print("opt results")
print(match)

plt.clf()
plt.plot(results["max"], label="max train size")
plt.plot(results["40"], label="40 train size", marker="*")
plt.plot(results["opt"], label="optimal train size")
plt.xlabel("pictures number")
plt.ylabel("accuracy")
plt.grid(True)
plt.legend()
plt.title("Accuracy on stack of pictures")
plt.savefig("figures2/parralelStack.png")
plt.pause(3)
print("----------------------------------------------------------------------")
plt.ioff()
plt.show()