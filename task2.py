import cv2
import matplotlib
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


def hist(image, columns):
    return cv2.calcHist([image], [0], None, [columns], [0, 256])


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


# def splitDataRandom(data, trainNum):
#     train = []
#     test = []
#     for person in data:
#         random.shuffle(person)
#         train.append(person[:trainNum])
#         test.append(person[trainNum:])
#     return train, test


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
    for i in range(len(test)):
        for img in test[i]:
            feature = methods[method](img, param)
            no += 1
            minLoss = np.inf
            res = 0
            for j in range(len(features)):
                for index, train in enumerate(features[j]):
                   loss = calcLoss(feature, train) 
                   if loss < minLoss:
                       minLoss = loss
                       res = j
            if res == i:
                yes += 1
        # print(f"predict {i}/40, {yes}, {no}, {res}, {i}")
    return yes / no


def predictStack(features, test, method, param):
    methods = {
        "hist": hist,
        "dft": dft,
        "dct": dct,
        "scale": scale,
        "grad": grad
    }
    yes = 0.0
    no = 0.0
    result = []
    for i in range(len(test)):
        for img in test[i]:
            feature = methods[method](img, param)
            no += 1
            minLoss = np.inf
            res = 0
            for j in range(len(features)):
                for index, train in enumerate(features[j]):
                   loss = calcLoss(feature, train) 
                   if loss < minLoss:
                       minLoss = loss
                       res = j
            if res == i:
                yes += 1
            result.append(yes / no) 
        # print(f"predict {i}/40, {yes}, {no}, {res}, {i}")
    return result


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
    for i in range(len(test)):
        for img in test[i]:
            no += 1
            feature = {}
            result = {}
            for mtd in methods.keys():
                feature = methods[mtd](img, params[mtd])
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

def predictParallelStack(features, test, params, weights):
    methods = {
        "hist": hist,
        "dft": dft,
        "dct": dct,
        "scale": scale,
        "grad": grad
    }
    yes = 0.0
    no = 0.0
    ret = []
    for i in range(len(test)):
        for img in test[i]:
            no += 1
            feature = {}
            result = {}
            for mtd in methods.keys():
                feature = methods[mtd](img, params[mtd])
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
            ret.append(yes/no)
    return ret




# here program starts
data = loadData("orl_faces")
mask = loadDataMask("orl_faces_with_mask")
fawkes = loadDataFawkes("orl_faces_high_cloaked")
print("----------------------------------------------------------------------")
# first test 1.0 on train and test
train, test = splitData(data, 9)
mtds = [
    ("scale", 0.2, 0.1, 5), 
    ("hist", 1, 1, 50),
    ("grad", 3, 2, 31), 
    ("dft", 1, 1, 15), 
    ("dct", 1, 1, 15)
]

resTrain = []
resTest = []
for (mtd, _, _, end) in mtds:
    print(f"train {mtd}")
    if mtd == "grad":
        end = 0
    features = fit(train, mtd, end)
    print(f"test {mtd} on train")
    resTrain.append(predict(features, train, mtd, end))
    print(f"result = {resTrain[-1]}")
    print(f"test {mtd} on test")
    print("----------------------------------------------------------------------")
    resTest.append(predict(features, test, mtd, end))

plt.ion()
plt.scatter(["scale", "hist", "grad", "dft", "dct"], resTrain, label="Train")
plt.scatter(["scale", "hist", "grad", "dft", "dct"], resTest, label="Test")
plt.xlabel("method")
plt.ylabel("accuracy")
plt.legend()    
plt.title("Test on train and test")
plt.grid(True)
plt.pause(3)
# plt.savefig(f"figures/firstCheck.png")
# print("----------------------------------------------------------------------")

# find optimal params
# optParam = {}
# for mtd, start, step, end in mtds:
#     print(f"train {mtd}")
#     s = start
#     accuracy = []
#     param = []
#     while s <= end:
#         print(f"train with param = {s}")
#         features = fit(train, mtd, s)
#         res = predict(features, test, mtd, s)
#         param.append(s)
#         accuracy.append(res)
#         s += step

#     print("----------------------------------------------------------------------")
#     plt.plot(param, accuracy)
#     plt.xlabel("param")
#     plt.ylabel("accuracy")
#     plt.title(mtd)
#     plt.savefig(f"figures/{mtd}r.jpg")
#     plt.grid(True)
#     plt.pause(3)
#     plt.clf()
#     # print(accuracy) 
#     # print(max(accuracy), accuracy.index(max(accuracy)))
#     optParam[mtd] = param[accuracy.index(max(accuracy))]
# print(optParam)
# print("----------------------------------------------------------------------")

# get oprimal train size
# optSize = {}
trainSize = [40 * i for i in range(1, 10)]
# metds = ["scale", "hist", "grad", "dft", "dct"]
# for mtd in metds:
#     accuracy = []
#     print(f"find optimal train size for {mtd}")
#     for index, size in enumerate(trainSize):
#         train, test = splitData(data, index + 1)
#         features = fit(train, mtd, optParam[mtd])
#         res = predict(features, test, mtd, optParam[mtd])
#         accuracy.append(res)
    
#     plt.plot(trainSize, accuracy)
#     plt.xlabel("train size")
#     plt.ylabel("accuracy")
#     plt.title(mtd)
#     plt.savefig(f"figures/trainSize_{mtd}.jpg")
#     plt.grid(True)
#     plt.pause(3)
#     plt.clf()
#     optSize[mtd] = trainSize[accuracy.index(max(accuracy))]
# print(optSize)
# print("----------------------------------------------------------------------")

# cross validation
print("cross validation")
# find optimal params for each method for diff train/test amount
# optParam = {}
# accuracy = {
#     "scale": [],
#     "hist": [],
#     "grad": [],
#     "dft": [],
#     "dct": []
# }
# params = {
#     "scale": [],
#     "hist": [],
#     "grad": [],
#     "dft": [],
#     "dct": []   
# }
# for i in range(1, 10):
#     train, test = splitData(data, i)
#     print(f"train with trainSize = {trainSize[i - 1]}")
#     print("----------------------------------------------------------------------")
#     for mtd, start, step, end in mtds:
#         print(f"train {mtd}")
#         s = start
#         accuracyTemp = []
#         param = []
#         while s <= end:
#             print(f"train with param = {s}")
#             features = fit(train, mtd, s)
#             res = predict(features, test, mtd, s)
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
# plt.savefig("figures/crossValidate.png")
# plt.pause(3)
# plt.clf()
accuracy = {
    "scale": [
        0.7222222222222222, 
        0.828125, 
        0.8714285714285714, 
        0.8958333333333334, 
        0.91, 
        0.95, 
        0.9666666666666667,
        0.9625, 
        0.95
    ],
    "hist": [
        0.6, 
        0.68125, 
        0.6964285714285714, 
        0.8041666666666667, 
        0.915, 
        0.9875,
        0.9833333333333333, 
        0.9875, 
        1.0
    ],
    "grad": [
        0.5333333333333333, 
        0.63125, 
        0.6642857142857143, 
        0.725, 
        0.705, 
        0.75,
        0.775, 
        0.8125, 
        0.825
    ],
    "dft": [
        0.7027777777777777, 
        0.821875,
        0.8321428571428572, 
        0.8833333333333333, 
        0.94, 
        0.9875, 
        0.9916666666666667, 
        1.0, 
        1.0
    ],
    "dct": [
        0.6888888888888889, 
        0.81875,
        0.8321428571428572, 
        0.875,
        0.91, 
        0.96875, 
        0.9583333333333334,
        0.9625,
        0.975
    ] 
}
params = {
    "scale": [
        4.899999999999999, 
        0.7999999999999999, 
        3.600000000000002, 
        4.200000000000001, 
        0.2, 
        1.0999999999999999, 
        4.200000000000001, 
        3.4000000000000017,
        3.4000000000000017
    ],
    "hist": [
        32, 32, 20, 19, 30, 17, 17, 13, 11
    ], 
    "grad": [
        17, 17, 17, 17, 17, 17, 17, 17, 17
    ], 
    "dft": [11, 15, 15, 15, 15, 6, 4, 11, 4], 
    "dct": [12, 14, 15, 14, 12, 6, 6, 5, 5]
}
optParam = {
    'scale': [280, 4.200000000000001], 
    'hist': [360, 11], 
    'grad': [360, 17], 
    'dft': [320, 11], 
    'dct': [360, 5]
}
print("----------------------------------------------------------------------")
# train all methods with optimal params and max train size
features = {}
mtds = ["scale", "hist", "grad", "dft", "dct"]
maskRes = []
fawkesRes = []
testRes = []
print("train all methods with optimal params on 360 train size")
for mtd in mtds:
    print(f"train {mtd}")
    train, test = splitData(data, 9)
    features[mtd] = fit(train, mtd, params[mtd][8])
    print(f"check {mtd} on test")
    testRes.append(predict(features[mtd], test, mtd, params[mtd][8]))
    print(f"check {mtd} on masks")
    maskRes.append(predict(features[mtd], mask, mtd, params[mtd][8]))
    print(f"check {mtd} on fawkes")
    fawkesRes.append(predict(features[mtd], fawkes, mtd, params[mtd][8])) 

plt.clf()
plt.plot(mtds, testRes, label="Test", marker="*")
plt.plot(mtds, maskRes, label="Masks", marker="+")
plt.plot(mtds, fawkesRes, label="Fawkes", marker="x")
plt.xlabel("method")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on max train size")
plt.savefig("figures/bestAccuracyMaxSize.png")
plt.pause(3)

# [0.95, 1.0, 0.825, 1.0, 0.975]
# [0.72, 0.4425, 0.94, 0.135, 0.17]
# [0.995, 0.995, 0.98, 1.0, 0.9975]

print("----------------------------------------------------------------------")
# train all methods with optimal params and optiman train size
features = {}
mtds = ["scale", "hist", "grad", "dft", "dct"]
maskResBest = []
fawkesResBest = []
testRes = []
print("train all methods with optimal params and optimal train size")
for mtd in mtds:
    print(f"train {mtd}")
    train, test = splitData(data, optParam[mtd][0] // 40)
    features[mtd] = fit(train, mtd, optParam[mtd][1])
    print(f"check {mtd} on test")
    testRes.append(predict(features[mtd], test, mtd, optParam[mtd][1]))
    print(f"check {mtd} on masks")
    maskResBest.append(predict(features[mtd], mask, mtd, optParam[mtd][1]))
    print(f"check {mtd} on fawkes")
    fawkesResBest.append(predict(features[mtd], fawkes, mtd, optParam[mtd][1])) 

plt.clf()
plt.plot(mtds, testRes, label="Test", marker="*")
plt.plot(mtds, maskResBest, label="Masks", marker="+")
plt.plot(mtds, fawkesResBest, label="Fawkes", marker="x")
plt.xlabel("method")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on optimal params")
plt.savefig("figures/bestAccuracy.png")
plt.pause(3)
# [0.9666666666666667, 1.0, 0.825, 1.0, 0.975]
# [0.685, 0.4425, 0.94, 0.15, 0.17]
# [0.99, 0.995, 0.98, 0.9975, 0.9975]
print("----------------------------------------------------------------------")
print("test methods on stack of people")
plotTest = {}
plotMasks = {}
plotFawkes = {}
train, test = splitData(data, 9)
for mtd in mtds:
    print(f"check {mtd} on test")
    plotTest[mtd] = predictStack(features[mtd], test, mtd, optParam[mtd][1])
    print(f"check {mtd} on masks")
    plotMasks[mtd] = predictStack(features[mtd], mask, mtd, optParam[mtd][1])
    print(f"check {mtd} on fawkes")
    plotFawkes[mtd] = predictStack(features[mtd], fawkes, mtd, optParam[mtd][1])

plt.clf()
for mtd in mtds:
    plt.plot(plotTest[mtd], label=mtd)
plt.xlabel("pictures amount")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on stack of people")
plt.savefig("figures/testStackMtds.png")
plt.pause(3)

plt.clf()
for mtd in mtds:
    plt.plot(plotMasks[mtd], label=mtd)
plt.xlabel("pictures amount")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.grid(True)
plt.title("Accuracy on stack of people with masks")
plt.savefig("figures/masksStackMtds.png")
plt.pause(3)

plt.clf()
for mtd in mtds:
    plt.plot(plotFawkes[mtd], label=mtd)
plt.xlabel("pictures amount")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on stack of people with fawkes")
plt.savefig("figures/fawkesStackMtds.png")
plt.pause(3)



print("----------------------------------------------------------------------")
# parallel system
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
    train, test = splitData(data, i)
    for mtd in mtds:
        print(f"train {mtd}")
        parallelFeatures[mtd] = fit(train, mtd, params[mtd][i-1])
        weights[mtd] = accuracy[mtd][i-1]
        param[mtd] = params[mtd][i-1]
    results.append(predictParallel(parallelFeatures, test, param, weights))
    print(f"accuracy {results[-1]}")
    print("----------------------------------------------------------------------")

plt.clf()
plt.plot(trainSize, results)
plt.xlabel("train size")
plt.ylabel("accuracy")
plt.grid(True)
plt.title("Parallel system accuracy")
plt.savefig("figures/parallelSystemSize.png")
plt.pause(3)

# print("----------------------------------------------------------------------")
# test with masks and fawkes
print("test with masks and fawkes")
param = {}
result = []
for i in range(5):
    param[mtds[i]] = params[mtds[i]][8]
print("trained parallel system with 360 train size and optimal params")

print("check parallel system on test")
result.append(predictParallel(parallelFeatures, test, param, weights))
print("check parallel system on masks")
result.append(predictParallel(parallelFeatures, mask, param, weights))
print("check parallel system on fawkes")
result.append(predictParallel(parallelFeatures, fawkes, param, weights))

# with 320 train size 
print("train system with 320 train size and optimal params")
parallelFeatures320 = {}
weights320 = {}
param320 = {}
for mtd in mtds:
    print(f"train {mtd}")
    parallelFeatures320[mtd] = fit(train, mtd, params[mtd][7])
    weights320[mtd] = accuracy[mtd][7]
    param320[mtd] = params[mtd][7]
print("check parallel system on test")
result.append(predictParallel(parallelFeatures320, test, param320, weights320))
print("check parallel system on masks")
result.append(predictParallel(parallelFeatures320, mask, param320, weights320))
print("check parallel system on fawkes")
result.append(predictParallel(parallelFeatures320, fawkes, param320, weights320))

# with optimal train size
print("train system with optimal train size and optimal params")
parallelFeaturesOpt = {}
paramOpt = {}
for mtd in mtds:
    train, test = splitData(data, optParam[mtd][0] // 40)
    print(f"train {mtd}")
    parallelFeaturesOpt[mtd] = fit(train, mtd, optParam[mtd][1])
    paramOpt[mtd] = optParam[mtd][1]

print("check parallel system on test")
result.append(predictParallel(parallelFeaturesOpt, test, paramOpt, weights))
print("check parallel system on masks")
result.append(predictParallel(parallelFeaturesOpt, mask, paramOpt, weights))
print("check parallel system on fawkes")
result.append(predictParallel(parallelFeaturesOpt, fawkes, paramOpt, weights))

plt.clf()
plt.plot(["test", "masks", "fawkes"], result[:3], marker="*", label="train size 360")
plt.plot(["test", "masks", "fawkes"], result[3:6], marker="+", label="train size 320")
plt.plot(["test", "masks", "fawkes"], result[6:], marker="x", label="optimal train size")
plt.xlabel("test group")
plt.ylabel("accuracy")
plt.grid(True)
plt.legend()
plt.title("Accuracy on optimal params and max train size")
plt.savefig("figures/parralelMaxAccuracy.png")
plt.pause(3)

print("----------------------------------------------------------------------")
print("test on mask and fawkes if change weights")
masksWeights = {}
param = {}
result = []
for i in range(5):
    masksWeights[mtds[i]] = maskRes[i]
    param[mtds[i]] = params[mtds[i]][8]
masksWeights = {
    "scale": 0.72,
    "hist": 0.44,
    "grad": 1.0,
    "dft": 0.55,
    "dct": 0.55
}
print("trained parallel system with 320 train size and optimal params")

print("check parallel system on test")
result.append(predictParallel(parallelFeatures320, test, param320, masksWeights))
print("check parallel system on masks")
result.append(predictParallel(parallelFeatures320, mask, param320, masksWeights))
print("check parallel system on fawkes")
result.append(predictParallel(parallelFeatures320, fawkes, param320, masksWeights))

plt.clf()
plt.plot("test", result[0], marker="*")
plt.plot("masks", result[1], marker="+")
plt.plot("fawkes", result[2], marker="x")
plt.xlabel("test group")
plt.ylabel("accuracy")
plt.grid(True)
plt.title("Accuracy on optimal params and max train size")
plt.savefig("figures/parralelMaxAccuracyWeigthed.png")
plt.pause(3)

print("----------------------------------------------------------------------")
print("test parallel system on stack of peoples")
print("check on test")
plt.clf()
plt.plot(
    predictParallelStack(parallelFeatures320, test, param320, masksWeights),
    label="test"
)
print("check on masks")
plt.plot(
    predictParallelStack(parallelFeatures320, mask, param320, masksWeights),
    label="masks"
)
print("check on fawkes")
plt.plot(
    predictParallelStack(parallelFeatures320, fawkes, param320, masksWeights),
    label="fawkes"
)
plt.xlabel("pictures amount")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.title("Accuracy on stack of people on parallel system")
plt.savefig("figures/parallelStack.png")
plt.pause(3)
print("----------------------------------------------------------------------")
plt.ioff()
plt.show()