import numpy as np
import cv2
import matplotlib.pyplot as plt

imageFolder = "images"
images = [
          '{}/img1.jpg'.format(imageFolder), 
          '{}/img2.jpg'.format(imageFolder), 
          '{}/img3.jpg'.format(imageFolder), 
          '{}/img4.jpg'.format(imageFolder), 
          '{}/img5.jpg'.format(imageFolder), 
          '{}/img6.jpg'.format(imageFolder), 
          '{}/img7.jpg'.format(imageFolder), 
          '{}/img8.jpg'.format(imageFolder)
         ]

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def faceDetection (image, image_gray, n):
    haar_cascade_face = cv2.CascadeClassifier('env/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)

    # Let us print the no. of faces found
    print('Faces found: ', len(faces_rects))

    for (x, y, w, h) in faces_rects:
         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)

    #convert image to RGB and show image
    plt.imshow(convertToRGB(image))
    figure_name = "VJ-" + str(n+1) + "-face"
    plt.savefig(figure_name)


def eyeDetection (image, image_gray, n):
    haar_cascade_face = cv2.CascadeClassifier('env/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml')
    faces_rects = haar_cascade_face.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)

    print('Eyes found: ', len(faces_rects))

    for (x, y, w, h) in faces_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)

    plt.imshow(convertToRGB(image))
    figure_name = "VJ-" + str(n+1) + "-eye"
    plt.savefig(figure_name)


for img in images:
    image = cv2.imread(img)
    image2 = image.copy()
    print(image.shape)
    # Converting to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Displaying the grayscale image
    plt.imshow(image_gray, cmap='gray')
    faceDetection(image, image_gray, images.index(img))
    eyeDetection(image2, image_gray, images.index(img))