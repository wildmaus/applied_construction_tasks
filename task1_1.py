import cv2
import numpy as np
from matplotlib import pyplot as plt
from time import time

imageFolder = "images"
templateFolder = "templates"
images = [
          '{}/img1.jpg'.format(imageFolder), 
          '{}/img2.jpg'.format(imageFolder), 
          '{}/img3.jpg'.format(imageFolder), 
          '{}/img4.jpg'.format(imageFolder), 
          '{}/img5.jpg'.format(imageFolder), 
          '{}/img6.jpg'.format(imageFolder), 
          '{}/img7.jpg'.format(imageFolder), 
          '{}/img8.jpg'.format(imageFolder),
          f"{imageFolder}/img9.jpg",
          f"{imageFolder}/img10.jpg",
          f"{imageFolder}/img1_cloaked.jpg",
          f"{imageFolder}/img3_cloaked.jpg"
         ]

templates = [
             '{}/tmp1.jpg'.format(templateFolder),
             '{}/tmp2.jpg'.format(templateFolder), 
             '{}/tmp3.jpg'.format(templateFolder)
            ]

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
           'cv2.TM_SQDIFF_NORMED']


def template_matching(image, template, n, m):
    w, h = template.shape[::-1]
    for meth in methods:
        method = eval(meth)
        img = image.copy()

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 5)

        figure_name = "res-" + str(n+1) + "-" + str(m+1) + "-" + str(method)
        plt.savefig("results/{}".format(figure_name))
        


for template in templates:
    templ = cv2.imread(template, 0)
    for image in images:
        img = cv2.imread(image, 0)
        template_matching(img, templ, images.index(image), templates.index(template))
