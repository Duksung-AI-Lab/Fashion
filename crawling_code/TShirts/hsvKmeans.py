# import the necessary packages
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys
import colorsys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# import visualize
sys.path.append(ROOT_DIR)


# HEX
def RGB2HEX(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# 각 label의 백분율
def centroid_histogram(labels):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(labels)) + 1)
    (hist, _) = np.histogram(labels, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

# rgb -> hsv
def rgb2hsv(rgb):
    result_hsv = []

    for one_rgb in rgb:
        # normalize
        rgb1 = one_rgb / 255.0
        r, g, b = rgb1

        v = max(rgb1)  # v

        # h, s
        if v == 0:
            h = 0
            s = 0
        else:
            min_rgb = min(rgb1)

            s  = 1 - (min_rgb / v)

            if v == r:
                h = 60 * (g - b) / (v - min_rgb)
            elif v == g:
                h = 120 + (60 * (b - r)) / (v - min_rgb)
            elif v == b:
                h = 240 + (60 * (r - g)) / (v - min_rgb)
            if h < 0:
                h += 360
            # h /= 360
        #print('HSV : ', h, s, v)
        result_hsv.append([h, s * 100, v * 100])  # h : 0~360도, s,v : 0 ~ 100%
    # print(result_hsv)
    return result_hsv

# color label
def color_label(hsv):
    h, s, v = hsv

    # Black, White, Gray
    if 0 <= v <= 10:
        return "Black"

    if s < 5 and 90 < v <= 100:
        return "White"

    if 10 < v <= 90 and s < 10:
        return "Gray"

    # h : 0도에 적색, 60도에 황색, 120도에 녹색, 180도에 시안(청록색), 240도에 청색, 300도에 마젠타(적보라색)
    # 30도씩 / Red, Orange, Yellow, Chartreuse_Green, Green, Spring Green, Cyan, Azure, Blue, Violet, Magenta, Rose
    h = round(h // 30, 0)
    if h == 1:
        color = "Orange"
    elif h == 2:
        color = "Yellow"
    elif h == 3:
        color = "Chartreuse_Green"
    elif h == 4:
        color = "Green"
    elif h == 5:
        color = "Spring Green"
    elif h == 6:
        color = "Cyan"
    elif h == 7:
        color = "Azure"
    elif h == 8:
        color = "Blue"
    elif h == 9:
        color = "Violet"
    elif h == 10:
        color = "Magenta"
    elif h == 11:
        color = "Rose"
    else :
        color = "Red"

    return color

def clustering(img):
    # bgra -> rgba
    b, g, r, a = cv.split(img)  # img파일을 b,g,r,a로 분리
    img = cv.merge([r, g, b, a])  # b, r을 바꿔서 Merge

    # reshape the image to be a list of pixels
    img2 = img.reshape(img.shape[0] * img.shape[1], 4)

    # alpha = 0인 픽셀 배열에서 제거
    image = []
    for i in img2:
        if i[3] == 0:
            continue
        else:
            image.append(i[:3])
    image = np.array(image)

    # 보다 편리한 데이타 Handling을 위해 DataFrame으로 변환
    feature_names = ['R', 'G', 'B']

    # cluster the pixel intensities
    k = 5
    clt = KMeans(init='k-means++', n_clusters=k)
    clt.fit(image)

    RGB = pd.DataFrame(data=image, columns=feature_names)
    labels = clt.labels_
    RGB['kmeans_cluster'] = labels

    # 각 레이블의 백분율 구하기
    labels_count = RGB['kmeans_cluster'].value_counts()
    # labels_values = labels_count.values.tolist()
    labels_keys= labels_count.keys().tolist()
    # print("labels_values", labels_values)
    #hist = centroid_histogram(labels)  # percentage

    # 색상 구하기 (HSV)
    center_colors = clt.cluster_centers_  # 5가지 주요 색상
    hsv_list = rgb2hsv(center_colors)  # rgb -> hsv

    main_color_rgb = center_colors[labels_keys[0]]
    sub_color_rgb = center_colors[labels_keys[1]]

    # RGB 16 진수로 변환
    main_color_h = RGB2HEX(main_color_rgb)
    sub_color_h = RGB2HEX(sub_color_rgb)
    # print("main:", main_color, "sub:", sub_color)

    # color labeling
    # main color
    main_color = color_label(hsv_list[0])
    sub_color = color_label(hsv_list[1])
    '''
    # sub color2
    print("sub2 hsv :", dic[2][2])
    sub_color2 = color_label(dic[2][2])
    '''
    return main_color, sub_color, main_color_h, sub_color_h
