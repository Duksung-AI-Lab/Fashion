import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from sklearn.cluster import KMeans
import cv2
import pandas as pd

MODEL_SETTED=False

# Path to weights file
WEIGHTS_PATH = "/home/guswl4174/1.15.0/web/static/models/mask_rcnn_topbottom_0030.h5" ### 학습된 상하의 인식 모델 경로 넣어주기

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = "../logs"  ### 경로 확인


############################################################
#  Configurations
############################################################

class ClothesConfig(Config):
    """Configuration for training on Clothes Dataset(Top, Botton category segmentation custom dataset)
    Derives from the base Config class and overrides values specific
    to the Clothes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + category(top&bottom)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    USE_MINI_MASK = True


class InferenceConfig(ClothesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Color Classification
############################################################
# HEX
def RGB2HEX(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def color_classification(pageNum, alpha_img_list, write_ws, load_wb):
    rowNum = (pageNum-1)*90+1
    print("color")
    for img in alpha_img_list:
        try:
            # show our image
            b, g, r, a = cv2.split(img)  # img파일을 b,g,r,a로 분리
            img = cv2.merge([r, g, b, a])  # b, r을 바꿔서 Merge

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
            # print(RGB)
            # print()

            # main, sub color 구하기
            labels_values = RGB['kmeans_cluster'].value_counts().index.tolist()
            # print(labels_values)
            center_colors = clt.cluster_centers_
            # print(center_colors)
            main_color = center_colors[labels_values[0]]
            sub_color = center_colors[labels_values[1]]
            # print(main_color, sub_color)

            # RGB 16 진수로 변환
            main_color = RGB2HEX(main_color)
            sub_color = RGB2HEX(sub_color)
            #print("main:", main_color, "sub:", sub_color)
            write_ws.cell(rowNum, 8, main_color)
            write_ws.cell(rowNum, 9, sub_color)

            load_wb.save("TShirts_DB3.xlsx")
            rowNum += 1
        except Exception as e:
            rowNum += 1
            print("Color result didn't be saved")
            print(str(e))



############################################################
#  Predict
############################################################

def predict_result(cls_index, model, crop_img, rowNum, calNum, write_ws, load_wb):
    # 예측
    result_classes = model.predict_classes(crop_img)
    result = model.predict(crop_img)
    #print('예측:', cls_index[result_classes[0]], 100 * max(result[0]))

    # Save output
    if max(result[0]) * 100 >= 70:
        write_ws.cell(rowNum, calNum, cls_index[result_classes[0]])
    else:
        write_ws.cell(rowNum, calNum, "etc")  # 예측 확률 70% 아래는 Etc

    load_wb.save("TShirts_DB3.xlsx")


def predict(pageNum, crop_img_list, write_ws, load_wb):
    ### 학습된 CNN모델 경로지정
    global neckline_model, printing_graphic_text_model, printing_plain_printing_model
    # class name
    # neckline_cls_index = ['round', 'vneck']
    printing_graphic_text_cls_index = ['graphic','text']
    printing_plain_printing_index = ['plain','printing']

    print('model 예측')

    for crop_img in crop_img_list:
        try:
            crop_img = cv.resize(crop_img, (224, 224))
            crop_img = crop_img.astype('float32') / 255.
            crop_img = crop_img.reshape((1, 224, 224, 3))

            predict_result(printing_cls_index, printing_model, crop_img, rowNum, 5, write_ws, load_wb)
            write_ws.cell(rowNum, 6, 'long') #반팔/긴팔 카테고리가 나누어져있어서 따로 모델 필요 X
            predict_result(neckline_cls_index, neckline_model, crop_img, rowNum, 7, write_ws, load_wb)

            rowNum += 1
        except Exception as e:
            rowNum += 1
            print("Predict result didn't be saved")
            print(str(e))


############################################################
#  model Detect
############################################################
## splash-aplha##
def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    # BGR to BGRA
    b_channel, g_channel, r_channel = cv.split(image)
    alpha_channel = np.full(b_channel.shape, 255, dtype=b_channel.dtype)
    image = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

    # alpha = 0
    # b_channel, g_channel, r_channel = cv.split(image)
    alpha_channel = np.zeros(b_channel.shape, dtype=b_channel.dtype)
    alpha = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, alpha).astype(np.uint8)
    else:
        splash = alpha.astype(np.uint8)

    return splash


def detect(pageNum, img_list, write_ws, load_wb):
    global MODEL_SETTED, model, neckline_model, printing_model
    if MODEL_SETTED is False:
        model, neckline_model, printing_model = model_set() # 카라 인식 모델 load
    crop_img_list = []  # 잘라낸 이미지 리스트
    alpha_img_list = [] # 투명 배경 이미지 리스트

    for image in img_list:
        try:
            # image 흰 배경과 합성해서 정사각형 이미지 생성 (for 비율유지)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            add_image = Image.fromarray(image)
            max_len = max(int(add_image.width), int(add_image.height))
            min_len = min(int(add_image.width), int(add_image.height))
            # print(max_len)
            if int(add_image.width) == int(add_image.height):
                image = add_image.convert("RGB")
            else:
                target_image = Image.open("white.jpg")
                target_image = target_image.resize((max_len, max_len))
                box_x = (max_len - min_len) // 2
                target_image.paste(im=add_image, box=(box_x, 0))
                image = target_image

            ## Read image
            image = np.array(image)
            # print(image.shape)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            # Detect objects
            r = model.detect([image], verbose=1)[0]

            # detect top & highest score bbox & crop image to the bbox & resize = (224, 224)
            class_id = r['class_ids']
            # print(class_id)

            idx = []
            for i in range(len(class_id)):
                # top index == 1
                if class_id[i] == 1:
                    idx.append(i)
            # print(idx)

            max_score_idx = idx[0]
            if len(idx) > 1:
                for i in range(1, len(idx)):
                    if r['scores'][idx[i]] > r['scores'][max_score_idx]:
                        max_score_idx = idx[i]
            # print(max_score_idx)

            # Color splash
            for i in range(r['masks'].shape[-1]):
                if i == max_score_idx:
                    continue
                r['masks'][:, :, i] = np.zeros(r['masks'][:, :, i].shape, dtype=r['masks'][:, :, i].dtype)
            alpha_image = color_splash(image, r['masks'])
            alpha_img_list.append(alpha_image)

            big_box = r['rois'][max_score_idx]
            x1, y1, x2, y2 = big_box
            x1 = x1 - 10 if x1 - 10 > 0 else 0
            y1 = y1 - 10 if y1 - 10 > 0 else 0
            x2 = x2 + 10 if x2 + 10 < image.shape[0] else image.shape[0]
            y2 = y2 + 10 if y2 + 10 < image.shape[1] else image.shape[1]

            width = x2 - x1
            height = y2 - y1
            if width > height:
                dif = width - height
                crop_img = image[x1:x2,
                           int(y1 - dif / 2) if int(y1 - dif / 2) >= 0 else 0:int(y2 + dif / 2) if (y2 + dif / 2) <=
                                                                                                   image.shape[1] else
                           image.shape[1]]
            else:
                dif = height - width
                crop_img = image[
                           int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(x2 + dif / 2) <=
                                                                                                    image.shape[0] else
                           image.shape[0], y1:y2]
            #print("bbox 가로 세로:", crop_img.shape)
            crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
            #print("resize후 가로 세로:", crop_img.shape)

            crop_img_list.append(crop_img)

        except Exception as e:
            crop_img_list.append([])
            alpha_img_list.append([])
            print("detect fail")
            print(str(e))

    predict(pageNum, crop_img_list, write_ws, load_wb)
    color_classification(pageNum, alpha_img_list, write_ws, load_wb)



############################################################
#  model setting
############################################################
def model_set():
    global MODEL_SETTED, neckline_model, printing_model, printing_graphic_text_model, printing_plain_printing_model
    if MODEL_SETTED is False:
        MODEL_SETTED=True
    # Configurations
    config = InferenceConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config,
                     model_dir=DEFAULT_LOGS_DIR)
    # Load weights
    model.load_weights(WEIGHTS_PATH, by_name=True)
    # neckline_model = load_model("./neckline_2classes.h5")
    printing_graphic_text_model = load_model("./printing_graphic-text_model.h5")
    printing_plain_printing_model = load_model("./printing_plain-printing_model.h5")
    print('**************************MODEL SETTED**********************')
    return model, printing_graphic_text_model, printing_plain_printing_model