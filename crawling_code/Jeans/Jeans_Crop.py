import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from pnslib import utils
import tensorflow as tf
from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from sklearn.cluster import KMeans
import cv2
import pandas as pd


# Path to weights file
WEIGHTS_PATH = "../model/mask_rcnn_clothes_0030.h5" ### 학습된 상하의 인식 모델 경로 넣어주기

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
#  Predict
############################################################

def predict_result(cls_index, model, crop_img, rowNum, calNum, write_ws, load_wb):
    # 예측
    result_classes = model.predict_classes(crop_img)
    result = model.predict(crop_img)
    print('예측:', cls_index[result_classes[0]], 100 * max(result[0]))

    # Save output
    if max(result[0]) * 100 >= 70:
        write_ws.cell(rowNum, calNum, cls_index[result_classes[0]])
    else:
        write_ws.cell(rowNum, calNum, "etc")  # 예측 확률 70% 아래는 Etc

    load_wb.save("Jeans_DB.xlsx")


def predict(rowNum, crop_img_list, write_ws, load_wb):
    ### 학습된 CNN모델 경로지정
    fit_model = load_model("../model/denim_fit_model.h5")
    color_model = load_model("../model/denim_color_model.h5")
    washing_model = load_model("../model/denim_washing_model.h5")
    damage_model = load_model("../model/denim_damage_model.h5")


    # class name
    fit_cls_index = ['butcut', 'regular', 'skinny', 'wide']
    washing_cls_index = ['brush', 'one']
    damage_cls_index = ['damage', 'nondamage']
    color_cls_index = ['DeepTone', 'LightTone', 'MiddleTone']

    for crop_img in crop_img_list:
        try:
            crop_img = cv.resize(crop_img, (224, 224))
            crop_img = crop_img.astype('float32') / 255.
            crop_img = crop_img.reshape((1, 224, 224, 3))

            predict_result(fit_cls_index, fit_model, crop_img, rowNum, 3, write_ws, load_wb)
            predict_result(color_cls_index, color_model, crop_img, rowNum, 4, write_ws, load_wb)
            predict_result(washing_cls_index, washing_model, crop_img, rowNum, 5, write_ws, load_wb)
            predict_result(damage_cls_index, damage_model, crop_img, rowNum, 6, write_ws, load_wb)

            rowNum += 1
        except Exception as e:
            print("rowNum", rowNum)
            rowNum += 1
            print("Predict result didn't be saved")
            print(str(e))


############################################################
#  model Detect
############################################################

def detect(num, img_list, write_ws, load_wb):

    model = model_set() # 모델 load
    crop_img_list = []  # 잘라낸 이미지 리스트

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
                # bottom index == 2
                if class_id[i] == 2:
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
            print("bbox 가로 세로:", crop_img.shape)
            crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
            print("resize후 가로 세로:", crop_img.shape)

            crop_img_list.append(crop_img)

        except Exception as e:
            crop_img_list.append([])
            print("detect fail")
            print(str(e))

    predict(num, crop_img_list, write_ws, load_wb)

############################################################
#  model setting
############################################################
def model_set():
    # Configurations
    config = InferenceConfig()
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config=config,
                     model_dir=DEFAULT_LOGS_DIR)
    # Load weights
    model.load_weights(WEIGHTS_PATH, by_name=True)

    return model