import os
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Path to weights file
WEIGHTS_PATH = "../../model/mask_rcnn_collar_0030.h5"  ### 학습된 카라 인식 모델 경로 넣어주기

# Directory to save logs and model checkpoints, if not provided
DEFAULT_LOGS_DIR = "../../../../logs"  ### 경로 확인


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
    NUM_CLASSES = 1 + 1  # Background + category(collar)

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

def predict(rowNum, crop_img_list, write_ws, load_wb):
    ### 학습된 CNN모델 경로지정
    collar_model = load_model("../../model/카라분류모델_4classes.h5")

    for crop_img in crop_img_list:
        try:
            crop_img = cv.resize(crop_img, (224, 224))
            crop_img = crop_img.astype('float32') / 255.
            crop_img = crop_img.reshape((1, 224, 224, 3))

            # collar 예측
            collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']
            collar_result_classes = collar_model.predict_classes(crop_img)
            collar_result = collar_model.predict(crop_img)
            print('예측:', collar_cls_index[collar_result_classes[0]], 100 * max(collar_result[0]))

            # Save output
            if max(collar_result[0]) * 100 >= 70:
                write_ws.cell(rowNum, 3, collar_cls_index[collar_result_classes[0]])  # Collar
            else:
                write_ws.cell(rowNum, 3, "etc")  # 예측 확률 70% 아래는 Etc

            load_wb.save("Shirts_DB.xlsx")
            rowNum += 1
        except Exception as e:
            print("rowNum", rowNum)
            rowNum += 1
            print(str(e))


############################################################
#  model Detect
############################################################
###  test
def detect(num, img_list, write_ws, load_wb):
    model = model_set()
    # 카라 인식 모델 load
    crop_img_list = []  # 잘라낸 이미지 리스트

    for image in img_list:
        # Detect objects
        try:
            r = model.detect([image], verbose=1)[0]

            # detect highest score bbox & crop image to the bbox & resize = (224, 224)
            if r['rois'].shape[0] > 0:
                id = np.argmax(r['scores'])
                print(r['rois'])
                big_box = r['rois'][id]
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
                               int(y1 - dif / 2) if int(y1 - dif / 2) >= 0 else 0:int(y2 + dif / 2) if (y2 + dif / 2) <= image.shape[1] else image.shape[1]]
                else:
                    dif = height - width
                    crop_img = image[
                               int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(x2 + dif / 2) <= image.shape[0] else image.shape[0], y1:y2]
                print("bbox 가로 세로:", crop_img.shape)
                crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
                print("resize후 가로 세로:", crop_img.shape)

                crop_img_list.append(crop_img)
                print("detect success")
            else:
                crop_img_list.append([])
                print("detect fail")

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
