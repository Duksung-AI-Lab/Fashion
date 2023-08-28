from PIL import Image
import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras.models import load_model
from . import hsvkmeans
import os
import sys
import time

sys.path.append('./mrcnn')
# from web.mrcnn.config import Config
# from web.mrcnn.model import MaskRCNN
from .mrcnn.config import Config
from .mrcnn.model import MaskRCNN

from tensorflow.python.keras.backend import set_session


############################################################
#  Configurations
############################################################

class ClothesConfig(Config):
    """Configuration for training on Clothes Dataset(Top, Botton category segmentation custom dataset)
    Derives from the base Config class and overrides values specific
    to the Clothes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Collar"

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
is_loaded = False
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

collar_model, pattern_model, collar_detect_model, topbottom_detect_model, pocket_model, denim_fit_model, \
denim_washing_model, denim_color_model, denim_damage_model, plain_pinting_model, graphic_text_model, sleeve_model, neckline_model = \
    None, None, None, None, None, None, None, None, None, None, None, None, None

user_img_root_path = "./media/"


def load_all_model():
    # executed only start program
    # load all model
    global is_loaded, collar_model, pattern_model, topbottom_detect_model, collar_detect_model, pocket_model, \
        denim_fit_model, denim_washing_model, denim_color_model, denim_damage_model, \
        plain_pinting_model, graphic_text_model, sleeve_model, neckline_model

    if is_loaded is False:
        CollarConfig = InferenceConfig()
        TopBottomConfig = InferenceConfig()
        CollarConfig.NUM_CLASSES = 2
        CollarConfig.NAME = "COLLAR"
        TopBottomConfig.NUM_CLASSES = 3
        TopBottomConfig.IMAGE_META_SIZE = 15
        TopBottomConfig.NAME = "TOPBOTTOM"

        # CollarConfig.display()
        # TopBottomConfig.display()

        # Create MaskRCNN model
        collar_detect_model = MaskRCNN(mode="inference", config=CollarConfig, model_dir="../ai_models/")
        topbottom_detect_model = MaskRCNN(mode="inference", config=TopBottomConfig, model_dir="../ai_models/")

        # Load weights
        print("***************** model loading *****************")
        t = time.time()

        # print(os.getcwd())

        path = "./static/models/"

        # Mask R-CNN 모델
        print("---mask rcnn collars model loading---")
        collar_detect_model.load_weights(path + "mask_rcnn_collars_0050.h5", by_name=True)
        print("---mask rcnn top-bottom model loading---")
        topbottom_detect_model.load_weights(path + 'mask_rcnn_topbottom_0030.h5', by_name=True)

        # 셔츠 모델
        print("---pocket model loading---")
        pocket_model = load_model(path + "pocket_model.h5")
        print("---pattern model loading---")
        pattern_model = load_model(path + 'pattern_5class.h5')
        print("---collars model loading---")
        collar_model = load_model(path + 'collars_4class.h5')

        # 티셔츠 모델
        print("---printing model1 loading---")
        plain_pinting_model = load_model(path + "printing_plain-printing_model.h5")
        print("---printing model2 loading---")
        graphic_text_model = load_model(path + "printing_graphic-text_model.h5")
        print("---sleeve model loading---")
        sleeve_model = load_model(path + "sleeve_2class_transfer_densenet2.h5")
        print("---neckline model loading---")
        neckline_model = load_model(path + "neckline_2classes.h5")

        # 청바지 모델
        print("---fit model loading---")
        denim_fit_model = load_model(path + "denim_fit_model.h5")
        print("---washing model loading---")
        denim_washing_model = load_model(path + "denim_washing_model.h5")
        print("---color model loading---")
        denim_color_model = load_model(path + "denim_color_model.h5")
        print("---damage model loading---")
        denim_damage_model = load_model(path + "denim_damage_model.h5")

        print("***************** loading success *****************")
        print('take ', (time.time() - t), 'secs')

    is_loaded = True


'''
def img_load_reshape():
    # 이미지 불러오기 코드 길어서 함수로 축약
    original_image = cv.imread('user_img.jpg', cv.IMREAD_COLOR)
    image = cv.resize(original_image, (224, 224))
    image = image.astype('float32') / 255.
    image = image.reshape((1, 224, 224, 3))
    return image
'''


def collar_detect(timestamp):
    with graph.as_default():
        set_session(sess)
        # mask-rcnn collar detect
        image = cv.imread(user_img_root_path + 'user_img_' + timestamp + '.jpg', cv.IMREAD_COLOR)
        r = collar_detect_model.detect([image], verbose=1)[0]
    # detect highest score bbox & crop image to the bbox & resize = (224, 224)
    if r['rois'].shape[0] > 0:
        id = np.argmax(r['scores'])
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
                       int(y1 - dif / 2) if int(y1 - dif / 2) >= 0 else 0:int(y2 + dif / 2) if (y2 + dif / 2) <=
                                                                                               image.shape[1] else
                       image.shape[1]]
        else:
            dif = height - width
            crop_img = image[
                       int(x1 - dif / 2) if int(x1 - dif / 2) >= 0 else 0: int(x2 + dif / 2) if int(x2 + dif / 2) <=
                                                                                                image.shape[0] else
                       image.shape[0], y1:y2]
        crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
        # cv.imwrite('./collar_crop_'+timestamp+'.jpg', crop_img)
    else:
        return None
    return crop_img


def topbottom_detect(top_or_bottom, timestamp):
    add_image = Image.open(user_img_root_path + 'user_img_' + timestamp + '.jpg').convert("RGB")
    max_len = max(int(add_image.width), int(add_image.height))
    min_len = min(int(add_image.width), int(add_image.height))
    if int(add_image.width) == int(add_image.height):
        image = add_image.convert("RGB")
    else:
        target_image = Image.open("./static/images/white.jpg")
        target_image = target_image.resize((max_len, max_len))
        box_x = (max_len - min_len) // 2
        if int(add_image.width) > int(add_image.height):
            target_image.paste(im=add_image, box=(box_x, 0))
        else:
            target_image.paste(im=add_image, box=(0, box_x))
        image = target_image

    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Detect objects
    with graph.as_default():
        set_session(sess)
        # mask-rcnn collar detect
        r = topbottom_detect_model.detect([image], verbose=1)[0]
    if r['rois'].shape[0] > 0:
        # detect top & highest score bbox & crop image to the bbox & resize = (224, 224)
        class_id = r['class_ids']
        idx = []
        for i in range(len(class_id)):
            # top index == 1
            # bottom index == 2
            # top_or_bottom >> 1 / 2
            if class_id[i] == top_or_bottom:
                idx.append(i)

        max_score_idx = idx[0]
        if len(idx) > 1:
            for i in range(1, len(idx)):
                if r['scores'][idx[i]] > r['scores'][max_score_idx]:
                    max_score_idx = idx[i]

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
        crop_img = cv.resize(crop_img, dsize=(224, 224), interpolation=cv.INTER_CUBIC)
        # if top_or_bottom == 1:
        #    cv.imwrite('./top_crop_'+timestamp+'.jpg', crop_img)
        # else:
        #    cv.imwrite('./bottom_crop_'+timestamp+'.jpg', crop_img)
    else:
        return None
    return crop_img


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
    '''
    # White background
    rgba = splash
    row, col, ch = rgba.shape
    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = (255, 255, 255)

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    '''
    return np.asarray(splash, dtype='uint8')
    # return splash


def get_top_alpha(image):
    with graph.as_default():
        set_session(sess)
        r = topbottom_detect_model.detect([image], verbose=1)[0]
    # Extract highest score mask
    class_id = r['class_ids']
    idx = []
    for i in range(len(class_id)):
        # top index add to idx
        if class_id[i] == 1:
            idx.append(i)

    max_score_idx = idx[0]
    if len(idx) > 1:
        for i in range(1, len(idx)):
            if r['scores'][idx[i]] > r['scores'][max_score_idx]:
                max_score_idx = idx[i]

    for i in range(r['masks'].shape[-1]):
        if i == max_score_idx:
            continue
        r['masks'][:, :, i] = np.zeros(r['masks'][:, :, i].shape, dtype=r['masks'][:, :, i].dtype)

    # Color splash
    crop_alpha_img = color_splash(image, r['masks'])
    crop_alpha_img = cv.resize(crop_alpha_img, dsize=(256, 256), interpolation=cv.INTER_CUBIC)
    return crop_alpha_img


def shirts_predict(timestamp):
    global sess, graph
    try:
        # Detect Collar and Crop
        collar_crop_img = collar_detect(timestamp=timestamp)
        if collar_crop_img is None:
            return None, None, None
        collar_crop_img = collar_crop_img.astype('float32') / 255.
        collar_crop_img = collar_crop_img.reshape((1, 224, 224, 3))

        # Detect Top and Crop
        top_crop_img = topbottom_detect(top_or_bottom=1, timestamp=timestamp)

        # collor Clustering
        crop_alpha_img = get_top_alpha(top_crop_img)
        # color = hsvkmeans.clustering(crop_alpha_img)
        main_color, sub_color = hsvkmeans.clustering(crop_alpha_img)

        if top_crop_img is None:
            return None, None, None, None
        top_crop_img = top_crop_img.astype('float32') / 255.
        top_crop_img = top_crop_img.reshape((1, 224, 224, 3))

        # Predict collar
        collar_cls_index = ['Band', 'ButtonDown', 'Notched', 'Regular']
        pocket_cls_index = ['0', '1', '2']
        pattern_cls_index = ['check', 'dot', 'floral', 'solid', 'stripe']

        # Predict Pattern, Pocket, Collar
        with graph.as_default():
            set_session(sess)
            collar_result = collar_model.predict(collar_crop_img)
            pocket_result = pocket_model.predict(top_crop_img)
            pattern_result = pattern_model.predict(top_crop_img)

        # print(collar_result)  # [[2.8315226e-09 1.6763654e-08 1.0000000e+00 1.0736538e-08]]
        # print(collar_result_classes)  # [2]

        acc_arr = np.array([max(collar_result[0]), max(pattern_result[0]), max(pocket_result[0])])
        # print('acc_arr: ', acc_arr)

        # if predict accuracy is under 0.6 -> etc
        for i in acc_arr:
            if i < 0.6:
                return None, None, None, None, None

        # 카라, 패턴, 포켓, 색상 예측 클래스 문자열 return
        return collar_cls_index[np.argmax(collar_result[0])], pattern_cls_index[np.argmax(pattern_result[0])], \
               pocket_cls_index[np.argmax(pocket_result[0])], main_color, sub_color

    except Exception as e:
        print("***************exception :", str(e))
        return None, None, None, None, None


def tshirts_predict(timestamp):
    global sess, graph
    try:
        # Detect Top and Crop
        top_crop_img = topbottom_detect(top_or_bottom=1, timestamp=timestamp)

        if top_crop_img is None:
            return None, None, None, None

        # color Clustering
        crop_alpha_img = get_top_alpha(top_crop_img)
        main_color, sub_color = hsvkmeans.clustering(crop_alpha_img)

        top_crop_img = top_crop_img.astype('float32') / 255.
        top_crop_img = top_crop_img.reshape((1, 224, 224, 3))

        # printing pattern
        ##순서 수정 필요!
        printing_cls_index = ['graphic', 'text']
        sleeve_cls_index = ['long', 'half']
        neckline_cls_index = ['round', 'vneck']
        with graph.as_default():
            set_session(sess)
            plain_pinting_result = plain_pinting_model.predict(top_crop_img)
            print("plain_printing_result=", plain_pinting_result)

            if plain_pinting_result[0][0] < plain_pinting_result[0][1]:
                graphic_text_result = graphic_text_model.predict_classes(top_crop_img)
                printing_result = printing_cls_index[graphic_text_result[0]]
            else:
                printing_result = 'plain'

            sleeve_result = sleeve_model.predict(top_crop_img)
            neckline_result = neckline_model.predict(top_crop_img)

        acc_arr = np.array([max(plain_pinting_result[0])])
        # print('acc_arr: ', acc_arr)

        # if predict accuracy is under 0.6 -> etc
        for i in acc_arr:
            if i < 0.6:
                return None, None, None, None, None

        # 프린팅, 소매, 넥라인, 색상 예측 클래스 문자열 return
        return printing_result, sleeve_cls_index[np.argmax(sleeve_result[0])], \
               neckline_cls_index[np.argmax(neckline_result[0])], main_color, sub_color

    except Exception as e:
        print(str(e))
        return None, None, None, None, None


def jeans_predict(timestamp):
    global sess, graph
    try:
        # Read image
        bottom_crop_img = topbottom_detect(top_or_bottom=2, timestamp=timestamp)
        if bottom_crop_img is None:
            return None, None, None, None
        bottom_crop_img = bottom_crop_img.astype('float32') / 255.
        bottom_crop_img = bottom_crop_img.reshape((1, 224, 224, 3))
        # color washing damage fit
        ##순서 확인 & 수정 필요!
        fit_cls_index = ['butcut', 'regular', 'skinny', 'wide']
        damage_cls_index = ['damage', 'nondamage']
        washing_cls_index = ['brush', 'one']
        color_cls_index = ['DeepTone', 'LightTone', 'MiddleTone']
        with graph.as_default():
            set_session(sess)
            fit_result_classes = denim_fit_model.predict_classes(bottom_crop_img)
            damage_result_classes = denim_damage_model.predict_classes(bottom_crop_img)
            washing_result_classes = denim_washing_model.predict_classes(bottom_crop_img)
            color_result_classes = denim_color_model.predict_classes(bottom_crop_img)

    except Exception as e:
        print(str(e))
        return None, None, None, None

    print(fit_cls_index[fit_result_classes[0]], damage_cls_index[damage_result_classes[0]],
          washing_cls_index[washing_result_classes[0]], color_cls_index[color_result_classes[0]])

    return fit_cls_index[fit_result_classes[0]], damage_cls_index[damage_result_classes[0]], washing_cls_index[
        washing_result_classes[0]], color_cls_index[color_result_classes[0]]


load_all_model()
