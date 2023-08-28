import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort

# parser = argparse.ArgumentParser(
#         description='This sample shows how to define custom OpenCV deep learning layers in Python. '
#                     'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
#                     'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
# parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
# parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
# parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
# parser.add_argument('--width', help='Resize input image to a specific width', default=256, type=int)
# parser.add_argument('--height', help='Resize input image to a specific height', default=256, type=int)
# parser.add_argument('--savefile', help='Specifies the output video path', default='output.mp4', type=str)
# args = parser.parse_args()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

# Load the model.
# net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
net = cv.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
cv.dnn_registerLayer('Crop', CropLayer)

#이미지 리스트 load
dir_path = "/home/lhy0807/python_maskrcnn/HED/data/pix2pix_data_pattern/"
save_path = "/home/lhy0807/python_maskrcnn/HED/output/pix2pix_data_canny/"
image_dir = os.listdir(dir_path)

## white background (black edge) ##
# for classes in image_dir:
#     images = os.listdir(dir_path+classes)
#     images = natsort.natsorted(images)
#     for img in images:
#         image = cv.imread(dir_path+classes+'/'+img)
#         (H,W) = image.shape[:2]
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         blurred = cv.GaussianBlur(gray, (5, 5), 0)
#         canny = cv.Canny(blurred, 30, 150)
#         inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
#                                    mean=(104.00698793, 116.66876762, 122.67891434),
#                                    swapRB=False, crop=False)
#         net.setInput(inp)
#         out = net.forward()
#         out = out[0, 0]
#         out = cv.resize(out, (image.shape[1], image.shape[0]))
#         out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
#         out = 255 * out
#         out = out.astype(np.uint8)
#         out2 = cv.bitwise_not(out)
#         cv.imwrite(save_path+'hed/'+img, out2)
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
#         canny2 = cv.bitwise_not(canny)
#         cv.imwrite(save_path+'canny/'+img, canny2)

## black background edge (white edge) ##
# for classes in image_dir:
#     images = os.listdir(dir_path+classes)
#     images = natsort.natsorted(images)
#     for img in images:
#         image = cv.imread(dir_path+classes+'/'+img)
#         (H,W) = image.shape[:2]
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         blurred = cv.GaussianBlur(gray, (5, 5), 0)
#         canny = cv.Canny(blurred, 50, 150)
#         inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
#                                    mean=(104.00698793, 116.66876762, 122.67891434),
#                                    swapRB=False, crop=False)
#         net.setInput(inp)
#         out = net.forward()
#         out = out[0, 0]
#         out = cv.resize(out, (image.shape[1], image.shape[0]))
#         out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
#         out = 255 * out
#         out = out.astype(np.uint8)
#         # cv.imwrite(save_path+'hed_black/'+img, out)
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#         canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
#
#         kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#         canny = cv.dilate(canny, kernel, iterations=1)  # 외곽선 두께 늘리기 (출처 : https://pythonq.com/so/python/1710597)
#
#         cv.imwrite(save_path+img, canny)


# for img in image_dir:
#     image = cv.imread(dir_path + '/' + img)
#     (H, W) = image.shape[:2]
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     blurred = cv.GaussianBlur(gray, (5, 5), 0)
#     canny = cv.Canny(blurred, 30, 150)
#     inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
#                                mean=(104.00698793, 116.66876762, 122.67891434),
#                                swapRB=False, crop=False)
#     net.setInput(inp)
#     out = net.forward()
#     out = out[0, 0]
#     out = cv.resize(out, (image.shape[1], image.shape[0]))
#     out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
#     out = 255 * out
#     out = out.astype(np.uint8)
#     out2 = cv.bitwise_not(out)
#     cv.imwrite(save_path + 'hed/' + img, out2)
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
#     canny2 = cv.bitwise_not(canny)
#     cv.imwrite(save_path + 'canny/' + img, canny2)

## white edge for single directory ##
for img in image_dir:
    image = cv.imread(dir_path + img)
    (H, W) = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 50, 150)
    inp = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (image.shape[1], image.shape[0]))
    out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    out = 255 * out
    out = out.astype(np.uint8)
    cv.imwrite(save_path + 'hed_black/' + img, out)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    canny = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    canny = cv.dilate(canny, kernel, iterations=1) # 외곽선 두께 늘리기 (출처 : https://pythonq.com/so/python/1710597)

    cv.imwrite(save_path + img, canny)


