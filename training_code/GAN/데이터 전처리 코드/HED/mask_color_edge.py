import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import natsort

## 색상 단순화 k-means ##
## 출처 : https://tech-diary.tistory.com/25 ##
def kmeansColorCluster(image, clusters, rounds = 1):
    """
    Parameters
        image <np.ndarray> : 이미지
        clusters <int> : 클러스터 개수 (군집화 개수)
        rounds <int> : 알고리즘을 몇 번 실행할지 (보통 1)
    returns
        clustered Image <np.ndarray> : 결과 이미지
        SSE <float> : 오차 제곱 합
    """

    height, width = image.shape[:2]
    samples = np.zeros([height * width, 3], dtype=np.float32)

    count = 0
    for x in range(height):
        for y in range(width):
            samples[count] = image[x][y]
            count += 1

    '''
    # compactness : SSE = 오차 제곱 합
    # labels : 레이블 배열 (0과 1로 표현)
    # centers : 클러스터 중심 좌표 (k개로 군집화된 색상들)
    '''
    compactness, labels, centers = cv2.kmeans(
        samples,  # 비지도 학습 데이터 정렬
        clusters,  # 군집화 개수
        None,  # 각 샘플의 군집 번호 정렬
        # criteria : kmeans 알고리즘 반복 종료 기준 설정
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                  10000,  # max_iter
                  0.0001),  # epsilon
        # attempts : 다른 초기 중앙값을 이용해 반복 실행할 횟수
        attempts=rounds,
        # flags : 초기 중앙값 설정 방법
        flags=cv2.KMEANS_PP_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]

    # 결과 이미지, 초기 중앙값, 오차제곱합 반환
    return res.reshape((image.shape)), centers, round(compactness, 4)

src_path = "/home/lhy0807/python_maskrcnn/HED/data/pix2pix_data_pattern/"
mask_path = "/home/lhy0807/python_maskrcnn/HED/output/pix2pix_data_canny/"
save_path = "/home/lhy0807/python_maskrcnn/HED/output/pix2pix_data_colorCanny/"
white = "/home/lhy0807/python_maskrcnn/HED/white.jpg"
image_dir = os.listdir(src_path)

image_dir = natsort.natsorted(image_dir)

# for img in image_dir:
#     src = cv2.imread(src_path + img)
#     mask = cv2.imread(mask_path + img, cv2.IMREAD_GRAYSCALE)
#     dst = cv2.imread(white)
#
#     if src is None or mask is None or dst is None:
#         print('Image load failed')
#         sys.exit()
#
#     ## 이미지 색상 단순화에 k-means 사용 ##
#     src, _, _ = kmeansColorCluster(src, 4)
#     # plt.imshow(src)
#     # plt.show()
#
#     src_h, src_w, src_c = src.shape
#     dst = cv2.resize(dst,dsize=(src_h,src_w))
#
#     ## 외곽선 두께 늘리는 코드 (edge detector 코드에 추가함) ##
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#     # mask = cv2.dilate(mask, kernel, iterations=1)
#     # plt.imshow(mask)
#     # plt.show()
#
#     ##  bitwise_and 연산이 안됨.. ##
#     # masked = cv2.bitwise_and(src,mask)
#     # plt.imshow(masked)
#     # plt.show()
#
#     cv2.copyTo(src, mask, dst) ## 출처 : https://deep-learning-study.tistory.com/104 ##
#
#     ## 색깔 edge에서 노이즈 부분 제거 필요 (안됨) ##
#     # dst = cv2.bilateralFilter(dst,-1,50,50)
#     # dst = cv2.GaussianBlur(dst, (3, 3), 5)
#     # plt.imshow(dst)
#     # plt.show()
#
#     cv2.imwrite(save_path + img, dst)
#     print(img)

## 이미지 색상 단순화에 k-means 사용 ##
src = cv2.imread(src_path+'check(90).jpg')
src, _, _ = kmeansColorCluster(src, 4)
cv2.imwrite('/home/lhy0807/python_maskrcnn/HED/data/check_k.jpg',src)
