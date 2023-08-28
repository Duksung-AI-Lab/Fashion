import os

import cv2
import numpy as np

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

dir_path = 'dataset/wshirt_crop/'
save_path = 'dataset/wshirt_crop_kmeans/'
img_list = os.listdir(dir_path)

for image in img_list:
    img = cv2.imread(dir_path+image)
    img, _, _ = kmeansColorCluster(img, 4)
    cv2.imwrite(save_path+image,img)

