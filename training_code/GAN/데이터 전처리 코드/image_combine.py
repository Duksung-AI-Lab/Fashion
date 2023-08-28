import cv2
import os

# dir_path = 'silhouette2_resize/'
# save_path = 'silhouette2_combine/'
#
# dir_list = os.listdir(dir_path)
#
# print(dir_list)
# i = 0
#
# for dirs in dir_list:
#     img_list = os.listdir(dir_path+dirs)
#     print(img_list)
#     silhouette_img = cv2.imread(dir_path+dirs+'/'+dirs+'.png')
#     for img in img_list:
#         fashion_img = cv2.imread(dir_path+dirs+'/'+img)
#
#         combine_img = cv2.hconcat([silhouette_img,fashion_img])
#
#         cv2.imwrite(save_path+dirs+'/'+str(i)+'.png',combine_img)
#
#         i+=1

dir_path = 'silhouette_testimg/'
save_path = 'silhouette_testimg/'

img_list = os.listdir(dir_path)

print(img_list)

white_img = cv2.imread('white.png')
i = 0
for img in img_list:
    silhouette_img = cv2.imread(dir_path+img)
    combine_img = cv2.hconcat([silhouette_img,white_img])
    cv2.imwrite(save_path+str(i)+'.png',combine_img)
    i+=1