from PIL import Image
import os
import natsort
import cv2
import numpy as np
# combine 된 이미지 잘라서 색상 입히기

img = Image.open('dataset/band1.png')
img2 = Image.open('dataset/band1.png')
print(img.size)

merge_img = Image.new('RGB',(img.size[0],img.size[1]))

img = img.convert("RGB")
img2 = img2.convert("RGB")

img = img.crop((0,0,256,256))
img2 = img2.crop((256,0,512,256))

for i in range(0, img.size[0]):
    for j in range(0, img.size[1]):
        rgb = img.getpixel((i,j)) # rgb tuple
        print(rgb)
        if max(rgb) > 50: # 검은색 부분 색 변경을 위한 if문,
            rgb2 = img2.getpixel((i,j))
            img.putpixel((i,j),rgb2)
# img2.save('dataset/colorbackground.png')

merge_img.paste(img,(0,0))
merge_img.paste(img2,(256,0))
merge_img.show()

merge_img.save('dataset/paste.png')

# dir_path = 'dataset/shirts_13000_crop/'
# edge_path = 'dataset/shirts_13000_crop_edge/'
# save_path = 'dataset/13000_colorbackground2/'
#
# imgs = os.listdir(dir_path)
# imgs = natsort.natsorted(imgs)
# for img in imgs:
#     edge = Image.open(edge_path+img)
#     color = Image.open(dir_path+img)
#     if color.size[1] == 2001:
#         continue
#     color = color.convert("RGB")
#     # rgb2 = color.getpixel((256,256))
#     # if min(rgb2) > 240:
#     #     edge.save(save_path+img)
#     #     continue
#     # print(rgb2)
#
#     edge = edge.convert("RGB")
#
#     for i in range(0, edge.size[0]):
#         for j in range(0, edge.size[1]):
#             rgb = edge.getpixel((i,j)) # rgb tuple
#             if max(rgb) > 100: # 흰색에 색상 픽셀 값 넣어주기
#                 rgb2 = color.getpixel((i, j))
#                 # rgbSum=tuple(sum(elem) for elem in zip(minus, rgb2))
#                 edge.putpixel((i,j),rgb2)
#                 # edge.putpixel((i,j),rgb2)
#     print(img)
#     edge.save(save_path+img)

# 참고 : https://blog.naver.com/jsk6824/222004059178
# 참고 : https://ddolcat.tistory.com/690