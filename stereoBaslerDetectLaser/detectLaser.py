import numpy as np
import os
import cv2
import glob

script_dir = os.path.dirname(os.path.realpath(__file__))

#frame 1 laser on
#frame 2 laser off
img_fns = glob.glob(script_dir + '\example\Basler_acA2440-20gm__22864917__20190712_162733221_*')

i = 0
j = 0
img_set_1_sum = 0
while i < len(img_fns):
    print(img_fns[i])
    img = cv2.imread(img_fns[i],cv2.IMREAD_GRAYSCALE)
    img_set_1_sum = img_set_1_sum + cv2.mean(img)[0]
    i = i + 2
    j = j + 1
img_set_1_mean = img_set_1_sum / j

i = 1
j = 0
img_set_2_sum = 0
while i < len(img_fns):
    img = cv2.imread(img_fns[i],cv2.IMREAD_GRAYSCALE)
    img_set_2_sum = img_set_2_sum + cv2.mean(img)[0]
    i = i + 2
    j = j + 1
img_set_2_mean = img_set_2_sum / j

print(img_set_1_mean)
print(img_set_2_mean)
if (img_set_1_mean > img_set_2_mean):
    print("image with laser is the odd frames e.g. 1,3,5,7...")
    print("image without laser is the even frames e.g. 2,4,6,8...")
elif (img_set_2_mean > img_set_1_mean):
    print("image with laser is the even frames e.g. 1,3,5,7...")
    print("image without laser is the odd frames e.g. 2,4,6,8...")
else:
    print("images have equal likelyhood of laser being odd")