import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#mask_dir = '/home/eren/Work/geomnet/data/motion_data/mask/train/'
#out_dir = '/home/eren/Work/geomnet/data/motion_data/mask_post/train/'

mask_dir = '/home/eren/Work/geomnet/data/smsnet_test/kitti/mask/'
out_dir = '/home/eren/Work/geomnet/data/smsnet_test/kitti/mask_post/'


for d in sorted(os.listdir(mask_dir)):
    for f in sorted(os.listdir(mask_dir+d)):
        img = cv2.imread(mask_dir+d+'/'+f, 0)
        if 'train' in mask_dir: # Postprocess masks for training data
            img[img!=151] = 0.0
            img[img==151] = 1.0
        else: # Postprocess masks for test data
            img[img!=2]=0
            img[img==2]=1
        cv2.imwrite(out_dir+d+'/'+f, img)


