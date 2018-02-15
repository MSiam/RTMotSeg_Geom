import numpy as np
import scipy.misc
import os
import sys
import cv2
import pdb
import matplotlib.pyplot as plt

def filter(box, mask, filtered_motion_mask):
    bb= mask[box[1]:box[3], box[0]:box[2]]
    intersection= len(bb[bb==255])/3.0
    area= (box[2]-box[0])*(box[3]-box[1])
    print('iou ', intersection/area)
    if intersection/area >0.2:
        filtered_motion_mask[box[1]:box[3], box[0]:box[2]]= bb
    return filtered_motion_mask

boxes_dir= 'yolo2_dets/'
out_dir= 'out_filtered_masks/'
counter = 0
for f in sorted(os.listdir(boxes_dir)):
    counter= f.split('.')[0].lstrip('0')
    if counter=='':
        counter='0'
    motion_mask= np.asarray(np.load('out_masks/'+str(counter)+'.npy')*255, dtype=np.uint8)
    print('Working on file ', f)
    temp= np.zeros_like(motion_mask)
    temp[motion_mask>127]=255
    motion_mask= temp
    filtered_motion_mask= np.zeros_like(motion_mask)

    det_file= open(boxes_dir+f, 'r')
    for line in det_file:
        tkns= line.split(' ')
        if tkns[0] not in ['Car', 'Truck', 'Van']:
            continue
        box= [int(float(t)) for t in tkns[1:]]
        filtered_motion_mask= filter(box, motion_mask, filtered_motion_mask)

    det_file.close()
    cv2.imshow('Mask', motion_mask)
    cv2.imshow('Filtered Mask', filtered_motion_mask)
    cv2.imwrite(out_dir+f.split('.')[0]+'.png', filtered_motion_mask)
    cv2.waitKey(10)




