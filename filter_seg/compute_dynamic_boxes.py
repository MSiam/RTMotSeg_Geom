import numpy as np
import scipy.misc
import os
import sys
import cv2
import pdb

def compute_iou(box, mask):
    bb= mask[box[1]:box[3], box[0]:box[2]]
    intersection= len(bb[bb==255])/3.0
    area= (box[2]-box[0])*(box[3]-box[1])
    if intersection>area:
        pdb.set_trace()
    return float(intersection/area)

boxes_dir= sys.argv[2]+'kittiboxes/'
imgs_dir= sys.argv[2]+'images/'
out_dir= sys.argv[1]+'shufflenet_dynamic_boxes/'
out2_dir= sys.argv[1]+'shufflenet_dynamic_images/'

f= open('val_kitti.txt', 'r')
files2= []
for line in f:
    files2.append(line.split(' ')[0].split('/')[-1])

counter = 0
#for root, dirs, files in os.walk(sys.argv[1]+'fcn8s_shufflenet_kitti_npy/'):
for f in files2:
    f= files2[counter]
    motion_mask= np.asarray(np.load(sys.argv[1]+'fcn8s_shufflenet_kitti_npy/'+str(counter)+'.npy')*255, dtype=np.uint8)
    counter+= 1
    print('Working on file ', f)
    img= cv2.imread(imgs_dir+f)
    motion_mask= cv2.resize(motion_mask, img.shape[:2][::-1])
    temp= np.zeros_like(motion_mask)
    temp[motion_mask>127]=255
    motion_mask= temp

    det_file= open(boxes_dir+f.split('.')[0]+'.txt', 'r')
    out_file= open(out_dir+f.split('.')[0]+'.txt', 'w')
    boxes= []
    dynamic=[]
    for line in det_file:
        tkns= line.split(' ')
        box= [int(float(t)) for t in tkns[2:]]
        boxes.append(box)
        iou = compute_iou(box, motion_mask)
        print('iou ', iou)
        if iou>0.2:
            dynamic.append('Dynamic')
            cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]),(0,0,255) )
            cv2.putText(img, 'Dynamic', (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX,
                    0.5,(0,0,255),1)
        else:
            dynamic.append('Static')
            cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]),(0,255,0) )
            cv2.putText(img, 'Static', (box[0],box[1]), cv2.FONT_HERSHEY_COMPLEX,
                    0.5,(0,255,0),1)

        out_file.write(dynamic[-1]+' Car '+str(box[0])+' '+
                str(box[1])+' '+str(box[2])+' '+str(box[3])+'\n')
    out_file.close()
    det_file.close()
    cv2.imshow('Mask', motion_mask)
    cv2.imshow('Static-Dynamic', img)
    cv2.imwrite(out2_dir+f, img)
    cv2.waitKey(10)




