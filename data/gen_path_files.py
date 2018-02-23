import numpy as np
import os
import sys
import pdb
import scipy.misc
import cv2

#def gen_masks(main_path, mask_path):
#    for f in sorted(os.listdir(main_path)):
#        cv2.imwrite(mask_path+f, np.zeros((384,1248)) )

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')

    img_dir= main_path+'/images/'
    short_img_dir= 'images/'
    label_dir= main_path+'/mask/'
    short_label_dir= 'mask/'
    flow_dir= main_path+'/flow/'
    short_flow_dir= 'flow/'

    imgs_files= sorted(os.listdir(img_dir))
    labels_files= sorted(os.listdir(label_dir))
    flow_files= sorted(os.listdir(flow_dir))
    for j in range(len(flow_files)):
         path_file.write(short_img_dir+'/'+imgs_files[j]+' '+short_flow_dir+'/'+flow_files[j]
                 +' '+short_label_dir+'/'+labels_files[j]+'\n')

    path_file.close()

main(sys.argv[1], '', sys.argv[2])

#gen_masks(sys.argv[1], sys.argv[2])
