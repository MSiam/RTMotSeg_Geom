import numpy as np
import os
import sys
import pdb
import scipy.misc

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')

    img_dir= main_path+split+'/images/'
    short_img_dir= split+'/images/'
    label_dir= main_path+split+'/mask/'
    short_label_dir= split+'/mask/'
    flow_dir= main_path+split+'/optflow/'
    short_flow_dir= split+'/optflow/'

    imgs_files= sorted(os.listdir(img_dir))
    labels_files= sorted(os.listdir(label_dir))
    flow_files= sorted(os.listdir(flow_dir))

    for j in range(len(labels_files)):
#            path_file.write(short_img_dir+imgs_folders[i]+'/'+labels_files[j]
#                    +' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')

        path_file.write(short_img_dir+labels_files[j]+' '+short_flow_dir+flow_files[j]
                +' '+short_label_dir+labels_files[j]+'\n')


    path_file.close()
main(sys.argv[1], sys.argv[2], sys.argv[3])
