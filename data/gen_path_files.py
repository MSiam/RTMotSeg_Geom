import numpy as np
import os
import sys
import pdb
import scipy.misc

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')

    img_dir= main_path+'images/'+split+'/'
    short_img_dir= 'images/'+split+'/'
    label_dir= main_path+'masks/'+split+'/'
    short_label_dir= 'masks/'+split+'/'

    imgs_folders= sorted(os.listdir(img_dir))
    labels_folders= sorted(os.listdir(label_dir))
    for i in range(len(imgs_folders)):
        imgs_files= sorted(os.listdir(img_dir+imgs_folders[i]))
        labels_files= sorted(os.listdir(label_dir+labels_folders[i]))

        for j in range(len(labels_files)):
            path_file.write(short_img_dir+imgs_folders[i]+'/'+labels_files[j]+' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')


    path_file.close()
main(sys.argv[1], sys.argv[2], sys.argv[3])
