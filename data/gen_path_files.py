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
    flow_dir= main_path+'warped_flow/'+split+'/'
    short_flow_dir= 'warped_flow/'+split+'/'

    imgs_folders= sorted(os.listdir(img_dir))
    labels_folders= sorted(os.listdir(label_dir))
    flow_folders= sorted(os.listdir(flow_dir))

    for i in range(len(imgs_folders)):
        imgs_files= sorted(os.listdir(img_dir+imgs_folders[i]))
        labels_files= sorted(os.listdir(label_dir+labels_folders[i]))
        flow_files= sorted(os.listdir(flow_dir+flow_folders[i]))

        for j in range(len(labels_files)):
#            path_file.write(short_img_dir+imgs_folders[i]+'/'+labels_files[j]
#                    +' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')

            fno= int(labels_files[j].split('frame')[1].split('.')[0])
            path_file.write(short_img_dir+imgs_folders[i]+'/'+labels_files[j]+' '+short_flow_dir+flow_folders[i]+'/'+'%06d.png'%fno
                    +' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')


    path_file.close()
main(sys.argv[1], sys.argv[2], sys.argv[3])
