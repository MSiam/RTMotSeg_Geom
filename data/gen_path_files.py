import numpy as np
import os
import sys
import pdb
import scipy.misc

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')

    img_dir= main_path+'images4/'+split+'/'
    short_img_dir= 'images4/'+split+'/'
    label_dir= main_path+'mask_post/'+split+'/'
    short_label_dir= 'mask_post/'+split+'/'
    flow_dir= main_path+'flow/'+split+'/'
    short_flow_dir= 'flow/'+split+'/'

    imgs_folders= sorted(os.listdir(img_dir))
    labels_folders= sorted(os.listdir(label_dir))
    flow_folders= sorted(os.listdir(flow_dir))

    for i in range(len(imgs_folders)):
        imgs_files= sorted(os.listdir(img_dir+imgs_folders[i]))
        labels_files= sorted(os.listdir(label_dir+labels_folders[i]))
        flow_files= sorted(os.listdir(flow_dir+flow_folders[i]))

        for j in range(len(labels_files)):
            if split == 'val':
                path_file.write(short_img_dir+imgs_folders[i]+'/'+labels_files[j]
                        +' '+short_flow_dir+flow_folders[i]+'/'+flow_files[j]
                        +' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')
            elif split == 'train':
                tokens= labels_files[j].split('_')
                fno = int(tokens[1])
                nn = int(tokens[2].split('.')[0])
                path_file.write(short_img_dir+imgs_folders[i]+'/'+tokens[0]+'_%06d_'%fno+'%06d_leftImg8bit.png'%nn
                        +' '+short_flow_dir+flow_folders[i]+'/'+tokens[0]+'_%06d_'%fno+'%06d_leftImg8bit.png'%(nn-1)
                        +' '+short_label_dir+labels_folders[i]+'/'+labels_files[j]+'\n')

    path_file.close()
main(sys.argv[1], sys.argv[2], sys.argv[3])
