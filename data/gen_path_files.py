import numpy as np
import os
import sys
import pdb
import scipy.misc

def main(main_path, split, out_path):

    path_file= open(out_path, 'w')

#    img_dir= main_path+'/pre_images3/'+split+'/'
#    short_img_dir= 'pre_images3/'+split+'/'
#    label_dir= main_path+'/mask/'+split+'/'
#    short_label_dir= 'mask/'+split+'/'
#    flow_dir= main_path+'/flow/'+split+'/'
#    short_flow_dir= 'flow/'+split+'/'
    img_dir= main_path+'/pre_images3/'
    short_img_dir= 'pre_images3/'
    label_dir= main_path+'/labels/'
    short_label_dir= 'labels/'
    flow_dir= main_path+'/flownet_flow/'
    short_flow_dir= 'flownet_flow/'

#    for folder in os.listdir(img_dir):
    imgs_files= sorted(os.listdir(img_dir))
    labels_files= sorted(os.listdir(label_dir))
    flow_files= sorted(os.listdir(flow_dir))
    for j in range(len(labels_files)):
#        path_file.write(short_img_dir+'/'+imgs_files[j]
#                +' '+short_label_dir+'/'+labels_files[j]+'\n')
         path_file.write(short_img_dir+'/'+imgs_files[j]+' '+short_flow_dir+'/'+flow_files[j]
                 +' '+short_label_dir+'/'+labels_files[j]+'\n')

#            tkns= imgs_files[j].split('_')
#            fr = tkns[1].lstrip('0')
#            sq= tkns[2].lstrip('0')
#            if fr == '':
#                fr= '0'
#            if sq == '':
#                sq= '0'
#            sq= str(int(sq)+1)
#            path_file.write(short_img_dir+folder+'/'+imgs_files[j]+' '+short_flow_dir+folder+'/'+flow_files[j]
#                    +' '+short_label_dir+folder+'/'+tkns[0]+'_'+fr+'_'+sq+'.png'+'\n')


    path_file.close()
#main(sys.argv[1], sys.argv[2], sys.argv[3])
main(sys.argv[1], '', sys.argv[2])
