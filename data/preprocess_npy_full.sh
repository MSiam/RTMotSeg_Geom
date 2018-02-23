#!/usr/bin/env bash

#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=train_2streamwarp_vivid.txt --out=vivid_2streamwarp_darpa/
#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=val_vivid.txt --out=vivid_darpa/

#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD/ --pathfile=train_kitti.txt --out=kitti_2stream/
#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD/ --pathfile=val_kitti.txt --out=kitti_2stream/

#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/ --pathfile=train_2stream_kittisms.txt --out=kitti_2stream_sms/
#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/ --pathfile=val_2stream_kittisms.txt --out=kitti_2stream_sms/

#python preprocess_npy.py --root=/home/eren/Data/motion_data/ --pathfile=train_2stream_citysms.txt --out=city_2stream_sms/
#python preprocess_npy.py --root=/home/eren/Data/smsnet_test/kitti/ --pathfile=val_2stream_citysms.txt --out=city_2stream_sms/
#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/testing_2/ --pathfile=val_2stream_kittisq.txt --out=kitti_2stream_sq/
#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/testing_2/ --pathfile=val_2stream_efskittisq.txt --out=kitti_2streamefs_sq/
#python preprocess_npy.py --root=/home/eren/Data/smsnet_test/kitti_mine/ --pathfile=val_2stream_sms_mine.txt --out=citykitti_2stream_sms/
python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/testing_3/ --pathfile=val_2stream_kittisq_3.txt --out=kitti_2stream_sq_3/
python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD_pre/testing_4/ --pathfile=val_2stream_kittisq_4.txt --out=kitti_2stream_sq_4/
