#!/usr/bin/env bash

#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=train_2streamwarp_vivid.txt --out=vivid_2streamwarp_darpa/
#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=val_vivid.txt --out=vivid_darpa/

python preprocess_npy.py --root=/home/eren/Work/geomnet/data/motion_data/ --pathfile=train_citykitti_2stream.txt --out=citykitti_2stream_sms/
python preprocess_npy.py --root=/home/eren/Work/geomnet/data/smsnet_test/kitti/ --pathfile=val_citykitti_2stream.txt --out=citykitti_2stream_sms/
