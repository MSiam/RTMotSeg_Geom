#!/usr/bin/env bash

#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=train_2streamwarp_vivid.txt --out=vivid_2streamwarp_darpa/
#python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=val_vivid.txt --out=vivid_darpa/

#python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD/ --pathfile=train_kitti.txt --out=kitti_2stream/
python preprocess_npy.py --root=/home/eren/Data/KITTI_MOD/ --pathfile=val_kitti.txt --out=kitti_2stream/
