#!/usr/bin/env bash

python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=train_2stream_vivid.txt --out=vivid_2stream_darpa/
python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=val_2stream_vivid.txt --out=vivid_2stream_darpa/
