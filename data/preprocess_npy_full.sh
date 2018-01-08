#!/usr/bin/env bash

python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=train_vivid.txt --out=vivid_darpa/
python preprocess_npy.py --root=/home/eren/Data/VIVID_DARPA/ --pathfile=val_vivid.txt --out=vivid_darpa/
