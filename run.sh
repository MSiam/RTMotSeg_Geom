#!/usr/bin/env bash

###################################### VGG16##################################################
#python3 main.py --load_config=fcn8s_2stream_vgg16_train.yaml train Train2Stream FCN8s2Stream
#python3 main.py --load_config=fcn8s_2stream_vgg16_test.yaml test Train2Stream FCN8s2Stream

python3 main.py --load_config=fcn8s_2stream_mobilenet_train.yaml train Train2Stream FCN8s2StreamMobileNet

###################################### ShuffleNet #################################################
#1- FCN8s ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_2stream_shufflenet_train.yaml train Train2Stream FCN8s2StreamShuffleNet

#2- FCN8s ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml test Train2Stream FCN8s2StreamShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml inference Train FCN8sShuffleNet
