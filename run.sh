#!/usr/bin/env bash

###################################### MobileNet ##################################################
#python3 main.py --load_config=fcn8s_mobilenet_train.yaml train Train FCN8sMobileNet
#python3 main.py --load_config=fcn8s_mobilenet_test.yaml test Train FCN8sMobileNet
#python3 main.py --load_config=fcn8s_2stream_mobilenet_train.yaml train Train2Stream FCN8s2StreamMobileNet

###################################### ShuffleNet ##################################################
#1- FCN8s ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_2stream_shufflenet_train.yaml train Train2Stream FCN8s2StreamShuffleNet2

#2- FCN8s ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml test Train2Stream FCN8s2StreamShuffleNet
#python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml test Train2Stream FCN8s2StreamShuffleNet2
python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml test_opt Train2Stream FCN8s2StreamShuffleNet2
#python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml inference Train2Stream FCN8s2StreamShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml inference Train FCN8sShuffleNet
