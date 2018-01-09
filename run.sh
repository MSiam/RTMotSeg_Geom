#!/usr/bin/env bash

###################################### MobileNet ##################################################
#1- FCN8s MobileNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_mobilenet_train.yaml train Train FCN8sMobileNet

#2- FCN8s MobileNet Test
#python3 main.py --load_config=fcn8s_mobilenet_test.yaml test Train FCN8sMobileNet

#3- UNet MobileNet Train Coarse+Fine
#python3 main.py --load_config=unet_mobilenet_traincoarse.yaml train Train UNetMobileNet
#python3 main.py --load_config=unet_mobilenet_train.yaml train Train UNetMobileNet

#4- UNet MobileNet Test 
#python3 main.py --load_config=unet_mobilenet_test.yaml test Train UNetMobileNet

#5- Dilation v1 MobileNet Train
#python3 main.py --load_config=dilation_mobilenet_train.yaml train Train DilationMobileNet

#6- Dilation v1 MobileNet Test

#7- Dilation v2 MobileNet Train 
#python3 main.py --load_config=dilationv2_mobilenet_train.yaml train Train DilationV2MobileNet

###################################### ShuffleNet #################################################
#1- FCN8s ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet
python3 main.py --load_config=fcn8s_2stream_shufflenet_train.yaml train Train2Stream FCN8s2StreamShuffleNet

#2- FCN8s ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml inference Train FCN8sShuffleNet

#3- UNet ShuffleNet Train Coarse+Fine
#python3 main.py --load_config=fcn8s_shufflenet_traincoarse.yaml train Train FCN8sShuffleNet
#python3 main.py --load_config=fcn8s_shufflenet_train.yaml train Train FCN8sShuffleNet

#4- UNet ShuffleNet Test
#python3 main.py --load_config=fcn8s_shufflenet_test.yaml test Train FCN8sShuffleNet

#5- Dilation v1 ShuffleNet Train
#python3 main.py --load_config=dilation_shufflenet_train.yaml train Train DilationShuffleNet
