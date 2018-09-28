# Real-time Segmentation using Appearance, Motion and Geometry

The repo contains the official implementation of real-time motion segmentation with geometric priors used in our IROS'18 paper:

Coming Soon ...

# Description

Real-time Segmentation is of crucial importance to robotics related applications such as autonomous driving, driving assisted systems, and traffic monitoring from unmanned aerial vehicles imagery. We propose a novel two-stream convolutional network for motion segmentation, which exploits flow and geometric cues to balance the accuracy and computational efficiency trade-offs. The geometric cues take advantage of the domain knowledge of the application. In case of mostly planar scenes from high altitude unmanned aerial vehicles (UAVs), homography compensated flow is used. While in the case of urban scenes in autonomous driving, with GPS/IMU sensory data available, sparse projected depth estimates and odometry information are used. The network provides 4x speedup over the state of the art networks in motion segmentation, at the expense of a reduction in the segmentation accuracy in terms of pixel boundaries. 

<div align="center">
<img src="https://github.com/MSiam/RTMotSeg_Geom/blob/master/figures/overview.png" width="70%" height="70%"><br><br>
</div>

# Dependencies

Python 3.5.2  
Tensorflow 1.4

# Usage

Use samples from run.sh

## Inference

```
python3 main.py --load_config=fcn8s_2stream_shufflenet_test.yaml test Train2Stream FCN8s2StreamShuffleNetLate
```

## Training

```
python3 main.py --load_config=fcn8s_2stream_shufflenet_train.yaml train Train2Stream FCN8s2StreamShuffleNetLate
```

# Data

[UAV Imagery VIVID Original Flow](https://drive.google.com/file/d/1WhSQMXmWGyxFKMW46I6-n7bHy3fv0R7B/view?usp=sharing)
[UAV Imagery VIVID Homography Compensated Flow](https://drive.google.com/file/d/17NZjIUz5tPhSIOh5Yp-9dKM7UzubWXWJ/view?usp=sharing)
[KITTI Motion Test](https://drive.google.com/open?id=1NqMN3iEC1G3JtiBGE8YGon_0S0EDkoOz)

# Weights

[Weights Trained on VIVID Original Flow](https://drive.google.com/open?id=18BM-i87dy35CiAM2mtMJWeKLvTx2PsVu)
[Weights Trained on VIVID Homography Compensated Flow](https://drive.google.com/open?id=1qt3oYR6ShmpVpeBbKE4qbouib1geD1C8)
[Cityscapes Motion Weights](https://drive.google.com/open?id=1HN7JmK9Bx6Cxp3nhO4aykpv_1EDWTwgC)

# Demo

[Video Demo](https://youtu.be/6PrmmkeFdS8)

