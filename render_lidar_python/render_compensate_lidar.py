"""
VISUALISE THE LIDAR DATA FROM THE KITTI DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912

Contains two methods of visualizing lidar data interactively.
 - Matplotlib - very slow, and likely to crash, so only 1 out of every 100
                points are plotted.
              - Also, data looks VERY distorted due to auto scaling along
                each axis. (this could potentially be edited)
 - Mayavi     - Much faster, and looks nicer.
              - Preserves actual scale along each axes so items look
                recognizable
"""
import pykitti  # install using pip install pykitti
import os
import numpy as np
import pdb
from mayavi import mlab
import wx
import time
import cv2
import itertools

# Chose which visualization library to use:  "mayavi" or "matplotlib"
# Raw Data directory information
basedir = '/home/menna/Datasets/KITTI_MOD/'
date = '2011_09_26'
drive = '0059'

R = np.array([9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01])
T= np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
calib_imu_2_velo= np.zeros((4,4))
calib_imu_2_velo[:3,:3]= R.reshape((3,3))
calib_imu_2_velo[:3,3]=T
calib_imu_2_velo[3,3]=1

# Optionally, specify the frame range to load
# since we are only visualizing one frame, we will restrict what we load
# Set to None to use all the data
frame_range = range(150, 151, 1)

# Load the data
dataset = pykitti.raw(basedir, date, drive)#, frame_range)
fig1 = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
fig2 = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))

velo = next(iter(itertools.islice(dataset.velo, 0, None)))
pose = next(iter(itertools.islice(dataset.oxts, 0, None))).T_w_imu
pose= calib_imu_2_velo*pose
velo_comp= np.ones((velo.shape[0],4), dtype= velo.dtype)
velo_comp[:,:3]= velo[:,:3]
velo_comp= np.dot(velo,pose)

plt = mlab.points3d(
    velo_comp[:, 0],   # x
    velo_comp[:, 1],   # y
    velo_comp[:, 2],   # z
    velo_comp[:, 2],   # Height data used for shading
    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
    #colormap='spectral',  # 'bone', 'copper',
    color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
    scale_factor=100,     # scale of the points
    line_width=10,        # Scale of the line, if any
    figure=fig1,
)
msplt = plt.mlab_source

velo = next(iter(itertools.islice(dataset.velo, 5, None)))
pose = next(iter(itertools.islice(dataset.oxts, 5, None))).T_w_imu
pose= calib_imu_2_velo*np.linalg.inv(pose)
velo_comp= np.ones((velo.shape[0],4), dtype= velo.dtype)
velo_comp[:,:3]= velo[:,:3]
velo_comp= np.dot(velo,pose)

plt2 = mlab.points3d(
    velo_comp[:, 0],   # x
    velo_comp[:, 1],   # y
    velo_comp[:, 2],   # z
    velo_comp[:, 2],   # Height data used for shading
    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
#    colormap='spectral',  # 'bone', 'copper',
    color=(1,0,0),
    scale_factor=100,     # scale of the points
    line_width=10,        # Scale of the line, if any
    figure=fig1,
)

#velo = next(iter(itertools.islice(dataset.velo, 2, None)))
#pose = next(iter(itertools.islice(dataset.oxts, 2, None))).T_w_imu
#pose= calib_imu_2_velo*pose
#velo_comp= np.ones((velo.shape[0],4), dtype= velo.dtype)
#velo_comp[:,:3]= velo[:,:3]
#velo_comp= np.dot(velo,pose)
#
#plt2 = mlab.points3d(
#    velo_comp[:, 0],   # x
#    velo_comp[:, 1],   # y
#    velo_comp[:, 2],   # z
#    velo_comp[:, 2],   # Height data used for shading
#    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
##    colormap='spectral',  # 'bone', 'copper',
#    color=(0,0,1),
#    scale_factor=100,     # scale of the points
#    line_width=10,        # Scale of the line, if any
#    figure=fig1,
#)
msplt2= plt2.mlab_source

#cv2.imshow('Camera ', cam2)
#cv2.waitKey(10)

#@mlab.animate(delay=100)
#def anim():
#    i= 1
#    for velo in dataset.velo:
#        pose = next(iter(itertools.islice(dataset.oxts, i, None))).T_w_imu
#        cam2 = next(iter(itertools.islice(dataset.cam2, i, None)))[:,:,::-1]
#        i+= 1
#        print('should be redering new scene', pose)
#        msplt.reset(x=velo[:, 0],   # x
#            y=velo[:, 1],   # y
#            z= velo[:, 2],   # z
#            scalars= velo[:, 2]   # Height data used for shading
#        )
#
#        pose = dataset.oxts.next().T_w_imu
#        pose= calib_imu_2_velo*pose
#        cam2 =dataset.cam2.next()[:,:,::-1]
#        velo_comp= np.ones((velo.shape[0],4), dtype= velo.dtype)
#        velo_comp[:,:3]= velo[:,:3]
#        velo_comp= np.dot(velo,pose)
#
#        msplt2.reset(
#            x= velo_comp[:, 0],   # x
#            y= velo_comp[:, 1],   # y
#            z= velo_comp[:, 2],   # z
#            scalars= velo_comp[:, 2],   # Height data used for shading
#            mode="point", # How to render each point {'point', 'sphere' , 'cube' }
#        )
#        print('should be redering new scene')
#        #cv2.imshow('Camera ', cam2)
#        #cv2.waitKey(10)
#        yield
#
#anim()
mlab.show()
