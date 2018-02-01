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

# Chose which visualization library to use:  "mayavi" or "matplotlib"
# Raw Data directory information
basedir = '/home/menna/Datasets/KITTI_MOD/'
date = '2011_09_26'
drive = '0059'

# Optionally, specify the frame range to load
# since we are only visualizing one frame, we will restrict what we load
# Set to None to use all the data
frame_range = range(150, 151, 1)

# Load the data
dataset = pykitti.raw(basedir, date, drive)#, frame_range)
fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
for velo in dataset.velo:
    if velo.shape is not None:
        break

plt = mlab.points3d(
    velo[:, 0],   # x
    velo[:, 1],   # y
    velo[:, 2],   # z
    velo[:, 2],   # Height data used for shading
    mode="point", # How to render each point {'point', 'sphere' , 'cube' }
    colormap='spectral',  # 'bone', 'copper',
    #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
    scale_factor=100,     # scale of the points
    line_width=10,        # Scale of the line, if any
    figure=fig,
)
msplt = plt.mlab_source

@mlab.animate(delay=100)
def anim():
    for velo in dataset.velo:
        print('should be redering new scene')
        msplt.reset(x=velo[:, 0],   # x
            y=velo[:, 1],   # y
            z= velo[:, 2],   # z
            scalars= velo[:, 2]   # Height data used for shading
        )
        yield

anim()
mlab.show()
