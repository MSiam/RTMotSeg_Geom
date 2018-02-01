from mayavi import mlab
from numpy import array, cos, sin, cos

x_coord = array([0.0, 1.0, 0.0, -1.0])
y_coord = array([1.0, 0.0, -1.0, 0.0])
z_coord = array([0.2, -0.2, 0.2, -0.2])

plt = mlab.points3d(x_coord, y_coord, z_coord)

msplt = plt.mlab_source
@mlab.animate(delay=100)
def anim():
    angle = 0.0
    while True:
        x_coord = array([sin(angle), cos(angle), -sin(angle), -cos(angle)])
        y_coord = array([cos(angle), -sin(angle), -cos(angle), sin(angle)])
        msplt.set(x=x_coord, y=y_coord)
        yield
        angle += 0.1

anim()
mlab.show()
