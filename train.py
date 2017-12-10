#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def rot(n, theta):
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K,K)


def get_bbox(p0, p1):
    '''
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    '''
    v = np.array([[p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
                  [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
                  [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]])
    e = np.array([[2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
                  [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]], dtype=np.uint8)

    return v, e


classes = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

# Set of all files in the training set
files = glob('deploy/trainval/*/*_image.jpg')
for snapshot in files:
    print(snapshot)
    img = plt.imread(snapshot)
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    bbox.resize([bbox.size // 11, 11])

    uv = np.dot(proj,np.vstack([xyz, np.ones_like(xyz[0, :])]))
    uv = uv / uv[2, :]
    
    clr = np.linalg.norm(xyz, axis=0)

    

    for k, b in enumerate(bbox):
        n = b[0:3]
        theta = np.linalg.norm(n)
        n /= theta
        R = rot(n, theta)
        t = b[3:6]
    
        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = np.dot(R,vert_3D) + t[:, np.newaxis]
    
        vert_2D = np.dot(proj,np.vstack([vert_3D, np.ones(8)]))
        vert_2D = vert_2D / vert_2D[2, :]

        c = classes[int(b[9])]
        ignore_in_eval = bool(b[10])
#        if ignore_in_eval:
#            ax2.text(t[0], t[1], t[2], c, color='r')
#        else:
#            ax2.text(t[0], t[1], t[2], c)
    