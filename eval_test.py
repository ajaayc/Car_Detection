#! /usr/bin/python3
import tensorflow as tf
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

class CarClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = 'fine_tuned_model/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            return boxes, scores, classes, num



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

#Output file
f = open('result.txt','w')
f.write('guid/image,N\n')

# Set of all files in the test set
#All x thousand test images
files = glob('rob599_dataset_deploy/test/*/*_image.jpg')
#Small subset of test images
#files = glob('deploy/test/*/*_image.jpg')

#Consider an object to be anything with a score greater than this
score_threshold = 0.05

maxcars = 12

model = CarClassifier()

for snapshot in files[6:11]:
    print(snapshot)
    img = plt.imread(snapshot)
    
    #RANDOM COUNT
    #numcars = random.randint(0,maxcars + 1)
    
    #Neural Network count
    m_output = model.get_classification(img)
    boxes = m_output[0]
    scores = m_output[1]
    classes = m_output[2]
    num = m_output[3]
    
    scores_subset = scores[0][0:maxcars]

    countif = np.where(scores_subset > score_threshold)
    
    numcars = np.shape(countif)[1]
    
    #Get image name
    name = snapshot.replace('rob599_dataset_deploy/test/','')
    name = name.replace('_image.jpg','')
    name = name + "," + str(numcars)
    
    f.write(name + '\n')
    
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz.resize([3, xyz.size // 3])
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

#    try:
#        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
#    except:
#        print('[*] bbox not found.')
#        bbox = np.array([], dtype=np.float32)
#    bbox.resize([bbox.size // 11, 11])

#    uv = np.dot(proj,np.vstack([xyz, np.ones_like(xyz[0, :])]))
#    uv = uv / uv[2, :]
#    
#    clr = np.linalg.norm(xyz, axis=0)
#    fig1 = plt.figure(1, figsize=(16, 9))
#    ax1 = fig1.add_subplot(1, 1, 1)
#    ax1.imshow(img)
#    ax1.scatter(uv[0, :], uv[1, :], c=clr, marker='.', s=1)
#    ax1.axis('scaled')
#    fig1.tight_layout()
    
#    fig2 = plt.figure(2, figsize=(8, 8))
#    ax2 = Axes3D(fig2)
#    ax2.set_xlabel('x')
#    ax2.set_ylabel('y')
#    ax2.set_zlabel('z')
    
#    step = 5
#    ax2.scatter(xyz[0, ::step], xyz[1, ::step], xyz[2, ::step], \
#        c=clr[::step], marker='.', s=1)
    
#    colors = ['C{:d}'.format(i) for i in range(10)]
#    for k, b in enumerate(bbox):
#        n = b[0:3]
#        theta = np.linalg.norm(n)
#        R = rot(n, theta)
#        t = b[3:6]
#        n /= theta
#    
#        sz = b[6:9]
#        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
#        vert_3D = np.dot(R,vert_3D) + t[:, np.newaxis]
#    
#        vert_2D = np.dot(proj,np.vstack([vert_3D, np.ones(8)]))
#        vert_2D = vert_2D / vert_2D[2, :]
    
#        clr = colors[k % len(colors)]
#        for e in edges.T:
#            ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)
#            ax2.plot(vert_3D[0, e], vert_3D[1, e], vert_3D[2, e], color=clr)
#    
#        c = classes[int(b[9])]
#        ignore_in_eval = bool(b[10])
#        if ignore_in_eval:
#            ax2.text(t[0], t[1], t[2], c, color='r')
#        else:
#            ax2.text(t[0], t[1], t[2], c)
    
#    ax2.auto_scale_xyz([-40, 40], [-40, 40], [0, 80])
#    ax2.view_init(elev=-30, azim=-90)
    
#    for e in np.identity(3):
#        ax2.plot([0, e[0]], [0, e[1]], [0, e[2]], color=e)
#    plt.show()
f.close()
