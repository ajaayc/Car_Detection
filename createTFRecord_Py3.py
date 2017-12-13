#!/usr/bin/python3


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import sys

from object_detection.utils import dataset_util

#flags = tf.app.flags
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#FLAGS = flags.FLAGS

class_strings = ['Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
           'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
           'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
           'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
           'Military', 'Commercial', 'Trains']

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

def create_tf_example(example,counter,datatype):
  print(example, ": " ,counter)
  img = plt.imread(example)
  
  proj = np.fromfile(example.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
  proj.resize([3, 4])
  
  if datatype == 'train':
      bbox = np.fromfile(example.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
      bbox.resize([bbox.size // 11, 11])
  
  # TODO(user): Populate the following variables from your example.
  #-----------------------------------------------------------------
  height = np.shape(img)[0] # Image height
  width = np.shape(img)[1] # Image width
  filename = str.encode(example) # Filename of the image. Empty if image is not from file
  #encoded_image_data = img.tostring() # Encoded image bytes
  #encoded_image_data = tf.gfile.FastGFile(filename, 'rb').read()

  #See https://github.com/swirlingsand/deeper-traffic-lights/blob/master/data_conversion_bosch.py for example of how to read in a file
  with tf.gfile.GFile(filename, 'rb') as fid:
      encoded_image_data = fid.read()
  
  image_format = 'jpg'.encode() # b'jpg' or b'png'

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [] # List of string class name of bounding box (1 per box)
  classes = [] # List of integer class id of bounding box (1 per box)
  
  #-----------------------------------------------------------------
  
  if datatype == 'train':
      for k, b in enumerate(bbox):
            ignore_in_eval = bool(b[10])
            if not ignore_in_eval:
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
                    
                #Divided by width and height since train.py requires these are normalized
                #Don't do integer division
                xmin = min(vert_2D[0]) / float(width)
                xmax = max(vert_2D[0]) / float(width)
                 
                ymin = min(vert_2D[1]) / float(height)
                ymax = max(vert_2D[1]) / float(height)
                
                
                xmax = min(xmax, 1.01)
                ymax = min(ymax, 1.01)
                
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                
                class_int = int(b[9])
                class_text = str.encode(class_strings[class_int])
        
                classes.append(class_int)
                classes_text.append(class_text)
        

  tf_example = ''
      
  if datatype == 'train':
      tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))

  if datatype == 'test':
    tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          #'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          #'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          #'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          #'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          #'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          #'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))

  return tf_example


def main(_):
  if len(sys.argv) != 2 or not(sys.argv[1] == 'test' or sys.argv[1] == 'train'):
      print('Error: Incorrect number of arguments')
      print('Usage: python createTFRecord_Py3.py <test OR train>')
      exit

  #Type of the data. 'test' or 'train'
  datatype = sys.argv[1]

  #writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
  # TODO(user): Write code to read in your dataset to examples variable
  if datatype == 'train':
      writer = tf.python_io.TFRecordWriter("train.record")
      examples = glob('deploy/trainval/*/*_image.jpg')

  if datatype == 'test':
      writer = tf.python_io.TFRecordWriter("test.record")
      examples = glob('deploy/test/*/*_image.jpg')

  counter = 1

  for example in examples:
    tf_example = create_tf_example(example,counter,datatype)
    counter += 1
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
