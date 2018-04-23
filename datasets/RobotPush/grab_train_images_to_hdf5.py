"""Code for converting TFRecords to HDF5"""

import os

import numpy as np
import tensorflow as tf
import h5py
import re

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile


FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3
BATCH_SIZE = 25

data_dir = 'push/push_train'
#dest_dir = '/media/haozekun/512SSD_2/push_jpg'
hdf5_path = '/media/haozekun/512SSD_2/robot_push_jpgs.h5'

f = h5py.File(hdf5_path, 'w', libver='latest') # Supports Single-Write-Multiple-Read
h5_push = f.require_group("push")
h5_push_train = h5_push.require_group("push_train")

        
    
def decode_proto(s_example, h5_push_train_vid):
    a = tf.train.Example()
    a.ParseFromString(s_example) # a: an example
    b = a.ListFields()[0][1].ListFields()[0][1]
    prog = re.compile('move/(\d+)/image/encoded')
    
    num_imgs = 0
    for key in b.keys():
        m = prog.match(key)
        if m:
            img_id = int(m.group(1))
            v = b[key]
            raw_data = v.ListFields()[0][1].ListFields()[0][1][0]
            
            h5_push_train_vid_jpg = h5_push_train_vid.require_dataset('{}.jpg'.format(img_id), shape=(len(raw_data),), dtype=np.uint8)
            h5_push_train_vid_jpg[:] = np.fromstring(raw_data, dtype=np.uint8)
            num_imgs = max(num_imgs, img_id)
    return num_imgs+1


filenames = gfile.Glob(os.path.join(data_dir, '*'))
if not filenames:
    raise RuntimeError('No data files found.')
vid_count = 0
for filename in filenames:
    for s_example in tf.python_io.tf_record_iterator(filename):
        h5_push_train_vid = h5_push_train.require_group(str(vid_count))
        num_imgs = decode_proto(s_example, h5_push_train_vid)
        h5_push_train_vid.attrs['frame_count'] = num_imgs
        vid_count += 1
        print(vid_count)

h5_push_train.attrs['video_count'] = vid_count

f.flush()
f.close()
