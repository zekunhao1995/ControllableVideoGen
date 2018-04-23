import h5py
import numpy as np
import cv2

data_dir = 'push/push_train'

f = h5py.File('robot_push_jpgs.h5', 'r')
for video_id in range(f['push/push_train'].attrs['video_count']):
    for img_id in range(f['push/push_train/{}'.format(video_id)].attrs['frame_count']):
        img = cv2.imdecode(f['push/push_train/{}/{}.jpg'.format(video_id, img_id)][()], -1)
        print(img.shape)
        cv2.imshow('image',img)
        cv2.waitKey(100)

cv2.destroyAllWindows()
