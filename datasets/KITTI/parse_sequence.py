#from os import listdir
#from os.path import isfile, join
import os
import re
import numpy as np

'dataset/sequences/[00 to 21]/image2/[000000 to n]'

kitti_path_prefix = '/data1/Video_Prediction/dataset/KITTI/dataset/sequences'

def get_num(x):
    return int(''.join(ele for ele in x if ele.isdigit()))
    
frame_count_stor = []
for vid_id in range(21):
    vid_path_prefix = os.path.join(kitti_path_prefix, '{:02d}'.format(vid_id), 'image_2')    
    video_file_list = os.listdir(vid_path_prefix)
    frame_count = 0
    for filename in video_file_list:
        frame_count = max(get_num(filename),frame_count)
    print(frame_count)
    frame_count_stor.append(frame_count)
    
# 16 / 5 split
# test: 15 11 7 5 4
frame_count_cumsum = np.cumsum(frame_count_stor)
print(frame_count_cumsum)


