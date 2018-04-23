#from os import listdir
#from os.path import isfile, join
import os
import re
import numpy as np
import cv2
# OpenBLAS screws up with CPU affinity
# Spawned process will inherit this
os.sched_setaffinity(0,range(os.cpu_count()))

'dataset/sequences/[00 to 21]/image_2/[000000 to n]'

kitti_path_prefix = '/data1/Video_Prediction/dataset/KITTI/dataset/sequences'

out_path_prefix = '/media/haozekun/512SSD_2/KITTI_bmp/dataset/sequences'

cam_names = ['image_2', 'image_3']

for vid_id in range(21):
    for cam_name in cam_names:
        vid_path_prefix = os.path.join(kitti_path_prefix, '{:02d}'.format(vid_id), cam_name)
        out_path = os.path.join(out_path_prefix, '{:02d}'.format(vid_id), cam_name)
        os.makedirs(out_path)
        video_file_list = os.listdir(vid_path_prefix)
        print('{} - {}'.format(vid_id, cam_name))
        for filename in video_file_list:
            png_full_path = os.path.join(vid_path_prefix,filename)
            out_full_path = os.path.join(out_path,filename.rsplit('.',1)[0]+'.bmp')
            frame = cv2.imread(png_full_path)
            frame2x = cv2.resize(frame, (845,256), interpolation=cv2.INTER_AREA)
            cv2.imwrite(out_full_path,frame2x)
            #cv2.imwrite(out_full_path,frame2x,[cv2.IMWRITE_JPEG_QUALITY, 100])
            #cv2.imwrite(out_full_path,frame2x,[cv2.IMWRITE_WEBP_QUALITY, 100])
            #print(out_full_path)

