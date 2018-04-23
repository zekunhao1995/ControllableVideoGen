
import os

import numpy as np
import h5py
import re

import cv2

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(8) 

# Load UCF101 dataset
DATASET_DIR = '/data2/UCF-101'
RAWFRAME_DIR = '/data1/UCF101seq'



def worker(action_dir):
    print(action_dir)
    video_files = os.listdir(os.path.join(DATASET_DIR, action_dir))
    action_out_dir = os.path.join(RAWFRAME_DIR, action_dir)
    os.mkdir(action_out_dir)
    for video_file in video_files:
        print(video_file)
        video_path = os.path.join(DATASET_DIR, action_dir, video_file)
        video_out_dir = os.path.join(RAWFRAME_DIR, action_dir, video_file)
        os.mkdir(video_out_dir)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Video open failed!!!')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_no in range(length):
            ret, frame = cap.read() # 320 by 240
            if not ret:
                print('Frame read error!')
                break
            frame_out_path = os.path.join(RAWFRAME_DIR, action_dir, video_file, str(frame_no)+'.jpg')
            #cv2.imwrite(frame_out_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 4])
            cv2.imwrite(frame_out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cap.release()
        

action_dir_list = os.listdir(DATASET_DIR)
#for action_dir in action_dir_list:
    
    
pool.map(worker, action_dir_list)
