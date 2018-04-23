import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# For trajectory storage
import h5py

# Setting parameters
TRAJ_H5_PATH = './traj_stor_train.h5'
DATASET_DIR = '/media/haozekun/512SSD_2/UCF101_seq/UCF-101'

f = h5py.File(TRAJ_H5_PATH, 'r', libver='latest')
# /PennActionTraj/by_video/%04d(videoNo)/%06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
db = f["/UCFTraj/by_clip"]

fig = plt.figure()

for clip_name in db.keys():
    video_path = db[clip_name].attrs['VidPath']
    print(video_path)
    clip_start = db[clip_name].attrs['StartFrame']
    clip_len = db[clip_name].attrs['TrajLen']
    clip_num_trajs = db[clip_name].attrs['TrajCount']
    clip_traj_data = db[clip_name]
    #cap = cv2.VideoCapture(video_path)
    #if not cap.isOpened():
    #    print('Video open failed!!!')
    #cap.set(cv2.CAP_PROP_POS_FRAMES ,clip_start)
    
    for ff in range(clip_len):
    #for ff in [0]:
        plt.clf()
        #ret, frame = cap.read() # 320 by 240
        #if not ret:
        #    print('Frame read error!')
        frame = cv2.imread(video_path+'/'+str(clip_start+ff)+'.jpg')
        img_data = cv2.resize(frame, (256,192))
        
        img_data = img_data[:,:,(2,1,0)] # h w c
        plt.imshow(img_data)
        for kk in range(clip_num_trajs):
            traj = clip_traj_data[kk,:,:]
            plt.scatter(traj[ff,0], traj[ff,1])
        print('Count: {}'.format(kk))
        fig.canvas.draw()
        plt.pause(0.001)
        #plt.waitforbuttonpress()
        #plt.show()
    #cap.release()
