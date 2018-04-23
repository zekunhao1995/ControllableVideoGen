import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# For trajectory storage
import h5py

# Setting parameters
TRAJ_H5_PATH = 'traj_stor_train.h5'
JPG_H5_PATH = '/media/haozekun/512SSD_2/robot_push_h5/robot_push_jpgs.h5'

f_traj = h5py.File(TRAJ_H5_PATH, 'r', libver='latest')
db_traj = f_traj["/RPTraj/by_clip"]

f_jpg = h5py.File(JPG_H5_PATH, 'r', libver='latest')
#f_jpg = f_jpg['push/push_train/']

fig = plt.figure()

for clip_name in db_traj.keys():
    video_id = db_traj[clip_name].attrs['VidId']
    clip_start = db_traj[clip_name].attrs['StartFrame']
    clip_len = db_traj[clip_name].attrs['TrajLen']
    clip_num_trajs = db_traj[clip_name].attrs['TrajCount']
    clip_traj_data = db_traj[clip_name]
    
    for ff in range(clip_len):
        plt.clf()
        img_id = ff + clip_start
        img_data = cv2.imdecode(f_jpg['push/push_train/{}/{}.jpg'.format(video_id, img_id)][()], -1)
        img_data = cv2.resize(img_data, (240,192))
        img_data = img_data[:,:,(2,1,0)] # h w c
        
        plt.imshow(img_data)
        for kk in range(clip_num_trajs):
            traj = clip_traj_data[kk,:,:]
            plt.scatter(traj[ff,0], traj[ff,1])
        fig.canvas.draw()
        plt.pause(0.001)
        #plt.waitforbuttonpress()
        #plt.show()
