import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# For trajectory storage
import h5py

# Setting parameters
DATASET_DIR = '../../../dataset/Penn_Action'

f = h5py.File("traj_stor.h5", 'r', libver='latest')
# /PennActionTraj/by_video/%04d(videoNo)/%06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
db = f["/PennActionTraj/by_video"]

fig = plt.figure()

for vid_name in db.keys():
    for clip_name in db[vid_name].keys():
        clip_start = db[vid_name][clip_name].attrs['StartFrame']
        clip_len = db[vid_name][clip_name].attrs['TrajLen']
        clip_num_trajs = db[vid_name][clip_name].attrs['TrajCount']
        clip_traj_data = db[vid_name][clip_name]
        for ff in range(clip_len):
            plt.clf()
            img_path = os.path.join(DATASET_DIR, 'frames', vid_name, '%06d.jpg' % (ff+clip_start))
            img_data = cv2.imread(img_path)[:,:,(2,1,0)] # h w c
            plt.imshow(img_data)
            for kk in range(clip_num_trajs):
                traj = clip_traj_data[kk,:,:]
                plt.scatter(traj[ff,0]*2, traj[ff,1]*2)
            fig.canvas.draw()
            plt.pause(0.001)
            #plt.waitforbuttonpress()
            #plt.show()
