import random
import numpy as np
from scipy import misc # for imread
from utils.find_border import find_border
import h5py

import math
import os

from scipy.cluster.vq import kmeans,kmeans2,vq

def filter_trajs_kmeans(trajs, num_centroids):
    num_trajs = trajs.shape[0]
    len_trajs = trajs.shape[1]
    traj_vec_stor = np.empty((num_trajs, (len_trajs-1)*2), np.float32)
    disp_stor = np.empty((num_trajs,), np.float32)
        
    for ii in range(num_trajs):
        traj = trajs[ii,:,:]  # n-by-2
        traj_vec_stor[ii,:] = (traj[1:,:] - traj[0,:]).flatten() # substract start point        
        disp_stor[ii] = np.sum(np.sqrt(np.sum((traj[1:,:]-traj[0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>0.4)
    traj_vec_stor = traj_vec_stor[good_trajs,:]
    
    if traj_vec_stor.shape[0] < num_centroids: # too few points
        #print("kmeans: TOO FEW USABLE KEYPOINTS")
        return good_trajs[np.arange(0,traj_vec_stor.shape[0]-1)] # try to use all of them
        
    # k-means on vectors
    #num_centroids = 10
    #centroids,_ = kmeans(traj_vec_stor,k_or_guess=num_centroids, iter=100)
    centroids,label = kmeans(traj_vec_stor,num_centroids, iter=20) # Label[i] is the cluster no that i-th datapoint belongs to
    
    # Sample
    # Find the nearest vectors to centroids
    rep = np.argmin(np.sum((traj_vec_stor[:,np.newaxis,:]-centroids[:,:])**2,2),0) # 10-dim
    
    rep = good_trajs[rep]
    
    return rep # return the index of K most representative trajectories
    
    
    
class UCFReader():
    TRAJ_H5_PATH = '/trajectories/ucf/traj_stor_test.h5'
    DATASET_DIR = '/datasets/UCF101/UCF-101'
    JPG_DIR = '/datasets/UCF101_seq/UCF-101'
    
    def __init__(self, num_frames=10):
        self._num_frames = num_frames
        self.height = 192
        self.width = 256
        
        traj_h5 = h5py.File(self.TRAJ_H5_PATH, 'r', libver='latest')
        traj_db = traj_h5["/UCFTraj/by_clip"]
        #traj_h5.close()
        self.clip_names = list(traj_db.keys())
        self.clip_num = len(self.clip_names)
        self.traj_db = traj_db
        print('[UCF Trajectory Statistics]')
        print('Clip count: %d' % (self.clip_num))
        
    def get_traj_input(self, trajs, start_frame, num_frames):
        num_trajs = trajs.shape[0]
        # Load annotations
        # Format: 2(frames), 3(T/F,dx,dy), H, W
        kpmap_seq = np.zeros([num_frames, 6,self.height,self.width], dtype=np.float32)
        
        #num_appear_trajs = min(num_trajs,10)
        num_appear_trajs = min(num_trajs,1)
        good_idx = filter_trajs_kmeans(trajs[:,start_frame:start_frame+num_frames,:], 10)
        
        appear_trajs = random.sample(range(num_trajs), num_appear_trajs)
        
        traj_list = trajs[appear_trajs, start_frame:start_frame+num_frames, :]
        for ff in range(num_frames):
            for traj_no in appear_trajs:
                kp_start_x = trajs[traj_no,start_frame,0]
                kp_start_y = trajs[traj_no,start_frame,1]
                kp_end_x = trajs[traj_no,start_frame+ff,0]
                kp_end_y = trajs[traj_no,start_frame+ff,1]

                kp_start_x_int = int(max(min(kp_start_x, self.width),0))
                kp_start_y_int = int(max(min(kp_start_y, self.height),0))
                kp_dx = kp_end_x - kp_start_x
                kp_dy = kp_end_y - kp_start_y
                kpmap_seq[ff, 0,kp_start_y_int,kp_start_x_int] = 1.0
                kpmap_seq[ff, 1,kp_start_y_int,kp_start_x_int] = kp_dy/16.
                kpmap_seq[ff, 2,kp_start_y_int,kp_start_x_int] = kp_dx/16.
                #vid_seq[0,1,kp_start_y,kp_start_x] = 0.5

                kp_end_x_int = int(max(min(kp_end_x, self.width),0))
                kp_end_y_int = int(max(min(kp_end_y, self.height),0))
                kp_dx2 = kp_start_x - kp_end_x
                kp_dy2 = kp_start_y - kp_end_y
                kpmap_seq[ff, 3,kp_end_y_int,kp_end_x_int] = 1.0
                kpmap_seq[ff, 4,kp_end_y_int,kp_end_x_int] = kp_dy2/16.
                kpmap_seq[ff, 5,kp_end_y_int,kp_end_x_int] = kp_dx2/16.
                
        return kpmap_seq, traj_list
        
    def __getitem__(self, idx):
        traj_db = self.traj_db
       
        if idx == -1:
            while True:
                idx = random.randint(0,self.clip_num-1)
                #if traj_db[self.clip_names[idx]].attrs['VidPath'].find("PushUps") != -1:
                break
        
        
        annot = traj_db[self.clip_names[idx]]
        vid_path = annot.attrs['VidPath']
        #vid_path = vid_path.replace('/datasets/UCF101/UCF-101', self.JPG_DIR) #
        annot_traj_len = annot.attrs['TrajLen']
        annot_clip_start = annot.attrs['StartFrame']
        num_trajs = annot.attrs['TrajCount']
        trajs = annot[()]
        
        num_frames = self._num_frames
        annot_start_frame = random.randint(0,annot_traj_len-num_frames) 

        # preallocate np array
        vid_seq = np.empty([num_frames,3,self.height,self.width], dtype=np.float32)
        for ff in range(num_frames):
            frame_no = annot_start_frame+annot_clip_start+ff
            try:
                frame = misc.imread(vid_path+'/'+str(frame_no)+'.jpg')
            except:
                print('Bad image found.')
                frame = np.zeros([self.width, self.height, 3], dtype=np.uint8)
            img = misc.imresize(frame, (self.height,self.width)) 
            vid_seq[ff,:,:,:] = np.transpose(img, (2,0,1))/255.0

        vid_mask = find_border(vid_seq[0,:,:,:], threshold=10/255)
        vid_seq = vid_seq - 0.5 # 2 C H W, [-0.5,0.5]
        vid_seq = vid_seq * vid_mask

        kpmap_seq, traj_list = self.get_traj_input(trajs, annot_start_frame, num_frames)
        
        print(idx, annot_start_frame)
        return vid_seq, kpmap_seq, traj_list
