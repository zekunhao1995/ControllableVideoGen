import os
import sys
import math
import numpy as np
import cv2
import random
from scipy import misc # for imread
import h5py

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
    
    


class KITTIReader():
    TRAJ_H5_PATH = '/trajectories/kitti/traj_stor_test.h5'
    DATASET_DIR = '/datasets/KITTI/dataset/sequences'

    def _calc_traj_len(self, traj): # [Traj_no, num_point, (x,y)]
        dx = np.sum((traj[:,0:-1,:]-traj[:,1:,:])**2, axis=2)

    def __init__(self, num_frames=10):
        self._clip_stor = []
        self._num_frames = num_frames
        self.height = 128
        self.width = 256
        
        traj_h5 = h5py.File(self.TRAJ_H5_PATH, 'r', libver='latest')
        traj_db = traj_h5["/KITTITraj/by_clip"]
        # Load all .mat files to memory
        print('Loading Trajectoriess for Penn Dataset...')
        for clip_name in traj_db.keys():
            clip_start = traj_db[clip_name].attrs['StartFrame']
            clip_len = traj_db[clip_name].attrs['TrajLen']
            clip_num_trajs = traj_db[clip_name].attrs['TrajCount']
            clip_traj_data = np.array(traj_db[clip_name])
            clip_video_id = traj_db[clip_name].attrs['VidNo']
            
            new_clip = {}
            new_clip['vid_name'] = clip_video_id
            new_clip['clip_start'] = clip_start
            new_clip['clip_len'] = clip_len
            new_clip['clip_num_trajs'] = clip_num_trajs
            new_clip['clip_trajs'] = clip_traj_data
            self._clip_stor.append(new_clip)

        print('[KITTI Trajectory Statistics]')
        print('Clip count: %d' % (len(self._clip_stor)))
        traj_h5.close()
            
    def get_traj_input(self, trajs, start_frame, num_frames):
        num_trajs = trajs.shape[0]
        # Load annotations
        # Format: 2(frames), 3(T/F,dx,dy), H, W
        kpmap_seq = np.zeros([num_frames, 6,self.height,self.width], dtype=np.float32)
        
        #num_appear_trajs = min(num_trajs,10)
        num_appear_trajs = min(num_trajs,1)
        #good_idx = filter_trajs_kmeans(trajs[:,start_frame:start_frame+num_frames,:], 10)
        
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
        if idx == -1:
            idx = random.randint(0,len(self._clip_stor))
        
        annot = self._clip_stor[idx] 
        
        vid_name = annot['vid_name']
        frame_count = annot['clip_len']
        clip_start = annot['clip_start']
        
        num_frames = self._num_frames
        # random start frame
        start_frame = random.randint(0,frame_count-num_frames) 
        
        # loading frames
        vid_seq = np.empty([num_frames,3,self.height,self.width], dtype=np.float32)
        for ff in range(num_frames): # only load two frames
            frame_no = start_frame+clip_start+ff
            img_path = os.path.join(self.DATASET_DIR, '{:02d}'.format(vid_name), 'image_2', '{:06d}.png'.format(frame_no))
            img_load = misc.imread(img_path) # h w c
            img = misc.imresize(img_load, (128,422))
            if ff == 0:
                img_2x = misc.imresize(img_load, (256,845))
                img_ori = img_2x.astype(np.float32)
            img = img[:,83:339,:]
            vid_seq[ff,:,:,:] = np.transpose(img, (2,0,1))/255.0

        img_ori = np.transpose(img_ori, (2,0,1))/255.0 - 0.5
        vid_seq = vid_seq - 0.5 # 2 C H W, [-0.5,0.5]
        
        num_trajs = annot['clip_num_trajs']
        trajs = annot['clip_trajs']
        kpmap_seq, traj_list = self.get_traj_input(trajs, start_frame, num_frames)
        

        print(idx, start_frame)
        return vid_seq, kpmap_seq, traj_list, img_ori

    
    
    
    
