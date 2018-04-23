import os
import numpy as np
from cffi import FFI
import cv2
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans,kmeans2,vq

# For trajectory storage
import h5py
import uuid

# For loading dataset MATLAB metadata
import scipy.io as sio

import random

# for Multi-threading
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(10) 



# =======================================================================
def filter_trajs_displacement(trajs):
    #print(trajs.shape)
    num_trajs = len(trajs)
    disp_stor = np.empty((num_trajs,), np.float32)
    for ii in range(num_trajs):
        disp_stor[ii] = np.sum(np.sqrt(np.sum((trajs[ii,1:,:]-trajs[ii,0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>3)
    
    return good_trajs
    

# =======================================================================
def filter_trajs_kmeans(trajs, dec_frames, num_centroids):
    num_trajs = len(trajs)
    traj_vec_stor = np.empty((num_trajs, (dec_frames-1)*2), np.float32)
    disp_stor = np.empty((num_trajs,), np.float32)
        
    for ii in range(num_trajs):
        traj = trajs[ii,0:dec_frames,:]  # n-by-2
        traj_vec_stor[ii,:] = (traj[1:,:] - traj[0,:]).flatten() # substract start point        
        disp_stor[ii] = np.sum(np.sqrt(np.sum((traj[1:,:]-traj[0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>0.4)
    traj_vec_stor = traj_vec_stor[good_trajs,:]
    
    if traj_vec_stor.shape[0] < num_centroids: # too few points
        print("kmeans: TOO FEW USABLE KEYPOINTS")
        return good_trajs[np.arange(0,traj_vec_stor.shape[0]-1)] # try to use all of them
        
    # k-means on vectors
    #num_centroids = 10
    #centroids,_ = kmeans(traj_vec_stor,k_or_guess=num_centroids, iter=100)
    centroids,_ = kmeans(traj_vec_stor,num_centroids, iter=100)
    
    # Find the nearest vectors to centroids
    rep = np.argmin(np.sum((traj_vec_stor[:,np.newaxis,:]-centroids[:,:])**2,2),0) # 10-dim
    
    rep = good_trajs[rep]
    
    return rep # return the index of K most representative trajectories
    
# ==========================================================================

# This time we don't do clustering
# Setting parameters
SAMPLES = 5000
CLIP_LENGTH = 20
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512

random.seed()

# Load video...
#for vid_idx in range(NUM_VIDEOS):
def worker(idx):
    print("Processing %d/%d" % (idx, len(job_stor)))
    vid_id, start_frame = job_stor[idx]
    
    for fram_no in range(CLIP_LENGTH):
        img_id = fram_no + start_frame
        #print('push/push_train/{}/{}.jpg'.format(vid_id, img_id))
        img = cv2.imdecode(h5f['push/push_train/{}/{}.jpg'.format(vid_id, img_id)][()], -1)
        img = cv2.resize(img, (240,192))
        if fram_no == 0:
            height = img.shape[0]
            width = img.shape[1]
            vid_seq = np.empty([CLIP_LENGTH,height,width,3], dtype=np.uint8)
        vid_seq[fram_no,:,:,:] = img
    
    # Calculate trajectories
    vid_seq_cptr = ffi.cast("char *", vid_seq.ctypes.data)
    traj_ret = ffi.new("Ret[]", 1)
    # note that a lot more parameters can be modified in DenseTrackStab.cpp.
    libtest.main_like(vid_seq_cptr, width, height, CLIP_LENGTH, traj_ret)
    #print(traj_ret[0].traj_length)
    #print(traj_ret[0].num_trajs)
    #print(traj_ret[0].out_trajs[0])
    trajs = np.frombuffer(ffi.buffer(traj_ret[0].out_trajs, traj_ret[0].traj_length*traj_ret[0].num_trajs*2*4), dtype=np.float32)
    trajs = np.resize(trajs,[traj_ret[0].num_trajs,traj_ret[0].traj_length,2])
    #print(trajs.shape)
    libtest.free_mem()

    #filtered_trajs = filter_trajs_kmeans(trajs, DEC_FRAMES, TRAJS_PER_VIDEO)
    filtered_trajs = filter_trajs_displacement(trajs)
    
    if len(filtered_trajs) == 0:
        print('No Trajectory detected!!!')
    else:
        # Write result to HDF5
        # %06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
        h5_rp_bc_traj = h5_rp_bc.require_dataset('%06d_%04d_%04d_%04d_%s' % (vid_id, start_frame, CLIP_LENGTH, filtered_trajs.size, uuid.uuid1()), shape=(filtered_trajs.size, CLIP_LENGTH, 2), dtype='float32')
        h5_rp_bc_traj[:,:,:] = trajs[filtered_trajs[:],:,:]
        h5_rp_bc_traj.attrs['VidId'] = vid_id
        h5_rp_bc_traj.attrs['StartFrame'] = start_frame
        h5_rp_bc_traj.attrs['TrajLen'] = CLIP_LENGTH
        h5_rp_bc_traj.attrs['TrajCount'] = filtered_trajs.size
        h5_rp_bc_traj.attrs['VidResH'] = height
        h5_rp_bc_traj.attrs['VidResW'] = width
        f.flush()
    
if __name__ == "__main__":
    # ========================================================================
    H5_PATH = '/media/haozekun/512SSD_2/robot_push_h5/robot_push_jpgs.h5' # [EDIT ME!]
    DATASET_PATH = 'push/push_train/'
    h5f = h5py.File(H5_PATH, 'r', libver='latest')
    video_count = h5f['push/push_train'].attrs['video_count'] # [EDIT ME!] push_test
    
    # Generating sample list...
    #video_list = random.sample(xrange(video_count), SAMPLES)
    print('Generating sample list...') 
    job_stor = []
    for vid_id in range(video_count):
        frame_count = h5f['push/push_train/{}'.format(vid_id)].attrs['frame_count'] # [EDIT ME!] push_test
        if frame_count < CLIP_LENGTH:
            continue
        start_frame = random.randint(0,frame_count-CLIP_LENGTH)
        job_stor.append((vid_id,start_frame))
    print('{} samples generated...'.format(len(job_stor)))

    # Load C extension......
    ffi = FFI()
    ffi.cdef('''
        typedef struct 
        {
            int traj_length;
            int num_trajs;
            float* out_trajs;
        } Ret;
        
        void free_mem();
        void main_like(char* in_video, int in_width, int in_height, int in_frames, Ret * ret);    
        ''')
    libtest = ffi.dlopen("./release/DenseTrackStab")

    # Load HDF5 database......
    f = h5py.File("traj_stor_train.h5", 'a', libver='latest') # Supports Single-Write-Multiple-Read # [EDIT ME!]
    h5_rp = f.require_group("RPTraj")
    h5_rp_bc = h5_rp.require_group("by_clip") # /KITTITraj/by_clip/%02d_%04d_%04d_uuid1(video, startframe, len)
    f.swmr_mode = True

    pool.map(worker, range(len(job_stor))) # sample 5000 clips each time
    #for ii in range(len(job_stor)):
    #    worker(ii)
    
    print('Done!!!!')

"""
# Now we plot the trajectory out
vid_h = height
vid_w = width
plt.figure()
plt.ylim(vid_h, 0)
plt.xlim(0, vid_w)
for ii in range(trajs.shape[0]):
    plt.plot(trajs[ii,:,0], trajs[ii,:,1])
    
plt.figure()
plt.imshow(vid_seq[0,:,:,:])
plt.ylim(vid_h, 0)
plt.xlim(0, vid_w)
for topk in range(12): # plot top-12 trajectories    
    traj = trajs[filtered_trajs[topk],:,:]
    #plt.plot(traj[0:4,0], traj[0:4,1])
    plt.plot(traj[:,0], traj[:,1])
plt.show()
"""
