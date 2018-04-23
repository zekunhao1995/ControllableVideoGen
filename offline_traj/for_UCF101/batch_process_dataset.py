import os
import numpy as np
from cffi import FFI
import cv2

from scipy.cluster.vq import kmeans,kmeans2,vq

# For trajectory storage
import h5py
import uuid

import re

# OpenBLAS affects CPU affinity
os.sched_setaffinity(0,range(os.cpu_count()))
def setaff():
    os.sched_setaffinity(0,range(os.cpu_count()))
    
# for Multi-threading
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(5, setaff)


# =======================================================================
def filter_trajs_displacement(trajs):
    #print(trajs.shape)
    num_trajs = len(trajs)
    disp_stor = np.empty((num_trajs,), np.float32)
    for ii in range(num_trajs):
        disp_stor[ii] = np.sum(np.sqrt(np.sum((trajs[ii,1:,:]-trajs[ii,0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>5)
    
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

CLIP_LENGTH = 11

# Load video...
#for vid_idx in range(NUM_VIDEOS):
def worker(idx):
    print("Processing %d/%d" % (idx, len(job_stor)))
    video_path, length, offset = job_stor[idx]
    
    #start_frame = random.randint(0,length-CLIP_LENGTH+1-1)
    start_frame = offset
    for fram_no in range(CLIP_LENGTH):
        frame = cv2.imread(video_path+'/'+str(start_frame+fram_no)+'.jpg')
        img = cv2.resize(frame, (256,192), interpolation=cv2.INTER_AREA)
        if fram_no == 0:
            height = img.shape[0]
            width = img.shape[1]
            vid_seq = np.empty([CLIP_LENGTH,height,width,3], dtype=np.uint8)
        vid_seq[fram_no,:,:,:] = img[:,:,:]

    # Calculate trajectories
    vid_seq_cptr = ffi.cast("char *", vid_seq.ctypes.data)
    traj_ret = ffi.new("Ret[]", 1)
    # note that a lot more parameters are hard-coded in DenseTrackStab.cpp due to laziness.
    libtest.main_like(vid_seq_cptr, width, height, CLIP_LENGTH, traj_ret)
    #print(traj_ret[0].traj_length)
    #print(traj_ret[0].num_trajs)
    #print(traj_ret[0].out_trajs[0])
    trajs = np.frombuffer(ffi.buffer(traj_ret[0].out_trajs, traj_ret[0].traj_length*traj_ret[0].num_trajs*2*4), dtype=np.float32)
    trajs = np.resize(trajs,[traj_ret[0].num_trajs,traj_ret[0].traj_length,2])
    #print(trajs.shape)
    libtest.free_mem()

    #filtered_trajs = filter_trajs_kmeans(trajs, 15, 10)
    filtered_trajs = filter_trajs_displacement(trajs)
    
    if len(filtered_trajs) == 0:
        print('No Trajectory detected!!!')
    else:
        # Write result to HDF5
        # %06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
        h5_ucf_bc_traj = h5_ucf_bc.require_dataset('%06d_%04d_%04d_%s' % (start_frame+1, CLIP_LENGTH, filtered_trajs.size, uuid.uuid1()), shape=(filtered_trajs.size, CLIP_LENGTH, 2), dtype='float32')
        h5_ucf_bc_traj[:,:,:] = trajs[filtered_trajs[:],:,:]
        h5_ucf_bc_traj.attrs['VidPath'] = video_path
        h5_ucf_bc_traj.attrs['StartFrame'] = start_frame
        h5_ucf_bc_traj.attrs['TrajLen'] = CLIP_LENGTH
        h5_ucf_bc_traj.attrs['TrajCount'] = filtered_trajs.size
        h5_ucf_bc_traj.attrs['VidResH'] = height
        h5_ucf_bc_traj.attrs['VidResW'] = width
        f.flush()
    
if __name__ == "__main__":
    # ========================================================================
    # Load UCF101 dataset
    DATASET_DIR = '/media/haozekun/512SSD_2/UCF101_seq/UCF-101' # [EDIT ME!]
    
    # Load split file:
    f = open('trainlist01.txt','r') # Sample: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi # [EDIT ME!]
    job_stor = []
    for line in f:
        vid_name = line.split()[0]
        video_path = os.path.join(DATASET_DIR, vid_name)
        img_list = os.listdir(video_path)
        frame_count = 0
        for filename in img_list:
            frame_count = max(frame_count, int(filename.split('.')[0]))
        frame_count += 1
        for offset in range(0, frame_count - CLIP_LENGTH + 1, 8): # Stride = 8
            job_stor.append((video_path, frame_count, offset))
    f.close()
            
    print('Job count: {:d}'.format(len(job_stor))) # 13320, or 9537

    # Load C extension......
    ffi = FFI()
    ffi.cdef('''
        typedef struct 
        {
            int traj_length;
            int num_trajs;
            float* out_trajs;
        } Ret;
        W
        void free_mem();
        void main_like(char* in_video, int in_width, int in_height, int in_frames, Ret * ret);    
        ''')
    libtest = ffi.dlopen("./release/DenseTrackStab")

    # Load HDF5 database......
    f = h5py.File("traj_stor_train.h5", 'a', libver='latest') # Supports Single-Write-Multiple-Read  # [EDIT ME!]
    h5_ucf = f.require_group("UCFTraj")
    #h5_kt_bv = h5_pa.require_group("by_video") # /KITTITraj/by_video/%04d(videoNo)/%06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
    h5_ucf_bc = h5_ucf.require_group("by_clip") # /KITTITraj/by_clip/%02d_%04d_%04d_uuid1(video, startframe, len)
    f.swmr_mode = True

    pool.map(worker, range(len(job_stor)))

