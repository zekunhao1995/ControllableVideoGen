import os
import numpy as np
from cffi import FFI
import cv2

from scipy.cluster.vq import kmeans,kmeans2,vq

# For trajectory storage
import h5py
import uuid

# OpenBLAS(used by OpenCV) changes CPU affinity
os.sched_setaffinity(0,range(os.cpu_count()))
def setaff():
    os.sched_setaffinity(0,range(os.cpu_count()))
    

# for Multi-threading
from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(5, setaff) 



# =======================================================================
def filter_trajs_displacement(trajs):
    num_trajs = len(trajs)
    disp_stor = np.empty((num_trajs,), np.float32)
    for ii in range(num_trajs):
        disp_stor[ii] = np.sum(np.sqrt(np.sum((trajs[1:,:]-trajs[0:-1,:])**2,1)))
    # Remove trajectories that have very low displacement
    good_trajs = np.flatnonzero(disp_stor>-1)
    
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
CLIP_LENGTH = 10


# Load video...
#for vid_idx in range(NUM_VIDEOS):
def worker(idx):
    print("Processing %d/%d" % (idx, len(job_stor)))
    
    vid_id, frame_count, cam_name, start_frame = job_stor[idx]
    
    for ff in range(CLIP_LENGTH):
        img_path = os.path.join(kitti_path_prefix, '{:02d}'.format(vid_id), cam_name, '{:06d}.png'.format(ff+start_frame))
        img_data = cv2.imread(img_path) # h w c
        img_data = cv2.resize(img_data, (422,128))
        img_data = img_data[:,83:339,:]
        
        #img_data = cv2.resize(img_data, dsize=None, fx=0.25, fy=0.25)
        if ff == 0:
            height = img_data.shape[0]
            width = img_data.shape[1]
            vid_seq = np.empty([CLIP_LENGTH,height,width,3], dtype=np.uint8)
        vid_seq[ff,:,:,:] = img_data
    
    # Calculate trajectories
    vid_seq_cptr = ffi.cast("char *", vid_seq.ctypes.data)
    traj_ret = ffi.new("Ret[]", 1)
    # note that a lot more parameters can be modified in DenseTrackStab.cpp.
    libtest.main_like(vid_seq_cptr, img_data.shape[1], img_data.shape[0], CLIP_LENGTH, traj_ret)
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
        h5_kt_bc_traj = h5_kt_bc.require_dataset('%06d_%04d_%04d_%s' % (start_frame+1, CLIP_LENGTH, filtered_trajs.size, uuid.uuid1()), shape=(filtered_trajs.size, CLIP_LENGTH, 2), dtype='float32')
        h5_kt_bc_traj[:,:,:] = trajs[filtered_trajs[:],:,:]
        h5_kt_bc_traj.attrs['VidNo'] = vid_id
        h5_kt_bc_traj.attrs['StartFrame'] = start_frame
        h5_kt_bc_traj.attrs['TrajLen'] = CLIP_LENGTH
        h5_kt_bc_traj.attrs['TrajCount'] = filtered_trajs.size
        h5_kt_bc_traj.attrs['CamName'] = cam_name
        h5_kt_bc_traj.attrs['VidResH'] = height
        h5_kt_bc_traj.attrs['VidResW'] = width
        f.flush()
    
if __name__ == "__main__":
    # ========================================================================
    # Load KITTI dataset
    kitti_path_prefix = '/data1/Video_Prediction/dataset/KITTI/dataset/sequences' # [EDIT ME!]
    def get_num(x):
        return int(''.join(ele for ele in x if ele.isdigit()))
    frame_count_stor = []
    for vid_id in range(21):
        vid_path_prefix = os.path.join(kitti_path_prefix, '{:02d}'.format(vid_id), 'image_2')    
        video_file_list = os.listdir(vid_path_prefix)
        frame_count = 0
        for filename in video_file_list:
            frame_count = max(get_num(filename),frame_count)
        print('Video {}, {} frames'.format(vid_id, frame_count))
        frame_count_stor.append(frame_count+1) # file name starts from 0
    # 16 / 5 split
    test_split = [15, 11, 7, 5, 4]
    train_split = list(set(range(21)) - set(test_split))
    frame_count_stor_train = [frame_count_stor[x] for x in train_split]
    frame_count_stor_test = [frame_count_stor[x] for x in test_split]
    ##frame_count_stor_train_cumsum = np.cumsum(frame_count_stor_train)
    ##train_vid_probs = frame_count_stor_train_cumsum/frame_count_stor_train_cumsum[-1]
    ##print(train_vid_probs)
    
    ## Dense sampling procedure
    print('Dense sampling videos......')
    job_stor = []
    for vid_id in train_split:  # [EDIT ME!] you might want test_split
        frame_count = frame_count_stor[vid_id]
        for offset in range(0, frame_count - CLIP_LENGTH + 1, 1):
            job_stor.append((vid_id, frame_count, 'image_2', offset))
            job_stor.append((vid_id, frame_count, 'image_3', offset))
    
    print(len(job_stor))
    

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
    f = h5py.File("traj_stor_train.h5", 'a', libver='latest') # Supports Single-Write-Multiple-Read # [EDIT ME!] this is the name of the produced file containing trajectories
    h5_kt = f.require_group("/KITTITraj")
    #h5_kt_bv = h5_pa.require_group("by_video") # /KITTITraj/by_video/%04d(videoNo)/%06d_%04d_%04d_uuid1(startFrame, trajLen, trajCount)
    h5_kt_bc = h5_kt.require_group("by_clip") # /KITTITraj/by_clip/%02d_%04d_%04d_uuid1(video, startframe, len)
    f.swmr_mode = True

    pool.map(worker, range(len(job_stor)))
    
