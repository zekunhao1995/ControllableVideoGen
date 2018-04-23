import numpy as np
import torch
from torch.autograd import Variable

def trajs2map(trajs, height, width): # traj: [N, S/E, X/Y] 
    #kpmap_seq = np.zeros([num_frames, 6,self.height,self.width], dtype=np.float32)

    #height = kpmap_seq.size(2)
    #width = kpmap_seq.size(3)
    kpmap_seq = Variable(torch.zeros(1,6,height,width).cuda())
    for traj_no in range(len(trajs)):
        kp_start_x = trajs[traj_no][0][0]
        kp_start_y = trajs[traj_no][0][1]
        kp_end_x = trajs[traj_no][1][0]
        kp_end_y = trajs[traj_no][1][1]

        kp_start_x_int = int(max(min(kp_start_x, width),0))
        kp_start_y_int = int(max(min(kp_start_y, height),0))
        kp_dx = kp_end_x - kp_start_x
        kp_dy = kp_end_y - kp_start_y
        kpmap_seq[0, 0,kp_start_y_int,kp_start_x_int] = 1.0
        kpmap_seq[0, 1,kp_start_y_int,kp_start_x_int] = kp_dy/16.
        kpmap_seq[0, 2,kp_start_y_int,kp_start_x_int] = kp_dx/16.
        #vid_seq[0,1,kp_start_y,kp_start_x] = 0.5

        kp_end_x_int = int(max(min(kp_end_x, width),0))
        kp_end_y_int = int(max(min(kp_end_y, height),0))
        kp_dx2 = kp_start_x - kp_end_x
        kp_dy2 = kp_start_y - kp_end_y
        kpmap_seq[0, 3,kp_end_y_int,kp_end_x_int] = 1.0
        kpmap_seq[0, 4,kp_end_y_int,kp_end_x_int] = kp_dy2/16.
        kpmap_seq[0, 5,kp_end_y_int,kp_end_x_int] = kp_dx2/16.
        
    return kpmap_seq
    
    
def trajs2map2(trajs, height, width): # traj: [N, S/E, X/Y] 
    #kpmap_seq = np.zeros([num_frames, 6,self.height,self.width], dtype=np.float32)

    #height = kpmap_seq.size(2)
    #width = kpmap_seq.size(3)
    kpmap_seq = Variable(torch.zeros(1,6,height,width).cuda())
    for traj_no in range(trajs.shape[0]):
        kp_start_x = trajs[traj_no,0,0]
        kp_start_y = trajs[traj_no,0,1]
        kp_end_x = trajs[traj_no,1,0]
        kp_end_y = trajs[traj_no,1,1]

        kp_start_x_int = int(max(min(kp_start_x, width),0))
        kp_start_y_int = int(max(min(kp_start_y, height),0))
        kp_dx = kp_end_x - kp_start_x
        kp_dy = kp_end_y - kp_start_y
        kpmap_seq[0, 0,kp_start_y_int,kp_start_x_int] = 1.0
        kpmap_seq[0, 1,kp_start_y_int,kp_start_x_int] = kp_dy/16.
        kpmap_seq[0, 2,kp_start_y_int,kp_start_x_int] = kp_dx/16.
        #vid_seq[0,1,kp_start_y,kp_start_x] = 0.5

        kp_end_x_int = int(max(min(kp_end_x, width),0))
        kp_end_y_int = int(max(min(kp_end_y, height),0))
        kp_dx2 = kp_start_x - kp_end_x
        kp_dy2 = kp_start_y - kp_end_y
        kpmap_seq[0, 3,kp_end_y_int,kp_end_x_int] = 1.0
        kpmap_seq[0, 4,kp_end_y_int,kp_end_x_int] = kp_dy2/16.
        kpmap_seq[0, 5,kp_end_y_int,kp_end_x_int] = kp_dx2/16.
        
    return kpmap_seq
