import cv2


def trajs2featmap(trajs, kpmap_seq):
    for traj_no in range(trajs.shape[0]):
        #cv2.circle(frame, tuple(trajs[traj_no, frame_no, :]), 2, (0.,1.,0.))
        kp_start_x = trajs[traj_no,0,0]
        kp_start_y = trajs[traj_no,0,1]
        kp_end_x = trajs[traj_no,1,0]
        kp_end_y = trajs[traj_no,1,1]

        kp_start_x_int = int(max(min(kp_start_x, kpmap_seq.shape[2]),0))
        kp_start_y_int = int(max(min(kp_start_y, kpmap_seq.shape[1]),0))
        kp_dx = kp_end_x - kp_start_x
        kp_dy = kp_end_y - kp_start_y
        kpmap_seq[0,kp_start_y_int,kp_start_x_int] = 1.0
        kpmap_seq[1,kp_start_y_int,kp_start_x_int] = kp_dy
        kpmap_seq[2,kp_start_y_int,kp_start_x_int] = kp_dx
        #vid_seq[0,1,kp_start_y,kp_start_x] = 0.5

        kp_end_x_int = int(max(min(kp_end_x, kpmap_seq.shape[2]),0))
        kp_end_y_int = int(max(min(kp_end_y, kpmap_seq.shape[1]),0))
        kp_dx2 = kp_start_x - kp_end_x
        kp_dy2 = kp_start_y - kp_end_y
        kpmap_seq[3,kp_end_y_int,kp_end_x_int] = 1.0
        kpmap_seq[4,kp_end_y_int,kp_end_x_int] = kp_dy2
        kpmap_seq[5,kp_end_y_int,kp_end_x_int] = kp_dx2
    return kpmap_seq
    
    
def drawtrajs(trajs, frame_no, img):
    for traj_no in range(trajs.shape[0]):
        cv2.circle(img, tuple(trajs[traj_no, frame_no, :]), 2, (0.,1.,0.))
    return img
