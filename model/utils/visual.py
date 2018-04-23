import numpy as np
from visdom import Visdom
import cv2
import os
# OpenBLAS screws up with CPU affinity
os.sched_setaffinity(0,range(os.cpu_count()))


class VisdomShow():
    def __init__(self, env_name):
        self.vis = Visdom(env=env_name)
        print('Visdom display initialized')
        
    def show_img(self, img):
        #img = img[(2,1,0),:,:]
        self.vis.image(np.clip(img,0,1))
        #self.vis.image(np.clip(img.data.cpu().numpy(),0,1))
    
    def show_vid(self, vid):
        vid = (np.clip(vid,0.,1.)*255.).astype(np.uint8)
        vid = np.transpose(vid[:,(2,1,0),:,:], (0,2,3,1))
        self.vis.video(vid, opts={'fps': 2})

    def add_text(self, img, text, color=(0,255,0)):
        img = np.transpose(img[(2,1,0),:,:], (1,2,0)).copy()
        cv2.putText(img, text, (2,24), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1, cv2.LINE_AA)
        img = np.transpose(img, (2,0,1))[(2,1,0),:,:]
        return img


def colorcode(flow_in): # N 1 H W, H S V=1
    #hsv = np.zeros((512, 512, 3))
    #hsv[..., 0] = np.linspace(0, 1, 512)
    #hsv[..., 1] = 1.
    #hsv[..., 2] = np.linspace(0, 1, 512)[:, np.newaxis]
    #rgb = hsv_to_rgb(hsv)
    flow_x = flow_in[0,:,:] / 5
    flow_y = flow_in[1,:,:] / 5
    shape = flow_x.shape
    H = np.arctan2(flow_x, flow_y) / (2.*np.pi) # [0,1)
    H = np.ravel(H)
    S = np.sqrt(flow_x**2+flow_y**2) # [0, len]
    S = np.ravel(S)
    
    i = np.int_(H*6.)
    f = H*6.-i

    q = f
    t = 1.-f
    i = np.ravel(i)
    f = np.ravel(f)
    i%=6
    t = np.ravel(t)
    q = np.ravel(q)
    v = 1
    clist = (1-S*np.vstack([np.zeros_like(f),np.ones_like(f),q,t]))*v

    #0:v 1:p 2:q 3:t
    order = np.array([[0,3,1],[2,0,1],[1,0,3],[1,2,0],[3,1,0],[0,1,2]])
    rgb = clist[order[i], np.arange(np.prod(shape))[:,None]]

    rgb = np.transpose(rgb.reshape(shape+(3,)),[2,0,1])
    return rgb

    
import sys
def pbar(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s epoch %s\r' % (bar, count, total, status))
    sys.stdout.flush()
