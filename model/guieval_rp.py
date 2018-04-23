import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
from reader.rp_reader import RPReader
from models.vgg_warper_weak_shortcut import VGG_Warper
from ops.flow_warper import FlowWarp
import matplotlib.pyplot as plt

import time
import itertools
import math

from scipy import misc

from utils.trajs2map import trajs2map
from utils.visual import colorcode

# Setup parameters
parser = argparse.ArgumentParser(description='Nothing')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50000, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
                    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.num_frames = 5

args.gpus = [0]
#torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

reader = RPReader(num_frames=20)

class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()
        self.warp_cnn = VGG_Warper(9)
        self.flow_warper = FlowWarp()
    
    def forward(self, img_input, warp_input, img_gt):
        warp_flow, masks, comp_imgs = self.warp_cnn(warp_input) # W*H*2
        warp_imgs = self.flow_warper(img_input, warp_flow)
        comp_imgs = F.hardtanh(comp_imgs,0.,1.)
        masks = F.sigmoid(masks)
        recon_img = torch.mul(warp_imgs, masks)+torch.mul(comp_imgs,1-masks)

        return recon_img, warp_flow, comp_imgs, masks, warp_imgs


mmodel = MModel()
mmodel.cuda(args.gpus[0])

def get_test_batch():
    vid_seq, kpmap_seq, traj_list = reader[-1]
    vid_seq = torch.from_numpy(vid_seq).unsqueeze(0)
    kpmap_seq = torch.from_numpy(kpmap_seq).unsqueeze(0)
    
    vid_seq = Variable(vid_seq, volatile=True)
    kpmap_seq = Variable(kpmap_seq, volatile=True)
    vid_seq = vid_seq.cuda(args.gpus[0])
    kpmap_seq = kpmap_seq.cuda(args.gpus[0])
    return vid_seq, kpmap_seq, traj_list
    # traj_list: Num, Len, x/y


# First click defines start point
# second click defines end point
# Click outside canvas to clear trajectories
# Press right mouse button to go to next image
def onclick(event):
    global sp, ep
    global clr, gonext
    global ix, iy
    if event.button == 3:
        gonext = True
        return
    ix, iy = event.xdata, event.ydata
    if ix is None:
        clr = True
        return 
    print('x = %d, y = %d'%(ix, iy))

    if sp is not None:
        if ep is not None:
            sp = (ix, iy)
            ep = None
        else:
            ep = (ix, iy)
    else:
        sp = (ix, iy)
            
            
    #if len(coords) == 2:
    #    fig.canvas.mpl_disconnect(cid)

    #return coords
    

def img_chooser():
    global sp, ep, clr, gonext
    sp = None
    ep = None
    clr = False
    gonext = False
    
    
    fig = plt.figure(1)
    ax = fig.add_subplot(231)
    ax.set_title('click to build line segments')
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    mmodel.eval()
    counter=0
    while True:
        #plt.clf()
        vid_seq, kpmap_seq, traj_list = get_test_batch()
        fram_stor = []
        img_input = vid_seq[:,0,:,:,:]
        trajs = []
        while  True:
            if gonext:
                gonext = False
                sp = None
                ep = None
                trajs = []
                break
            if sp is not None and ep is not None:
                print('Move!')
                trajs.append((sp,ep))
            if clr:
                print('Clr!')
                clr = False
                sp = None
                ep = None
                trajs = []
            kpmap_seq = trajs2map( trajs, img_input.size(2), img_input.size(3))
            warp_input = torch.cat((img_input, kpmap_seq), dim=1)
            recon_img, warp_flow, comp, alpha, warp_img = mmodel(img_input, warp_input, None)
            
            #img_gt = vid_seq[:,ff,:,:,:]
            #fram_stor.append(recon_img)
            
            fram = np.transpose(recon_img[0,:,:,:].data.cpu().numpy()+0.5, [1,2,0])
            framin = np.transpose(img_input[0,:,:,:].data.cpu().numpy()+0.5, [1,2,0])
            warpimga = np.transpose(warp_img[0,:,:,:].data.cpu().numpy()+0.5, [1,2,0])
            #misc.imsave('./FirstImage/{}.png'.format(counter), fram)
            counter += 1
            ax.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            ax.imshow(framin)
            #ax6.imshow(framin)
            ax6.imshow(warpimga)
            ax5.imshow(fram)
            
            max_flow = torch.sqrt(torch.max(warp_flow[0,0,:,:]**2 + warp_flow[0,1,:,:]**2)).data.cpu().numpy()
            warp_flow_c = np.clip(colorcode(warp_flow.data.cpu().numpy()[0,0,:,:]/max_flow, warp_flow.data.cpu().numpy()[0,1,:,:]/max_flow),0,1)
            ax2.imshow(np.transpose(warp_flow_c,[1,2,0]))
            ax3.imshow(np.transpose(comp[0,:,:,:].data.cpu().numpy()+0.5, [1,2,0]))
            ax4.imshow(alpha[0,0,:,:].data.cpu().numpy()+0.5, cmap=plt.get_cmap('Greys'))
            
            
            for arr in trajs:
                ax.arrow( arr[0][0], arr[0][1], arr[1][0]-arr[0][0], arr[1][1]-arr[0][1], fc="g", ec="g",head_width=5, head_length=5 )
            fig.canvas.draw()
            fig.savefig('user_out/{}.png'.format(counter), bbox_inches='tight', pad_inches=0)
            
            plt.waitforbuttonpress()


ckpt = torch.load('./ckpt_RP.pth')
mmodel.load_state_dict(ckpt['mmodel_state_dict'])
img_chooser()

