import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
    
import numpy as np

from dataset.data_loader_kitti_reimpl import KITTIReader_traj
from models.vgg_warper_weak_shortcut_nobn import VGG_Warper
from utils.visual import colorcode, VisdomShow, pbar

from ops.flow_warper_pad_2x import FlowWarp
from ops.hardshinkloss import HardshinkLoss
from ops.laplace2d import Laplace2D

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse

args = {}
args['gpus'] = [0]
args['seed'] = 12345
torch.backends.cudnn.benchmark = True


# Initialize Pytorch Dataloader
datareader = KITTIReader_traj(is_test=True, max_interval=10, min_ntraj=10, max_ntraj=10, is_eval=True)
train_loader = torch.utils.data.DataLoader(
    datareader, batch_size=4, shuffle=False, collate_fn=datareader.collate_fn, worker_init_fn=datareader.worker_init_fn, num_workers=4, pin_memory=True, drop_last = True)
    
class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()
        self.warp_cnn = VGG_Warper(9)
        self.flow_warper = FlowWarp()
        self.mseloss = nn.MSELoss(size_average=True, reduce=True)
        self.hardshrinkloss = HardshinkLoss(0., 1.)
    
    def forward(self, img_input, warp_input, img_gt):
        warp_flow, masks, comp_imgs = self.warp_cnn(warp_input) # W*H*2
        warp_imgs = self.flow_warper(img_input, warp_flow, padl=83)
        comp_imgs = F.hardtanh(comp_imgs,0.,1.)
        masks = F.sigmoid(masks)
        recon_img = torch.mul(warp_imgs, masks)+torch.mul(comp_imgs,1-masks)

        return recon_img, warp_flow, comp_imgs, masks

mmodel = MModel()
mmodel.cuda()
mmodel = nn.DataParallel(mmodel, device_ids=[0])

visual = VisdomShow('kitti_eval_10')

def test():
    print('\n\n=========================== Testing ============================')
    mmodel.eval()
    mse_stor = []
    ssim_stor = []
    for batch_idx, (img_input, warp_input, img_gt, vid_mask, img_input_2x) in enumerate(train_loader):
        img_input = Variable(img_input, volatile=True).cuda(args['gpus'][0])
        img_input_2x = Variable(img_input_2x).cuda(args['gpus'][0])
        warp_input = Variable(warp_input, volatile=True).cuda(args['gpus'][0])
        img_gt = Variable(img_gt, volatile=True).cuda(args['gpus'][0])
        vid_mask = Variable(vid_mask, volatile=True).cuda(args['gpus'][0])


        # warp_input : [interval-1, 9, H, W]
        # print(warp_input.shape) # ([1, 9, 9, 192, 256])
        recon_img, warp_flow, comp_imgs, masks = mmodel(img_input_2x, warp_input, img_gt)
        recon_img *= vid_mask
        img_gt *= vid_mask
        
        gen_seq = recon_img.data.cpu().numpy()
        gt_seq = img_gt.data.cpu().numpy()
        mses = np.zeros(gen_seq.shape[0])
        ssims = np.zeros(gen_seq.shape[0])
        for i in range(gen_seq.shape[0]):
            gen = np.transpose(gen_seq[i,:,:,:], [1,2,0])
            gt = np.transpose(gt_seq[i,:,:,:], [1,2,0])
            mses[i] = mse(gen,gt)
            ssims[i] = ssim(gt, gen, data_range=1., multichannel=True)
        
        mse_stor.append(mses.reshape([-1,9]))
        ssim_stor.append(ssims.reshape([-1,9]))
        
        
        if batch_idx%1 == 0:
            pbar(batch_idx, len(train_loader), 0)
        
        if batch_idx%10 == 0:
            mse_a = np.concatenate(mse_stor, axis=0)
            ssim_a = np.concatenate(ssim_stor, axis=0)
            psnr_all = -10*np.log(np.mean(mse_a, axis=0))/np.log(10)
            ssim_all = np.mean(ssim_a, axis=0)
            
            print('PSNR')
            print(psnr_all)
            print('SSIM')
            print(ssim_all)
        
        if batch_idx%10 == 0:
            out_seq = torch.cat((img_input[(0,),:,:,:],recon_img), dim=0).data.cpu().numpy()
            for i in range(out_seq.shape[0]):
                out_seq[i,:,:,:] = visual.add_text(out_seq[i,:,:,:], str(i), (0,1,1))
            out_gt = torch.cat((img_input[(0,),:,:,:],img_gt), dim=0).data.cpu().numpy()
            for i in range(out_gt.shape[0]):
                out_gt[i,:,:,:] = visual.add_text(out_gt[i,:,:,:], 'GT', (0,1,0))
                
            out_seq = np.concatenate((out_seq,out_gt), axis=3)
            visual.show_vid(out_seq)
    
    mse_a = np.concatenate(mse_stor, axis=0)
    ssim_a = np.concatenate(ssim_stor, axis=0)
    psnr_all = -10*np.log(np.mean(mse_a, axis=0))/np.log(10)
    ssim_all = np.mean(ssim_a, axis=0)
    print('\nPSNR SSIM')
    for i in range(psnr_all.size):
        print('{} {}'.format(psnr_all[i], ssim_all[i]))
                        
def restore(ckpt_file):
    ckpt = torch.load(ckpt_file)
    mmodel.module.load_state_dict(ckpt['mmodel_state_dict'])
    #optimizer.load_state_dict(ckpt['optimizer'])
    #hist = ckpt['hist']
    print('Restored from {}'.format(ckpt_file))
    
restore('./snapshots/kitti/ckpt_e0_b0_rev2.pth')
test()
    

