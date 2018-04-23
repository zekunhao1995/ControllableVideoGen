import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
    
import numpy as np

from dataset.data_loader_rp_reimpl import RPReader_traj
from models.vgg_warper_weak_shortcut import VGG_Warper
from utils.visual import colorcode, VisdomShow, pbar

from ops.flow_warper import FlowWarp
from ops.hardshinkloss import HardshinkLoss
from ops.laplace2d import Laplace2D


args = {}
args['gpus'] = [0]
args['seed'] = 12345
args['batch_size'] = 32
torch.backends.cudnn.benchmark = True

# Initialize Pytorch Dataloader
datareader = RPReader_traj(is_test=False, max_interval=10, min_ntraj=1, max_ntraj=5)  # change to min_ntraj=10, max_ntraj=10 for autoencoding (video prediction) evaluation
train_loader = torch.utils.data.DataLoader(
    datareader, batch_size=args['batch_size'], shuffle=True, collate_fn=datareader.collate_fn, worker_init_fn=datareader.worker_init_fn, num_workers=4, pin_memory=True)
    

class MModel(nn.Module):
    def __init__(self):
        super(MModel, self).__init__()
        self.warp_cnn = VGG_Warper(9)
        self.flow_warper = FlowWarp()
        self.mseloss = nn.MSELoss(size_average=True, reduce=True)
        self.hardshrinkloss = HardshinkLoss(0., 1.)
    
    def forward(self, img_input, warp_input, img_gt):
        warp_flow, masks, comp_imgs = self.warp_cnn(warp_input) # W*H*2
        warp_imgs = self.flow_warper(img_input, warp_flow)
        comp_imgs = F.hardtanh(comp_imgs,0.,1.)
        masks = F.sigmoid(masks)
        recon_img = torch.mul(warp_imgs, masks)+torch.mul(comp_imgs,1-masks)

        return recon_img, warp_flow, comp_imgs, masks
    
# Charbonnier penalty function
# Φ(x) = (x^2 + \epsilon^2)^{1/2}
class CPF(nn.Module):
    def __init__(self):
        super(CPF, self).__init__()
    def forward(self, x, mean=True):
        eps = 0.0001
        eps2 = eps**2
        if mean:
            loss = torch.mean(torch.sqrt(x**2+eps2))
        else:
            loss = torch.sum(torch.sqrt(x**2+eps2))
        
        return loss

mmodel = MModel()
mmodel.cuda()
mmodel = nn.DataParallel(mmodel, device_ids=[0,1])

#reconstruction_function = nn.BCELoss()
#reconstruction_function = nn.L1Loss()
mseloss = nn.MSELoss()
#mseloss.size_average = True
cpfloss = CPF()
hardshrinkloss = HardshinkLoss(0., 1.)
#sl1loss = nn.SmoothL1Loss(size_average=False)
optimizer = optim.Adam(mmodel.parameters(), lr=1e-3, weight_decay=0)

visual = VisdomShow('rp_train_humaneval')

def train(epoch):
    print('\n\n=========================== Epoch {} ============================'.format(epoch))
    mmodel.train()
    for batch_idx, (img_input, warp_input, img_gt, vid_mask) in enumerate(train_loader):
        img_input = Variable(img_input).cuda(args['gpus'][0])
        warp_input = Variable(warp_input).cuda(args['gpus'][0])
        img_gt = Variable(img_gt).cuda(args['gpus'][0])
        vid_mask = Variable(vid_mask).cuda(args['gpus'][0])

        optimizer.zero_grad()
        recon_img, warp_flow, comp_imgs, masks = mmodel(img_input, warp_input, img_gt)
        
        loss_comp_pen = hardshrinkloss(comp_imgs)
        loss_recon = cpfloss((recon_img-img_gt)*vid_mask)
        #loss_recon = mseloss(recon_img*vid_mask,img_gt*vid_mask)
        loss_mask_pen = torch.mean((masks-1.)**2)
        
        loss = loss_recon + 0.1*loss_comp_pen + 0.01*loss_mask_pen
        loss.backward()
        optimizer.step()
        
        hist['loss'].append(loss_recon.data.cpu().numpy()[0])
        hist['comp_pen'].append(loss_comp_pen.data.cpu().numpy()[0])
        
        if batch_idx%10 == 0:
            pbar(batch_idx, len(train_loader), epoch)
            
        if batch_idx%200 == 0:
            img_out = visual.add_text(recon_img[0,:,:,:].data.cpu().numpy(), 'Out', (0,0,1))
            img_in = visual.add_text(img_input[0,:,:,:].data.cpu().numpy(), 'In', (0,1,0))
            img_gt = visual.add_text(img_gt[0,:,:,:].data.cpu().numpy(), 'GT', (1,0,0))
            comp_out = visual.add_text(comp_imgs[0,:,:,:].data.cpu().numpy(), 'Comp', (0,1,1))
            mask_bw = masks[0,:,:,:].data.cpu().numpy()
            mask_out = visual.add_text(np.concatenate((mask_bw,mask_bw,mask_bw),0), 'Mask', (1,0,0))
            warp_out = visual.add_text(colorcode(warp_flow[0,:,:,:].data.cpu().numpy()), 'Flow', (0,0,0))
            
            visual.show_img(comp_out)
            visual.show_img(mask_out)
            visual.show_img(warp_out)
            vid = np.stack((img_in, img_out, img_gt, img_in, img_out, img_gt, img_gt), axis=0)
            visual.show_vid(vid)
        if batch_idx%2000 == 0:
            ckpt = {
                'mmodel_state_dict': mmodel.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'hist': hist
                }
            torch.save(ckpt, './snapshots/rp/ckpt_e{}_b{}.pth'.format(epoch, batch_idx))

def restore(ckpt_file):
    ckpt = torch.load(ckpt_file)
    mmodel.module.load_state_dict(ckpt['mmodel_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    #hist = ckpt['hist']
    print('Restored from {}'.format(ckpt_file))
    
hist = {}
hist['loss'] = []
hist['comp_pen'] = []

#restore('./snapshots2/ckpt_e1_b44000.pth')
for epoch in range(0, 20):
    #test(epoch)
    train(epoch)
    

