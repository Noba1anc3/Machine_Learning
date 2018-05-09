from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import visdom
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
parser.add_argument('--naf', type=int, default=96)
parser.add_argument('--nepoch', type=int, default=450, help='number of epochs to train for')
parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netA', default='', help="path to netA (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

## set the manualSeed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1,10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##load the dataset, ToTensor means map the value range from (0 to 255) to (0 to 1)
## normalize transform the channel to (channel-mean) /std  normalize(mean,std)
dataset = dset.ImageFolder(root=opt.dataroot,
    transform=transforms.Compose([
         transforms.Scale(opt.imageSize),
         transforms.CenterCrop(opt.imageSize), 
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5),
    ]))

##piece the dataset in the guidance of batch_size，shffule=True means that during
##each epoches for training, the sequence will be disordered
##num_workers=2 means that use two progress to load the data

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
naf = int(opt.naf)
nc = 1

# custom weights initialization called on netG, netD and netA
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential()
        
            # input is nc, going into a convolution
            self.main.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
            self.main.add_module('input-relu',nn.LeakyReLU(0.2, inplace=True))
            
            # state size. (ngf) x 32 x 32
            for i in range 3:
                self.main.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
                self.main.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
                self.main.add_module('pyramid.{0}.relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))
            
            # state size. (ngf*8) x 4 x 4
            for i in range (3,0,-1):
                self.main.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, ngf * 2**(i-1), 4, 0.5, 1, bias=False))
                self.main.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
                self.main.add_module('pyramid.{0}.relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))
            
            # state size. (ngf) x 32 x 32
                self.main.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 0.5, 1, bias=False))
                self.main.add_module('output-tanh', nn.Tanh())
                
            # state size. (nc) x 64 x 64
        

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            ##use multiple gpus to run the forward function
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    ##Copies parameters and buffers from state_dict(torch.load(opt.netG)) into this module and its descendants.
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential()
        
             # input is (nc) x 64 x 64
             self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
             self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

             # state size. (ndf) x 32 x 32
             for i in range 3:
                 self.main.add_module('pyramid.{0}-{1}.conv'.format(ndf*2**(i), ndf * 2**(i+1)), nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
                 self.main.add_module('pyramid.{0}.batchnorm'.format(ndf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
                 self.main.add_module('pyramid.{0}.relu'.format(ndf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))
                    
             # state size. (ndf*8) x 4 x 4
             self.main.add_module('output-conv', nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
             self.main.add_module('output-sigmoid', nn.Sigmoid())
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            ##torch.view: to transform the scale of the tensor to other appearance
        return output.view(-1, 1)

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

class _netA(nn.Module):
    def __init__(self, ngpu):
        super(_netA, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential()
        
             # input is (nc*2) x 64 x 64
             self.main.add_module('input-conv', nn.Conv2d(nc*2, naf, 4, 2, 1, bias=False))
             self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

             # state size. (naf) x 32 x 32
             for i in range 3:
                 self.main.add_module('pyramid.{0}-{1}.conv'.format(naf*2**(i), naf * 2**(i+1)), nn.Conv2d(naf * 2 ** (i), naf * 2 ** (i+1), 4, 2, 1, bias=False))
                 self.main.add_module('pyramid.{0}.batchnorm'.format(naf * 2**(i+1)), nn.BatchNorm2d(naf * 2 ** (i+1)))
                 self.main.add_module('pyramid.{0}.relu'.format(naf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

             # state size. (naf*8) x 4 x 4
             self.main.add_module('output-conv', nn.Conv2d(naf * 8, 1, 4, 1, 0, bias=False))
             self.main.add_module('output-sigmoid', nn.Sigmoid())
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)

netA = _netA(ngpu)
netA.apply(weights_init)
if opt.netA != '':
    netA.load_state_dict(torch.load(opt.netA))
print(netA)

criterion = nn.BCELoss()

input_img = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
ass_label = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
noass_label = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize, 1)	
real_label = 1
fake_label = 0

input_img = Variable(input_img)
ass_label = Variable(ass_label)
noass_label = Variable(noass_label)
label = Variable(label)

if opt.cuda:
    netD.cuda()
    netA.cuda()
    netG.cuda()
    criterion.cuda()
    input_img, label = input_img.cuda(), label.cuda()
    ass_label, noass_label = ass_label.cuda(), noass_label.cuda()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerA = optim.Adam(netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
for epoch in range(opt.nepoch):
    for i, (images,_) in enumerate(dataloader):
        ###########################
        # (1) Update D network
        ###########################
        netD.zero_grad()

        # train with real(associated)
        # resize real because last batch may has less than
        # opt.batchSize images
        batch_size = images.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input_img.data.resize_(images.size()).copy_(images)
        label.data.resize_(batch_size).fill_(real_label)

        output = netD(ass_label)
        errD_real1 = criterion(output, label)
        errD_real1.backward()
        ##D_x1 = output.data.mean()

        # train with real(not associated)
        output = netD(noass_label)
        errD_real2 = criterion(output, label)
        errD_real2.backward()
        ##D_x2 = output.data.mean()
        
        # train with fake
        label.data.fill_(fake_label)
        fake = netG(input_img)   
        # detach gradients here so that gradients of G won't be updated
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        ##D_G_z1 = output.data.mean()

        errD = errD_real1 + errD_real2 + errD_fake
        optimizerD.step()           
        
        ###########################
        # (1) Update A network
        ###########################
        netA.zero_grad()

        fake = netG(input_img)
        assd = torch.cat(input_img, ass_label, 2)
        noassd = torch.cat(input_img, noass_label, 2)
        faked = torch.cat(input_img, fake, 2)

        # train with associated
        label.data.fill_(real_label)
        output = netA(assd)
        errA_real1 = criterion(output, label)
        errA_real1.backward()
        ##D_G_z1 = output.data.mean()

        # train with not associated
        label.data.fill_(fake_label)
        output = netA(noassd)
        errA_real2 = criterion(output, label)
        errA_real2.backward()
        ##D_G_z2 = output.data.mean()

        # train with fake
        output = netA(faked)
        errA_fake = criterion(output, label)
        errA_fake.backward()
        ##D_G_z3 = output.data.mean()	   

        errA = errA_real1 + errA_real2 + errA_fake
        optimizerA.step()

        ###########################
        # (1) Update G network
        ###########################

        netG.zero_grad()
        label.data.fill_(real_label)
        faked = torch.cat(input_img, fake, 2)

        output = netD(faked)
        errGD = criterion(output, label)
        errGD.backward()
        ##local df_dg = netD:updateGradInput(fake, df_do)

        output = netA(faked)
        errGA = criterion(output, label)
        errGA.backward()
        ##local df_dg2 = netA:updateGradInput(faked, df_do)
        ##local df_dg = df_dg2[{{},{4,6}}]

        errG = errGA + errGD
        optimizerG.step()

        ########### Logging #########
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_A: %.4f Loss_G: %.4f '
           % (epoch, opt.niter, i, len(dataloader),errD.data[0], errA.data[0],errG.data[0]))

        ########## Visualize #########
        if(i % 50 == 0):
            vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netA.state_dict(), '%s/netA_epoch_%d.pth' % (opt.outf, epoch))
