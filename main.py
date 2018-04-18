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

vis = visdom.Visdom()
vis.env = 'vae_dcgan'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='folder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=96)
parser.add_argument('--ndf', type=int, default=96)
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

dataset = dset.ImageFolder(root=opt.dataroot,
    transform=transforms.Compose([
         transforms.Scale(opt.imageSize),
         transforms.CenterCrop(opt.imageSize),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

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
        self.main = nn.Sequential(
            # input is nc, going into a convolution
            nn.ConvTranspose2d(nc, ngf, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ngf) x 32 x 32
            for i in range 3:
                nn.ConvTranspose2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False),
	nn.BatchNorm2d(ngf * 2**(i+1)),
                nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ngf*8) x 4 x 4
            for i in range (3,0,-1):
	  nn.ConvTranspose2d(ngf*2**(i), ngf * 2**(i-1), 4, 0.5, 1, bias=False),
	  nn.BatchNorm2d(ngf * 2**(i-1)),
	  nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ngf) x 32 x 32
	  nn.ConvTranspose2d(ngf,nc, 4, 0.5, 1, bias=False),
                  nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential()
        # input is (nc) x 64 x 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf) x 32 x 32
        for i in range 3:
               nn.Conv2d(ndf*2**(i), ndf * 2**(i+1), 4, 2, 1, bias=False),
               nn.BatchNorm2d(ndf * 2**(i+1)),
               nn.LeakyReLU(0.2, inplace=True),
               
        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

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
        nn.Conv2d(nc*2, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf) x 32 x 32
        for i in range 3:
	nn.Conv2d(ndf*2**(i), ndf * 2**(i+1), 4, 2, 1, bias=False),
	nn.BatchNorm2d(ndf * 2**(i+1)),
                nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*8) x 4 x 4
        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
        
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

input_img = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
ass_label = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noass_label = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize, 1)	
real_label = 1
fake_label = 0

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
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        
        # train with real(associated)
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        ############################
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        ############################
        ass_labelv = Variable(ass_label)     
        labelv = Variable(label.fill_(real_label))
        output = netD(ass_labelv)
        errD_real1 = criterion(output, labelv)
        errD_real1.backward()
        ##netD:backward(ass_label, df_do)
        D_x1 = output.data.mean()

        # train with real(not associated)
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        ############################
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        ############################
        noass_labelv = Variable(noass_label) 
        labelv = Variable(label.fill_(real_label))
        output = netD(noass_labelv)
        errD_real2 = criterion(output, labelv)
        errD_real2.backward()
        ##netD:backward(noass_label, df_do)
        D_x2 = output.data.mean()

        # train with fake
        input_imgv = Variable(input_img)
        fake = netG(input_imgv)   
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        ##netD:backward(fake, df_do)
        D_G_z1 = output.data.mean()

        errD = (errD_real1 + errD_real2 + errD_fake) / 3
        optimizerD.step()

        ###########################
        # (1) Update A network: 
        ###########################
        netA.zero_grad()

        assd = torch.cat(input_img, ass_label, 2)
        noassd = torch.cat(input_img, noass_label, 2)
        input_imgv = Variable(input_img)
        fake = netG(input_imgv)
        faked = torch.cat(input_img, fake, 2)
        
        assdv = Variable(assd)
        noassdv = Variable(noass)
        fakedv = Variable(faked)

        # train with associated
        labelv = Variable(label.fill_(real_label))
        output = netA(assdv)
        errA_real1 = criterion(output, labelv)
        errA_real1.backward()
        D_G_z1 = output.data.mean()

        # train with not associated
        labelv = Variable(label.fill_(fake_label))
        output = netA(noassdv)
        errA_real2 = criterion(output, labelv)
        errA_real2.backward()
        D_G_z1 = output.data.mean()
  
        # train with fake
        labelv = Variable(label.fill_(fake_label))
        output = netA(fakedv)
        errA_fake = criterion(output, labelv)
        errA_fake.backward()
        D_G_z1 = output.data.mean()	   

        errA = (errA_real1 + errA_real2 + errA_fake) / 3
        optimizerA.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        input_imgv = Variable(input_img)
        fake = netG(input_imgv)
        output = netD(fake)
        labelv = Variable(label.fill_(real_label))  
        errGD = criterion(output, labelv)
        errGD.backward()
        ##local df_dg = netD:updateGradInput(fake, df_do)
        ##netG:backward(input_img, df_dg)
        D_G_z2 = output.data.mean()

        faked = torch.cat(input_img, fake, 2)
        fakedv = Variable(faked)
        output = netA(fakedv)
        labelv = Variable(label.fill_(real_label))  
        errGA = criterion(output, labelv)
        errGA.backward()
        ##local df_dg2 = netA:updateGradInput(faked, df_do)
        ##local df_dg = df_dg2[{{},{4,6}}]
        ##netG:backward(input_img, df_dg)
        errG = (errGA + errGD)/2
        optimizerG.step()


        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.nepoch, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
