import torch
import torch.nn as nn
from torch.nn import init
import torch.optim.lr_scheduler
from torch.autograd import Variable
import itertools, functools
from utils import Pool, GANLoss, View, EnergyLoss
from collections import OrderedDict
import os
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class CycleGAN(object):

    def __init__(self, lr, lamb_A, lamb_B, lamb_I, lamb_E, output_dir, shape,
                    batch_size, pool_size, gpu_id=None, is_training=True, fc=[0, 2], mc=[1, 3]):

        self.tensor = torch.cuda.FloatTensor if gpu_id else torch.Tensor
        self.is_training = is_training
        self.output_dir = output_dir
        self.shape = shape
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.lr = lr
        self.lamb_A = lamb_A
        self.lamb_B = lamb_B
        self.lamb_I = lamb_I
        self.lamb_E = lamb_E
        self.gpu_id = gpu_id
        self.fc = fc
        self.mc = mc

        self.input_A = self.tensor(batch_size, 4, self.shape[0], self.shape[1])
        self.input_B = self.tensor(batch_size, 4, self.shape[0], self.shape[1])

        self.gen_A = self.build_G(2, 2, 8, self.gpu_id)
        self.gen_B = self.build_G(2, 2, 8, self.gpu_id)

        if self.is_training:
            self.dis_A = self.build_D(2, 4, self.gpu_id)
            self.dis_B = self.build_D(2, 4, self.gpu_id)

            # fake pool
            self.fplA = Pool(self.pool_size)
            self.fplB = Pool(self.pool_size)
            # real pool
            self.rplA = Pool(self.pool_size)
            self.rplB = Pool(self.pool_size)

            self.GAN_loss = GANLoss(tensor=self.tensor)
            self.cycle_loss = torch.nn.L1Loss()
            self.identity_loss = torch.nn.L1Loss() # if the input is from its target domain, it should output an identity one.
            self.energy_loss = EnergyLoss()

            self.optim_G = torch.optim.Adam(
                    itertools.chain(self.gen_A.parameters(), self.gen_B.parameters()),
                    lr=self.lr, betas=(0.5, 0.999)
                )
            """
            self.optim_D = torch.optim.Adam(
                    lr=self.lr,
                    itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()),
                )
            """
            self.optim_D_A = torch.optim.Adam(self.dis_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optim_D_B = torch.optim.Adam(self.dis_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optims = [self.optim_G, self.optim_D_A, self.optim_D_B]
            self.lr_scheds = []
            for optim in self.optims:
                epochs = np.linspace(100, 1000, 10, dtype=int)
                self.lr_scheds.append(
                    torch.optim.lr_scheduler.MultiStepLR(optim, epochs)
                )

    def build_D(self, input_nc, n_layers, gpu_ids):
        if not(isinstance(gpu_ids ,list)):
            gpu_ids = [gpu_ids]
        D = NLayerDiscriminator(input_nc, n_layers=n_layers, gpu_ids=gpu_ids)
        if len(gpu_ids) > 0:
            D.cuda(gpu_ids[0])
        D.apply(weights_init)
        return D


    def build_G(self, input_nc, output_nc, num_downs, gpu_ids):
        if not(isinstance(gpu_ids ,list)):
            gpu_ids = [gpu_ids]
        #G = UNetGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids)
        G = AxisCompressGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids)
        if len(gpu_ids) > 0:
            G.cuda(gpu_ids[0])
        G.apply(weights_init)
        return G

    def set_input(self, inputs):
        self.input_A.copy_(inputs["A"])
        self.input_B.copy_(inputs["B"])

    def optimize(self, n_G=1, n_D=1):
        self.forward()
        self.backward_G(n_G)
        self.backward_D(n_D)

    def sched_step(self):
        for s in self.lr_scheds:
            s.step()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def _backward_D(self, dis, real, fake):
        pred_real = dis(real)
        D_loss_real = self.GAN_loss(pred_real, True)
        pred_fake = dis(fake.detach())
        D_loss_fake = self.GAN_loss(pred_fake, False)
        D_loss = (D_loss_real + D_loss_fake) * 0.5
        D_loss.backward()
        return D_loss

    def backward_D(self, n_D=1, step=True):
        # put data into pool
        fake_A = self.fplA.draw(self.fake_A)
        fake_B = self.fplB.draw(self.fake_B)
        self.rplA.draw(self.input_A.clone())
        self.rplB.draw(self.input_B.clone())
        real_A = self.real_A[:, self.fc, ...]
        real_B = self.real_B[:, self.fc, ...]

        D_loss_As = []
        D_loss_Bs = []

        for i in range(n_D):
            # reset gradient
            self.optim_D_A.zero_grad()
            self.optim_D_B.zero_grad()

            D_loss_A = self._backward_D(self.dis_A, real_A, fake_A)
            if step: self.optim_D_A.step()

            D_loss_B = self._backward_D(self.dis_B, real_B, fake_B)
            if step: self.optim_D_B.step()

            D_loss_As.append(D_loss_A.data[0])
            D_loss_Bs.append(D_loss_B.data[0])

            # resample
            fake_A = Variable(self.fplA.get_samples(self.batch_size))
            fake_B = Variable(self.fplB.get_samples(self.batch_size))
            real_A = Variable(self.rplA.get_samples(self.batch_size))[:, self.fc, ...]
            real_B = Variable(self.rplB.get_samples(self.batch_size))[:, self.fc, ...]

        self.D_loss_A = np.mean(D_loss_As)
        self.D_loss_B = np.mean(D_loss_Bs)

    def backward_G(self, n_G=1, step=True):
        # reset gradient
        self.optim_G.zero_grad()

        # discriminator A
        fake_A = self.gen_A(self.real_B[:, self.mc, ...])
        pred_A = self.dis_A(fake_A)
        G_loss_A = self.GAN_loss(pred_A, True) # fool dis_A to think it's true

        # discriminator B
        fake_B = self.gen_B(self.real_A[:, self.mc, ...])
        pred_B = self.dis_B(fake_B)
        G_loss_B = self.GAN_loss(pred_B, True) # fool dis_B to think it's true


        # cycle loss
        # caculate real/imaginary part respectively
        rec_A = self.gen_A(fake_B)
        cyc_r_loss_A = self.lamb_A * self.cycle_loss(rec_A[:, 0, :, :], 
                          self.real_A[:, self.fc[0], :, :])
        cyc_i_loss_A = self.lamb_A * self.cycle_loss(rec_A[:, 1, :, :], 
                          self.real_A[:, self.fc[1], :, :])

        rec_B = self.gen_B(fake_A)
        cyc_r_loss_B = self.lamb_B * self.cycle_loss(rec_B[:, 0, :, :], 
                          self.real_B[:, self.fc[0], :, :])
        cyc_i_loss_B = self.lamb_B * self.cycle_loss(rec_B[:, 1, :, :], 
                          self.real_B[:, self.fc[1], :, :])

        # energy loss
        ene_A = self.lamb_E * self.energy_loss(fake_A, 
                  self.real_B[:, self.fc, ...])
        ene_B = self.lamb_E * self.energy_loss(fake_B, 
                  self.real_A[:, self.fc, ...])

        # calculate identity loss
        if self.lamb_I > 0.0:
            idt_A = self.gen_A(self.real_A[:, self.fc, ...])
            idt_B = self.gen_B(self.real_B[:, self.fc, ...])
            idt_loss_A = self.lamb_A * self.lamb_I * self.identity_loss(idt_A, 
                            self.real_A[:, self.fc, ...])
            idt_loss_B = self.lamb_B * self.lamb_I * self.identity_loss(idt_B, 
                            self.real_B[:, self.fc, ...])

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data

        self.idt_loss_A = idt_loss_A.data[0] if self.lamb_I > 0.0 else 0.
        self.idt_loss_B = idt_loss_B.data[0] if self.lamb_I > 0.0 else 0.

        G_loss = G_loss_A + G_loss_B +  \
                 (cyc_r_loss_A + cyc_i_loss_A) * 0.5 + \
                 (cyc_r_loss_B + cyc_i_loss_B) * 0.5 + \
                 idt_loss_A + idt_loss_B + \
                 (ene_A + ene_B) * 0.5
        G_loss.backward()
        if step: self.optim_G.step()

        self.fake_A = fake_A.data
        self.fake_B = fake_B.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.G_loss_A = G_loss_A.data[0]
        self.G_loss_B = G_loss_B.data[0]
        self.cyc_r_loss_A = cyc_r_loss_A.data[0]
        self.cyc_i_loss_A = cyc_i_loss_A.data[0]
        self.cyc_r_loss_B = cyc_r_loss_B.data[0]
        self.cyc_i_loss_B = cyc_i_loss_B.data[0]
        self.ene_A = ene_A.data[0]
        self.ene_B = ene_B.data[0]

    def report_errors(self):
        report = OrderedDict([
                    ("G_A", self.G_loss_A), ("G_B", self.G_loss_B),
                    ("D_A", self.D_loss_A), ("D_B", self.D_loss_B),
                    ("Cyc_r_A", self.cyc_r_loss_A), ("Cyc_r_B", self.cyc_r_loss_B),
                    ("Cyc_i_A", self.cyc_i_loss_A), ("Cyc_i_B", self.cyc_i_loss_B),
                    ("Ene_A", self.ene_A), ("Ene_B", self.ene_B)])
        if self.lamb_I > 0.0:
            report["idt_A"] = self.idt_loss_A
            report["idt_B"] = self.idt_loss_B
        return report

    def get_samples(self, n_sample):
        fake_As = self.fplA.get_samples(n_sample)
        fake_Bs = self.fplB.get_samples(n_sample)
        return fake_As, fake_Bs

    def generate(self):
        real_A = Variable(self.input_A, volatile=True)
        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.gen_A(real_B[:, self.fc, ...])
        fake_B = self.gen_B(real_A[:, self.fc, ...])
        recn_A = self.gen_A(fake_B)
        recn_B = self.gen_B(fake_A)

        return fake_A.data[0], fake_B.data[0], recn_A.data[0], recn_B.data[0]

    def save(self, label):
        self.save_network(self.gen_A, "G_A", label, self.gpu_id)
        self.save_network(self.dis_A, "D_A", label, self.gpu_id)
        self.save_network(self.gen_B, "G_B", label, self.gpu_id)
        self.save_network(self.dis_B, "D_B", label, self.gpu_id)

    def save_network(self, network, network_label, epoch_label, gpu_id=None):
        save_filename = "{}_net_{}.pth".format(epoch_label, network_label)
        save_path = os.path.join(self.output_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_id) > 0 and torch.cuda.is_available():
            network.cuda(gpu_id[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = "{}_net_{}.pth".format(epoch_label, network_label)
        save_path = os.path.join(self.output_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=32, n_layers=3, k_size=4, stride=(2,2), padding=1, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            # args are partially set.
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        seq = [nn.Conv2d(input_nc, ndf, kernel_size=k_size,
                            stride=stride, padding=padding),
               nn.LeakyReLU(0.2, True)]
        nf_mul = nf_mul_prev = 1
        for i in range(1, n_layers):
            nf_mul_prev = nf_mul
            nf_mul = min(2**i, 2)
            seq += [nn.Conv2d(ndf * nf_mul_prev, ndf * nf_mul, kernel_size=k_size,
                            stride=stride, padding=padding, bias=use_bias),
                    norm_layer(ndf * nf_mul),
                    nn.LeakyReLU(0.2, True)]
        nf_mul_prev = nf_mul
        nf_mul = min(2**n_layers, 2)
        seq += [nn.Conv2d(ndf * nf_mul_prev, ndf * nf_mul, kernel_size=k_size,
                            stride=stride, padding=padding, bias=use_bias),
                norm_layer(ndf * nf_mul),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mul, 1, kernel_size=k_size, stride=1, padding=padding),
                View(-1, 93),
                nn.Linear(93, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            r = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            r = self.model(input)
        return r

class AxisCompressGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                  norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        self.gpu_ids = gpu_ids
        super(AxisCompressGenerator, self).__init__()

        # (256, 32, 32)
        unet_block = UNetBlock(ngf * 8, ngf * 8, input_nc=None,
                    submodule=None, norm_layer=norm_layer,
                    k_size=(1, 2), stride=(1, 1), padding=(0, 1), pos="innermost")

        # (256, 32, 64)
        unet_block = UNetBlock(ngf * 8, ngf * 8, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(1, 4), stride=(1, 2), padding=(0, 1))

        # (256, 32, 128)
        unet_block = UNetBlock(ngf * 8, ngf * 8, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(1, 16), stride=(1, 2), padding=(0, 7))

        # (128, 32, 256)
        unet_block = UNetBlock(ngf * 4, ngf * 8, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(2, 1), stride=(2, 1), padding=(0, 0))

        # (128, 64, 256)
        unet_block = UNetBlock(ngf * 2, ngf * 4, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(4, 1), stride=(2, 1), padding=(1, 0))

        # (128, 128, 256)
        unet_block = UNetBlock(ngf * 1, ngf * 2, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(16, 1), stride=(2, 1), padding=(7, 0))

        # (64, 256, 256)
        unet_block = UNetBlock(output_nc, ngf, input_nc=input_nc,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(64, 1), stride=(4, 1), padding=(31, 0), pos="outermost", transpose=(30, 0))

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UNetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        unet_block = UNetBlock(ngf * 4, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer, pos="innermost", stride=(2,1))
        for i in range(num_downs - 4):
            unet_block = UNetBlock(ngf * 4, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        #unet_block = UNetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, pos="outermost")

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class UNetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                k_size=(4,4), stride=(2,2), padding=(1,1),
                submodule=None, pos=None, norm_layer=nn.BatchNorm2d,
                transpose=None):
        super(UNetBlock, self).__init__()
        self.pos = pos
        self.outermost = False
        if type(norm_layer) == functools.partial:
            # args are partially set.
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc == None:
            input_nc = outer_nc
        if transpose == None:
            transpose = padding

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=k_size,
                             stride=stride, padding=padding, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if pos == "outermost":
            # inner_nc * 2: it takes as input cat([x, submodule(x)])
            self.outermost = True
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif pos == "innermost":
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # inner_nc * 2: it takes as input cat([x, submodule(x)])
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)

