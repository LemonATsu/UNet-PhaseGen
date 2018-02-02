import torch
import torch.nn as nn
from torch.nn import init
import torch.optim.lr_scheduler
from torch.autograd import Variable
import itertools, functools
from utils import Pool, GANLoss, View, EnergyLoss, Transpose
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
                    batch_size, pool_size, gpu_id=None, is_training=True, fc=[0, 2], mc=[1, 3], 
                    lamb_k=0.0001, gamma=0.9, mtype=None, lamb_G=2.):

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
        self.lamb_k = lamb_k
        self.gamma = gamma
        self.fc = fc
        self.mc = mc
        self.lamb_G = lamb_G
        if mtype == "BE":
            self.kt_A = 0.
            self.kt_B = 0.

        self.input_A = self.tensor(batch_size, 2, self.shape[0], self.shape[1])
        self.input_B = self.tensor(batch_size, 2, self.shape[0], self.shape[1])

        self.gen_A = self.build_G(2, 2, 8, mtype, self.gpu_id)
        self.gen_B = self.build_G(2, 2, 8, mtype, self.gpu_id)

        if self.is_training:
            self.dis_A = self.build_D(2, 3, mtype, self.gpu_id)
            self.dis_B = self.build_D(2, 3, mtype, self.gpu_id)

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
            self.BE_loss = torch.nn.L1Loss()

            self.optim_G = torch.optim.Adam(
                    itertools.chain(self.gen_A.parameters(), self.gen_B.parameters()),
                    lr=self.lr, betas=(0.5, 0.999)
                )
            self.optim_D_A = torch.optim.Adam(self.dis_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optim_D_B = torch.optim.Adam(self.dis_B.parameters(), lr=self.lr, betas=(0.5, 0.999))
            self.optims = [self.optim_G, self.optim_D_A, self.optim_D_B]
            self.lr_scheds = []
            for optim in self.optims:
                epochs = np.linspace(100, 1000, 10, dtype=int)
                self.lr_scheds.append(
                    torch.optim.lr_scheduler.MultiStepLR(optim, epochs)
                )

    def build_D(self, input_nc, n_layers, mtype, gpu_ids):
        if not(isinstance(gpu_ids ,list)):
            gpu_ids = [gpu_ids]
        print("Build D: type {}".format(mtype))

        if mtype is None:
            D = NLayerDiscriminator(input_nc, n_layers=n_layers, gpu_ids=gpu_ids)
        elif mtype == "BE":
            D = AxisCompressGenerator(input_nc, input_nc, 8, gpu_ids=gpu_ids)
            #D = AEModel(input_nc, input_nc, gpu_ids=gpu_ids, batch_size=self.batch_size)
        else:
            raise NotImplementedError

        if len(gpu_ids) > 0:
            D.cuda(gpu_ids[0])
        D.apply(weights_init)
        return D


    def build_G(self, input_nc, output_nc, num_downs, mtype, gpu_ids):
        if not(isinstance(gpu_ids ,list)):
            gpu_ids = [gpu_ids]

        #G = UNetGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids)
        print("Build G: type {}".format(mtype))
        if mtype is None:
            G = AxisCompressGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids, gen_mask=True)
        elif mtype == "BE":
            G = AxisCompressGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids, gen_mask=True)
            #G = AEModel(input_nc, input_nc, gpu_ids=gpu_ids, batch_size=self.batch_size)
        else:
            raise NotImplementedError

        if len(gpu_ids) > 0:
            G.cuda(gpu_ids[0])
        G.apply(weights_init)
        return G

    def set_input(self, inputs):
        self.input_A.copy_(inputs["A"])
        self.input_B.copy_(inputs["B"])

    def optimize(self, n_G=1, n_D=1, mtype=None):
        self.forward()
        if mtype is None:
            self.backward_G(n_G)
            self.backward_D(n_D)
        elif mtype == "BE":
            self.backward_BE()
        else:
            raise NotImplementedError

    def sched_step(self):
        for s in self.lr_scheds:
            s.step()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def _get_D_loss_BE(self, dis, real, fake, kt):
        # detach the fake variable to prevent gradient flowing back
        AE_x = dis(real)
        AE_g = dis(fake)
        D_loss_real = self.BE_loss(AE_x, real)
        D_loss_fake = self.BE_loss(AE_g, fake)
        D_loss = D_loss_real - kt * D_loss_fake 
        return D_loss, D_loss_real

    def _get_G_loss_BE(self, dis, fake):
        AE_g = dis(fake).detach()
        G_loss = self.BE_loss(fake, AE_g) * self.lamb_G
        return G_loss

    def _get_cyc_loss(self, gen, real, fake, lamb):
        rec = gen(fake)
        cyc_r_loss = lamb * self.cycle_loss(rec[:, 0, ...], real[:, 0, ...])
        cyc_i_loss = lamb * self.cycle_loss(rec[:, 1, ...], real[:, 1, ...])
        return cyc_r_loss, cyc_i_loss

    def _get_idt_loss(self, gen, real, lamb):
        idt = gen(real)
        idt_loss = lamb * self.lamb_I * self.identity_loss(idt, real)
        return idt_loss

    def backward_BE(self):

        # Let's try full-mix first
        real_A = self.real_A[:, self.fc, ...]
        real_B = self.real_B[:, self.fc, ...]

        # Get fake things
        fake_A = self.gen_A(real_B)
        fake_B = self.gen_B(real_A)

        # Put things into pool
        sample_A = self.fplA.draw(fake_A.data)
        sample_B = self.fplB.draw(fake_B.data)
        self.rplA.draw(self.input_A.clone())
        self.rplB.draw(self.input_B.clone())

        # Get D loss
        # train with samples
        D_loss_A, D_loss_real_A = self._get_D_loss_BE(self.dis_A, real_A, sample_A, self.kt_A)
        D_loss_B, D_loss_real_B = self._get_D_loss_BE(self.dis_B, real_B, sample_B, self.kt_B)


        # Get G loss
        G_loss_A = self._get_G_loss_BE(self.dis_A, fake_A)
        G_loss_B = self._get_G_loss_BE(self.dis_B, fake_B)

        # update k
        balance_A = (self.gamma * D_loss_real_A - G_loss_A).data[0] 
        balance_B = (self.gamma * D_loss_real_B - G_loss_B).data[0]
        self.kt_A = min(max(self.kt_A + self.lamb_k * balance_A, 0.01), 1) 
        self.kt_B = min(max(self.kt_B + self.lamb_k * balance_B, 0.01), 1) 

        # cycle loss
        cyc_r_loss_A, cyc_i_loss_A = self._get_cyc_loss(self.gen_A, real_A, fake_B, self.lamb_A)
        cyc_r_loss_B, cyc_i_loss_B = self._get_cyc_loss(self.gen_B, real_B, fake_A, self.lamb_B)

        # identity loss
        idt_loss_A = self._get_idt_loss(self.gen_A, real_A, self.lamb_A)
        idt_loss_B = self._get_idt_loss(self.gen_B, real_B, self.lamb_B)
        
        # energy loss
        ene_A = self.lamb_E * self.energy_loss(fake_A, real_B)
        ene_B = self.lamb_E * self.energy_loss(fake_B, real_A)

        G_loss = G_loss_A + G_loss_B + \
                     (cyc_r_loss_A + cyc_i_loss_A) * 0.5 + \
                     (cyc_r_loss_B + cyc_i_loss_B) * 0.5 + \
                     idt_loss_A + idt_loss_B + \
                     ene_A + ene_B
        # take the optimization step    
        self.optim_G.zero_grad()
        G_loss.backward()
        self.optim_G.step()

        self.optim_D_A.zero_grad()
        self.optim_D_B.zero_grad()
        D_loss_A.backward()
        D_loss_B.backward()
        self.optim_D_A.step()
        self.optim_D_B.step()

        # BE measure
        self.M_A = D_loss_real_A.data[0] + abs(balance_A)
        self.M_B = D_loss_real_B.data[0] + abs(balance_B)

        self.D_loss_A = D_loss_A.data[0]
        self.D_loss_B = D_loss_B.data[0]
        self.G_loss_A = G_loss_A.data[0]
        self.G_loss_B = G_loss_B.data[0]
        self.idt_loss_A = idt_loss_A.data[0]
        self.idt_loss_B = idt_loss_B.data[0]
        self.cyc_r_loss_A = cyc_r_loss_A.data[0]
        self.cyc_i_loss_A = cyc_i_loss_A.data[0]
        self.cyc_r_loss_B = cyc_r_loss_B.data[0]
        self.cyc_i_loss_B = cyc_i_loss_B.data[0]
        self.ene_loss_A = ene_A.data[0]
        self.ene_loss_B = ene_B.data[0]

    def _backward_D(self, dis, real, fake):
        pred_real = dis(real)
        D_loss_real = self.GAN_loss(pred_real, True)
        pred_fake = dis(fake.detach()) # stop gradient from flowing back to G 
        D_loss_fake = self.GAN_loss(pred_fake, False)
        D_loss = (D_loss_real + D_loss_fake) * 0.5
        D_loss.backward()
        return D_loss

    def backward_D(self, n_D=1, step=True):
        # put data into pool
        D_loss_As, D_loss_Bs = [], []
        fake_A = self.fplA.draw(self.fake_A)
        fake_B = self.fplB.draw(self.fake_B)
        self.rplA.draw(self.input_A.clone())
        self.rplB.draw(self.input_B.clone())
        real_A = self.real_A[:, self.fc, ...]
        real_B = self.real_B[:, self.fc, ...]

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
        real_A = self.real_A
        real_B = self.real_B

        G_loss_As, G_loss_Bs = [], []
        cyc_r_loss_As, cyc_i_loss_As = [], []
        cyc_r_loss_Bs, cyc_i_loss_Bs = [], []
        ene_As, ene_Bs = [], []


        for i in range(n_G):
            self.optim_G.zero_grad()
            fake_A = self.gen_A(real_B[:, self.mc, ...])
            pred_A = self.dis_A(fake_A)
            G_loss_A = self.GAN_loss(pred_A, True) # fool dis_A

            fake_B = self.gen_B(real_A[:, self.mc, ...])
            pred_B = self.dis_B(fake_B)
            G_loss_B = self.GAN_loss(pred_B, True) # fool dis_B

            # cycle loss
            # caculate real/imag part respectively
            rec_A = self.gen_A(fake_B)
            cyc_r_loss_A = self.lamb_A * self.cycle_loss(rec_A[:, 0, ...], real_A[:, self.fc[0], ...])
            cyc_i_loss_A = self.lamb_A * self.cycle_loss(rec_A[:, 1, ...], real_A[:, self.fc[1], ...])

            # reconstruction loss
            rec_B = self.gen_B(fake_A)
            cyc_r_loss_B = self.lamb_B * self.cycle_loss(rec_B[:, 0, ...], real_B[:, self.fc[0], ...])
            cyc_i_loss_B = self.lamb_B * self.cycle_loss(rec_B[:, 1, ...], real_B[:, self.fc[1], ...])

            # energy loss
            ene_A = self.lamb_E * self.energy_loss(fake_A, real_B[:, self.fc, ...])
            ene_B = self.lamb_E * self.energy_loss(fake_B, real_A[:, self.fc, ...])

            if self.lamb_I > 0.:
                idt_A = self.gen_A(real_A[:, self.fc, ...])
                idt_B = self.gen_B(real_B[:, self.fc, ...])
                idt_loss_A = self.lamb_A * self.lamb_I * self.identity_loss(idt_A, real_A[:, self.fc, ...])
                idt_loss_B = self.lamb_B * self.lamb_I * self.identity_loss(idt_B, real_B[:, self.fc, ...])

                self.idt_A = idt_A.data
                self.idt_B = idt_B.data

            self.idt_loss_A = idt_loss_A.data[0]
            self.idt_loss_B = idt_loss_B.data[0]

            # sum the loss and back-prop
            G_loss = G_loss_A + G_loss_B + \
                     (cyc_r_loss_A + cyc_i_loss_A) * 0.5 + \
                     (cyc_r_loss_B + cyc_i_loss_B) * 0.5 + \
                     idt_loss_A + idt_loss_B + \
                     (ene_A + ene_B) * 0.5
            G_loss.backward()
            if step: self.optim_G.step()
            
            if i == 0:
                # the data is sampled from dataset when i==0.
                # therefore, set self.fake here, and it will be put
                # into pool when the Ds are backwarding
                self.fake_A = fake_A.data
                self.fake_B = fake_B.data
                self.rec_A = rec_A.data
                self.rec_B = rec_B.data

            G_loss_As.append(G_loss_A.data[0])
            G_loss_Bs.append(G_loss_B.data[0])
            cyc_r_loss_As.append(cyc_r_loss_A.data[0])
            cyc_i_loss_As.append(cyc_i_loss_A.data[0])
            cyc_r_loss_Bs.append(cyc_r_loss_B.data[0])
            cyc_i_loss_Bs.append(cyc_i_loss_B.data[0])
            ene_As.append(ene_A.data[0])
            ene_Bs.append(ene_B.data[0])

            if (n_G > 1) and (self.rplA.n > n_G) and (self.rplB.n > n_G): 
                # sample new data
                real_A = Variable(self.rplA.get_samples(self.batch_size))
                real_B = Variable(self.rplB.get_samples(self.batch_size))
            else:
                break
        
        self.G_loss_A = np.mean(G_loss_As)
        self.G_loss_B = np.mean(G_loss_Bs)
        self.cyc_r_loss_A = np.mean(cyc_r_loss_As)
        self.cyc_i_loss_A = np.mean(cyc_i_loss_As)
        self.cyc_r_loss_B = np.mean(cyc_r_loss_Bs)
        self.cyc_i_loss_B = np.mean(cyc_i_loss_Bs)
        self.ene_A = np.mean(ene_As)
        self.ene_B = np.mean(ene_Bs)

    def report_errors(self):
        report = OrderedDict([
                    ("G_A", self.G_loss_A), ("G_B", self.G_loss_B),
                    ("D_A", self.D_loss_A), ("D_B", self.D_loss_B),
                    ("Cyc_r_A", self.cyc_r_loss_A), ("Cyc_r_B", self.cyc_r_loss_B),
                    ("Cyc_i_A", self.cyc_i_loss_A), ("Cyc_i_B", self.cyc_i_loss_B)])
        if self.lamb_I > 0.0:
            report["idt_A"] = self.idt_loss_A
            report["idt_B"] = self.idt_loss_B
        if self.lamb_E > 0.0:
            report["ene_A"] = self.ene_loss_A
            report["ene_B"] = self.ene_loss_B
        report["M_A"] = self.M_A
        report["M_B"] = self.M_B
        report["kt_A"] = self.kt_A
        report["kt_B"] = self.kt_B
        return report

    def get_samples(self, n_sample):
        fake_As = self.fplA.get_samples(n_sample)
        fake_Bs = self.fplB.get_samples(n_sample)
        return fake_As, fake_Bs

    def generate(self):
        real_A = Variable(self.input_A, volatile=True)
        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.gen_A(real_B)
        fake_B = self.gen_B(real_A)
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
                View(-1, 441),
                nn.Linear(441, 1)]
        self.model = nn.Sequential(*seq)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            r = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            r = self.model(input)
        return r

class AxisCompressGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32,
                  norm_layer=nn.BatchNorm2d, gpu_ids=[], gen_mask=False):
        self.gpu_ids = gpu_ids
        self.gen_mask = gen_mask
        super(AxisCompressGenerator, self).__init__()

        if self.gen_mask:
            print("build with difference.")

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

        # (128, 128, 128)
        unet_block = UNetBlock(ngf * 1, ngf * 2, input_nc=None,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(16, 1), stride=(2, 1), padding=(7, 0))

        # (64, 256, 128)
        unet_block = UNetBlock(output_nc, ngf, input_nc=input_nc,
                    submodule=unet_block, norm_layer=norm_layer,
                    k_size=(16, 4), stride=(4, 1), padding=(7, 1), pos="outermost", transpose=(30, 0))

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            f = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            f = self.model(input)
        #if self.gen_mask:
        #    f = f + input
        return f


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


class AEModel(nn.Module):
    def __init__(self, input_nc, output_nc, n_f=3, n_t=2, batch_size=1, hidden=256, gpu_ids=[], norm_layer=nn.BatchNorm2d, trp=False):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        super(AEModel, self).__init__()
        if type(norm_layer) == functools.partial:
            # args are partially set.
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            # compress freq
            View(batch_size, input_nc, 128),
            nn.Conv1d(input_nc, input_nc // 2, kernel_size=16, stride=2, padding=7), # 1024, 64
            norm_layer(input_nc // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(input_nc // 2, input_nc // 2, kernel_size=8, stride=2, padding=3), # 512, 32
            norm_layer(input_nc // 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(input_nc // 2, input_nc // 4, kernel_size=4, stride=2, padding=1), # 256, 16
            norm_layer(input_nc // 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(input_nc // 4, input_nc // 4, kernel_size=2, stride=2, padding=0), # 128, 8
            norm_layer(input_nc // 4),
            nn.LeakyReLU(0.2, True),
            View(batch_size, -1),
            nn.Linear(input_nc // 4 * 8, 256) if not trp else nn.Linear(input_nc // 4 * 64, 256)
        ]

        decoder = [
            nn.Linear(256, input_nc // 4  * 8) if not trp else nn.Linear(256, input_nc // 4 * 64),
            View(batch_size, input_nc // 4, 8) if not trp else View(batch_size, input_nc // 4, 64),
            # decode time
            nn.ConvTranspose1d(input_nc // 4, input_nc // 4, kernel_size=2, stride=2, padding=0), # 256, 16
            norm_layer(input_nc // 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(input_nc // 4, input_nc // 2, kernel_size=4, stride=2, padding=1), # 512, 32
            norm_layer(input_nc // 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(input_nc // 2, input_nc // 2, kernel_size=8, stride=2, padding=3), # 1024, 64
            norm_layer(input_nc // 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(input_nc // 2, input_nc, kernel_size=16, stride=2, padding=7), 
            norm_layer(input_nc),
            nn.Tanh(),
            View(batch_size, input_nc, 128) if not trp else View(batch_size, 128, -1),
        ]
        if not trp:
            seq = encoder + decoder
        else:
            print("TRP!")
            seq = [Transpose(1,2)] + encoder + decoder + [Transpose(2, 1)]
        self.model = nn.Sequential(*seq)
            
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

class XModel(nn.Module):
    def __init__(self, input_nc, output_nc, batch_size=1, hidden=256, gpu_ids=[], norm_layer=nn.BatchNorm2d):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        super(XModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            YModel(input_nc, input_nc // 2, k=16, s=2, p=7, kb=(31, 15), sb=(1, 1), pb=(15, 7), batch_size=batch_size), # 2, 512, 64
            YModel(input_nc // 2, input_nc // 4, k=16, s=2, p=7, kb=(15, 7), sb=(1, 1), pb=(7, 3), batch_size=batch_size), # 2, 256, 32
            YModel(input_nc // 4, input_nc // 16, k=8, s=2, p=3, kb=(7, 3), sb=(1, 1), pb=(3, 1), batch_size=batch_size), # 2, 64, 16
            View(batch_size, -1),
            nn.Linear(2048, 256),
        ]

        decoder = [
            nn.Linear(256, 2048),
            View(batch_size, 2, 64, 16),
            YModel(input_nc // 16, input_nc // 4, k=8, s=2, p=3, kb=(7, 3), sb=(1, 1), pb=(3, 1), 
                       batch_size=batch_size, up_sampling=True), # 2, 256, 32
            YModel(input_nc // 4, input_nc // 2, k=16, s=2, p=7, kb=(15, 7), sb=(1, 1), pb=(7, 3), 
                       batch_size=batch_size, up_sampling=True), # 2, 256, 32
            YModel(input_nc // 2, input_nc, k=16, s=2, p=7, kb=(31, 15), sb=(1, 1), pb=(15, 7), 
                       batch_size=batch_size, up_sampling=True), # 2, 256, 32
        ]

        self.model = nn.Sequential(*(encoder + decoder))

    def forward(self, input):
        return self.model(input)

class YModel(nn.Module):
    def __init__(self, input_nc, output_nc, k, s, p, kb, sb, pb, batch_size=1, up_sampling=False, gpu_ids=[], norm_layer=nn.BatchNorm2d):
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.up_sampling = up_sampling
        cnn = nn.Conv1d if not up_sampling else nn.ConvTranspose1d
        cnn2d = nn.Conv2d if not up_sampling else nn.ConvTranspose2d
        relu = nn.LeakyReLU if not up_sampling else nn.ReLU
        relu_args = [0.2, True] if not up_sampling else [True]
        super(YModel, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.view_1 = View(self.batch_size, 2, input_nc, -1)
        cnn_real = cnn(input_nc, output_nc, kernel_size=k, stride=s, padding=p)
        cnn_imag = cnn(input_nc, output_nc, kernel_size=k, stride=s, padding=p)
        norm_1r = norm_layer(output_nc)
        norm_1i = norm_layer(output_nc)
        relu_1 = relu(*relu_args)
        self.view_2 = View(self.batch_size, 2, output_nc, -1)

        self.real_net = nn.Sequential(*[cnn_real, norm_1r, relu_1])
        self.imag_net = nn.Sequential(*[cnn_imag, norm_1i, relu_1])
       
        #cnn_comb = cnn(output_nc * 2, output_nc * 2, kernel_size=kb, stride=sb, padding=pb) 
        cnn_comb = cnn2d(2, 2, kernel_size=kb, stride=sb, padding=pb) 
        norm_comb = norm_layer(2)
        relu_2 = relu(*relu_args)
        self.comb_net = nn.Sequential(*[cnn_comb, norm_comb, relu_2])

    def forward(self, input):
        #re = self.view_1(input)
        if not self.up_sampling:
            real, imag = self.real_net(input[:, 0, ...]), self.imag_net(input[:, 1, ...])
            out = self.comb_net(torch.cat([real.unsqueeze(1), imag.unsqueeze(1)], dim=1))
        else:
            comb = self.comb_net(input)
            out= torch.cat([self.real_net(comb[:, 0, ...]).unsqueeze(1), self.imag_net(comb[:, 1, ...]).unsqueeze(1)], dim=1)
        return out



