import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import itertools, functools
from utils import Pool, GANLoss, View
from collections import OrderedDict

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

    def __init__(self, lr, lamb_A, lamb_B, lamb_I, output_dir, shape,
                    batch_size, pool_size, gpu_id=None, is_training=True):

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
        self.gpu_id = gpu_id

        self.input_A = self.tensor(batch_size, 2, self.shape[0], self.shape[1])
        self.input_B = self.tensor(batch_size, 2, self.shape[0], self.shape[1])

        self.gen_A = self.build_G(2, 2, 8, self.gpu_id)
        self.gen_B = self.build_G(2, 2, 8, self.gpu_id)

        if self.is_training:
            self.dis_A = self.build_D(2, 4, self.gpu_id)
            self.dis_B = self.build_D(2, 4, self.gpu_id)

            # fake pool
            self.fplA = Pool(self.pool_size)
            self.fplB = Pool(self.pool_size)
            self.GAN_loss = GANLoss(tensor=self.tensor)
            self.cycle_loss = torch.nn.L1Loss()
            self.identity_loss = torch.nn.L1Loss() # if the input is from its target domain, it should output an identity one.

            self.optim_G = torch.optim.Adam(
                    itertools.chain(self.gen_A.parameters(), self.gen_B.parameters()),
                    lr=self.lr,
                )
            """
            self.optim_D = torch.optim.Adam(
                    lr=self.lr,
                    itertools.chain(self.dis_A.parameters(), self.dis_B.parameters()),
                )
            """
            self.optim_D_A = torch.optim.Adam(self.dis_A.parameters(), lr=self.lr)
            self.optim_D_B = torch.optim.Adam(self.dis_B.parameters(), lr=self.lr)
            self.optims = [self.optim_G, self.optim_D_A, self.optim_D_B]

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
        G = UNetGenerator(input_nc, output_nc, num_downs, gpu_ids=gpu_ids)
        if len(gpu_ids) > 0:
            G.cuda(gpu_ids[0])
        G.apply(weights_init)
        return G


    def set_input(self, inputs):
        self.input_A.copy_(inputs["A"])
        self.input_B.copy_(inputs["B"])

    def optimize(self):
        self.optim_G.zero_grad()
        self.optim_D_A.zero_grad()
        self.optim_D_B.zero_grad()

        self.forward()
        self.backward_G()
        self.backward_D()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def _backward_D(self, dis, real, fake):
        pred_real = dis(real)
        D_loss_real = self.GAN_loss(pred_real, True)
        pred_fake = dis(fake)
        D_loss_fake = self.GAN_loss(pred_fake, False)
        D_loss = (D_loss_real + D_loss_fake) * 0.5
        D_loss.backward()
        return D_loss

    def backward_D(self, step=True):
        fake_A = self.fplA.draw(self.fake_A)
        D_loss_A = self._backward_D(self.dis_A, self.real_A, fake_A)
        if step: self.optim_D_A.step()

        fake_B = self.fplB.draw(self.fake_B)
        D_loss_B = self._backward_D(self.dis_B, self.real_B, fake_B)
        if step: self.optim_D_B.step()

        self.D_loss_A = D_loss_A.data[0]
        self.D_loss_B = D_loss_B.data[0]


    def backward_G(self, step=True):

        # discriminator A
        fake_A = self.gen_A(self.real_B)
        pred_A = self.dis_A(fake_A)
        G_loss_A = self.GAN_loss(pred_A, True) # fool dis_A to think it's true

        # discriminator B
        fake_B = self.gen_B(self.real_A)
        pred_B = self.dis_B(fake_B)
        G_loss_B = self.GAN_loss(pred_B, True) # fool dis_B to think it's true


        # cycle loss
        # caculate real/imaginary part respectively
        rec_A = self.gen_A(fake_B)
        cyc_r_loss_A = self.cycle_loss(rec_A[:, :, :, 0], self.real_A[:, :, :, 0]) * self.lamb_A
        cyc_i_loss_A = self.cycle_loss(rec_A[:, :, :, 1], self.real_A[:, :, :, 1]) * self.lamb_A

        rec_B = self.gen_B(fake_A)
        cyc_r_loss_B = self.cycle_loss(rec_B[:, :, :, 0], self.real_B[:, :, :, 0]) * self.lamb_B
        cyc_i_loss_B = self.cycle_loss(rec_B[:, :, :, 1], self.real_B[:, :, :, 1]) * self.lamb_B


        # calculate identity loss
        if self.lamb_I > 0.0:
            idt_A = self.gen_A(self.real_A)
            idt_B = self.gen_B(self.real_B)
            idt_loss_A = self.identity_loss(idt_A, self.real_A) * self.lamb_A * self.lamb_I
            idt_loss_B = self.identity_loss(idt_A, self.real_B) * self.lamb_B * self.lamb_I

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data

        self.idt_loss_A = idt_loss_A.data[0] if self.lamb_I > 0.0 else 0.
        self.idt_loss_B = idt_loss_B.data[0] if self.lamb_I > 0.0 else 0.

        G_loss = G_loss_A + G_loss_B +  \
                 cyc_r_loss_A + cyc_i_loss_A + \
                 cyc_r_loss_B + cyc_i_loss_B + \
                 idt_loss_A + idt_loss_B
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

    def report_errors(self):
        report = OrderedDict([("D_A", self.D_loss_A), ("D_B", self.D_loss_B),
                    ("Cyc_r_A", self.cyc_r_loss_A), ("Cyc_r_B", self.cyc_r_loss_B),
                    ("Cyc_i_A", self.cyc_i_loss_A), ("Cyc_i_B", self.cyc_i_loss_B)])
        if self.lamb_I > 0.0:
            report["idt_A"] = self.idt_loss_A
            report["idt_B"] = self.idt_loss_B
        return report

    def get_samples(self, n_sample):
        fake_As = self.fplA.get_samples(n_sample)
        fake_Bs = self.fplB.get_samples(n_sample)
        return fake_As, fake_Bs


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
                View(-1, 217),
                nn.Linear(217, 1)]
        self.model = nn.Sequential(*seq)

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
                submodule=None, pos=None, norm_layer=nn.BatchNorm2d):
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
                                        stride=stride, padding=padding, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif pos == "innermost":
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=padding, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # inner_nc * 2: it takes as input cat([x, submodule(x)])
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=padding, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)

