import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


# GTSRB-CNN-1
class GtsrbCNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.color_map = nn.Conv2d(3, 3, (1, 1), stride=(1, 1), padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14336, 1024, bias=True), nn.ReLU(), nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        x = self.color_map(x)
        branch1 = self.module1(x)
        branch2 = self.module2(branch1)
        branch3 = self.module3(branch2)

        branch1 = branch1.reshape(1, -1)
        branch2 = branch2.reshape(1, -1)
        branch3 = branch3.reshape(1, -1)

        concat = torch.cat([branch1, branch2, branch3], 1)

        out = self.fc1(concat)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


# GTSRB-CNN-2/Lisa-CNN-2
class Net(nn.Module):
    def __init__(self, class_n):
        super(Net, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 100, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(250)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(250 * 2 * 2, 350)
        self.fc2 = nn.Linear(350, class_n)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)
class LisaCNN(nn.Module):

    def __init__(self, n_class):

        super().__init__()                                               #32323
        self.conv1 = nn.Conv2d(3, 64, (8, 8), stride=(2, 2), padding=3)  #161664
        self.conv2 = nn.Conv2d(64, 128, (6, 6), stride=(2, 2), padding=0) #66128
        self.conv3 = nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=0) #22128
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):

        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


###GAN###
class Generator(nn.Module):
    def __init__(self, gen_input_nc, image_nc, target="Auto"):
        super(Generator, self).__init__()

        encoder_lis = [
            # input:4*128*128
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*126*126
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*62*62
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*30*30
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64*14*14
        ]

        bottle_neck_lis = [
            ResnetBlock(64),
            ResnetBlock(64),
            ResnetBlock(64),
            ResnetBlock(64),
        ]

        if target == "HighResolution":
            decoder_lis = [
                nn.ConvTranspose2d(
                    64, 32, kernel_size=3, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
                # state size. 32 x 29 x 29
                nn.ConvTranspose2d(
                    32, 16, kernel_size=4, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 60 x 60
                nn.ConvTranspose2d(
                    16, 8, kernel_size=5, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 123 x 123
                nn.ConvTranspose2d(
                    8, image_nc, kernel_size=6, stride=1, padding=0, bias=False
                ),
                # nn.Tanh()
                # state size. 3 x 128 x 128
            ]
        else:
            decoder_lis = [
                nn.ConvTranspose2d(
                    64, 32, kernel_size=3, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
                # state size. 32 x 29 x 29
                nn.ConvTranspose2d(
                    32, 16, kernel_size=4, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 60 x 60
                nn.ConvTranspose2d(
                    16, 8, kernel_size=5, stride=2, padding=0, bias=False
                ),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 123 x 123
                nn.ConvTranspose2d(
                    8, 8, kernel_size=6, stride=1, padding=0, bias=False
                ),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # nn.Tanh()
                # state size. 8 x 128 x 128
            ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

        self.last_conv2d = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x_ = self.encoder(x)
        x_ = self.bottle_neck(x_)
        x_ = self.decoder(x_)  # 8*128*128

        orig_img = x[:, :3, :, :]  # N*3*128*128
        out = torch.cat([x_, x], dim=1)  # 12*128*128

        out = self.last_conv2d(out)
        out = self.Tanh(out)

        return out + orig_img

    def eval_forward(self, x):
        return torch.clamp(self.forward(x), min=0, max=1)


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type="reflect",
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        use_bias=False,
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
