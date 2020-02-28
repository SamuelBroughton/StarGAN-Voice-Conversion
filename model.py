import torch
import torch.nn as nn
import numpy as np
import argparse
from data_loader import get_loader, to_categorical


class DownSampleBlock(nn.Module):
    """Down-sampling layers."""
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(DownSampleBlock, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=dim_in,
                                    out_channels=dim_out,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias)

        self.batch_norm = nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class UpSampleBlock(nn.Module):
    """Up-sampling layers."""
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bias):
        super(UpSampleBlock, self).__init__()

        self.conv_layer = nn.ConvTranspose2d(in_channels=dim_in,
                                             out_channels=dim_out,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             bias=bias)

        self.batch_norm = nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x


class Generator(nn.Module):
    """Generator network."""
    def __init__(self):
        super(Generator, self).__init__()

        # Down-sampling layers.
        self.down_sample_1 = DownSampleBlock(dim_in=1,
                                             dim_out=32,
                                             kernel_size=(3, 9),
                                             stride=(1, 1),
                                             padding=(1, 4),
                                             bias=False)

        self.down_sample_2 = DownSampleBlock(dim_in=32,
                                             dim_out=64,
                                             kernel_size=(4, 8),
                                             stride=(2, 2),
                                             padding=(1, 3),
                                             bias=False)

        self.down_sample_3 = DownSampleBlock(dim_in=64,
                                             dim_out=128,
                                             kernel_size=(4, 8),
                                             stride=(2, 2),
                                             padding=(1, 3),
                                             bias=False)

        self.down_sample_4 = DownSampleBlock(dim_in=128,
                                             dim_out=64,
                                             kernel_size=(3, 5),
                                             stride=(1, 1),
                                             padding=(1, 2),
                                             bias=False)

        self.down_sample_5 = DownSampleBlock(dim_in=64,
                                             dim_out=5,
                                             kernel_size=(9, 5),
                                             stride=(9, 1),
                                             padding=(1, 2),
                                             bias=False)

        # Up-sampling layers.
        self.up_sample_1 = UpSampleBlock(dim_in=9,
                                         dim_out=64,
                                         kernel_size=(9, 5),
                                         stride=(9, 1),
                                         padding=(0, 2),
                                         bias=False)

        self.up_sample_2 = UpSampleBlock(dim_in=68,
                                         dim_out=128,
                                         kernel_size=(3, 5),
                                         stride=(1, 1),
                                         padding=(1, 2),
                                         bias=False)

        self.up_sample_3 = UpSampleBlock(dim_in=132,
                                         dim_out=64,
                                         kernel_size=(4, 8),
                                         stride=(2, 2),
                                         padding=(1, 3),
                                         bias=False)

        self.up_sample_4 = UpSampleBlock(dim_in=68,
                                         dim_out=32,
                                         kernel_size=(4, 8),
                                         stride=(2, 2),
                                         padding=(1, 3),
                                         bias=False)

        # Deconv
        self.deconv_layer = nn.ConvTranspose2d(in_channels=36,
                                               out_channels=1,
                                               kernel_size=(3, 9),
                                               stride=(1, 1),
                                               padding=(1, 4),
                                               bias=False)

    def forward(self, x, c):
        # Replicate spatially..
        c = c.view(c.size(0), c.size(1), 1, 1)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)
        x = self.down_sample_5(x)

        # concat domain specific information
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.up_sample_1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up_sample_2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up_sample_3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.up_sample_4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.deconv_layer(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_speakers):
        super(Discriminator, self).__init__()
        i = num_speakers + 1

        # Downsample
        self.down_sample_1 = DownSampleBlock(dim_in=i,
                                             dim_out=32,
                                             kernel_size=(3, 9),
                                             stride=(1, 1),
                                             padding=(1, 4),
                                             bias=False)

        self.down_sample_2 = DownSampleBlock(dim_in=36,
                                             dim_out=32,
                                             kernel_size=(3, 8),
                                             stride=(1, 1),
                                             padding=(1, 3),
                                             bias=False)

        self.down_sample_3 = DownSampleBlock(dim_in=36,
                                             dim_out=32,
                                             kernel_size=(3, 8),
                                             stride=(1, 1),
                                             padding=(1, 3),
                                             bias=False)

        self.down_sample_4 = DownSampleBlock(dim_in=36,
                                             dim_out=32,
                                             kernel_size=(3, 6),
                                             stride=(1, 1),
                                             padding=(1, 2),
                                             bias=False)

        self.conv_layer = nn.Conv2d(in_channels=36,
                                    out_channels=1,
                                    kernel_size=(36, 5),
                                    stride=(36, 1),
                                    padding=(0, 2),
                                    bias=False)

        self.pool = nn.AvgPool2d(kernel_size=(1, 64))

    def forward(self, x, c):
        # Replicate spatially..
        c = c.view(c.size(0), c.size(1), 1, 1)

        # concat domain specific information
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.down_sample_1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.down_sample_2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.down_sample_3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.down_sample_4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv_layer(x)

        x = self.pool(x)
        x = torch.squeeze(x)
        x = torch.sigmoid(x)

        return x


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()

        # Down sample.
        self.down_sample_1 = DownSampleBlock(dim_in=1,
                                             dim_out=8,
                                             kernel_size=(4, 4),
                                             stride=(2, 2),
                                             padding=(5, 1),
                                             bias=False)

        self.down_sample_2 = DownSampleBlock(dim_in=8,
                                             dim_out=16,
                                             kernel_size=(4, 4),
                                             stride=(2, 2),
                                             padding=(1, 1),
                                             bias=False)

        self.down_sample_3 = DownSampleBlock(dim_in=16,
                                             dim_out=32,
                                             kernel_size=(4, 4),
                                             stride=(2, 2),
                                             padding=(0, 1),
                                             bias=False)

        self.down_sample_4 = DownSampleBlock(dim_in=32,
                                             dim_out=16,
                                             kernel_size=(3, 4),
                                             stride=(1, 2),
                                             padding=(1, 1),
                                             bias=False)

        self.conv_layer = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.pool = nn.AvgPool2d((1, 16))
        self.softmax = nn.Softmax()

    def forward(self, x):
        # slice
        x = x[:, :, 0:8, :]

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)

        x = self.conv_layer(x)
        x = self.pool(x)
        x = self.softmax(x)

        x = x.view(x.size(0), x.size(1))

        return x


# Just for testing shapes of architecture.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test G and D architecture')

    train_dir_default = '../data/VCTK-Data/mc/train'
    speaker_default = 'p229'

    # Data config.
    parser.add_argument('--train_dir', type=str, default=train_dir_default, help='Train dir path')
    parser.add_argument('--speakers', type=str, nargs='+', required=True, help='Speaker dir names')
    num_speakers = 4

    argv = parser.parse_args()
    train_dir = argv.train_dir
    speakers_using = argv.speaker

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    generator = Generator().to(device)
    discriminator = Discriminator(num_speakers=num_speakers).to(device)
    classifier = DomainClassifier().to(device)

    # Load data
    train_loader = get_loader(speakers_using, train_dir, 8, 'train', num_workers=1)
    data_iter = iter(train_loader)

    mc_real, spk_label_org, spk_c_org = next(data_iter)
    mc_real.unsqueeze_(1)  # (B, D, T) -> (B, 1, D, T) for conv2d

    spk_c = np.random.randint(0, num_speakers, size=mc_real.size(0))
    spk_c_cat = to_categorical(spk_c, num_speakers)
    spk_label_trg = torch.LongTensor(spk_c)
    spk_c_trg = torch.FloatTensor(spk_c_cat)

    mc_real = mc_real.to(device)              # Input mc.
    spk_label_org = spk_label_org.to(device)  # Original spk labels.
    spk_c_org = spk_c_org.to(device)          # Original spk acc conditioning.
    spk_label_trg = spk_label_trg.to(device)  # Target spk labels for classification loss for G.
    spk_c_trg = spk_c_trg.to(device)          # Target spk conditioning.

    print('Testing Domain Classifier')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    cls_real = classifier(mc_real)
    print(f'Shape out: {cls_real.shape}')
    print('------------------------')

    print('Testing Discriminator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    dis_real = discriminator(mc_real, spk_c_org)
    print(f'Shape out: {dis_real.shape}')
    print('------------------------')

    print('Testing Generator')
    print('-------------------------')
    print(f'Shape in: {mc_real.shape}')
    mc_fake = generator(mc_real, spk_c_trg)
    print(f'Shape out: {mc_fake.shape}')
    print('------------------------')
