from torch import nn


class Discriminator_quick_fully(nn.Module):
    def __init__(self):
        super(Discriminator_quick_fully, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(True)
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2)
        self.BatchNorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2)
        self.BatchNorm4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3136, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.LeakyReLU(out)
        out = self.conv2(out)
        out = self.BatchNorm2(out)
        out = self.LeakyReLU(out)
        out = self.conv3(out)
        out = self.BatchNorm3(out)
        out = self.LeakyReLU(out)

        out = self.conv4(out)
        out = self.BatchNorm4(out)
        out = self.LeakyReLU(out)
        out = out.view(-1, 64 * 7 * 7)
        out = self.fc1(out)
        out = self.sigmoid(out)
        return out


class Encoder_residual(nn.Module):
    def __init__(self):
        super(Encoder_residual, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.MaxPool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv1_0 = nn.Conv2d(3, 32, 1)
        self.conv1_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.BatchNorm1 = nn.BatchNorm2d(32)

        self.conv2_0 = nn.Conv2d(32, 64, 1)
        self.conv2_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_3 = nn.Conv2d(64, 64, 5, padding=2)
        self.BatchNorm2 = nn.BatchNorm2d(64)

        self.conv3_0 = nn.Conv2d(64, 128, 1)
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(128)

    def forward(self, x):

        resi = self.conv1_0(x)
        resi = self.BatchNorm1(resi)
        resi = self.ReLU(resi)

        out = self.conv1_1(resi)
        out = self.BatchNorm1(out)
        out = self.ReLU(out)
        out = self.conv1_2(out)
        out = self.BatchNorm1(out)

        resi2 = resi + out

        resi2 = self.conv2_0(resi2)
        resi2 = self.BatchNorm2(resi2)
        resi2 = self.ReLU(resi2)
        resi2, indices1 = self.MaxPool(resi2)
        out = self.conv2_1(resi2)
        out = self.BatchNorm2(out)
        out = self.ReLU(out)
        out = self.conv2_2(out)
        out = self.BatchNorm2(out)
        resi3 = resi2 + out

        resi3 = self.conv3_0(resi3)
        resi3 = self.BatchNorm3(resi3)
        resi3 = self.ReLU(resi3)
        resi3, indices2 = self.MaxPool(resi3)
        out = self.conv3_1(resi3)
        out = self.BatchNorm3(out)
        out = self.ReLU(out)
        out = self.conv3_2(out)
        out = self.BatchNorm3(out)
        resi_last = resi3 + out
        size = resi_last.size()
        resi_last, indices3 = self.MaxPool(resi_last)

        return resi_last, (indices1, indices2, indices3), size


class Encoder_residual4(nn.Module):
    def __init__(self):
        super(Encoder_residual4, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(True)
        self.MaxPool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv1_0 = nn.Conv2d(3, 32, 1)
        self.conv1_1 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv1_3 = nn.Conv2d(32, 32, 5, padding=2)
        self.BatchNorm1 = nn.BatchNorm2d(32)

        self.conv2_0 = nn.Conv2d(32, 64, 1)
        self.conv2_1 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv2_3 = nn.Conv2d(64, 64, 5, padding=2)
        self.BatchNorm2 = nn.BatchNorm2d(64)

        self.conv3_0 = nn.Conv2d(64, 128, 1)
        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(128)

        self.conv4_0 = nn.Conv2d(128, 256, 3)
        self.conv4_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.BatchNorm4 = nn.BatchNorm2d(256)

    def forward(self, x):

        resi = self.conv1_0(x)
        resi = self.BatchNorm1(resi)
        resi = self.LeakyReLU(resi)

        out = self.conv1_1(resi)
        out = self.BatchNorm1(out)
        out = self.LeakyReLU(out)
        out = self.conv1_2(out)
        out = self.BatchNorm1(out)

        resi2 = resi + out
        resi2 = self.conv2_0(resi2)
        resi2 = self.BatchNorm2(resi2)
        resi2 = self.LeakyReLU(resi2)
        resi2, indices1 = self.MaxPool(resi2)

        out = self.conv2_1(resi2)
        out = self.BatchNorm2(out)
        out = self.LeakyReLU(out)
        out = self.conv2_2(out)
        out = self.BatchNorm2(out)

        resi3 = resi2 + out
        resi3 = self.conv3_0(resi3)
        resi3 = self.BatchNorm3(resi3)
        resi3 = self.LeakyReLU(resi3)
        resi3, indices2 = self.MaxPool(resi3)

        out = self.conv3_1(resi3)
        out = self.BatchNorm3(out)
        out = self.LeakyReLU(out)
        out = self.conv3_2(out)
        out = self.BatchNorm3(out)

        resi4 = resi3 + out
        resi4, indices3 = self.MaxPool(resi4)
        size = resi4.size()
        resi4 = self.conv4_0(resi4)
        resi4 = self.BatchNorm4(resi4)
        resi4 = self.LeakyReLU(resi4)

        out = self.conv4_1(resi4)
        out = self.BatchNorm4(out)
        out = self.LeakyReLU(out)
        out = self.conv4_2(out)
        out = self.BatchNorm4(out)
        resi_last = resi4 + out

        return resi_last, (indices1, indices2, indices3), size


class Decoder_residual_real(nn.Module):
    def __init__(self):
        super(Decoder_residual_real, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.MaxUnPool = nn.MaxUnpool2d(2, stride=2)

        self.deconv1_0 = nn.ConvTranspose2d(128, 64, 6)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(64)

        self.deconv2_0 = nn.ConvTranspose2d(64, 32, 7)
        self.deconv2_1 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.deconv2_2 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.BatchNorm2 = nn.BatchNorm2d(32)

        self.deconv3_0 = nn.ConvTranspose2d(32, 3, 7)
        self.deconv3_1 = nn.ConvTranspose2d(3, 3, 5, padding=2)
        self.deconv3_2 = nn.ConvTranspose2d(3, 3, 5, padding=2)
        self.BatchNorm3 = nn.BatchNorm2d(3)

    def forward(self, x, indices, size):
        out = self.MaxUnPool(x, indices[-1], output_size=size)

        resi = self.deconv1_0(out)
        resi = self.BatchNorm1(resi)
        resi = self.ReLU(resi)

        out = self.deconv1_1(resi)
        out = self.BatchNorm1(out)
        out = self.ReLU(out)
        out = self.deconv1_2(out)
        out = self.BatchNorm1(out)
        resi2 = resi + out

        resi2 = self.deconv2_0(resi2)
        resi2 = self.BatchNorm2(resi2)
        resi2 = self.ReLU(resi2)
        out = self.deconv2_1(resi2)
        out = self.BatchNorm2(out)
        out = self.ReLU(out)
        out = self.deconv2_2(out)
        out = self.BatchNorm2(out)
        resi3 = resi2 + out

        resi3 = self.deconv3_0(resi3)

        out = self.deconv3_1(resi3)
        out = self.BatchNorm3(out)
        out = self.ReLU(out)
        out = self.deconv3_2(out)
        out = self.BatchNorm3(out)
        resi_last = resi3 + out

        return resi_last


class Decoder_residual_real4(nn.Module):
    def __init__(self):
        super(Decoder_residual_real4, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(True)
        self.MaxUnPool = nn.MaxUnpool2d(2, stride=2)

        self.deconv0_0 = nn.ConvTranspose2d(256, 128, 3)
        self.deconv0_1 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.deconv0_2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.BatchNorm0 = nn.BatchNorm2d(128)

        self.deconv1_0 = nn.ConvTranspose2d(128, 64, 6)
        self.deconv1_1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.deconv1_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.BatchNorm1 = nn.BatchNorm2d(64)

        self.deconv2_0 = nn.ConvTranspose2d(64, 32, 7)
        self.deconv2_1 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.deconv2_2 = nn.ConvTranspose2d(32, 32, 5, padding=2)
        self.BatchNorm2 = nn.BatchNorm2d(32)

        self.deconv3_0 = nn.ConvTranspose2d(32, 3, 7)
        self.deconv3_1 = nn.ConvTranspose2d(3, 3, 5, padding=2)
        self.deconv3_2 = nn.ConvTranspose2d(3, 3, 5, padding=2)
        self.BatchNorm3 = nn.BatchNorm2d(3)

    def forward(self, x, indices, size):

        resi0 = self.deconv0_0(x)
        resi0 = self.BatchNorm0(resi0)
        resi0 = self.LeakyReLU(resi0)

        out = self.deconv0_1(resi0)
        out = self.BatchNorm0(out)
        out = self.LeakyReLU(out)
        out = self.deconv0_2(out)
        out = self.BatchNorm0(out)

        resi_tmp = resi0 + out
        out = self.MaxUnPool(resi_tmp, indices[-1])

        resi = self.deconv1_0(out)
        resi = self.BatchNorm1(resi)
        resi = self.LeakyReLU(resi)

        out = self.deconv1_1(resi)
        out = self.BatchNorm1(out)
        out = self.LeakyReLU(out)
        out = self.deconv1_2(out)
        out = self.BatchNorm1(out)

        resi2 = resi + out
        resi2 = self.deconv2_0(resi2)
        resi2 = self.BatchNorm2(resi2)
        resi2 = self.LeakyReLU(resi2)

        out = self.deconv2_1(resi2)
        out = self.BatchNorm2(out)
        out = self.LeakyReLU(out)
        out = self.deconv2_2(out)
        out = self.BatchNorm2(out)
        resi3 = resi2 + out
        resi3 = self.deconv3_0(resi3)
        out = self.deconv3_1(resi3)
        out = self.BatchNorm3(out)
        out = self.LeakyReLU(out)
        out = self.deconv3_2(out)
        out = self.BatchNorm3(out)
        resi_last = resi3 + out

        return resi_last
