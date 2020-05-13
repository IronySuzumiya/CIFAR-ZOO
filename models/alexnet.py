# -*-coding:utf-8-*-
import torch.nn as nn

from pytorx.layer import crxb_Conv2d, crxb_Linear


__all__ = ['alexnet', 'alexnet_pytorx']


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet(num_classes):
    return AlexNet(num_classes=num_classes)


class AlexNetPytorx(nn.Module):
    def __init__(self, config):
        super(AlexNetPytorx, self).__init__()
        self.conv1 = crxb_Conv2d(3, 64, kernel_size=11, stride=4, padding=5, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
        self.conv2 = crxb_Conv2d(64, 192, kernel_size=5, padding=2, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
        self.conv3 = crxb_Conv2d(192, 384, kernel_size=3, padding=1, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
        self.conv4 = crxb_Conv2d(384, 256, kernel_size=3, padding=1, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
        self.conv5 = crxb_Conv2d(256, 256, kernel_size=3, padding=1, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = crxb_Linear(50, config.num_classes, crxb_size=config.crxb_size, scaler_dw=config.scaler_dw,
                                gwire=config.gwire, gload=config.gload, gmax=config.gmax, gmin=config.gmin, vdd=config.vdd, freq=config.freq, temp=config.temp,
                                enable_SAF=config.enable_SAF, enable_ec_SAF=config.enable_ec_SAF,
                                enable_noise=config.enable_noise, ir_drop=config.ir_drop)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def alexnet_pytorx(config):
    return AlexNetPytorx(config)