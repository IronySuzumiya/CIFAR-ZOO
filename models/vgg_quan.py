# -*-coding:utf-8-*-
import torch.nn as nn
from layer_quan import QConv2d, QLinear

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',
          512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512,
          512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, features, config):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = QLinear(512, config.num_classes,
                            wl_input = config.quantization.wl_activate, wl_activate=config.quantization.wl_activate,
                            wl_error = config.quantization.wl_error, wl_weight= config.quantization.wl_weight,
                            onoffratio = config.quantization.onoffratio, cellBit = config.quantization.cellBit,
                            subArray = config.quantization.subArray, ADCprecision = config.quantization.ADCprecision,
                            vari = config.quantization.vari, t = config.quantization.t, v = config.quantization.v,
                            detect = config.quantization.detect, target = config.quantization.target, name = 'FC_')
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, QLinear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, config, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QConv2d(in_channels, v, kernel_size=3, padding=1,
                             wl_input = config.quantization.wl_activate, wl_activate=config.quantization.wl_activate,
                             wl_error = config.quantization.wl_error, wl_weight= config.quantization.wl_weight,
                             onoffratio = config.quantization.onoffratio, cellBit = config.quantization.cellBit,
                             subArray = config.quantization.subArray, ADCprecision = config.quantization.ADCprecision,
                             vari = config.quantization.vari, t = config.quantization.t, v = config.quantization.v,
                             detect = config.quantization.detect, target = config.quantization.target, name = 'Conv'+str(i)+'_')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(config):
    return VGG(make_layers(cfg['A'], config, batch_norm=True), config)


def vgg13(config):
    return VGG(make_layers(cfg['B'], config, batch_norm=True), config)


def vgg16(config):
    return VGG(make_layers(cfg['D'], config, batch_norm=True), config)


def vgg19(config):
    return VGG(make_layers(cfg['E'], config, batch_norm=True), config)
