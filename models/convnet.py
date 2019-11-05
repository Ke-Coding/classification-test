from collections import OrderedDict
import torch as tc
import torch.nn as nn
import torchsummary
import os


__all__ = ['ConvNet']


class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes,
                 channels=[32, 48, 48, 64],
                 strides=[1, 2, 1, 2],
                 dropout_p=None
                 ):
        r"""
        :param input_channels(int): num of input channels
        :param num_classes(int): num of classes
        :param channels(List[int]): hidden layers' channels
        :param strides(List[int]): hidden layers' strides
            - len(input_channels): the number of the hidden layers
        """
        super(ConvNet, self).__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.strides = strides
        self.using_dropout = dropout_p is not None
        self.num_layers = len(self.channels)
        assert len(self.strides) == self.num_layers
        
        self.bn0 = nn.BatchNorm2d(input_channels)  # this layer has a huge influence on acc
        self.backbone = self._make_backbone(
            [input_channels] + channels[:-1],
            channels
        )
        if self.using_dropout:
            self.dropout_p = dropout_p
            self.dropout = nn.Dropout(p=dropout_p)
        last_ch = 128
        self.conv_last = nn.Sequential(
            nn.Conv2d(channels[-1], last_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(last_ch, num_classes, bias=True)
        
        self._init_params()
    
    def forward(self, x):
        x = self.bn0(x)
        features = self.backbone(x)
        features = self.conv_last(features)
        if self.using_dropout:
            features = self.dropout(features)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)
        return logits
    
    def _make_backbone(self, in_channels, out_channels):
        backbone = OrderedDict()
        name_prefix = 'conv2d'
        for i in range(self.num_layers):
            name = name_prefix + "_%d" % i
            backbone[name] = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels[i], kernel_size=3,
                          stride=self.strides[i], padding=1, bias=False),
                nn.BatchNorm2d(out_channels[i]),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(backbone)
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def save(self, final_acc_str):
        save_path = 'saves/conv_saves/' + final_acc_str
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = 'channels[%s]_%s.pkl' % ('-'.join([str(e) for e in self.channels]),
                                             'dropout%g' % self.dropout_p if self.using_dropout else 'no_dropout')
        file_path = os.path.join(save_path, file_name)
        tc.save(self.state_dict(), file_path)
        print('model saved successfully at %s' % file_path)
    
    def load(self, final_acc_str):
        save_path = 'saves/conv_saves/' + final_acc_str
        assert os.path.exists(save_path)
        file_name = 'channels[%s]_%s.pkl' % ('-'.join([str(e) for e in self.channels]),
                                             'dropout%g' % self.dropout_p if self.using_dropout else 'no_dropout')
        file_path = os.path.join(save_path, file_name)
        self.load_state_dict(tc.load(file_path))
        print('model loaded successfully at %s' % file_path)


if __name__ == '__main__':
    net: ConvNet = ConvNet(input_channels=1, num_classes=10, dropout_p=0.2)
    torchsummary.summary(net, (1, 28, 28))
