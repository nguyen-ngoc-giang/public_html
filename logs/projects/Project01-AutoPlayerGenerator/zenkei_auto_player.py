import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

N_ANGLE = 16

dummy_input = (torch.rand(1,3,128, 128), torch.rand(1,3,64,128))

# Residual Block
class ResidualBlock(nn.Module):
    """ the unit is
    bn - relu - conv - bn - relu - dropout - conv - add
    """
    def __init__(self, in_channels, out_channels, stride=1, p=0.0):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False)

        self.drop_rate = p
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += residual
        return out

class _ConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2, p=0.0):
        super(_ConvLayer, self).__init__()

        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=stride)

        self.drop_rate = p
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.maxpool(x)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x

class PlayerNet(nn.Module):
    """PlayerNet

    Args:
        ch_fp: Number of channels to embed for floorplan images. (Default value is 0, which means no embedding)
        ch_pn: Number of channels to embed for panorama images. (Default value is 0, which means no embedding)
        n_xy: Number of ResNet units in each ResNet part for (x, y).
        n_angle: Number of ResNet units in each ResNet part for (angle).
        dropout_xy: Dropout probability for (x, y).
        dropout_angle: Dropout probability for (angle).
    """
    def __init__(
        self,
        ch_fp=0, ch_pn=0,
        n_xy=4, n_angle=4,
        dropout_xy=0.1, dropout_angle=0.1):

        super(PlayerNet, self).__init__()

        # RGB to embeddings with 10-dim vector
        self.conv_fp_ch = None
        self.conv_pn_ch = None
        if ch_fp > 0:
            self.conv_fp_ch = nn.Conv2d(
                3, ch_fp, kernel_size=1, padding=0, stride=1, bias=False)
        else:
            # for later use
            ch_fp = 3
        if ch_pn > 0:
            self.conv_pn_ch = nn.Conv2d(
                3, ch_pn, kernel_size=1, padding=0, stride=1, bias=False)
        else:
            # for later use
            ch_pn = 3

        # ch_fp, 128, 128
        self.conv_fp_1 = _ConvLayer(ch_fp, 16, 2)
        # 16, 64, 64
        self.conv_fp_2 = _ConvLayer(16, 32, 2)
        # 32, 32, 32
        self.conv_fp_3 = _ConvLayer(32, 64, 2)
        # 64, 16, 16
        self.conv_fp_4 = _ConvLayer(64, 64, 2)
        # 64, 8, 8

        # ch_pn, 64, 128
        self.conv_pn_1 = _ConvLayer(ch_pn, 32, 2)
        # 32, 32, 64
        self.conv_pn_2 = _ConvLayer(32, 64, 2)
        # 64, 16, 32
        self.conv_pn_3 = _ConvLayer(64, 64, 2)
        # 64, 8, 16
        self.conv_pn_4 = _ConvLayer(64, 64, (1, 2))
        # 64, 8, 8
        
        self.xy_res = self._res_block(128, 128, n_xy, p=dropout_xy)
        self.xy_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.xy_fc = nn.Linear(128, 2)

        # 256, 8, 8
        self.angle_res_1 = self._res_block(256, 256, n_angle, p=dropout_angle)
        # 256, 8, 8
        self.angle_conv_1 = _ConvLayer(256, 256, 2, p=dropout_angle)
        # 256, 4, 4
        self.angle_res_2 = self._res_block(256, 256, n_angle, p=dropout_angle)
        # 256, 4, 4
        self.angle_conv_2 = _ConvLayer(256, N_ANGLE, 2, p=dropout_angle)
        # N_ANGLE, 2, 2
        self.angle_res_3 = self._res_block(N_ANGLE, N_ANGLE, n_angle, p=dropout_angle)

        self.angle_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # N_ANGLE, 1, 1

    def _res_block(self, ch_in, ch_out, blocks, p=0.0):
        layers = []
        layers.append(ResidualBlock(ch_in, ch_out, p=p))
        for i in range(1, blocks):
            layers.append(ResidualBlock(ch_out, ch_out, p=p))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        fp, pn = inputs

        # RGB to embeddings
        if not self.conv_fp_ch is None:
            fp = self.conv_fp_ch(fp)
        if not self.conv_pn_ch is None:
            pn = self.conv_pn_ch(pn)

        fp = self.conv_fp_1(fp)
        fp = self.conv_fp_2(fp)
        fp = self.conv_fp_3(fp)
        fp = self.conv_fp_4(fp)

        pn = self.conv_pn_1(pn)
        pn = self.conv_pn_2(pn)
        pn = self.conv_pn_3(pn)
        pn = self.conv_pn_4(pn)

        x = torch.cat([fp, pn], 1)
        # 128, 8, 8

        x_xy = x
        # 128, 8, 8
        x_xy = self.xy_res(x_xy)
        x_xy_bypass = x_xy
        
        x_xy = F.relu(x_xy)
        x_xy = self.xy_avgpool(x_xy)
        # 128, 1, 1
        x_xy = x_xy.view(x_xy.size(0), -1)
        x_xy = self.xy_fc(x_xy)


        # angle
        x_angle = torch.cat([x, x_xy_bypass], 1)
        # 256, 8, 8
        x_angle = self.angle_res_1(x_angle)
        x_angle = self.angle_conv_1(x_angle)
        # 256, 4, 4
        x_angle = self.angle_res_2(x_angle)
        x_angle = self.angle_conv_2(x_angle)
        # N_ANGLE, 2, 2
        x_angle = self.angle_res_3(x_angle)

        x_angle = self.angle_avgpool(x_angle)
        # N_ANGLE, 1, 1
        x_angle = x_angle.view(x_angle.size(0), -1)

        return x_xy, x_angle


model = PlayerNet(n_xy=4, n_angle=6)

with SummaryWriter(comment='zenkei_auto_player') as w:
    w.add_graph(model, (dummy_input, ))