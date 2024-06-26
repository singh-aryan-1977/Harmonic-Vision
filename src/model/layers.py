import torch
import torch.nn as nn


class UpResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, cond_dim, sn, bias, w_init, first=False):
        super().__init__()
        self.skip_upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.skip_conv = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=1, sn=sn, bias=bias, w_init=w_init)

        self.bn1 = CondBatchNorm2d(cond_dim=cond_dim, num_features=in_ch)
        self.relu1 = nn.ReLU()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.bn2 = CondBatchNorm2d(cond_dim=cond_dim, num_features=out_ch)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)

    def forward(self, x, cond):
        x_skip = self.skip_upsample(x)
        x_skip = self.skip_conv(x_skip)

        y = self.bn1(x, cond)
        y = self.relu1(y)
        y = self.upsample(y)
        y = self.conv1(y)

        y = self.bn2(y, cond)
        y = self.relu2(y)
        y = self.conv2(y)

        y = y + x_skip
        return y


class DownResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sn, bias, w_init):
        super().__init__()
        self.skip_conv = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1,
            stride=1, sn=sn, bias=bias, w_init=w_init)
        self.skip_downsample = nn.AvgPool2d(kernel_size=ks, stride=2, padding=1)

        self.relu1 = nn.ReLU()
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.relu2 = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.downsample = nn.AvgPool2d(kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        x_skip = self.skip_conv(x)
        x_skip = self.skip_downsample(x_skip)

        y = self.relu1(x)
        y = self.conv1(y)

        y = self.relu2(y)
        y = self.conv2(y)
        y = self.downsample(y)
        y = y + x_skip
        return y


class ConstResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sn, bias, w_init):
        super().__init__()
        self.conv1 = Conv2dSN(
            in_channels=in_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)
        self.relu = nn.ReLU()
        self.conv2 = Conv2dSN(
            in_channels=out_ch, out_channels=out_ch, kernel_size=ks,
            stride=1, padding=1, sn=sn, bias=bias, w_init=w_init)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        return y + x


class SelfAttn(nn.Module):
    def __init__(self, in_dim, sn):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, sn=sn)
        self.key_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, sn=sn)
        self.value_conv = Conv2dSN(in_channels=in_dim, out_channels=in_dim, kernel_size=1, sn=sn)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class CondBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, cond_dim, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.bn_weight = nn.Linear(in_features=cond_dim, out_features=num_features)
            self.bn_bias = nn.Linear(in_features=cond_dim, out_features=num_features)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.bn_weight.weight.data.normal_(1, 0.02)
            self.bn_bias.weight.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.bn_weight(cats).view(shape)
            bias = self.bn_bias(cats).view(shape)
            out = out * weight + bias

        return out


class SpectralLayer(nn.Module):
    def __init__(self, layer, sn, w_init, *args, **kwargs):
        super().__init__()
        self.layer = layer
        if w_init is not None: w_init(self.layer.weight)
        self.layer = nn.utils.spectral_norm(self.layer) if sn else self.layer

    def forward(self, x):
        return self.layer(x)


class LinearSN(SpectralLayer):
    def __init__(self, sn, w_init=None, *args, **kwargs):
        layer = nn.Linear(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)


class Conv2dSN(SpectralLayer):
    def __init__(self, sn, w_init=None, *args, **kwargs):
        layer = nn.Conv2d(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)


class ConvTranspose2dSN(SpectralLayer):
    def __init__(self, *args, sn, w_init=None, **kwargs):
        layer = nn.ConvTranspose2d(*args, **kwargs)
        super().__init__(layer, sn, w_init, *args, **kwargs)
