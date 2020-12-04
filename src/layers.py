import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def xavier_init(module, gain=1., bias=0., distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0., std=1., bias=0.):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=not use_spectral_norm)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.gate_conv = spectral_norm(self.gate_conv)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class CARAFEPack(nn.Module):
    """ A unified package of CARAFE upsampler that contains:
    1) channel compressor 2) content encoder 3) CARAFE op
    Official implementation of ICCV 2019 paper
    CARAFE: Content-Aware ReAssembly of FEatures
    Please refer to https://arxiv.org/abs/1905.02188 for more details.
    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor
    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels,
                 output_channels,
                 scale_factor,
                 up_kernel=5,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64):
        super(CARAFEPack, self).__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=self.up_kernel, dilation=self.scale_factor,
                                padding=self.up_kernel // 2 * self.scale_factor)
        self.proj = nn.Conv2d(channels, output_channels, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)

    def kernel_normalizer(self, kernel):
        kernel = F.pixel_shuffle(kernel, self.scale_factor)  # [B, K_up*K_up, HS, WS]
        n, kernel_c, h, w = kernel.size()
        kernel_channel = int(kernel_c / (self.up_kernel * self.up_kernel))
        kernel = kernel.view(n, kernel_channel, -1, h, w)  # [B, 1, K_up*K_up, HS, WS]

        kernel = F.softmax(kernel, dim=2)
        kernel = kernel.view(n, kernel_c, h, w).contiguous()  # [B, K_up*K_up, HS, WS]

        return kernel

    def feature_reassemble(self, x, kernel):
        n, c, h, w = x.size()
        x = self.upsample(x)  # [B, C, HS, WS]
        x = self.unfold(x)  # [B, K_up*K_up*C, HS, WS]
        x = x.view(n, c, -1, h * self.scale_factor, w * self.scale_factor)  # [B, C, Kup*Kup, HS, WS]
        x = torch.einsum('bkhw,bckhw->bchw', [kernel, x])
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)  # [B, CM, H, W]
        kernel = self.content_encoder(compressed_x)  # [B, K_up*K_up*S*S, H, W]
        kernel = self.kernel_normalizer(kernel)  # [B, K_up*K_up, HS, WS]

        x = self.feature_reassemble(x, kernel)
        x = self.proj(x)
        return x


class CarafeConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, **kwargs):
        super(CarafeConv, self).__init__()
        self.carafe = CARAFEPack(channels=in_channels, output_channels=out_channels, scale_factor=stride)

    def forward(self, x):
        return self.carafe(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
