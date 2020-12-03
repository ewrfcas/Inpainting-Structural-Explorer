import torch
import torch.nn as nn
import os
import torch.optim as optim
from src.layers import GateConv, ResnetBlock, Conv, spectral_norm
from src.loss import VGG19, PerceptualLoss, StyleLoss, AdversarialLoss
from utils.utils import torch_show_all_params, get_lr_schedule_with_steps
import torch.nn.functional as F

try:
    from apex import amp

    amp.register_float_function(torch, 'matmul')
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def get_generator(config, input_channel):
    if config.model_type == 'EC':
        return ECGenerator(config, input_channel=input_channel)
    else:
        raise NotImplementedError


def get_conv(conv_type):
    if conv_type == 'gate':
        return GateConv
    elif conv_type == 'normal':
        return Conv
    else:
        raise NotImplementedError


def get_norm(norm_type):
    if norm_type == 'IN':
        return nn.InstanceNorm2d
    elif norm_type == 'BN':
        return nn.BatchNorm2d
    else:
        raise NotImplementedError


class ECGenerator(nn.Module):
    def __init__(self, config, input_channel=4):
        super(ECGenerator, self).__init__()

        econv = get_conv(config.econv_type)
        norm = get_norm(config.norm_type)
        ch = config.dim

        # encoder
        encoder = [nn.ReflectionPad2d(3),
                   econv(in_channels=input_channel, out_channels=ch, kernel_size=7,
                         padding=0, use_spectral_norm=config.gen_spectral_norm),
                   norm(ch),
                   nn.ReLU(True)]
        for _ in range(config.layer_nums[0] - 1):
            ch *= 2
            encoder.extend([econv(in_channels=ch // 2, out_channels=ch, kernel_size=4,
                                  stride=2, padding=1, use_spectral_norm=config.gen_spectral_norm),
                            norm(ch),
                            nn.ReLU(True)])
        self.encoder = nn.Sequential(*encoder)

        # middle
        blocks = []
        for _ in range(config.layer_nums[1]):
            blocks.append(ResnetBlock(ch, 2))
        self.middle = nn.Sequential(*blocks)

        # decoder
        dconv = get_conv(config.dconv_type)
        decoder = []
        for _ in range(config.layer_nums[2] - 1):
            decoder.extend([dconv(in_channels=ch, out_channels=ch // 2, kernel_size=4, stride=2,
                                  padding=1, transpose=True, use_spectral_norm=config.gen_spectral_norm),
                            norm(ch // 2),
                            nn.ReLU(True)])
            ch = ch // 2
        decoder.extend([nn.ReflectionPad2d(3),
                        nn.Conv2d(in_channels=ch, out_channels=3, kernel_size=7, padding=0),
                        nn.Tanh()])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, mask=None):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config, in_channels):
        super(Discriminator, self).__init__()

        dis_conv = get_conv(config.dis_conv_type)
        self.conv1 = self.features = nn.Sequential(
            dis_conv(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            dis_conv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            dis_conv(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            dis_conv(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1,
                     use_spectral_norm=config.dis_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1,
                                    bias=not config.dis_spectral_norm), config.dis_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Model(nn.Module):
    def __init__(self, config, logger=None):
        super(Model, self).__init__()
        self.config = config
        self.iteration = 0
        self.name = config.model_type
        self.g_path = os.path.join(config.path, self.name + '_g')
        self.d_path = os.path.join(config.path, self.name + '_d')

        self.g_model = get_generator(config, input_channel=4).to(config.device)
        self.d_model = Discriminator(config, in_channels=3).to(config.device)
        if logger is not None:
            logger.info('Generator Parameters:{}'.format(torch_show_all_params(self.g_model)))
            logger.info('Discriminator Parameters:{}'.format(torch_show_all_params(self.d_model)))
        else:
            print('Generator Parameters:{}'.format(torch_show_all_params(self.g_model)))
            print('Discriminator Parameters:{}'.format(torch_show_all_params(self.d_model)))

        # loss
        self.l1_loss = nn.L1Loss(reduction='none').to(config.device)
        vgg = VGG19(pretrained=True, vgg_norm=config.vgg_norm).to(config.device)
        self.perceptual_loss = PerceptualLoss(vgg, weights=config.vgg_weights, reduction='none').to(config.device)
        self.style_loss = StyleLoss(vgg).to(config.device)
        self.adversarial_loss = AdversarialLoss(type=config.gan_type).to(config.device)
        self.eps = 1e-7

        self.g_opt = optim.Adam(params=self.g_model.parameters(),
                                lr=float(config.g_lr), betas=(config.beta1, config.beta2))
        self.d_opt = optim.Adam(params=self.d_model.parameters(),
                                lr=float(config.d_lr), betas=(config.beta1, config.beta2))
        self.d_sche = get_lr_schedule_with_steps(config.decay_type,
                                                 self.d_opt,
                                                 drop_steps=config.drop_steps,
                                                 gamma=config.drop_gamma)
        self.g_sche = get_lr_schedule_with_steps(config.decay_type,
                                                 self.g_opt,
                                                 drop_steps=config.drop_steps,
                                                 gamma=config.drop_gamma)

        if config.float16:
            self.float16 = True
            [self.g_model, self.d_model], [self.g_opt, self.d_opt] = amp.initialize([self.g_model, self.d_model],
                                                                                    [self.g_opt, self.d_opt],
                                                                                    num_losses=2, opt_level='O1')
        else:
            self.float16 = False

    def forward(self, img, mask):
        # 被mask区域全部为1
        img_masked = img * (1 - mask) + mask
        inputs = torch.cat([img_masked, mask], dim=1)
        outputs = self.g_model(inputs)
        return outputs

    def get_losses(self, meta):
        self.iteration += 1
        real_img = meta['img']
        mask = meta['mask']

        g_loss = 0
        d_loss = 0

        # process outputs
        fake_img = self.forward(real_img, mask)

        # discriminator loss
        d_input_real = real_img
        d_input_fake = fake_img.detach()
        d_real, _ = self.d_model(d_input_real)
        d_fake, _ = self.d_model(d_input_fake)
        d_real_loss = self.adversarial_loss(d_real, True, True)
        d_fake_loss = self.adversarial_loss(d_fake, False, True)
        d_loss += (d_real_loss + d_fake_loss) / 2

        # generator adversarial loss
        g_input_fake = fake_img
        g_fake, _ = self.d_model(g_input_fake)
        g_gan_loss = self.adversarial_loss(g_fake, True, False) * self.config.adv_loss_weight
        g_loss += g_gan_loss

        # generator l1 loss
        g_l1_loss = self.l1_loss(fake_img, real_img)  # [bs, 3, H, W]
        mask_ = mask.repeat(1, 3, 1, 1)
        mask_sum = torch.sum(mask_, dim=[2, 3]) + self.eps
        valid_sum = torch.sum(1 - mask_, dim=[2, 3]) + self.eps
        mask_g_l1_loss = torch.mean(torch.sum(g_l1_loss * mask_, dim=[2, 3]) / mask_sum)
        valid_g_l1_loss = torch.mean(torch.sum(g_l1_loss * (1 - mask_), dim=[2, 3]) / valid_sum)
        g_l1_loss = mask_g_l1_loss * self.config.mask_l1_loss_weight + \
                    valid_g_l1_loss * self.config.valid_l1_loss_weight
        g_loss += g_l1_loss

        # generator perceptual loss
        perceptual_losses = self.perceptual_loss(fake_img, real_img)
        perceptual_loss = 0.0
        for perceptual_loss_ in perceptual_losses:
            mask_ = F.interpolate(mask, size=(perceptual_loss_.shape[2], perceptual_loss_.shape[3]))
            mask_ = mask_.repeat(1, perceptual_loss_.shape[1], 1, 1)
            mask_sum = torch.sum(mask_, dim=[2, 3]) + self.eps
            valid_sum = torch.sum(1 - mask_, dim=[2, 3]) + self.eps
            mask_perceptual_loss = torch.mean(torch.sum(perceptual_loss_ * mask_, dim=[2, 3]) / mask_sum)
            valid_perceptual_loss = torch.mean(torch.sum(perceptual_loss_ * (1 - mask_), dim=[2, 3]) / valid_sum)
            perceptual_loss += (mask_perceptual_loss * self.config.mask_vgg_loss_weight + \
                                valid_perceptual_loss * self.config.valid_vgg_loss_weight)
        g_loss += perceptual_loss

        # generator style loss
        g_style_loss = self.style_loss(fake_img * mask, real_img * mask)
        g_style_loss = g_style_loss * self.config.style_loss_weight
        g_loss += g_style_loss

        # create logs
        logs = [
            ("d_loss", d_loss.item()),
            ("g_loss", g_gan_loss.item()),
            ("l1_loss", g_l1_loss.item()),
            ("vgg_loss", perceptual_loss.item()),
            ("sty_loss", g_style_loss.item()),
        ]

        return fake_img, g_loss, d_loss, logs

    def backward(self, g_loss=None, d_loss=None):
        self.d_opt.zero_grad()
        if d_loss is not None:
            if self.float16:
                with amp.scale_loss(d_loss, self.d_opt, loss_id=0) as d_loss_scaled:
                    d_loss_scaled.backward()
            else:
                d_loss.backward()
        self.d_opt.step()

        self.g_opt.zero_grad()
        if g_loss is not None:
            if self.float16:
                with amp.scale_loss(g_loss, self.g_opt, loss_id=1) as g_loss_scaled:
                    g_loss_scaled.backward()
            else:
                g_loss.backward()
        self.g_opt.step()

        self.d_sche.step()
        self.g_sche.step()

    def load(self, is_test=False):
        g_path = self.g_path + '_last.pth'
        if self.config.restore or is_test:
            if is_test and os.path.exists(g_path.replace('_last.pth', '_best_fid.pth')):
                g_path = g_path.replace('_last.pth', '_best_fid.pth')
            if os.path.exists(g_path):
                print('Loading %s generator...' % g_path)
                if torch.cuda.is_available():
                    data = torch.load(g_path)
                else:
                    data = torch.load(g_path, map_location=lambda storage, loc: storage)
                self.g_model.load_state_dict(data['g_model'])
                if self.config.restore:
                    self.g_opt.load_state_dict(data['g_opt'])
                self.iteration = data['iteration']
            else:
                print(g_path, 'not Found')
                raise FileNotFoundError

        d_path = self.d_path + '_last.pth'
        if self.config.restore and not is_test:  # 判别器不加载best_fid，因为只会在训练的时候用到
            if os.path.exists(d_path):
                print('Loading %s discriminator...' % d_path)
                if torch.cuda.is_available():
                    data = torch.load(d_path)
                else:
                    data = torch.load(d_path, map_location=lambda storage, loc: storage)
                self.d_model.load_state_dict(data['d_model'])
                if self.config.restore:
                    self.d_opt.load_state_dict(data['d_opt'])
            else:
                print(d_path, 'not Found')
                raise FileNotFoundError
        else:
            print('No need for discriminator during testing')

    def save(self, prefix=None):
        if prefix is not None:
            save_g_path = self.g_path + "_{}.pth".format(prefix)
            save_d_path = self.d_path + "_{}.pth".format(prefix)
        else:
            save_g_path = self.g_path + ".pth"
            save_d_path = self.d_path + ".pth"

        print('\nsaving {} {}...\n'.format(self.name, prefix))
        torch.save({'iteration': self.iteration,
                    'g_model': self.g_model.state_dict(),
                    'g_opt': self.g_opt.state_dict()}, save_g_path)

        torch.save({'d_model': self.d_model.state_dict(),
                    'd_opt': self.d_opt.state_dict()}, save_d_path)
