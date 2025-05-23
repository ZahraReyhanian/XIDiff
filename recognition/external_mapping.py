from torch import nn as nn
import torch
import torch.nn.functional as F
import math


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def get_dim(style_dims=[]):
    cdim = 0
    for index in style_dims:
        if index == 2:
            cdim += 64
        if index == 4:
            cdim += 128
        if index == 6:
            cdim += 256
        if index == 8:
            cdim += 512
    return cdim


def get_spatial(style_dims=[]):
    spatial_dim = []
    for index in style_dims:
        if index == 2:
            spatial_dim.append((56, 56))
        if index == 4:
            spatial_dim.append((28, 28))
        if index == 6:
            spatial_dim.append((14, 14))
        if index == 8:
            spatial_dim.append((7, 7))
    return spatial_dim


class ExternalMappingV4Dropout(nn.Module):

    def __init__(self, return_spatial, out_size=(16, 16), out_channel=64, dropout_prob=0.0):

        super(ExternalMappingV4Dropout, self).__init__()
        self.return_spatial = return_spatial
        self.channels = [get_dim([idx]) for idx in self.return_spatial]
        self.spatial_dims = get_spatial(self.return_spatial)
        self.out_size = tuple(out_size)

        for i, chn in enumerate(self.channels):
            conv = nn.Conv2d(in_channels=chn, out_channels=out_channel, kernel_size=1, stride=1)
            norm = nn.BatchNorm2d(out_channel)
            relu = nn.PReLU()
            channel_weight = nn.ParameterList([nn.Parameter(torch.Tensor(out_channel, 2))])
            channel_weight[0].data.fill_(0.5)
            dropout = nn.Dropout(dropout_prob)
            setattr(self, 'conv_{}'.format(i), conv)
            setattr(self, 'norm_{}'.format(i), norm)
            setattr(self, 'relu_{}'.format(i), relu)
            setattr(self, 'channel_weight_{}'.format(i), channel_weight)
            setattr(self, 'dropout_{}'.format(i), dropout)

            pos_emb_init = positionalencoding1d(out_channel, (self.out_size[0] * self.out_size[1]) + 1) - 0.5
            pos_embedding = nn.ParameterList([nn.Parameter(pos_emb_init)])
            setattr(self, 'pos_emb_{}'.format(i), pos_embedding)

            ln = nn.LayerNorm(out_channel)
            setattr(self, 'ln_{}'.format(i), ln)

        self.cross_attn_adapter = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)

    def forward(self, spatial_features):
        assert len(self.channels) == len(spatial_features)
        outs = []
        for i, feature in enumerate(spatial_features):
            relu = getattr(self, 'relu_{}'.format(i))
            conv = getattr(self, 'conv_{}'.format(i))
            norm = getattr(self, 'norm_{}'.format(i))
            channel_weight = getattr(self, 'channel_weight_{}'.format(i))[0]
            dropout = getattr(self, 'dropout_{}'.format(i))
            pos_emb = getattr(self, 'pos_emb_{}'.format(i))[0]
            ln = getattr(self, 'ln_{}'.format(i))

            if int(feature.shape[-1] // self.out_size[0] * self.out_size[0]) != feature.shape[-1]:
                divisible_side = int(feature.shape[-1] // self.out_size[0] * self.out_size[0])
                feature = F.interpolate(feature, size=(divisible_side, divisible_side))

            feature = dropout(feature)

            feature = norm(conv(relu(feature)))
            side = feature.shape[-1]
            patch_side = side // self.out_size[0]
            B, C = feature.shape[0], feature.shape[1]

            global_mean = nn.AvgPool2d(side, side)(feature).view(B, C, -1)
            global_var = global_mean - nn.AvgPool2d(side, side)(feature ** 2).view(B, C, -1)
            patch_mean = nn.AvgPool2d(patch_side, patch_side)(feature).view(B, C, -1)
            patch_var = patch_mean - nn.AvgPool2d(patch_side, patch_side)(feature ** 2).view(B, C, -1)

            global_stat = torch.stack([global_mean, global_var], dim=2)
            patch_stat = torch.stack([patch_mean, patch_var], dim=2)
            global_info = (global_stat * channel_weight[None, :, :, None]).sum(2)
            patch_info = (patch_stat * channel_weight[None, :, :, None]).sum(2)
            info = torch.cat([global_info, patch_info], dim=2).transpose(1, 2)
            info = info + pos_emb[None, :, :]
            info = ln(info)

            outs.append(info)
        outs = torch.cat(outs, dim=2)
        return outs


def make_external_mapping(config, unet_config):
    if config["version"] == None:
        return None
    elif config["version"] == 'v4_dropout':
        print('v4_dropout external', config["dropout_prob"])
        assert unet_config['params']["condition_type"] in ['cross_attn', 'crossatt_and_stylemod']
        assert unet_config['params']["condition_source"] in ['patchstat_spatial_and_linear_label_center',
                                                             'patchstat_spatial_and_image',
                                                             'image_and_patchstat_spatial']
        ext_mapping = ExternalMappingV4Dropout(return_spatial=config['return_spatial'],
                                               out_size=(config["spatial_dim"], config["spatial_dim"]),
                                               out_channel=config["out_channel"],
                                               dropout_prob=config["dropout_prob"])
    else:
        raise ValueError('')
    return ext_mapping


if __name__ == '__main__':
    aux = ExternalMappingV4Dropout(return_spatial=[2], out_size=(16, 16), out_channel=64)
    aux([torch.randn(4, 64, 56, 56)]).shape
