import numpy as np
import torch



def make_condition(pl_module, batch):

    result = {'cross_attn': None, 'concat': None, 'add': None, 'center_emb': None}

    id_feat, id_cross_att = pl_module.id_extractor(batch["id_img"]) # id_image

     # style image batch["exp_img"]
    # kps_src, kps_driv, expression_vector = pl_module.expression_encoder.extract_keypoints_and_expression(batch["id_img"], batch["exp_img"])
    #
    # cross_attn = torch.cat([id_cross_att, expression_vector], dim=1).transpose(1,2)
    # result['cross_attn'] = pl_module.expression_encoder.cross_attn_adapter(cross_attn)

    _, spatial = pl_module.recognition_model(batch["exp_img"])  # exp_image
    ext_mapping = pl_module.external_mapping(spatial)

    cross_attn = torch.cat([id_cross_att, ext_mapping], dim=1).transpose(1, 2)
    result['cross_attn'] = pl_module.external_mapping.cross_attn_adapter(cross_attn)

    result['stylemod'] = id_feat

    class_label = batch["src_label"].to(pl_module.device)
    center_emb = pl_module.recognition_model.center(class_label).unsqueeze(1)

    result['center_emb'] = center_emb

    return result


def mix_hidden_states(encoder_hidden_states, mixing_hidden_states,
                      mixing_method='label_interpolate', source_alpha=0.0, pl_module=None):

    result = {'cross_attn': None, 'concat': None, 'add': None, 'center_emb': None}
    condition_type = 'crossatt_and_stylemod'
    condition_source = 'patchstat_spatial_and_image'
    source_label, source_spatial = split_label_spatial(condition_type, condition_source, encoder_hidden_states, pl_module)
    mixing_label, mixing_spatial = split_label_spatial(condition_type, condition_source, mixing_hidden_states, pl_module)

    source_label_feat, source_label_spat = source_label
    mixing_label_feat, mixing_label_spat = mixing_label

    ##########
    mixed_label_spat = source_alpha * source_label_spat + (1 - source_alpha) * mixing_label_spat
    mixed_label_feat = source_alpha * source_label_feat + (1 - source_alpha) * mixing_label_feat
    mixed_cross_attn = torch.cat([mixed_label_spat, source_spatial], dim=2)
    mixed_stylemod = mixed_label_feat

    result['cross_attn'] = mixed_cross_attn
    result['stylemod'] = mixed_stylemod
    source_class_emb = encoder_hidden_states['center_emb']
    mixing_class_emb = mixing_hidden_states['center_emb']

    if 'spatial' in mixing_method:
        mixed_class_emb = source_class_emb
    else:
        mixed_class_emb = source_alpha * source_class_emb + (1 - source_alpha) * mixing_class_emb
        mixed_class_emb = mixed_class_emb / torch.norm(mixed_class_emb, 2, -1, keepdim=True)
    result['center_emb'] = mixed_class_emb
    return result


def split_label_spatial(condition_type, condition_source, encoder_hidden_states, pl_module=None):

    if condition_type == 'cross_attn' and condition_source == 'label_center':
        label = encoder_hidden_states['cross_attn']
        spatial = None

    elif condition_type == 'cross_attn' and condition_source == 'spatial_and_label_center':
        label_dim = 512
        label = encoder_hidden_states['cross_attn'][:,:,:label_dim]
        spatial = encoder_hidden_states['cross_attn'][:,:,label_dim:]

    elif condition_type == 'add_and_cat' and condition_source == 'spatial_and_label_center':
        label = encoder_hidden_states['add']
        spatial = encoder_hidden_states['concat']

    elif condition_type == 'cross_attn' and condition_source == 'patchstat_spatial_and_linear_label_center':
        num_label_features = pl_module.id_extractor.pos_emb[0].shape[0]
        label = encoder_hidden_states['cross_attn'][:, :num_label_features, :]
        spatial = encoder_hidden_states['cross_attn'][:, num_label_features:, :]
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'patchstat_spatial_and_image':
        num_label_features = pl_module.id_extractor.pos_emb[0].shape[0] - 1
        label_feat = encoder_hidden_states['stylemod']
        label_spat = encoder_hidden_states['cross_attn'][:, :, :num_label_features]
        spatial = encoder_hidden_states['cross_attn'][:, :, num_label_features:]
        label = [label_feat, label_spat]
    elif condition_type == 'crossatt_and_stylemod' and condition_source == 'image_and_patchstat_spatial':
        spatial = encoder_hidden_states['stylemod']
        label = encoder_hidden_states['cross_attn']
    else:
        raise ValueError('not implemented')

    return label, spatial
