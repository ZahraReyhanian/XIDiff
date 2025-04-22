from torch import nn
import torch
import torch.nn.functional as F

from expression.keypoint_detector import KPDetector


class ExpressionEncoder(nn.Module):
    """
    Extracts the latent expression vector.
    """

    def __init__(self, kp_detector,
                 in_channels = 32,
                 num_kp=10,
                 expression_size_per_kp=32,
                 expression_size=256,
                 pad=0,
                 n_tokens=26,
                 token_dim=512):
        super(ExpressionEncoder, self).__init__()

        self.kp_detector = kp_detector
        self.expression_size = expression_size #Output dimension
        self.expression_size_per_kp = expression_size_per_kp #Number of output features of the convolutional layer for each keypoint 
        self.num_kp = num_kp
        self.expression = nn.Conv2d(in_channels=in_channels,
                                      out_channels=num_kp * self.expression_size_per_kp , kernel_size=(7, 7), padding=pad)

        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.expression_mlp = nn.Sequential(
            nn.Linear(self.expression_size_per_kp*self.num_kp, 640),
            nn.ReLU(),
            nn.Linear(640, 1280),
            nn.ReLU(),
            nn.Linear(1280, 640),
            nn.ReLU(),
            nn.Linear(640, self.expression_size),
        )

        self.post_proj = nn.Linear(expression_size,
                                   n_tokens * token_dim)

        self.cross_attn_adapter = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1)

 
    def forward(self, feature_map, heatmap):
        latent_expression_feat = self.expression(feature_map)
        final_shape = latent_expression_feat.shape
        latent_expression_feat = latent_expression_feat.reshape(final_shape[0], self.num_kp, self.expression_size_per_kp , final_shape[2], final_shape[3])

        heatmap = heatmap.unsqueeze(2)
        latent_expression = heatmap * latent_expression_feat
        latent_expression = latent_expression.view(final_shape[0], self.num_kp, self.expression_size_per_kp , -1)
        latent_expression = latent_expression.sum(dim=-1).view(final_shape[0],-1)
        latent_expression = self.expression_mlp(latent_expression)

        expr_tokens = self.post_proj(latent_expression)  # [B,13312]
        expr_tokens = expr_tokens.view(latent_expression.size(0),  # [B,26,512]
                                       self.n_tokens,
                                       self.token_dim)
        
        return expr_tokens

    def extract_keypoints_and_expression(self, img_src, img_driv):
        '''
        Shapes:
            img_src:       [bs,nsrc,3,h,w]
            img_driv:      [bs,(ndriv),3,h,w]
        '''
        assert self.kp_detector is not None
        if len(img_driv.shape) == 4:
            img_driv = img_driv.unsqueeze(1)

        if len(img_src.shape) == 4:
            img_src = img_src.unsqueeze(1)

        bs, nsrc, c, h, w = img_src.shape
        nkp = self.kp_detector.num_kp
        ndriv = img_driv.shape[1]
        img = torch.cat([img_src, img_driv], dim=1).view(-1, c, h, w)


        with torch.no_grad():
            kps, latent_dict = self.kp_detector(img)
            kps = kps.view(bs, nsrc + ndriv, nkp, 2)
            heatmaps = latent_dict['heatmap'].view(bs, nsrc + ndriv, nkp, latent_dict['heatmap'].shape[-2],
                                                   latent_dict['heatmap'].shape[-1])
            feature_maps = latent_dict['feature_map'].view(bs, nsrc + ndriv, latent_dict['feature_map'].shape[-3],
                                                           latent_dict['feature_map'].shape[-2],
                                                           latent_dict['feature_map'].shape[-1])



        kps_src, kps_driv = torch.split(kps, [nsrc, ndriv], dim=1)
        _, heatmap_driv = torch.split(heatmaps, [nsrc, ndriv], dim=1)
        _, feature_map_driv = torch.split(feature_maps, [nsrc, ndriv], dim=1)

        if kps_driv.shape[1] == 1:
            kps_driv = kps_driv.squeeze(1)

        expression_vector_src, expression_vector = torch.split(
            self.forward(feature_maps.flatten(0, 1), heatmaps.flatten(0, 1)).view(bs,nsrc+ndriv,-1), [nsrc,ndriv],dim = 1)

        if expression_vector.shape[1] == 1:
            expression_vector = expression_vector.squeeze(1)
            expression_vector = expression_vector.reshape(bs, self.n_tokens, self.token_dim)

        return kps_src, kps_driv, expression_vector

def create_expression_encoder(device, root, expression_size=256):
    kp_detector = KPDetector().to(device)
    kp_detector.load_state_dict(torch.load(f'{root}/fsrt_checkpoints/kp_detector.pt'))
    kp_detector.eval()
    expression_encoder = ExpressionEncoder(kp_detector=kp_detector, expression_size=expression_size, in_channels=kp_detector.predictor.out_filters)

    return expression_encoder
