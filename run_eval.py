import os
import json
import numpy as np
import torch
from PIL import Image
import argparse
from Trainer import Trainer as MyModelTrainer
from utils.evaluation import *

def eval(source_root_path, target_root_path, pl_module):
    images = sorted(os.listdir(source_root_path))
    targets = sorted(os.listdir(target_root_path))
    psnr = []
    ssim = []
    id_cs = []
    lpips = []
    it = 0
    new_size = (112, 112)

    for image_path, target_path in zip(images, targets):
        print(it)
        image_path = os.path.join(source_root_path, image_path)
        target_path = os.path.join(target_root_path, target_path)
        print(image_path, target_path)


        target_image = Image.open(target_path).convert("RGB").resize(new_size)
        image = Image.open(image_path).convert("RGB").resize(new_size)



        cs = compute_id_cosine_similarity(target_image, Image.fromarray(np.uint8(image)), pl_module)
        id_cs.append(cs)
        print(cs)
        lpips.append(compute_lpips(target_image, Image.fromarray(np.uint8(image))))

        target_image = np.array(target_image)
        image = np.array(image)

        psnr.append(compute_PSNR(target_image, image))
        ssim.append(compute_SSIM(target_image, image))

        it+=1

    print("PSNR: ", np.array(psnr).mean())
    print("SSIM: ", np.array(ssim).mean())
    print("ID Cosine Similarity: ", np.array(id_cs).mean())
    print("LPIPS: ", np.array(lpips).mean())


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    root = cfg['root']

    with open('config/general.json') as f:
        general_cfg = json.load(f)
    checkpoint_path = 'E:/uni/Articles/pretrained_models/dcface_3x3.ckpt'
    print("loading from: ", checkpoint_path)
    ckpt = torch.load(checkpoint_path, weights_only=False)

    pl_module = MyModelTrainer(unet_config=general_cfg['unet_config'],
                               use_pretrained=True,
                               lr=cfg['lr'],
                               recognition=general_cfg['recognition'],
                               recognition_eval=general_cfg['recognition_eval'],
                               label_mapping=general_cfg['label_mapping'],
                               external_mapping=general_cfg['external_mapping'],
                               output_dir=cfg["output_dir"],
                               perceptual_loss_weight=cfg['perceptual_loss_weight'],
                               sampler=general_cfg['sampler'],
                               attention_on_style=False,  #
                               random_alpha=False,
                               num_classes=cfg['num_classes'],
                               batch_size=1,
                               root=root)
    pl_module.load_state_dict(ckpt['state_dict'], strict=True)
    pl_module.to('cuda')
    pl_module.eval()
    ###############################

    parser = argparse.ArgumentParser(description="inference script")

    ######### General ###########
    parser.add_argument('--source_path', type=str, required=True, help="path to source generated images")
    parser.add_argument('--target_path', type=str, required=True, help="path to target images")

    args = parser.parse_args()
    args = vars(args)  # convert to dictionary
    print(args)


    eval(args['source_path'], args['target_path'], pl_module)



if __name__ == "__main__":
    main()