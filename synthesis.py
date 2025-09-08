import os
import json
import argparse
import numpy as np
import torch
from Trainer import Trainer as MyModelTrainer
from datamodules.face_datamodule import FaceDataModule
from utils.generation_utils import generate_image
from utils.interpolation_utils import prepare_text_img
from utils.os_utils import get_latest_file

image_size = 112


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    root = cfg['root']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get checkpoint path
    model_ckpt_path = root + cfg["ckpt_path"]
    name = get_latest_file(model_ckpt_path)
    print(name)
    # name = 'last-v10.ckpt'


    parser = argparse.ArgumentParser(description="inference script")

    ######### General ###########
    parser.add_argument('--model_path', type=str,
                        default=name,
                        help="path to pretrained model")
    parser.add_argument('--dcface', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True, help="interpolation or generation")

    args = parser.parse_args()
    print(args)

    if args['dcface']==0:
        # load pl_module
        checkpoint_path = os.path.join(model_ckpt_path, args['model_path'])
        print("loading from: ", checkpoint_path)

        pl_module = MyModelTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path)
        pl_module.to(device)
        pl_module.eval()

    elif args['dcface']==1:
        # To load dcface
        with open('config/general.json') as f:
            general_cfg = json.load(f)
        # checkpoint_path = '/opt/data/reyhanian/pretrained_models/dcface_5x5.ckpt'
        checkpoint_path = os.path.join(model_ckpt_path, args['model_path'])
        print("loading from: ", checkpoint_path)
        ckpt = torch.load(checkpoint_path)
        # model_hparam = ckpt['hyper_parameters']
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
                               attention_on_style=False, #
                               random_alpha=False,
                               num_classes=cfg['num_classes'],
                               batch_size=1,
                               root=root)
        pl_module.load_state_dict(ckpt['state_dict'], strict=True)
        pl_module.to('cuda')
        pl_module.eval()
        ###############################



    # load dataset
    json_path = os.path.join(root, cfg["json_path"])

    if args['mode']=='generation':
        # image generation
        batch_size1 = 1
        datamodule = FaceDataModule(json_path=json_path,
                                    img_size=(image_size, image_size),
                                    batch_size=batch_size1)
        datamodule.setup()

        generate_image(pl_module=pl_module,
                       save_root="generated_images",
                       batch_size=batch_size1,
                       datamodule=datamodule,
                       total_images=1000)

    else:
        # interpolation
        batch_size2 = 4
        datamodule = FaceDataModule(json_path=json_path,
                                    img_size=(image_size, image_size),
                                    batch_size=batch_size2)
        datamodule.setup()
        alphas = np.linspace(1, 0, 11).round(2)
        generate_image(pl_module=pl_module,
                       save_root="interpolation",
                       batch_size=batch_size2,
                       datamodule=datamodule,
                       alphas=alphas)




if __name__ == "__main__":
    main()
