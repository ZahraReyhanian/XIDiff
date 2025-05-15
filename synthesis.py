import os
import torch
from Trainer import Trainer as MyModelTrainer
from datamodules.face_datamodule import FaceDataModule
from utils.generation_utils import generate_image
import re
import json

from utils.os_utils import get_latest_file

image_size = 256

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    root = cfg['root']

    ## config from dcface
    # ckpt = torch.load(os.path.join(root, 'pretrained_models/dcface_3x3.ckpt'))
    # model_hparam = ckpt['hyper_parameters']
    # model_hparam['unet_config']['params']['pretrained_model_path'] = None
    # model_hparam['_target_'] = 'src.trainer.Trainer'
    # model_hparam['_partial_'] = True

    # load pl_module
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get checkpoint path
    model_ckpt_path = root + cfg["ckpt_path"]
    name = get_latest_file(model_ckpt_path)
    print(name)
    checkpoint_path = os.path.join(model_ckpt_path, name)
    print("loading from: ", checkpoint_path)

    pl_module = MyModelTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path)
    pl_module.to('cuda')
    pl_module.eval()

    # load dataset
    print("loading dataset ........")

    dataset_path = os.path.join(root, cfg["dataset_path"])
    bs = 1
    datamodule = FaceDataModule(dataset_path=dataset_path, img_size=(image_size, image_size), batch_size=bs)
    datamodule.setup()

    generate_image(pl_module=pl_module,
                   save_root="generated_images",
                   batch_size=bs,
                   dataloader=datamodule.test_dataloader(),
                   device=device)


if __name__ == "__main__":
    main()
