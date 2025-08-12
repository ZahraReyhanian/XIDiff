import os
import re
import json
import numpy as np
import torch

from Trainer import Trainer as MyModelTrainer
from datamodules.face_datamodule import FaceDataModule
from utils.generation_utils import generate_image
from utils.os_utils import get_latest_file

image_size = 112

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    root = cfg['root']

    # load pl_module
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get checkpoint path
    model_ckpt_path = root + cfg["ckpt_path"]
    # name = get_latest_file(model_ckpt_path)
    # print(name)
    name = 'last-v4.ckpt'
    checkpoint_path = os.path.join(model_ckpt_path, name)
    print("loading from: ", checkpoint_path)

    pl_module = MyModelTrainer.load_from_checkpoint(checkpoint_path=checkpoint_path)
    pl_module.to(device)
    pl_module.eval()

    # load dataset
    print("loading dataset ........")

    json_path = os.path.join(root, cfg["json_path"])
    batch_size = 4 #4 for interpolation 1 for generation
    datamodule = FaceDataModule(json_path=json_path,
                                img_size=(image_size, image_size),
                                batch_size=batch_size)
    datamodule.setup()

    # generate_image(pl_module=pl_module,
    #                save_root="generated_images",
    #                batch_size=batch_size,
    #                datamodule=datamodule,
    #                total_images=800)

    # interpolation
    alphas = np.linspace(1, 0, 11).round(2)
    generate_image(pl_module=pl_module,
                   save_root="interpolation",
                   batch_size=batch_size,
                   datamodule=datamodule,
                   alphas=alphas)




if __name__ == "__main__":
    main()
