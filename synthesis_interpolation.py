import os
import json
import torch

from utils.os_utils import get_latest_file
from Trainer import Trainer as MyModelTrainer
from datamodules.face_datamodule import FaceDataModule
from utils.interpolation_utils import generate_interpolation

image_size = 112

def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    root = cfg['root']
    num_img_per_subject = 4
    num_subjects = 4
    batch_size=num_img_per_subject*num_subjects*2

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
    pl_module.to(device)
    pl_module.eval()

    # load dataset
    print("loading dataset ........")

    json_path = os.path.join(root, cfg["json_path"])
    datamodule = FaceDataModule(json_path=json_path,
                                img_size=(image_size, image_size),
                                batch_size=batch_size,
                                shuffle=False)
    datamodule.setup()

    for it, batch in enumerate(datamodule.test_dataloader()):
        generate_interpolation(batch, pl_module, num_img_per_subject=4, num_subjects=4,
                               mixing_method='spatial_interpolate',
                               save_root='interpolation', it=it)
        print(it)

if __name__ == "__main__":
    main()