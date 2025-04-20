import os
import torch
from Trainer import Trainer as MyModelTrainer
from utils.generation_utils import generate_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import re
import json

root = '/opt/data/reyhanian'

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main():
    with open('config/config.json') as f:
        cfg = json.load(f)

    ckpt = torch.load(os.path.join(root, 'pretrained_models/dcface_3x3.ckpt'))
    model_hparam = ckpt['hyper_parameters']
    model_hparam['unet_config']['params']['pretrained_model_path'] = None
    model_hparam['recognition']['ckpt_path'] = os.path.join(root, model_hparam['recognition']['ckpt_path'])
    model_hparam['recognition']['center_path'] = os.path.join(root, model_hparam['recognition']['center_path'])
    model_hparam['recognition_eval']['center_path'] = os.path.join(root, model_hparam['recognition_eval']['center_path'])

    model_hparam['_target_'] = 'src.trainer.Trainer'
    model_hparam['_partial_'] = True

    # load pl_module
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    last = True

    pl_module = MyModelTrainer(**model_hparam, ckpt_path=cfg['ckpt_path'], last=last, device=device)
    print('Instantiated ', model_hparam['_target_'])
    # load model

    pl_module.load_state_dict(ckpt['state_dict'], strict=True)
    pl_module.to('cuda')
    pl_module.eval()


    transform = transforms.Compose([transforms.Resize((48,48)),
                                    transforms.ToTensor()])

    # load dataset
    print("loading dataset ........")

    path = cfg["dataset_path"]
    data_test = datasets.ImageFolder(f'{path}test', transform=transform)
    bs = 1
    test_loader = DataLoader(data_test, batch_size=1)

    generate_image(pl_module=pl_module,
                   save_root="generated_images",
                   batch_size=bs,
                   dataloader=test_loader,
                   device=device)


if __name__ == "__main__":
    main()
