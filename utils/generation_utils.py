import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torchvision
from utils import sample_visual
from torchvision import transforms
from utils.interpolation_utils import prepare_text_img


def plot_generated_image(dataloader, pl_module, seed, save_root, total_images):
    it = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Images: '):
        print('target_label:', batch['target_label'])

        original_image = Image.open(batch['src_path'][0]).convert("RGB")
        target_image = Image.open(batch['target_path'][0]).convert("RGB")

        plotting_images = sample_batch(batch, pl_module, seed=seed)

        for i, image in enumerate(plotting_images):
            save_name = f"img_{i}_{it}.jpg"
            # save image
            save_path = os.path.join(save_root, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)

            save_path = os.path.join(save_root, f"img_{i}_{it}_{batch['target_label'].item()}-exp.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            target_image.save(save_path)

            save_path = os.path.join(save_root, f"img_{i}_{it}-id.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            original_image.save(save_path)

            it += 1

        if it > total_images:
            break


def plot_expr_interpolation(dataloader, alphas, pl_module, seed, save_root):
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None

    it = 0
    num_alphas = len(alphas)
    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Images: '):
        vis = []

        for id_img, exp_img, target_label, src_path, target_path in zip(batch['id_img'], batch['exp_img'], batch['target_label'], batch['src_path'], batch['target_path']):
            batch_data = {'id_img': id_img.unsqueeze(0), 'exp_img': exp_img.unsqueeze(0), 'target_label': target_label.unsqueeze(0)}

            sub_pred_image = []
            for alpha in alphas:
                pred_images = sample_visual.render_condition(batch_data, pl_module,
                                                             sampler='ddim', between_zero_and_one=True,
                                                             show_progress=False, generator=generator,
                                                             mixing_batch=None,
                                                             source_alpha=alpha,
                                                             return_x0_intermediates=False)
                sub_pred_image.append(pred_images.squeeze(0))

            sub_pred_image = torch.tensor(np.array(sub_pred_image))

            original_image = Image.open(src_path).convert("RGB")
            target_image = Image.open(target_path).convert("RGB")
            transform = transforms.Compose([transforms.Resize((112,112)),
                                            transforms.ToTensor()])
            original_image = transform(original_image)
            target_image = transform(target_image)

            row_images = np.concatenate((original_image.unsqueeze(0),
                                         target_image.unsqueeze(0),
                                         sub_pred_image.permute(0, 3, 1, 2)), axis=0)

            grid = torchvision.utils.make_grid(torch.tensor(row_images), nrow=num_alphas+2)
            row_grid = sample_visual.to_image_npy_uint8(grid.detach().cpu().numpy().transpose(1, 2, 0))

            vis.append(row_grid)

        vis = np.concatenate(vis, axis=0)
        sample_visual.save_uint8(vis, path='{}/{}.jpg'.format(save_root, it))
        it+=1


def generate_image(pl_module, datamodule, batch_size=1, save_root='./', seed=42, total_images=200, alphas=None):
    os.makedirs(save_root, exist_ok=True)
    print(save_root)

    dataloader = datamodule.test_dataloader()
    print(datamodule.data_test.idx_to_label)

    if alphas is not None:
        plot_expr_interpolation(dataloader, alphas, pl_module, seed, save_root)
    else:
        plot_generated_image(dataloader, pl_module, seed, save_root, total_images)


def sample_batch(batch, pl_module, seed=None):

    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    pred_images = sample_visual.render_condition(batch, pl_module, between_zero_and_one=True,
                                   show_progress=False, generator=generator, mixing_batch=None,
                                   return_x0_intermediates=False)

    # select which time to plot
    plotting_images = pred_images * 255
    return plotting_images[:, :, :, ::-1]


