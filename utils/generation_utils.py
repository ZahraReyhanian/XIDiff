import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torchvision
from utils import sample_visual
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
    pred_images_all = []
    it = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc='Generating Images: '):
        for alpha in alphas:
            pred_images = sample_visual.render_condition(batch, pl_module,
                                                         sampler='ddim', between_zero_and_one=True,
                                                         show_progress=False, generator=generator,
                                                         mixing_batch=None,
                                                         source_alpha=alpha,
                                                         return_x0_intermediates=False)

            pred_images_all.append(pred_images)

        orig_images = batch['id_img']
        extra_image = batch['exp_img']
        num_img_per_subject = len(batch)
        for i in range(len(batch)):
            sub_orig_images = orig_images[i * num_img_per_subject: (i + 1) * num_img_per_subject]
            orig_grid = torchvision.utils.make_grid(sub_orig_images * 0.5 + 0.5, nrow=num_img_per_subject)
            orig_grid_uint8 = sample_visual.to_image_npy_uint8(orig_grid.detach().cpu().numpy().transpose(1, 2, 0))
            orig_text = prepare_text_img('ID image', height=orig_grid_uint8.shape[0], width=340, )
            orig_grid_uint8 = np.concatenate([orig_text, orig_grid_uint8], axis=1)

            sub_extra_image = extra_image[i * num_img_per_subject: (i + 1) * num_img_per_subject]
            extra_grid = torchvision.utils.make_grid(sub_extra_image * 0.5 + 0.5, nrow=num_img_per_subject)
            extra_grid_uint8 = sample_visual.to_image_npy_uint8(extra_grid.detach().cpu().numpy().transpose(1, 2, 0))
            extra_text = prepare_text_img('Expr image', height=orig_grid_uint8.shape[0], width=340, )
            extra_grid_uint8 = np.concatenate([extra_text, extra_grid_uint8], axis=1)
            vis = [orig_grid_uint8, extra_grid_uint8]

            for alpha, pred_images in zip(alphas, pred_images_all):
                sub_pred_images = pred_images[i * num_img_per_subject: (i + 1) * num_img_per_subject]
                grid = torchvision.utils.make_grid(torch.tensor(sub_pred_images.transpose(0, 3, 1, 2)),
                                                   nrow=num_img_per_subject)
                grid_uint8 = sample_visual.to_image_npy_uint8(grid.detach().cpu().numpy().transpose(1, 2, 0))
                new_text = prepare_text_img('Generated alpha:{:.2f}'.format(alpha), height=orig_grid_uint8.shape[0],
                                            width=340, )
                grid_uint8 = np.concatenate([new_text, grid_uint8], axis=1)
                vis.append(grid_uint8)

            vis = np.concatenate(vis, axis=0)
            sample_visual.save_uint8(vis, path='{}/{}-{}.jpg'.format(save_root, i, it))
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


