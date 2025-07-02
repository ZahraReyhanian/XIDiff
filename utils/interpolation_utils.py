import torch
from torchvision.utils import make_grid
import torchvision
import numpy as np
import os
import io
import cv2
import imageio
import matplotlib.pyplot as plt
from utils import sample_visual

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def prepare_text_img(text, height=300, width=30, fontsize=16, textcolor='C1', fontweight='normal', bg_color='white'):
    text_kwargs = dict(ha='center', va='center', fontsize=fontsize, color=textcolor, fontweight=fontweight)
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(width*px, height*px), facecolor=bg_color)
    plt.text(0.5, 0.5, text, **text_kwargs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_facecolor(bg_color)
    array = get_img_from_fig(fig)
    plt.clf()
    array = cv2.resize(array, (width, height))
    return array


def generate_interpolation(batch, pl_module, num_img_per_subject, num_subjects, mixing_method,
                           save_root='interpolation',
                           return_x0_intermediates=False):

    label_batch, extra_batch = divide_batch(batch, half=num_img_per_subject*num_subjects)

    os.makedirs(save_root, exist_ok=True)

    alphas = np.linspace(1, 0, 10).round(2)
    pred_images_all = []
    for alpha in alphas:
        generator = torch.manual_seed(0)
        pred_images = sample_visual.render_condition(label_batch, pl_module,
                                                     sampler='ddim', between_zero_and_one=True,
                                                     show_progress=False, generator=generator,
                                                     mixing_batch=extra_batch,
                                                     mixing_method=mixing_method, source_alpha=alpha,
                                                     return_x0_intermediates=return_x0_intermediates)

        if return_x0_intermediates:
            pred_images, x0_intermediates = pred_images
            pred_images_grid = torchvision.utils.make_grid(torch.tensor(pred_images.transpose(0, 3, 1, 2)), nrow=8)
            pred_images_grid_uint8 = sample_visual.to_image_npy_uint8(pred_images_grid.detach().cpu().numpy().transpose(1,2,0))
            sample_visual.save_uint8(pred_images_grid_uint8, path='{}/{}.jpg'.format(save_root, f'all_alpha_{alpha:.2f}.jpg'))

            for i in range(num_subjects):
                interms = torch.tensor(np.array([val[i] for _, val in x0_intermediates.items()]).transpose(0, 3, 1, 2))
                interms_grid = torchvision.utils.make_grid(interms, nrow=10)
                interms_grid_uint8 = sample_visual.to_image_npy_uint8(interms_grid.detach().cpu().numpy().transpose(1,2,0))
                sample_visual.save_uint8(interms_grid_uint8, path='{}/{}.jpg'.format(save_root, f'interms_{i}_alpha_{alpha:.2f}.jpg'))

        pred_images_all.append(pred_images)

    orig_images = label_batch['exp_img']
    extra_image = extra_batch['exp_img']
    for i in range(num_subjects):
        sub_orig_images = orig_images[i*num_img_per_subject: (i+1)*num_img_per_subject]
        orig_grid = torchvision.utils.make_grid(sub_orig_images * 0.5 + 0.5, nrow=num_img_per_subject)
        orig_grid_uint8 = sample_visual.to_image_npy_uint8(orig_grid.detach().cpu().numpy().transpose(1,2,0))
        orig_text = prepare_text_img('Subject 1', height=orig_grid_uint8.shape[0], width=340,)
        orig_grid_uint8 = np.concatenate([orig_text, orig_grid_uint8], axis=1)

        sub_extra_image = extra_image[i*num_img_per_subject: (i+1)*num_img_per_subject]
        extra_grid = torchvision.utils.make_grid(sub_extra_image * 0.5 + 0.5, nrow=num_img_per_subject)
        extra_grid_uint8 = sample_visual.to_image_npy_uint8(extra_grid.detach().cpu().numpy().transpose(1,2,0))
        extra_text = prepare_text_img('Subject 2', height=orig_grid_uint8.shape[0], width=340,)
        extra_grid_uint8 = np.concatenate([extra_text, extra_grid_uint8], axis=1)
        vis = [orig_grid_uint8, extra_grid_uint8]

        for alpha, pred_images in zip(alphas, pred_images_all):
            sub_pred_images = pred_images[i*num_img_per_subject: (i+1)*num_img_per_subject]
            grid = torchvision.utils.make_grid(torch.tensor(sub_pred_images.transpose(0, 3, 1, 2)), nrow=num_img_per_subject)
            grid_uint8 = sample_visual.to_image_npy_uint8(grid.detach().cpu().numpy().transpose(1,2,0))
            new_text = prepare_text_img('Generated alpha:{:.2f}'.format(alpha), height=orig_grid_uint8.shape[0], width=340,)
            grid_uint8 = np.concatenate([new_text, grid_uint8], axis=1)
            vis.append(grid_uint8)

        vis = np.concatenate(vis, axis=0)
        sample_visual.save_uint8(vis, path='{}/{}.jpg'.format(save_root, i))


def divide_batch(batch, half=4):
    label_batch = {}
    extra_batch = {}
    for key, val in batch.items():
        label_batch[key] = val[:half]
        extra_batch[key] = val[half:]
    return label_batch, extra_batch
