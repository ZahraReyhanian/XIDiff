# -*- coding: utf-8 -*-

# Section 2: Implementing One


# @title Install requirements
# !pip install datasets &>> install.log
#
# !pip install --upgrade torch torchvision torchaudio

#@title imports and utility functions
from datasets import load_dataset
from PIL import Image
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_url


def img_to_tensor(im):
  return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

# 2.1 Dataset

im_size = 48
transform = transforms.Compose([transforms.Resize((im_size, im_size)),
                                 transforms.ToTensor()])

# load dataset
train = datasets.ImageFolder('data/train', transform=transform)
valid = datasets.ImageFolder('data/valid', transform=transform)
test = datasets.ImageFolder('data/test', transform=transform)


## 2.2 Adding Noise


n_steps = 100
beta = torch.linspace(0.0001, 0.04, n_steps)

def q_xt_xtminus1(xtm1, t):
    beta_t = gather(beta, t)
    mean = (1. - beta_t).sqrt() * xtm1
    std = beta_t.sqrt()
    return mean + std * torch.randn(*xtm1.shape)


n_steps = 100
beta = torch.linspace(0.0001, 0.04, n_steps)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

def q_xt_x0(x0, t):
    alpha_bar_t = gather(alpha_bar, t)
    mean = (alpha_bar_t).sqrt()*x0
    std = (1- alpha_bar_t).sqrt()
    return mean + std * torch.randn(*x0.shape)

## 2.3 UNETs


#@title Unet Definition

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

# A fancy activation function
class Swish(nn.Module):
    """
    ### Swish actiavation function or SilU
    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)

# The time embedding
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings

        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb

# Residual blocks include 'skip' connections
class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)

# Ahh yes, magical attention...
class AttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)

# The core class definition (aka the important bit)
class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))


# 2.4 Training

#data loader for train and valid
batch_size = 16 # Lower this if hitting memory issues

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

print("size of train data:", len(train_loader))
print("size of valid data:", len(valid_loader))

# path to save model
PATH = './model_ddpm.pth'

torch.cuda.empty_cache()
# Set up some parameters
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");
print("device: ", device)

n_steps = 1000
beta = torch.linspace(0.0001, 0.04, n_steps, device=device)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# return the noise itself as well
def q_xt_x0(x0, t):
    alpha_bar_t = gather(alpha_bar, t)
    mean = (alpha_bar_t).sqrt()*x0

    std = (1-alpha_bar_t).sqrt()
    noise = torch.randn_like(x0).to(device)

    x_t = mean + std * noise

    return x_t, noise

################################################################################
# Create the model
n_channels = 32
unet = UNet(n_channels=n_channels).cuda()
# unet = UNet(n_channels=n_channels).cuda()
# unet.load_state_dict(torch.load(PATH))
# Training params
lr = 1e-4 # Explore this - might want it lower when training on the full dataset

epochs = 100
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr) # Optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8) #learning rate scheduler
denoising_loss = F.mse_loss #use mean squared error fuction to calculate loss

n = len(train)


# Training Loop
losses = [] # Store losses for later plotting
val_losses = []

for epoch in range(epochs):
    #train step
    unet.train()
    train_loss = 0

    #number of iteration
    it = len(train_loader)

    progress_bar = tqdm(it, total=it)
    progress_bar.set_description(f"Epoch {epoch+1}")
    prev_loss = 1000000

    for idx, (x, y) in enumerate(train_loader):
        b_size = x.shape[0]
        x = x.reshape(b_size,3,im_size,im_size).to(device=device)
        optimizer.zero_grad()

        t = torch.randint(0, n_steps, (b_size,), dtype=torch.long).to(device=device)
        x_t, noise = q_xt_x0(x, t)

        #predict noise using unet model
        pred_noise = unet(x_t.float(), t)

        #calculate loss
        loss = denoising_loss(noise.reshape(b_size,3,im_size*im_size).float(), pred_noise.reshape(b_size,3,im_size*im_size))
        train_loss += loss.item()

        #backpropagation
        loss.backward()

        #optimize parameters
        optimizer.step()

        progress_bar.set_postfix({"loss": train_loss / (idx + 1), "lr": scheduler.get_last_lr()})
        progress_bar.update(1)

    scheduler.step()
    losses.append(train_loss/it)
    progress_bar.close()

    print(f"Epoch {epoch+1}: Train Loss: {train_loss/it} ")

    if train_loss / it < prev_loss:
        torch.save(unet.state_dict(), PATH)
    prev_loss = train_loss / it

    # validation step
    unet.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, (xv, yv) in enumerate(valid_loader):
            b_size = xv.shape[0]
            xv, yv = xv.reshape(b_size,3,im_size,im_size).to(device=device), yv.to(device=device)

            t = torch.randint(0, n_steps, (b_size,), dtype=torch.long).to(device=device)
            x_t, noise = q_xt_x0(xv, t)

            pred_noise = unet(x_t.float(), t)

            # calculate validation loss
            loss = denoising_loss(noise.reshape(b_size,3,im_size*im_size), pred_noise.reshape(b_size,3,im_size*im_size))
            val_loss += loss.item()

        val_losses.append(val_loss/len(valid_loader))
        print(f"Epoch {epoch+1}: Validation Loss: {val_loss/len(valid_loader)} ")


    # End
################################################################################



"""## plot train and validation loss"""

import matplotlib.pyplot as plt

plt.plot(range(epochs), losses, label="Train loss")
plt.plot(range(epochs), val_losses, label="Validation loss")
plt.ylabel("loss/error")
plt.xlabel('Epochs')
plt.title("Train and valid loss during training")
plt.legend()
plt.show()
plt.savefig('training.png')


"""### save model"""

torch.save(unet.state_dict(), './final_model.pth')

"""## 2.5 The Reverse Step

Now we need to define the reverse step $p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$

See that little $_\theta$? That often indicates 'learned parameters' - in this case our unet model! We use our model to predict the noise and then 'undo' the forward noise steps one at a time to go from an image that is pure noise to one that (hopefully) looks like a real image:
"""

torch.cuda.empty_cache()

"""### load model"""

model = UNet(n_channels=32).cuda()
model.load_state_dict(torch.load(PATH))
model.eval()

"""### revese step distibution"""

def p_xt(xt, noise, t):
################################################################################
    # TODO: complete the code here
    # Write this function based on the formula given in the paper for the reverse step
    alpha_t = gather(alpha, t)
    alpha_bar_t = gather(alpha_bar, t)
    beta_t = gather(beta, t)

    eps_coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    mu = (xt - eps_coef * noise) / (torch.sqrt(alpha_t))
    z = torch.randn_like(xt)
    std = torch.sqrt(beta_t)

    xt_1 = mu + z * std
    return xt_1

    # End
################################################################################


"""### *Generate* and Save Samples"""


import os
fake_image_path = './images_fakes'
os.mkdir(fake_image_path)

from torchvision.utils import save_image

# generate 100 samples
progress_bar = tqdm(n_steps, total=n_steps)
n_images = 100
x = torch.randn(n_images, 3, im_size, im_size).cuda() # Start with random noise
for i in range(n_steps):
    t = torch.tensor(n_steps-i-1, dtype=torch.long).cuda()
    with torch.no_grad():
        pred_noise = model(x.float(), t.unsqueeze(0))
        x = p_xt(x, pred_noise, t.unsqueeze(0))

        progress_bar.update(1)
        # save the generated sample (x) in a directory
        # See this link: https://pytorch.org/vision/stable/generated/torchvision.utils.save_image.html
        # Be sure to assign a different name to each image!
    # fake_images_tensor.append()
for i in range(n_images):
    save_image(tensor=x[i], fp=f'{fake_image_path}/img_{i}.png')
