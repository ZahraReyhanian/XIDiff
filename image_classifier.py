# -*- coding: utf-8 -*-
"""image_classifier.ipynb

"""# Import libraries"""

import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn

"""# Preprocess dataset"""

dir = '/opt/data/reyhanian/data/affectnet/train'
dir_valid = '/opt/data/reyhanian/data/affectnet/valid'
# find the class names so in prediction time we can map the predictions to the painters properly


transform = transforms.Compose([transforms.Resize(self.img_size),
                                            transforms.ToTensor()])

# load dataset
data_train = datasets.ImageFolder(dir, transform=transform)
data_val = datasets.ImageFolder(dir_valid, transform=transform)


"""# Create Model"""
# model = keras.applications.ResNet50(
#     include_top=False,
#     weights="imagenet",
#     input_shape=(256,256,3),
#     pooling='avg',
#     classes=7,
# )
# model.trainable = False
# x = keras.layers.Flatten()(model.output)
# x = keras.layers.Dense(256,activation='relu')(x)
# x = keras.layers.Dense(100,activation='relu')(x)
# x = keras.layers.Dense(7, activation='softmax')(x)
# model = keras.models.Model(model.input, x)
class ExpressionClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

lr = 2e-4

epochs = 30
checkpoint_cb = 5

def scheduler(epoch, lr):
     if epoch % 2:
         return lr * 0.1
     else:
         return lr

model = ExpressionClassifier()

