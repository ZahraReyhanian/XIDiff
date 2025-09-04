from torch.utils.data import Dataset
import json
from PIL import Image
import torch
import random

class WrapperDataset(Dataset):
    """
    each sample {
        "id_img" : id image,
        "exp_img" : expression image,
        "target_label" : target label
        "exp_path"
        "id_path"
    }
    """
    def __init__(self, json_path, stage, transform=None, shuffle=False, seed=42):
        self.transform = transform
        self.stage = stage

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        if shuffle:
            random.seed(seed)
            random.shuffle(self.data)

        all_labels = sorted(set([d[2] for d in self.data]))
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def get_image(self, neutral_path, emotion_path):
        neutral_img = Image.open(neutral_path).convert('RGB')
        emotion_img = Image.open(emotion_path).convert('RGB')

        if self.transform:
            neutral_img = self.transform(neutral_img)
            emotion_img = self.transform(emotion_img)

        return neutral_img, emotion_img

    def __getitem__(self, idx):
        if self.stage == 'train':
            neutral_path, emotion_path, emotion_label = self.data[idx]
            label_idx = self.label_to_idx[emotion_label]

            neutral_img, emotion_img = self.get_image(neutral_path, emotion_path)

            return {
                "id_img": neutral_img, #identity image
                "target_label": label_idx, #target label
                "exp_img": emotion_img, #image for target label
                "exp_path": emotion_path,
                "id_path": neutral_path,
            }
        else:
            neutral_path, emotion_path, emotion_label, target_path = self.data[idx]
            label_idx = self.label_to_idx[emotion_label]

            neutral_img, emotion_img = self.get_image(neutral_path, emotion_path)

            return {
                "id_img": neutral_img,  # identity image
                "target_label": label_idx,  # target label
                "exp_img": emotion_img,  # image for target label
                "exp_path": emotion_path,
                "id_path": neutral_path,
                "target_path": target_path # image with id and expression that we want to generate (ground truth)
            }