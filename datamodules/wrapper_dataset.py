# paired_dataset.py
import random
from torch.utils.data import Dataset

class WrapperDataset(Dataset):
    """
    each sample {
        "id_img" : id image,
        "src_label" : source label,
        "exp_img" : expression image,
        "target_label" : target label
    }
    """
    def __init__(self, base_dataset):
        super().__init__()
        self.base = base_dataset                # datasets.ImageFolder
        self.num_classes = len(self.base.classes)

        # class_id
        self.class_to_indices = {i: [] for i in range(self.num_classes)}
        for idx, (_, label) in enumerate(self.base.samples):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        id_img, src_label = self.base[idx]

        # ---------- choice target_label ≠ src_label ----------
        # all classes except src_label
        candidate_labels = list(range(self.num_classes))
        candidate_labels.remove(src_label)
        target_label = random.choice(candidate_labels)

        target_idx = random.choice(self.class_to_indices[target_label])
        exp_img, _ = self.base[target_idx]

        return {
            "id_img": id_img,
            "src_label": src_label,
            "target_label": target_label,
            "exp_img": exp_img
        }
