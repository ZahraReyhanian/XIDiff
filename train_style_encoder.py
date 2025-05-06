import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from datamodules.face_datamodule import FaceDataModule
from recognition.external_mapping import ExternalMappingV4Dropout
from recognition.recognition_helper import RecognitionModel, make_recognition_model

import warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

class StyleEncoderContrastive(nn.Module):
    def __init__(self, external_mapping, z_dim=64, num_classes=7):
        super(StyleEncoderContrastive, self).__init__()
        self.external_mapping = external_mapping

        self.classifier = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        z = self.external_mapping(features)  # z: (B, seq_len, z_dim)
        z = z.mean(dim=1)  # Global average pooling → (B, z_dim)

        logits = self.classifier(z)
        return logits, z


# Example usage:
if __name__ == "__main__":

    with open('config/config.json') as f:
        cfg = json.load(f)

    with open('config/general.json') as f:
        general_cfg = json.load(f)

    root = cfg['root']

    dirc = root + cfg['style_ckpt_path']

    if not os.path.exists(dirc):
        os.makedirs(dirc)

    PATH = dirc + '/external_mapping_model.pth'

    # external_mapping
    external_mapping = ExternalMappingV4Dropout(return_spatial=[2], out_size=(16, 16), out_channel=64)
    model = StyleEncoderContrastive(external_mapping)

    # recognition model, it's frozen
    recognition = general_cfg['recognition']
    recognition['ckpt_path'] = os.path.join(root, recognition['ckpt_path'])
    recognition['center_path'] = os.path.join(root, recognition['center_path'])
    recognition_model: RecognitionModel = make_recognition_model(recognition, root, enable_training=False)

    dataset_path = os.path.join(root, cfg["dataset_path"])
    transforms_setting = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    datamodule = FaceDataModule(dataset_path=dataset_path,
                                img_size=(cfg["image_size"], cfg["image_size"]),
                                batch_size=8, transforms_setting=transforms_setting)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    mse_loss_fn = nn.MSELoss()
    loss_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for batch in train_dataloader:
        _, spatial = recognition_model(batch["exp_img"])

        logits, embedding = model(spatial)  # خروجی (B, 7), (B, z_dim)
        labels = batch['target_label']  # لیبل 7 کلاسه

        ce_loss = F.cross_entropy(logits, labels)
        one_hot = F.one_hot(labels, num_classes=7).float().to(embedding.device)
        padded_one_hot = F.pad(one_hot, (0, embedding.shape[1] - one_hot.shape[1]))
        mse_loss = mse_loss_fn(embedding, padded_one_hot)
        loss = ce_loss + 0.1 * mse_loss
        loss.backward()

        loss_history.append(loss.item())
        print("Loss:", loss.item())

    torch.save(external_mapping.state_dict(), PATH)

    # === validation ===
    with torch.no_grad():
        for batch in val_dataloader:
            _, spatial = recognition_model(batch["exp_img"])
            logits, embedding = model(spatial)
            labels = batch['target_label']

            ce_loss = F.cross_entropy(logits, labels)
            val_loss = ce_loss.item()
            val_loss_history.append(val_loss)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean().item()
            val_acc_history.append(acc)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), PATH.replace(".pth", "_best.pth"))
                print(f"New best model saved with val_loss = {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # === loss , accuracy plots ===
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss over Batches')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(val_acc_history, label='Validation Accuracy', color='green')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Batches')
    plt.legend()
    plt.grid(True)
    plt.savefig('Accuracy.png')
    plt.close()

    # === t-SNE visualization ===
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            _, spatial = recognition_model(batch["exp_img"])
            _, embedding = model(spatial)
            all_embeddings.append(embedding.cpu())
            all_labels.append(batch['target_label'].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    tsne = TSNE(n_components=2, perplexity=6, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i in range(7):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.legend()
    plt.title('t-SNE visualization of expression embeddings')
    plt.savefig('tSNE.png')
    plt.close()
