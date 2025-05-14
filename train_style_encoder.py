import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch import flatten
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from datamodules.face_datamodule import FaceDataModule
from recognition.external_mapping import make_external_mapping
from recognition.recognition_helper import RecognitionModel, make_recognition_model


class StyleEncoderContrastive(nn.Module):
    def __init__(self, external_mapping, channels=512, features=26,  num_classes=7):
        # features is K*k +1 which here k is 5
        super(StyleEncoderContrastive, self).__init__()
        self.external_mapping = external_mapping
        self.channels = channels
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(channels*features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        # features[0] shape: torch.Size([8, 64, 56, 56])
        # print("feature[0] shape: ", features[0].shape)
        z = self.external_mapping(features)  # spatial: (B, C, H, W)

        # print("z shape", z.shape)
        # z shape: torch.Size([8, 26, 512])
        z = z.reshape(-1, self.channels*self.features)
        # print("z shape", z.shape)

        logits = self.classifier(z)
        return logits, z


# Example usage:
if __name__ == "__main__":
    # load configs and initialize hyper parameter
    with open('config/config.json') as f:
        cfg = json.load(f)

    with open('config/general.json') as f:
        general_cfg = json.load(f)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    root = cfg['root']
    dirc = root + cfg['style_ckpt_path']
    if not os.path.exists(dirc):
        os.makedirs(dirc)

    PATH = dirc + '/external_mapping_model.pth'
    PATH_model = dirc + '/style_model.pth'
    batch_size = 8
    channels = general_cfg['external_mapping']["out_channel"]
    dim = general_cfg['external_mapping']["spatial_dim"]
    features = dim*dim + 1
    num_classes = cfg['num_classes']
    lr = 2e-3
    num_epochs = 50
    step_size = 5
    gamma = 0.5

    # initialize model and style encoder
    recognition = general_cfg['recognition']
    recognition['ckpt_path'] = os.path.join(root, recognition['ckpt_path'])
    recognition['center_path'] = os.path.join(root, recognition['center_path'])
    recognition_model: RecognitionModel = make_recognition_model(recognition, root, enable_training=False).to(device)

    external_mapping = make_external_mapping(general_cfg['external_mapping'], general_cfg['unet_config']).to(device)
    model = StyleEncoderContrastive(external_mapping,
                                    channels=channels,
                                    features=features,
                                    num_classes=num_classes).to(device)

    # Load dataset
    dataset_path = os.path.join(root, cfg["dataset_path"])
    # transforms_setting = transforms.Compose([
    #     transforms.Resize((cfg["image_size"], cfg["image_size"])),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    datamodule = FaceDataModule(dataset_path=dataset_path,
                                batch_size=batch_size)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # set an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_history = []
    val_loss_history = []
    val_acc_history = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Train model
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        mean_train_loss = 0
        corrects = 0
        model.train()
        train_pbar = tqdm(train_dataloader, desc="Training", ncols=100)
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            _, spatial = recognition_model(batch["exp_img"])
            logits, _ = model(spatial)
            labels = F.one_hot(batch['target_label'], num_classes=num_classes).float()

            ce_loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            train_loss = ce_loss.item()
            mean_train_loss += train_loss

            preds = torch.argmax(logits, dim=1)
            corrects += (preds == batch['target_label']).float().sum()
            train_pbar.set_postfix({"Loss": f"{train_loss:.4f}", "lr": optimizer.param_groups[0]["lr"]})

        mean_train_loss = mean_train_loss / len(train_dataloader)
        loss_history.append(mean_train_loss)
        print("Mean of Train loss:", mean_train_loss)

        accuracy = 100 * corrects / len(datamodule.data_train)
        print("Accuracy = {}".format(accuracy))
        if epoch < 30:
            scheduler.step()

        # Eval model
        model.eval()
        mean_val_loss = 0
        corrects = 0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc="Validation", ncols=100)
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, spatial = recognition_model(batch["exp_img"])
                logits, _ = model(spatial)
                labels = F.one_hot(batch['target_label'], num_classes=num_classes).float()

                ce_loss = F.cross_entropy(logits, labels)
                val_loss = ce_loss.item()
                mean_val_loss += val_loss

                preds = torch.argmax(logits, dim=1)
                corrects += (preds == batch['target_label']).float().sum().item()


                val_pbar.set_postfix({"ValLoss": f"{val_loss:.4f}"})

            mean_val_loss = mean_val_loss/len(val_dataloader)
            val_loss_history.append(mean_val_loss)
            print("Mean of validation loss:", mean_val_loss)

            accuracy = 100 * corrects / len(datamodule.data_val)
            print("Accuracy = {}".format(accuracy))
            val_acc_history.append(accuracy)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(external_mapping.state_dict(), PATH.replace(".pth", "_best.pth"))
            torch.save(model.state_dict(), PATH_model)


    # === loss , accuracy plots ===
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(val_acc_history, label='Validation Accuracy', color='green')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Accuracy.png')
    plt.close()

    # === t-SNE visualization ===
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, spatial = recognition_model(batch["exp_img"])
            _, embedding = model(spatial)
            all_embeddings.append(embedding.cpu())
            all_labels.append(batch['target_label'].cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i in range(7):
        idx = labels == i
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {i}', alpha=0.7)
    plt.legend()
    plt.title('t-SNE visualization of expression embeddings')
    plt.savefig('tSNE.png')
    plt.close()
