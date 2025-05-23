import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch import flatten
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

from datamodules.face_datamodule import FaceDataModule
from recognition.external_mapping import make_external_mapping
from recognition.recognition_helper import RecognitionModel, make_recognition_model

continue_training = False

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class StyleEncoderContrastive(nn.Module):
    def __init__(self, external_mapping, channels=512, features=26, num_classes=7):
        super(StyleEncoderContrastive, self).__init__()
        self.external_mapping = external_mapping
        self.channels = channels
        self.features = features

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # convert (B, C, T) to (B, C)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(features * features),
            nn.Linear(features * features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, features):
        z = self.external_mapping(features)  # shape: (B, 26, 512)
        z_t = z.transpose(1, 2)  # (B, 512, 26)
        z_pooled = self.global_pool(z_t).squeeze(-1)  # (B, 512)

        z_out = torch.bmm(z, z_pooled.unsqueeze(2)).squeeze(-1)  # (B, 26)
        z_out = torch.bmm(z_out.unsqueeze(2), z_out.unsqueeze(1))  # (B, 26, 26)
        z_out = z_out.view(z_out.size(0), -1)  # Flatten to (B, 676)

        logits = self.classifier(z_out)
        return logits, z_out


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

    PATH_model = dirc + '/style_model.pth'
    channels = general_cfg['external_mapping']["out_channel"]
    dim = general_cfg['external_mapping']["spatial_dim"]
    features = dim*dim + 1
    num_classes = cfg['num_classes']
    batch_size = 16
    lr = 1e-3
    num_epochs = 60
    patience = 5
    factor = 0.5
    best_val_loss = float('inf')
    s_epoch = 0

    loss_history = []
    val_loss_history = []
    acc_history = []
    val_acc_history = []

    # initialize model and style encoder
    recognition = general_cfg['recognition']
    recognition['ckpt_path'] = os.path.join(root, recognition['ckpt_path'])
    recognition['center_path'] = os.path.join(root, recognition['center_path'])
    recognition_model: RecognitionModel = make_recognition_model(recognition, root, enable_training=False).to(device)

    external_mapping = make_external_mapping(general_cfg['external_mapping'], general_cfg['unet_config']).to(device)
    model = StyleEncoderContrastive(external_mapping,
                                    channels=channels,
                                    features=features,
                                    num_classes=num_classes,
                                    # batch_size=batch_size
                                    ).to(device)

    # set an optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=10, delta=0.01)

    if continue_training:
        print('loading model from:', PATH_model)
        checkpoint = torch.load(PATH_model, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        s_epoch = checkpoint['epoch']
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        val_acc_history = checkpoint['val_acc_history']
        acc_history = checkpoint['acc_history']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = checkpoint['scheduler']

    # Load dataset
    dataset_path = os.path.join(root, cfg["dataset_path"])

    datamodule = FaceDataModule(dataset_path=dataset_path,
                                batch_size=batch_size)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    print(model)

    for epoch in range(s_epoch, num_epochs):
        # Train model
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        mean_train_loss = 0
        corrects = 0
        model.train()
        train_pbar = tqdm(train_dataloader, desc="Training", ncols=100)
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            _, spatial = recognition_model(batch["id_img"])
            logits, _ = model(spatial)
            labels = F.one_hot(batch['src_label'], num_classes=num_classes).float()

            ce_loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()

            train_loss = ce_loss.item()
            mean_train_loss += train_loss

            preds = torch.argmax(logits, dim=1)
            corrects += (preds == batch['src_label']).float().sum().item()
            train_pbar.set_postfix({"Loss": f"{train_loss:.4f}", "lr": optimizer.param_groups[0]["lr"]})

        mean_train_loss = mean_train_loss / len(train_dataloader)
        loss_history.append(mean_train_loss)
        print("Mean of Train loss:", mean_train_loss)

        accuracy = 100 * corrects / (len(train_dataloader)*batch_size)
        print("Accuracy = {}".format(accuracy))
        acc_history.append(accuracy)

        # Eval model
        model.eval()
        mean_val_loss = 0
        corrects = 0

        with torch.no_grad():
            val_pbar = tqdm(val_dataloader, desc="Validation", ncols=100)
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, spatial = recognition_model(batch["id_img"])
                logits, _ = model(spatial)
                labels = F.one_hot(batch['src_label'], num_classes=num_classes).float()

                ce_loss = F.cross_entropy(logits, labels)
                val_loss = ce_loss.item()
                mean_val_loss += val_loss

                preds = torch.argmax(logits, dim=1)
                corrects += (preds == batch['src_label']).float().sum().item()


                val_pbar.set_postfix({"ValLoss": f"{val_loss:.4f}"})

            mean_val_loss = mean_val_loss/len(val_dataloader)
            val_loss_history.append(mean_val_loss)
            print("Mean of validation loss:", mean_val_loss)

            accuracy = 100 * corrects / (len(val_dataloader)*batch_size)
            print("Accuracy = {}".format(accuracy))
            val_acc_history.append(accuracy)

        scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(external_mapping.state_dict(), f"{dirc}/external_mapping_{epoch}_best.pth")

        if epoch%5==0:
            torch.save(external_mapping.state_dict(), f"{dirc}/external_mapping_{epoch}_epoch.pth")

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss_history': loss_history,
                    'val_loss_history': val_loss_history,
                    'acc_history': acc_history,
                    'val_acc_history': val_acc_history,
                    'scheduler': scheduler
                    }, PATH_model)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.load_best_model(model)

    # Final evaluation on the test set
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, spatial = recognition_model(batch["id_img"])
            logits, _ = model(spatial)

            preds = torch.argmax(logits, dim=1)

            y_true+=batch['src_label'].cpu()
            y_pred+=preds.cpu()

    print(classification_report(y_true, y_pred))

    # === loss , accuracy plots ===
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(val_acc_history, label='Validation Accuracy', color='green')
    plt.plot(acc_history, label='Train Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('Accuracy.png')
    plt.close()

    # === TODO t-SNE visualization ===
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, spatial = recognition_model(batch["id_img"])
            _, embedding = model(spatial)
            all_embeddings.append(embedding.cpu())
            all_labels.append(batch['src_label'].cpu())

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
