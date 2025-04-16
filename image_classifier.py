# -*- coding: utf-8 -*-
"""image_classifier.ipynb

"""# Import libraries"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


"""# Preprocess dataset"""

dir = '/opt/data/reyhanian/data/affectnet/train'
dir_valid = '/opt/data/reyhanian/data/affectnet/valid'
# find the class names so in prediction time we can map the predictions to the painters properly

img_size = 112
transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                            transforms.ToTensor()])

# load dataset
data_train = datasets.ImageFolder(dir, transform=transform)
data_val = datasets.ImageFolder(dir_valid, transform=transform)

train_loader = DataLoader(data_train, batch_size=32, shuffle=True)
val_loader = DataLoader(data_val, batch_size=32, shuffle=True)

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

def evaluate_metrics(model, data_loader, device, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # محاسبه‌ی معیارها
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    # گزارش کامل
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return accuracy, precision, recall, f1


model = ExpressionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")


class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise' ]

evaluate_metrics(model, val_loader, device, class_names)