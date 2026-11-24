import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm

weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(weights=weights)

model.fc = nn.Identity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

print("Device:", device)

preprocess = weights.transforms()
print(preprocess)

# df
df = pd.read_csv("../inat_birds/dataset_slim.csv")

df["image_path"] = df["image_path"].apply(
    lambda path: os.path.join("../inat_birds", path)
)

class BirdImageDataset(Dataset):

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, idx

image_paths = df["image_path"].tolist()

dataset = BirdImageDataset(
    image_paths=image_paths,
    transform=preprocess
)

# dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

embeddings = np.zeros(
    (len(dataset), 2048),
    dtype=np.float32
)

with torch.inference_mode():
    for images, indices in tqdm(dataloader):

        images = images.to(device, non_blocking=True)

        features = model(images)

        embeddings[indices.numpy()] = features.cpu().numpy()


embedding_df = pd.DataFrame(
    embeddings,
    columns=[f"embedding_{i}" for i in range(2048)]
)

embedding_df.insert(0, "species_scientific", df["species_scientific"].values)
embedding_df.insert(0, "image_path", df["image_path"].values)

embedding_df.to_csv(
    "../inat_birds/resnet50_embedding.csv",
    index=False
)