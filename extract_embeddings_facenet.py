"""extract_embeddings_facenet.py
Reads aligned/<student>/*.jpg images, computes face embeddings using facenet-pytorch
(InceptionResnetV1), and saves embeddings and labels to disk.

Usage:
  python extract_embeddings_facenet.py --input ./aligned --out embeddings_facenet.npz
"""

import os
import argparse
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

def collect_embeddings(input_folder, device):
    # Facenet model (pretrained on VGGFace2)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Transform: resize to 160x160, convert to tensor, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])       # -> [-1,1]
    ])

    embeddings = []
    labels = []

    for student in sorted(os.listdir(input_folder)):
        student_dir = os.path.join(input_folder, student)
        if not os.path.isdir(student_dir):
            continue

        for fname in sorted(os.listdir(student_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(student_dir, fname)
            try:
                img = Image.open(path).convert('RGB')
            except Exception as e:
                print("Failed to open:", path, "->", e)
                continue

            img_t = transform(img).unsqueeze(0).to(device)  # shape [1,3,160,160]

            with torch.no_grad():
                emb = model(img_t).cpu().numpy()[0]         # 512-d vector

            embeddings.append(emb)
            labels.append(student)

    return np.array(embeddings), np.array(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./aligned",
                        help="Aligned faces folder (per-student subfolders)")
    parser.add_argument("--out", default="embeddings_facenet.npz",
                        help="Output .npz file (embeddings+labels)")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    X, y = collect_embeddings(args.input, device)
    if len(X) == 0:
        print("No embeddings collected. Check your aligned/ folder.")
    else:
        np.savez_compressed(args.out, embeddings=X, labels=y)
        print(f"Saved embeddings: {args.out}  (samples={len(X)})")
