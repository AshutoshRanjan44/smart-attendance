"""train_classifier.py
Loads embeddings.npz (from extract_embeddings_facenet.py), trains an SVM classifier,
and saves classifier.pkl and label_encoder.pkl.

Usage:
  python train_classifier.py --in embeddings_facenet.npz --out model_facenet
This will create model_facenet/classifier.pkl and model_facenet/label_encoder.pkl
"""

import os
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main(inp, out_dir):
    data = np.load(inp)
    X = data["embeddings"]
    y = data["labels"]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # simple train/test split for quick evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Saved classifier and label encoder to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="embeddings_facenet.npz", help="Input .npz with embeddings+labels")
    parser.add_argument("--out", dest="out_dir", default="model_facenet", help="Output folder to save model files")
    args = parser.parse_args()
    main(args.inp, args.out_dir)
