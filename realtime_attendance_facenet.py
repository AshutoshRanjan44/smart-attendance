"""realtime_attendance_facenet.py
Real-time webcam recognition using MTCNN + InceptionResnetV1 (facenet-pytorch)
and a trained SVM classifier (model_facenet/*).

Usage:
  python realtime_attendance_facenet.py --model model_facenet --threshold 0.6 --camera 0
"""

import os
import argparse
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import pickle

import torch
from torchvision import transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1

DB_PATH = "attendance.db"
EVIDENCE_DIR = "evidence"

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        timestamp TEXT,
        image_path TEXT,
        confidence REAL
    )
    """)
    conn.commit()
    conn.close()

def log_attendance(name, image_path, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO attendance (name, timestamp, image_path, confidence) VALUES (?,?,?,?)",
        (name, datetime.now().isoformat(timespec='seconds'), image_path, float(confidence))
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_facenet",
                        help="Folder containing classifier.pkl and label_encoder.pkl")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Probability threshold to accept prediction")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    args = parser.parse_args()

    clf_path = os.path.join(args.model, "classifier.pkl")
    le_path = os.path.join(args.model, "label_encoder.pkl")
    if not (os.path.exists(clf_path) and os.path.exists(le_path)):
        print("Model files not found in", args.model)
        raise SystemExit(1)

    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    # MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device)

    # Facenet model for embeddings
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Transform
    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5])
    ])

    ensure_db()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        raise SystemExit(1)

    recent_log = {}         # name -> last log time
    DUPLICATE_SECONDS = 30  # do not re-log same person within this many seconds

    print("Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert OpenCV BGR frame to RGB PIL image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Detect faces
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None:
            cv2.imshow("Attendance (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for box, prob in zip(boxes, probs):
            if box is None or prob is None:
                continue

            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop face from original frame
            face_rgb = rgb_frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if face_rgb.size == 0:
                continue

            face_pil = Image.fromarray(face_rgb)
            face_t = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = resnet(face_t).cpu().numpy()[0]

            # Predict using classifier
            probs_clf = clf.predict_proba([emb])[0]
            best_idx = int(np.argmax(probs_clf))
            best_prob = float(probs_clf[best_idx])
            pred_name = le.inverse_transform([best_idx])[0]

            if best_prob >= args.threshold:
                label = f"{pred_name} ({best_prob:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Unknown ({best_prob:.2f})"
                color = (0, 0, 255)

            # Draw box + label on BGR frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log attendance if confident and not recently logged
            if best_prob >= args.threshold:
                now = time.time()
                last = recent_log.get(pred_name, 0)
                if now - last > DUPLICATE_SECONDS:
                    date_folder = datetime.now().strftime("%Y-%m-%d")
                    dest_dir = os.path.join(EVIDENCE_DIR, date_folder)
                    os.makedirs(dest_dir, exist_ok=True)

                    timestamp = datetime.now().strftime("%H%M%S")
                    img_path = os.path.join(dest_dir, f"{pred_name}_{timestamp}.jpg")
                    cv2.imwrite(img_path, frame)

                    log_attendance(pred_name, img_path, best_prob)
                    recent_log[pred_name] = now
                    print(f"Logged {pred_name} @ {img_path} (conf={best_prob:.2f})")

        cv2.imshow("Attendance (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
