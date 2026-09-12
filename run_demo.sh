#!/bin/bash
echo "Ensure you have a Python venv active."
echo "Install torch (CPU) and requirements as instructed in README."
echo "Capture images: python capture_images.py --name rahul --count 30 --output ./data"
echo "Align: python detect_align.py --input ./data --output ./aligned"
echo "Extract embeddings: python extract_embeddings_facenet.py --input ./aligned --out embeddings_facenet.npz"
echo "Train: python train_classifier.py --in embeddings_facenet.npz --out model_facenet"
echo "Run attendance: python realtime_attendance_facenet.py --model model_facenet --threshold 0.6 --camera 0"
