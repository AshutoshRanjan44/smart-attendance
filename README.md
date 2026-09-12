# Smart AI Attendance System (Facenet pipeline)

**Contents**
- capture_images.py         : Capture labeled face images from webcam (per-student folders)
- detect_align.py          : Run face detection and produce aligned faces (160x160)
- extract_embeddings_facenet.py : Produce 512-d embeddings using facenet-pytorch
- train_classifier.py      : Train SVM classifier on embeddings
- realtime_attendance_facenet.py : Real-time webcam recognition + SQLite logging (Facenet)
- app.py                   : Minimal Flask dashboard to view/export attendance and serve evidence images
- requirements.txt         : Python dependencies (facenet-pytorch + torch)
- .gitignore               : common ignores
- run_demo.sh              : Simple run helper (Linux/macOS)

## Quick start (CPU)
1. Create a Python venv:
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. Install dependencies (CPU):
   pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt

3. Capture images (example for student rahul, 30 images):
   python capture_images.py --name rahul --count 30 --output ./data

4. Align detected faces:
   python detect_align.py --input ./data --output ./aligned

5. Extract embeddings (Facenet):
   python extract_embeddings_facenet.py --input ./aligned --out embeddings_facenet.npz

6. Train classifier:
   python train_classifier.py --in embeddings_facenet.npz --out model_facenet

7. Run realtime attendance:
   python realtime_attendance_facenet.py --model model_facenet --threshold 0.6 --camera 0

8. Run dashboard (in separate terminal):
   export FLASK_APP=app.py
   flask run
   # Open http://127.0.0.1:5000

## Notes
- If you have trouble installing `dlib` or `face_recognition`, this pipeline uses `facenet-pytorch` (no dlib).
- For better accuracy, collect 20-50 images per student across multiple sessions (lighting/pose).
- The evidence images are saved under `evidence/YYYY-MM-DD/` and can be viewed from the dashboard.
