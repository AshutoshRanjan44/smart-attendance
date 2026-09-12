"""detect_align.py
Run face detection on the collected images and produce aligned/resized face images.
Usage:
  python detect_align.py --input ./data --output ./aligned
"""

import cv2
import os
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(img_path, out_path, face_cascade):
    img = cv2.imread(img_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    if len(faces) == 0:
        # if no face found, save resized whole image as fallback
        fallback = cv2.resize(img, (160,160))
        cv2.imwrite(out_path, fallback)
        return True
    # choose largest face
    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    # optionally add a small margin
    margin = int(0.2 * min(w,h))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)
    face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (160,160))
    cv2.imwrite(out_path, face)
    return True

def main(input_folder, output_folder):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    ensure_dir(output_folder)

    for student in sorted(os.listdir(input_folder)):
        student_in = os.path.join(input_folder, student)
        if not os.path.isdir(student_in):
            continue
        student_out = os.path.join(output_folder, student)
        ensure_dir(student_out)
        for fname in sorted(os.listdir(student_in)):
            if not fname.lower().endswith(('.jpg','.png','.jpeg')):
                continue
            in_path = os.path.join(student_in, fname)
            out_fname = fname
            out_path = os.path.join(student_out, out_fname)
            ok = process_image(in_path, out_path, face_cascade)
            if not ok:
                print("Failed:", in_path)

    print("Alignment finished. Aligned images saved to:", output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data", help="Input data folder (per-student folders)")
    parser.add_argument("--output", default="./aligned", help="Output folder for aligned faces")
    args = parser.parse_args()
    main(args.input, args.output)
