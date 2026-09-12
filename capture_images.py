"""capture_images.py
Usage:
  python capture_images.py --name "Rahul" --count 30 --output ./data

Press:
  c  -> capture a face image (when a face is detected)
  a  -> toggle auto-capture mode (captures automatically when face detected)
  q  -> quit
"""

import cv2
import os
import argparse
import time

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(name, count, output_folder, camera_index=0):
    # Use OpenCV's Haar cascade for quick face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    student_folder = os.path.join(output_folder, name)
    ensure_dir(student_folder)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    idx = len([f for f in os.listdir(student_folder) if f.lower().endswith(('.jpg','.png'))])
    print(f"Starting capture for '{name}'. Already {idx} images found in folder.")
    auto_mode = False
    last_capture_time = 0
    auto_interval = 0.8  # seconds between auto captures

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: empty frame.")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detectMultiScale params can be tuned
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        # draw boxes
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        info = f"Images: {idx}/{count}  |  Press 'c' to capture, 'a' auto:{auto_mode}, 'q' quit"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Capture (press c/a/q)", frame)
        key = cv2.waitKey(1) & 0xFF

        if auto_mode and len(faces)>0 and (time.time() - last_capture_time) > auto_interval:
            # auto-capture the largest face
            faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
            x,y,w,h = faces_sorted[0]
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (160,160))
            save_path = os.path.join(student_folder, f"{name}_{idx:03d}.jpg")
            cv2.imwrite(save_path, face_img)
            idx += 1
            last_capture_time = time.time()
            print(f"[AUTO] Saved {save_path}")

        if key == ord('c'):
            if len(faces) == 0:
                print("No face detected — move closer to the camera and try again.")
            else:
                # pick largest face
                faces_sorted = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x,y,w,h = faces_sorted[0]
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (160,160))
                save_path = os.path.join(student_folder, f"{name}_{idx:03d}.jpg")
                cv2.imwrite(save_path, face_img)
                idx += 1
                print(f"Saved {save_path}")

        if key == ord('a'):
            auto_mode = not auto_mode
            print("Auto-capture:", auto_mode)

        if key == ord('q'):
            print("Quitting capture.")
            break

        if idx >= count:
            print(f"Reached target count ({count}).")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Student name (no spaces recommended)")
    parser.add_argument("--count", type=int, default=30, help="Number of images to collect")
    parser.add_argument("--output", default="./data", help="Output folder for collected images")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    main(args.name, args.count, args.output, args.camera)
