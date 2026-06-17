"""
tools/enroll_face.py
---------------------
Enroll a person into the face recognition database.

USAGE EXAMPLES:

  # Enroll from a photo
  python tools/enroll_face.py --name "John Doe" --id staff_001 --source data/john.jpg

  # Enroll from a video (extracts best face frame automatically)
  python tools/enroll_face.py --name "John Doe" --id staff_001 --source data/john_intro.mp4

  # Enroll from your webcam (takes a snapshot when you press SPACE)
  python tools/enroll_face.py --name "John Doe" --id staff_001 --source webcam

WHAT IT DOES:
  1. Extracts face embedding using DeepFace
  2. Saves embedding as data/enrolled_faces/{id}.npy
  3. Prints the YAML block to add to config/identities.yaml

NOTES:
  - For best results: good lighting, face looking directly at camera
  - Takes 3-5 photos from slightly different angles for more robust matching
  - Run once per person; the .npy file is reused at runtime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def check_deepface() -> bool:
    try:
        from deepface import DeepFace
        return True
    except ImportError:
        print("ERROR: DeepFace not installed.")
        print("Run: pip install deepface")
        return False


def extract_embedding(
    image: np.ndarray,
    model_name: str = "VGG-Face",
    detector: str = "opencv",
) -> np.ndarray:
    from deepface import DeepFace
    result = DeepFace.represent(
        img_path=image,
        model_name=model_name,
        detector_backend=detector,
        enforce_detection=True,
    )
    if not result:
        raise ValueError("No face detected in image")
    return np.array(result[0]["embedding"])


def enroll_from_image(
    source_path: str,
    person_id: str,
    person_name: str,
    save_dir: str = "data/enrolled_faces",
    model: str = "VGG-Face",
) -> bool:
    """Enroll from a photo file."""
    image = cv2.imread(source_path)
    if image is None:
        print(f"Error: Cannot read image: {source_path}")
        return False

    print(f"Extracting face embedding for {person_name}...")
    try:
        embedding = extract_embedding(image, model_name=model)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the image has a clearly visible face.")
        return False

    return _save_and_print(embedding, person_id, person_name, save_dir)


def enroll_from_video(
    video_path: str,
    person_id: str,
    person_name: str,
    save_dir: str = "data/enrolled_faces",
    model: str = "VGG-Face",
    max_frames: int = 30,
) -> bool:
    """
    Enroll from a video — tries multiple frames and picks the one
    with the highest face detection confidence.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames)

    best_embedding = None
    best_score = -1
    frames_tried = 0

    print(f"Scanning {min(max_frames, total_frames)} frames for best face...")
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            from deepface import DeepFace
            result = DeepFace.represent(
                img_path=frame,
                model_name=model,
                detector_backend="opencv",
                enforce_detection=True,
            )
            if result:
                # Use face area as proxy for "best" (larger face = clearer)
                facial_area = result[0].get("facial_area", {})
                score = facial_area.get("w", 0) * facial_area.get("h", 0)
                if score > best_score:
                    best_score = score
                    best_embedding = np.array(result[0]["embedding"])
                    print(f"  Frame {i}: face area = {score}px² ✓")
                frames_tried += 1
        except Exception:
            pass

    cap.release()

    if best_embedding is None:
        print(f"Error: No face found in any of {frames_tried} sampled frames.")
        return False

    print(f"Best face found (area={best_score}px²)")
    return _save_and_print(best_embedding, person_id, person_name, save_dir)


def enroll_from_webcam(
    person_id: str,
    person_name: str,
    save_dir: str = "data/enrolled_faces",
    model: str = "VGG-Face",
    device: int = 0,
) -> bool:
    """Capture from webcam — press SPACE to take snapshot."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam {device}")
        return False

    print(f"\nWebcam open. Press SPACE to capture, ESC to cancel.")
    print("Position your face clearly in the frame.")

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.ellipse(display, (cx, cy), (120, 160), 0, 0, 360, (0, 255, 0), 2)
        cv2.putText(display, "Align face in oval. Press SPACE to capture.",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(f"Enroll: {person_name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            captured_frame = frame.copy()
            print("Snapshot taken!")
            break
        elif key == 27:
            print("Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        return False

    print("Extracting face embedding...")
    try:
        embedding = extract_embedding(captured_frame, model_name=model)
    except Exception as e:
        print(f"Error: {e}")
        return False

    return _save_and_print(embedding, person_id, person_name, save_dir)


def _save_and_print(
    embedding: np.ndarray,
    person_id: str,
    person_name: str,
    save_dir: str,
) -> bool:
    save_path = Path(save_dir) / f"{person_id}.npy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(save_path), embedding)

    print(f"\n✓ Embedding saved: {save_path}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm:  {np.linalg.norm(embedding):.4f}")

    print("\n" + "="*60)
    print("ADD THIS TO config/identities.yaml (under 'persons:'):")
    print("="*60)
    print(f"""
  - id: {person_id}
    name: "{person_name}"
    role: staff
    embedding_path: "{save_path}"
    allowed_zones: []       # fill in zone labels
    restricted_zones: []    # fill in restricted zone labels
    use_cases: [housekeeping_validation, identity_restriction]
    active: true
""")
    return True


def main():
    parser = argparse.ArgumentParser(description="Enroll a face for Vision AI identity recognition")
    parser.add_argument("--name",     required=True, help="Person's full name")
    parser.add_argument("--id",       required=True, dest="person_id", help="Unique person ID (e.g. staff_001)")
    parser.add_argument("--source",   required=True,
                        help="Path to image/video file, or 'webcam' for webcam capture")
    parser.add_argument("--save-dir", default="data/enrolled_faces", help="Where to save embeddings")
    parser.add_argument("--model",    default="VGG-Face",
                        choices=["VGG-Face", "Facenet", "ArcFace", "Facenet512"],
                        help="DeepFace model to use (must match face_recognizer setting)")
    args = parser.parse_args()

    if not check_deepface():
        sys.exit(1)

    source = args.source
    success = False

    if source == "webcam":
        success = enroll_from_webcam(args.person_id, args.name, args.save_dir, args.model)
    elif source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        success = enroll_from_video(source, args.person_id, args.name, args.save_dir, args.model)
    else:
        success = enroll_from_image(source, args.person_id, args.name, args.save_dir, args.model)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
